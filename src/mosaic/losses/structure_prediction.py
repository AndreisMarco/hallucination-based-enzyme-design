import equinox as eqx
import jax
from jaxtyping import Float, Array, Int

from collections.abc import Callable
import jax.numpy as jnp
import numpy as np

from ..common import LossTerm

# 64 bins, 0.0–32.0 Å, shared across all model backends
PAE_BINS = np.arange(start=0.25, stop=32.0, step=0.5)


class StructureModelOutput(eqx.Module):
    distogram_logits: Float[Array, "N N Bins"]
    distogram_bins: Float[Array, "Bins"]
    plddt: Float[Array, "N"]
    pae: Float[Array, "N N"]
    pae_logits: Float[Array, "N N Bins"]
    pae_bins: Float[Array, "Bins"]
    structure_coordinates: Array  # all-atom coords, shape varies by model
    backbone_coordinates: Float[Array, "N 4 3"]
    full_sequence: Float[Array, "N 20"]
    asym_id: Float[Array, "N"]
    residue_idx: Int[Array, "N"]
    # Canonical atom37 view of the predicted heavy atoms (see
    # `mosaic.losses.atom37.ATOM37_NAMES`). Populated by every model wrapper.
    atom37_coords: Float[Array, "N 37 3"]
    atom37_mask: Float[Array, "N 37"]



def interaction_prediction_score(
    logits: jnp.ndarray,
    bin_centers: jnp.ndarray,
    asym_id: jnp.ndarray | None = None,
    pae_cutoff: float = 10.0,
) -> jnp.ndarray:
    probs = jax.nn.softmax(logits, axis=-1)
    pae = jnp.sum(probs * bin_centers, axis=-1)

    pair_mask = jnp.ones_like(pae, dtype=bool)
    pair_mask *= asym_id[:, None] != asym_id[None, :]

    # only include residue pairs below the pae_cutoff
    pair_mask *= pae < pae_cutoff
    n_residues = jnp.sum(pair_mask, axis=-1, keepdims=True)

    # Compute adjusted d_0(num_res) per residue  as defined by eqn. (15) in
    # Dunbrack, R., "What's wrong with AlphaFold’s ipTM score and how to fix it."
    # 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC11844409/
    d0 = 1.24 * (jnp.clip(n_residues, min=27) - 15) ** (1.0 / 3) - 1.8

    tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))
    predicted_tm_term = jnp.sum(probs * tm_per_bin, axis=-1)

    normed_residue_mask = pair_mask / (1e-8 + n_residues)
    per_alignment = jnp.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return per_alignment


def predicted_tm_score(
    logits: jnp.ndarray,
    bin_centers: jnp.ndarray,
    pair_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    num_res = logits.shape[0]
    # Clip num_res to avoid negative/undefined d0.
    clipped_num_res = max(num_res, 19)

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    # Convert logits to probs.
    probs = jax.nn.softmax(logits, axis=-1)

    # TM-Score term for every bin.
    tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))
    # E_distances tm(distance).
    predicted_tm_term = jnp.sum(probs * tm_per_bin, axis=-1)

    if pair_mask is None:
        pair_mask = jnp.ones(shape=(num_res, num_res), dtype=bool)

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask
    normed_residue_mask = pair_residue_weights / (
        1e-8 + jnp.sum(pair_residue_weights, axis=-1, keepdims=True)
    )
    per_alignment = jnp.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return per_alignment


def contact_cross_entropy(
    distogram_logits: Float[Array, "N N Bins"],
    contact_dist: float,
    bins: Float[Array, "Bins"],
) -> Float[Array, "... N N"]:
    """Compute partial entropy (under distogram) that D_ij < contact_dist."""
    assert bins.shape[-1] == distogram_logits.shape[-1]
    assert distogram_logits.ndim == 3

    distogram_logits = jax.nn.log_softmax(distogram_logits)

    mask = bins < contact_dist

    px_ = jax.nn.softmax(distogram_logits, axis=-1, where=mask)

    return (px_ * distogram_logits).sum(-1)


def contact_log_probability(
    distogram_logits: Float[Array, "... N N 64"],
    contact_dist: float,
    bins: Float[Array, "Bins"],
) -> Float[Array, "... N N"]:
    """Compute log probability (under distogram) that D_ij < contact_dist."""
    assert bins.shape[-1] == distogram_logits.shape[-1]
    assert distogram_logits.ndim == 3
    distogram_logits = jax.nn.log_softmax(distogram_logits)
    mask = bins < contact_dist
    return jax.nn.logsumexp(distogram_logits, where=mask, axis=-1)


class WithinBinderContact(LossTerm):
    max_contact_distance: float = 14.0
    min_sequence_separation: int = 8
    num_contacts_per_residue: int = 25
    name: str = "intra_contact"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        binder_len = sequence.shape[0]
        log_contact_intra = contact_cross_entropy(
            output.distogram_logits[:binder_len, :binder_len],
            self.max_contact_distance,
            bins=output.distogram_bins,
        )
        # only count binder-binder contacts with sequence sep > min_sequence_separation
        within_binder_mask = (
            jnp.abs(jnp.arange(binder_len)[:, None] - jnp.arange(binder_len)[None, :])
            > self.min_sequence_separation
        )
        # for each position in binder find positions most likely to make contact

        # JAX/XLO has a bizarre issue with top_k when used inside vmap _only_ when used on multiple GPUs.
        # so for now we sort instead of using top_k
        # binder_binder_max_p, _ = jax.vmap(
        #     lambda lcp: jax.lax.top_k(lcp, self.num_contacts_per_residue)
        # )(log_contact_intra + (1 - within_binder_mask) * -30)
        # average_log_prob = binder_binder_max_p.mean()

        sorted_log_probs = jnp.sort(
            log_contact_intra + (1 - within_binder_mask) * -30, descending=True, axis=-1
        )
        top_k_log_probs = sorted_log_probs[:, : self.num_contacts_per_residue]
        top_k_mean = top_k_log_probs.mean(axis=-1)

        average_log_prob = top_k_mean.mean()
        return -average_log_prob, {self.name: average_log_prob}


class ColabDesignContactLoss(LossTerm):
    cutoff: float = 14.0
    binary: bool = False
    seqsep: int = 9
    num: int = 2
    num_pos: float = float("inf")
    name: str = "con"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        binder_len = sequence.shape[0]
        dgram = output.distogram_logits[:binder_len, :binder_len]
        bins = output.distogram_bins

        bin_mask = (bins < self.cutoff).astype(jnp.float32)
        px = jax.nn.softmax(dgram)
        px_ = jax.nn.softmax(dgram - 1e7 * (1 - bin_mask))

        if self.binary:
            p = -jnp.log((bin_mask * px + 1e-8).sum(-1))
        else:
            p = -(px_ * jax.nn.log_softmax(dgram)).sum(-1)

        idx = jnp.arange(binder_len)
        seqsep_mask = jnp.abs(idx[:, None] - idx[None, :]) >= self.seqsep

        p_masked = jnp.where(seqsep_mask, p, jnp.inf)
        p_sorted = jnp.sort(p_masked, axis=-1)
        k = min(self.num, binder_len)
        k_mask = jnp.arange(p_sorted.shape[-1]) < k
        per_res = jnp.where(k_mask, p_sorted, 0.0).sum(-1) / jnp.maximum(k_mask.sum(), 1)

        if self.num_pos < binder_len:
            per_res_sorted = jnp.sort(per_res)
            n = int(self.num_pos)
            con = per_res_sorted[:n].mean()
        else:
            con = per_res.mean()

        return con, {self.name: con}


class BinderTargetContact(LossTerm):
    paratope_idx: list[int] | None = None
    paratope_size: int | None = None
    contact_distance: float = 20.0
    epitope_idx: list[int] | None = None
    name: str = "target_contact"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        binder_len = sequence.shape[0]
        log_contact_inter = contact_cross_entropy(
            output.distogram_logits[:binder_len, binder_len:],
            self.contact_distance,
            bins=output.distogram_bins,
        )
        if self.epitope_idx is not None:
            log_contact_inter = log_contact_inter[:, self.epitope_idx]

        # see above note about JAX/XLO issue with top_k inside vmap
        # binder_target_max_p = jax.vmap(lambda v: jax.lax.top_k(v, 3)[0])(
        #     log_contact_inter
        # ).mean(-1)
        sorted_log_probs = jnp.sort(log_contact_inter, descending=True, axis=-1)
        binder_target_max_p = sorted_log_probs[:, :3].mean(axis=-1)

        # log probability of contacting target for each position in binder
        if self.paratope_idx is not None:
            binder_target_max_p = binder_target_max_p[self.paratope_idx]
        if self.paratope_size is not None:
            binder_target_max_p = jax.lax.top_k(
                binder_target_max_p, self.paratope_size
            )[0]

        average_log_prob = binder_target_max_p.mean()
        return -average_log_prob, {self.name: average_log_prob}


class HelixLoss(LossTerm):
    max_distance: float = 6.0
    target_value: float = -2.0
    name: str = "helix"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        binder_len = sequence.shape[0]
        log_contact = contact_log_probability(
            output.distogram_logits[:binder_len, :binder_len],
            self.max_distance,
            bins=output.distogram_bins,
        )
        value = jnp.diagonal(log_contact, 3).mean()

        loss = jax.nn.elu(self.target_value - value)

        return loss, {self.name: loss}


class DistogramRadiusOfGyration(LossTerm):
    target_radius: float | None = None
    name: str = "radius_of_gyration"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        # TODO: Why RMSE instead of MAE?
        binder_len = sequence.shape[0]
        dgram_radius_of_gyration = jnp.sqrt(
            jnp.fill_diagonal(
                (
                    jax.nn.softmax(output.distogram_logits)[:binder_len, :binder_len]
                    * (output.distogram_bins[None, None, :] ** 2)
                ).sum(-1),  # expected squared distance
                0,
                inplace=False,
            ).mean()
            + 1e-8
        )

        rg_th = (
            2.38 * binder_len**0.365
            if self.target_radius is None
            else self.target_radius
        )
        return jax.nn.elu(dgram_radius_of_gyration - rg_th), {
            self.name: dgram_radius_of_gyration
        }


class MAERadiusOfGyration(LossTerm):
    target_radius: float | None = None
    name: str = "radius_of_gyration"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        binder_len = sequence.shape[0]

        dgram_radius_of_gyration = jnp.fill_diagonal(
            (
                jax.nn.softmax(output.distogram_logits)[:binder_len, :binder_len]
                * (output.distogram_bins[None, None, :])
            ).sum(-1),  # expected squared distance
            0,
            inplace=False,
        ).mean()

        rg_th = (
            2.38 * binder_len**0.365
            if self.target_radius is None
            else self.target_radius
        )
        return jax.nn.elu(dgram_radius_of_gyration - rg_th), {
            self.name: dgram_radius_of_gyration
        }


class DistogramCE(LossTerm):
    f: Float[Array, "... Bins"]
    name: str = "distogram_ce"
    l: float = -np.inf
    u: float = np.inf

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        binder_len = sequence.shape[0]
        # expand dims so self.f is broadcastable to network_output["pdistogram"] of size (N, N, Bins)
        f = jnp.expand_dims(self.f, [i for i in range(3 - self.f.ndim)])

        ce = -jnp.fill_diagonal(
            (
                jax.nn.log_softmax(output.distogram_logits)[:binder_len, :binder_len]
                * f
            ).sum(-1),
            0,
            inplace=False,
        ).mean()

        return ce.clip(self.l, self.u), {self.name: ce}


class PLDDTLoss(LossTerm):
    name: str = "plddt"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        binder_len = sequence.shape[0]
        plddt = output.plddt[:binder_len].mean()
        return -plddt, {self.name: plddt}


class WithinBinderPAE(LossTerm):
    name: str = "bb_pae"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        binder_len = sequence.shape[0]
        pae_within = jnp.fill_diagonal(
            output.pae[:binder_len, :binder_len], 0, inplace=False
        ).mean()
        return pae_within, {self.name: pae_within}


class BinderTargetPAE(LossTerm):
    name: str = "bt_pae"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        binder_len = sequence.shape[0]
        pae = output.pae[:binder_len, binder_len:].mean()
        return pae, {self.name: pae}


class TargetBinderPAE(LossTerm):
    name: str = "tb_pae"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        binder_len = sequence.shape[0]
        pae = output.pae[binder_len:, :binder_len].mean()
        return pae, {self.name: pae}


class IPTMLoss(LossTerm):
    name: str = "iptm"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        # binder - target iptm -- we override asym-id in the case of multi-chain targets
        N = output.full_sequence.shape[0]
        asym_id = jnp.concatenate(
            (jnp.zeros(sequence.shape[0]), jnp.ones(N - sequence.shape[0]))
        ).astype(jnp.int32)
        pair_mask = asym_id[:, None] != asym_id[None, :]
        iptm = predicted_tm_score(
            logits=output.pae_logits,
            bin_centers=output.pae_bins,
            pair_mask=pair_mask,
        ).max()
        return -iptm, {self.name: iptm}


class BinderTargetIPTM(LossTerm):
    name: str = "bt_iptm"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        # binder - target iptm -- we override asym-id in the case of multi-chain targets
        N = output.full_sequence.shape[0]
        asym_id = jnp.concatenate(
            (jnp.zeros(sequence.shape[0]), jnp.ones(N - sequence.shape[0]))
        ).astype(jnp.int32)
        pair_mask = asym_id[:, None] != asym_id[None, :]
        bt_iptm = predicted_tm_score(
            logits=output.pae_logits,
            bin_centers=output.pae_bins,
            pair_mask=pair_mask,
        )[: sequence.shape[0]].max()  # limit to binder index
        return -bt_iptm, {self.name: bt_iptm}


class BinderPTMLoss(LossTerm):
    name: str = "binder_ptm"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        binder_len = sequence.shape[0]
        num_res = output.full_sequence.shape[0]
        pair_mask = jnp.zeros(shape=(num_res, num_res), dtype=bool)
        pair_mask = pair_mask.at[:binder_len, :binder_len].set(True)
        ptm = predicted_tm_score(
            logits=output.pae_logits,
            bin_centers=output.pae_bins,
            pair_mask=pair_mask,
        ).max()
        return -ptm, {self.name: ptm}


class BinderTargetIPSAE(LossTerm):
    reduce: Callable = jnp.max
    name: str = "bt_ipsae"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        N = output.full_sequence.shape[0]
        binder_len = sequence.shape[0]
        # override asym-id in the case of multi-chain targets
        asym_id = jnp.concatenate(
            (jnp.zeros(binder_len), jnp.ones(N - binder_len))
        ).astype(jnp.int32)
        bt_ipsae = self.reduce(
            interaction_prediction_score(
                asym_id=asym_id,
                logits=output.pae_logits,
                bin_centers=output.pae_bins,
                pae_cutoff=10.0,
            )[:binder_len]
        )
        return -bt_ipsae, {self.name: bt_ipsae}


class TargetBinderIPSAE(LossTerm):
    reduce: Callable = jnp.max
    name: str = "tb_ipsae"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        N = output.full_sequence.shape[0]
        binder_len = sequence.shape[0]
        # override asym-id in the case of multi-chain targets
        asym_id = jnp.concatenate(
            (jnp.zeros(binder_len), jnp.ones(N - binder_len))
        ).astype(jnp.int32)
        tb_ipsae = self.reduce(
            interaction_prediction_score(
                asym_id=asym_id,
                logits=output.pae_logits,
                bin_centers=output.pae_bins,
                pae_cutoff=10.0,
            )[binder_len:]
        )
        return -tb_ipsae, {self.name: tb_ipsae}


class IPSAE_min(LossTerm):
    name: str = "ipsae_min"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        bt_ipsae = (
            -1
            * (
                BinderTargetIPSAE()(
                    sequence=sequence,
                    output=output,
                    key=key,
                )
            )[0]
        )
        tb_ipsae = (
            -1
            * (
                TargetBinderIPSAE()(
                    sequence=sequence,
                    output=output,
                    key=key,
                )
            )[0]
        )
        ipsae_min = jnp.minimum(bt_ipsae, tb_ipsae)

        return -ipsae_min, {self.name: ipsae_min}


class ActualRadiusOfGyration(LossTerm):
    target_radius: float
    name: str = "actual_rg"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        binder_len = sequence.shape[0]
        first_atom_coords = output.backbone_coordinates[:binder_len, 1]
        rg = jnp.sqrt(
            ((first_atom_coords - first_atom_coords.mean(0)) ** 2).sum(-1).mean()
        )

        return jax.nn.elu(rg - self.target_radius), {self.name: rg}


class pTMEnergy(LossTerm):
    name: str = "pTMEnergy"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        len_binder = sequence.shape[0]
        logits = output.pae_logits
        num_res = logits.shape[0]
        # Clip num_res to avoid negative/undefined d0.
        clipped_num_res = max(num_res, 19)

        d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

        pae_bin_centers = output.pae_bins

        g_d_b = 1.0 / (1 + jnp.square(pae_bin_centers) / jnp.square(d0))
        energy = jax.scipy.special.logsumexp(a=logits, b=g_d_b, axis=-1)
        # return negative mean over cross-chain pairs
        binder_target = energy[:len_binder, len_binder:].mean()
        target_binder = energy[len_binder:, :len_binder].mean()
        E = -(binder_target + target_binder) / 2
        return E, {self.name: E}
