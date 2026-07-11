# TODO: figure out how to NOT produce MSA for a target chain
# Note we use a vanilla ODE sampler for the structure module by default!
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from protenix.backend import load_model as backend_load_model
from pathlib import Path
from jaxtyping import Array, Float, PyTree

from protenix.data.template import ChainInput, featurize
from protenix.data.constants import PRO_STD_RESIDUES


from mosaic.losses.protenix import (
    MultiSampleProtenixLoss,
    biotite_array_to_gemmi_struct,
    get_trunk_state,
    protenix_forward_from_trunk,
    set_binder_sequence,
)
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.structure_prediction import (
    PolymerType,
    StructurePrediction,
    StructurePredictionModel,
    TargetChain,
)


_PROTEIN_UNK_IDX = PRO_STD_RESIDUES["UNK"]
def _apply_partial_template_mask(features_dict: dict, chains: list[TargetChain]) -> dict:
    '''
    Protenix does not natively support partial templating, this function manually modifies the protenix features
    setting to zero the influence of template positions specified by the TargetChain.template_mask.
    '''
    # concatenate masks for all chains
    keep_segments = []
    for c in chains:
        n = len(c.sequence)
        if c.template_chain is not None and c.template_mask is not None:
            m = np.asarray(c.template_mask, dtype=bool)
            if m.shape != (n,):
                raise ValueError(
                    f"template_mask shape {m.shape} does not match sequence length {n}"
                )
            keep_segments.append(m)
        else:
            keep_segments.append(np.ones(n, dtype=bool))
    keep = np.concatenate(keep_segments, axis=0)
    # if no position is masked
    if keep.all():
        return features_dict
    # build and apply 2D mask to template features
    pair_keep = (keep[:, None] & keep[None, :]).astype(np.float32)
    features_dict["template_pseudo_beta_mask"]    = features_dict["template_pseudo_beta_mask"]    * pair_keep[None]
    features_dict["template_backbone_frame_mask"] = features_dict["template_backbone_frame_mask"] * pair_keep[None]
    features_dict["template_distogram"]           = features_dict["template_distogram"]           * pair_keep[None, :, :, None]
    features_dict["template_unit_vector"]         = features_dict["template_unit_vector"]         * pair_keep[None, :, :, None]
    # aatype to UNK in masked positions
    aatype = np.asarray(features_dict["template_aatype"]).copy()
    aatype[:, ~keep] = _PROTEIN_UNK_IDX
    features_dict["template_aatype"] = aatype
    return features_dict


def load_model(name="protenix_mini_default_v0.5.0", bf16=False):
    jax_model = backend_load_model(name, bf16=bf16)
    # set gamma0, step_scale_eta, and N_steps to match the vanilla ODE sampler settings
    jax_model = eqx.tree_at(lambda m: (m.gamma0, m.step_scale_eta, m.noise_scale_lambda, m.N_steps), jax_model, (0.0, 1.0, 1.0, 20))

    return jax_model


class Protenix(StructurePredictionModel):
    protenix: eqx.Module
    default_sample_steps: int
    name: str

    def target_only_features(self, chains: list[TargetChain]):
        for c in chains:
            if c.polymer_type != PolymerType.PROTEIN:
                assert False, (
                    "Protenix interface only supports Protein chains. Manually build features for more complex targets. "
                )

        features_dict, atom_array, _ = featurize(
            [
                ChainInput(
                    sequence=c.sequence,
                    compute_msa=c.use_msa,
                    template=c.template_chain,
                )
                for c in chains
            ]
        )
        features_dict = _apply_partial_template_mask(features_dict, chains)

        return features_dict, atom_array

    def binder_features(self, binder_length, chains: list[TargetChain], binder_chain: TargetChain | None=None):
        if binder_chain is None:
            sequence = "X" * binder_length
            binder_chain = TargetChain(sequence=sequence, use_msa=False)
        else:
            if binder_length != len(binder_chain.sequence):
                raise ValueError(f"Specified init_sequence length ({len(binder_chain.sequence)}) does not match specified binder_length ({binder_length})")
        return self.target_only_features([binder_chain] + chains)

    def build_loss(
        self,
        *,
        loss,
        features,
        recycling_steps=1,
        sampling_steps=None,
        gradient_steps=None,
        name: str | None = None,
        initial_recycling_state=None,
        features_to_log: list[str] | None = None,
        restype_scale: float = 1.0,
        msa_fix: bool = False,
        confidence_stop_gradient: bool = False,
        diffusion_stop_gradient: bool = False,
        use_dropout: bool = False,
    ):
        return self.build_multisample_loss(
            loss=loss,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            gradient_steps=gradient_steps,
            name=name if name is not None else self.name,
            num_samples=1,
            initial_recycling_state=initial_recycling_state,
            features_to_log=features_to_log,
            restype_scale=restype_scale,
            msa_fix=msa_fix,
            confidence_stop_gradient=confidence_stop_gradient,
            diffusion_stop_gradient=diffusion_stop_gradient,
            use_dropout=use_dropout,
        )

    def build_multisample_loss(
        self,
        *,
        loss,
        features,
        recycling_steps=1,
        num_samples: int = 4,
        sampling_steps=None,
        gradient_steps=None,
        name: str | None = None,
        reduction=jnp.mean,
        initial_recycling_state=None,
        features_to_log: list[str] | None = None,
        restype_scale: float = 1.0,
        msa_fix: bool = False,
        confidence_stop_gradient: bool = False,
        diffusion_stop_gradient: bool = False,
        use_dropout: bool = False,
    ):
        if features_to_log is not None:
            not_found = [f for f in features_to_log if f not in features.keys()]
            if len(not_found) != 0:
                print(f"The following features are not registered in the current model: {not_found}")

        if sampling_steps is None:
            sampling_steps = self.default_sample_steps
        return MultiSampleProtenixLoss(
            model=self.protenix,
            features=features,
            loss=loss,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            backward_steps=gradient_steps,
            name=name if name is not None else self.name,
            num_samples=num_samples,
            reduction=reduction,
            initial_recycling_state=initial_recycling_state,
            features_to_log=features_to_log,
            restype_scale=restype_scale,
            msa_fix=msa_fix,
            confidence_stop_gradient=confidence_stop_gradient,
            diffusion_stop_gradient=diffusion_stop_gradient,
            use_dropout=use_dropout,
        )

    @eqx.filter_jit
    def model_output(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=1,
        sampling_steps=None,
        gradient_steps=None,
        initial_recycling_state=None,
        key,
    ):
        if sampling_steps is None:
            sampling_steps = self.default_sample_steps
        features = set_binder_sequence(PSSM, features) if PSSM is not None else features

        initial_embedding, trunk_state = get_trunk_state(
            model=self.protenix,
            features=features,
            initial_recycling_state=initial_recycling_state,
            recycling_steps=recycling_steps,
            key=key,
        )

        return protenix_forward_from_trunk(
            model=self.protenix,
            features=features,
            initial_embedding=initial_embedding,
            trunk_state=trunk_state,
            sampling_steps=sampling_steps,
            backward_steps=gradient_steps,
            key=key,
        )

    def predict(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        writer,
        recycling_steps=1,
        sampling_steps=None,
        initial_recycling_state=None,
        key,
    ):
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            initial_recycling_state=initial_recycling_state,
            key=key,
        )
        seq = PSSM if PSSM is not None else jnp.zeros((0, 20))
        iptm = -IPTMLoss()(seq, output, key=jax.random.key(0))[0]
        return StructurePrediction(
            st=biotite_array_to_gemmi_struct(writer, np.array(output.structure_coordinates[0])),
            plddt=output.plddt,
            pae=output.pae,
            iptm=iptm,
            model_output=output,
        )


def ProtenixMini(bf16=False):
    return Protenix(load_model(name="protenix_mini_default_v0.5.0", bf16=bf16), 2, name="ProtenixMini")


def ProtenixTiny(bf16=False):
    return Protenix(load_model(name="protenix_tiny_default_v0.5.0", bf16=bf16), 2, name="ProtenixTiny")


def ProtenixBase(bf16=False):
    return Protenix(load_model(name="protenix_base_default_v1.0.0", bf16=bf16), 20, name="ProtenixBase")


def Protenix2025(bf16=False):
    return Protenix(load_model(name="protenix_base_20250630_v1.0.0", bf16=bf16), 20, name="Protenix2025")

def ProtenixV2(bf16=False):
    return Protenix(load_model(name="protenix-v2", bf16=bf16), 20, name="ProtenixV2")
