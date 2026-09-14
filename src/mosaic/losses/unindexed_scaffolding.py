"""Unindexed scaffold: motif geometry without position assignment.

Stores M motif residues extracted from a PDB. Unlike IndexedScaffold,
positions in the design are unknown — an assignment matrix (N, M) maps
predicted positions to motif residues during optimization.

Losses receive assignment as a keyword argument via **kwargs, composing
normally with LinearCombination and indexed losses (which ignore it).
"""
from itertools import permutations

import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
import gemmi

from mosaic.common import TOKENS, LossTerm
from mosaic.losses.atom37 import ATOM37_INDEX
from mosaic.losses.indexed_scaffolding import IndexedScaffold
from mosaic.structure_prediction import StructureModelOutput
from mosaic.util import gram_schmidt, kabsch


class UnindexedScaffold:

    def __init__(self,
                 path_to_structure: str,
                 keep_intervals: str,
                 length: int):
        structure = gemmi.read_structure(str(path_to_structure))
        intervals = IndexedScaffold._parse_intervals(keep_intervals)

        all_a37, all_a37m, motif_seq, residue_labels = [], [], [], []
        for start, end in intervals:
            for chain in structure[0]:
                for res in chain.first_conformer():
                    if gemmi.find_tabulated_residue(res.name).is_amino_acid() \
                       and start <= res.seqid.num <= end:
                        a37, a37m, seq = IndexedScaffold._extract_residue(
                            structure, chain.name, res.seqid.num)
                        all_a37.append(a37)
                        all_a37m.append(a37m)
                        motif_seq.append(seq)
                        residue_labels.append(f"{chain.name}{res.seqid.num}")

        if not motif_seq:
            raise ValueError(f"No motif residues found in intervals: {keep_intervals}")

        self._atom37_coords = jnp.array(np.concatenate(all_a37, axis=0))  # (M, 37, 3)
        self._atom37_mask = jnp.array(np.concatenate(all_a37m, axis=0))   # (M, 37)
        self._motif_sequence = "".join(motif_seq)
        self._residue_labels = residue_labels
        self._design_length = length

    @property
    def n_motif(self) -> int:
        return len(self._motif_sequence)

    @property
    def sequence(self) -> str:
        return "X" * self._design_length

    @property
    def motif_sequence(self) -> str:
        return self._motif_sequence

    @property
    def atom37_coordinates(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self._atom37_coords, self._atom37_mask

    def __len__(self) -> int:
        return self._design_length

    def pssm(self) -> jnp.ndarray:
        """(M, 20) one-hot of motif residue identities."""
        aa_indices = jnp.array([TOKENS.index(aa) for aa in self._motif_sequence])
        return jax.nn.one_hot(aa_indices, num_classes=20)

    def backbone_coordinates(self) -> jnp.ndarray:
        bb_idx = jnp.array([ATOM37_INDEX[a] for a in ("N", "CA", "C", "O")])
        return self._atom37_coords[:, bb_idx, :]  # (M, 4, 3)

    def distogram(self) -> jnp.ndarray:
        ca = self._atom37_coords[:, ATOM37_INDEX["CA"], :]
        cb = self._atom37_coords[:, ATOM37_INDEX["CB"], :]
        has_cb = self._atom37_mask[:, ATOM37_INDEX["CB"]].astype(bool)
        pb = jnp.where(has_cb[:, None], cb, ca)
        diff = pb[:, None, :] - pb[None, :, :]
        return jnp.linalg.norm(diff, axis=-1)  # (M, M)

    def backbone_frames(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        bb = self.backbone_coordinates()
        R = gram_schmidt(v1=bb[:, 2, :] - bb[:, 1, :], v2=bb[:, 0, :] - bb[:, 1, :])
        return bb[:, 1, :], R  # (M, 3), (M, 3, 3)


class UnindexedRMSD(LossTerm):
    """RMSD through a soft assignment matrix.

    Soft-selects M predicted positions via einsum with the (N, M)
    assignment matrix, Kabsch-aligns to ground truth, returns RMSD.

    Modes ca_only and backbone work throughout optimization.
    Modes all_atom and side_chain only produce meaningful gradients
    once the assignment is near one-hot and amino acid identities
    are correct at motif positions.
    """
    _gt: jnp.ndarray
    _gt_ca: jnp.ndarray | None
    _atom_idx: jnp.ndarray | None
    _weights: jnp.ndarray | None
    _mode: str
    name: str = "unindexed_rmsd"

    def __init__(self, gt_coords, name: str = "unindexed_rmsd", mode: str = "ca_only",
                 weighted: bool = False, gt_atom37_mask=None):
        self.name = name
        self._mode = mode
        self._atom_idx = None
        self._weights = None
        self._gt_ca = None

        if mode in ("all_atom", "side_chain"):
            mask37 = gt_atom37_mask.copy()
            if mode == "side_chain":
                backbone_idx = jnp.array([0, 2, 4])  # zero N, C, O — keep CA
                mask37 = mask37.at[:, backbone_idx].set(0.0)
            flat_mask = mask37.reshape(-1)
            n_atoms = int(flat_mask.sum())
            self._atom_idx = jnp.where(flat_mask, size=n_atoms)[0]
            self._gt = gt_coords.reshape(-1, 3)[self._atom_idx]
            self._gt_ca = gt_coords[:, ATOM37_INDEX["CA"], :]
            if weighted:
                n_res = mask37.shape[0]
                atoms_per_res = mask37.sum(-1, keepdims=True)
                per_atom_w = jnp.where(mask37, 1.0 / (n_res * atoms_per_res + 1e-8), 0.0)
                self._weights = per_atom_w.reshape(-1)[self._atom_idx][..., None]
        elif mode == "ca_only":
            self._gt = gt_coords[:, 1, :]  # (M, 3)
        elif mode == "backbone":
            self._gt = gt_coords.reshape(-1, 3)  # (M*4, 3)
        else:
            raise ValueError(f"Unknown mode: {mode}. Available: ca_only, backbone, all_atom, side_chain")

        if weighted and self._weights is None:
            length = self._gt.shape[0]
            self._weights = (jnp.ones(length) / length)[..., None]

    @classmethod
    def from_scaffold(cls, scaffold: UnindexedScaffold, mode: str = "ca_only",
                      weighted: bool = False, name: str = "unindexed_rmsd"):
        if mode in ("all_atom", "side_chain"):
            coords, atom37_mask = scaffold.atom37_coordinates
            return cls(gt_coords=coords, name=name, mode=mode,
                       weighted=weighted, gt_atom37_mask=atom37_mask)
        return cls(gt_coords=scaffold.backbone_coordinates(), name=name,
                   mode=mode, weighted=weighted)

    def __call__(self, sequence, output: StructureModelOutput, key, assignment, **kwargs):
        N = assignment.shape[0]
        if self._mode in ("all_atom", "side_chain"):
            pred_a37 = jnp.einsum("nm,nad->mad", assignment, output.atom37_coords[:N])
            pred_ca = pred_a37[:, ATOM37_INDEX["CA"], :]
            R, t = kabsch(pred_ca, self._gt_ca, weights=self._weights)
            pred = pred_a37.reshape(-1, 3)[self._atom_idx]
        elif self._mode == "ca_only":
            pred = jnp.einsum("nm,nd->md", assignment, output.backbone_coordinates[:N, 1, :])
            R, t = kabsch(pred, self._gt, weights=self._weights)
        else:  # backbone
            pred = jnp.einsum("nm,nkd->mkd", assignment, output.backbone_coordinates[:N]).reshape(-1, 3)
            R, t = kabsch(pred, self._gt, weights=self._weights)

        aligned = pred @ R + t
        if self._weights is not None:
            msd = (self._weights * jnp.square(aligned - self._gt)).sum((-1, -2))
        else:
            msd = jnp.mean(jnp.sum((aligned - self._gt) ** 2, axis=-1))
        rmsd = jnp.sqrt(msd)
        return rmsd, {self.name: rmsd}


class UnindexedDistogramCCE(LossTerm):
    """Distogram cross-entropy through a soft assignment matrix.

    Double einsum extracts (M, M, n_bins) motif-pair logits from the
    full (N, N, n_bins) prediction, then computes CCE against the
    ground-truth motif distogram.
    """
    _gt_distogram: jnp.ndarray
    name: str = "unindexed_dgram_cce"

    def __init__(self, gt_distogram, name: str = "unindexed_dgram_cce"):
        self._gt_distogram = gt_distogram
        self.name = name

    @classmethod
    def from_scaffold(cls, scaffold: UnindexedScaffold, name: str = "unindexed_dgram_cce"):
        return cls(gt_distogram=scaffold.distogram(), name=name)

    def __call__(self, sequence, output: StructureModelOutput, key, assignment, **kwargs):
        N = assignment.shape[0]
        pred_logits = output.distogram_logits[:N, :N]  # (N, N, n_bins)
        num_bins = pred_logits.shape[-1]

        motif_logits = jnp.einsum(
            "ni,nkb,kj->ijb", assignment, pred_logits, assignment
        )  # (M, M, n_bins)

        bin_edges = jnp.linspace(
            output.distogram_bins[0], output.distogram_bins[-1], num_bins - 1
        )
        gt_indices = (self._gt_distogram[..., None] > bin_edges).sum(-1)
        gt_one_hot = nn.one_hot(gt_indices, num_classes=num_bins)

        loss = -jnp.sum(gt_one_hot * nn.log_softmax(motif_logits, axis=-1), axis=-1)
        cce = jnp.mean(loss)
        return cce, {self.name: cce}

# ============================================================================
# GeometricSearch LossTerm
# ============================================================================

def _compute_cb_coords(atom37_coords, atom37_mask):
    ca = atom37_coords[:, ATOM37_INDEX["CA"], :]
    cb = atom37_coords[:, ATOM37_INDEX["CB"], :]
    has_cb = atom37_mask[:, ATOM37_INDEX["CB"]].astype(bool)
    return jnp.where(has_cb[:, None], cb, ca)


def _pairwise_distances(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return jnp.linalg.norm(diff + 1e-10, axis=-1)


def _find_best_positions(pred_dist, gt_dist, top_k):
    """Find M design positions whose internal geometry best matches the GT motif.

    Anchor-based sequential search: tries all N positions as motif 0,
    then narrows candidates by distance matching. Total candidates:
    N * K^(M-1), evaluated via full [M, M] sub-distogram Frobenius norm.
    """
    N = pred_dist.shape[0]
    M = gt_dist.shape[0]

    partial = jnp.arange(N)[:, None]  # [N, 1]

    for m in range(1, M):
        def _score_next(partial_row):
            assigned_dists = pred_dist[partial_row]  # [m, N]
            gt_dists_to_m = gt_dist[:m, m]  # [m]
            errors = jnp.abs(assigned_dists - gt_dists_to_m[:, None])  # [m, N]
            total_error = errors.sum(axis=0)  # [N]
            total_error = total_error.at[partial_row].set(1e6)
            return total_error

        all_errors = jax.vmap(_score_next)(partial)  # [n_cand, N]
        _, top_k_per_cand = jax.lax.top_k(-all_errors, top_k)  # [n_cand, K]

        partial_expanded = jnp.repeat(partial, top_k, axis=0)
        new_col = top_k_per_cand.reshape(-1, 1)
        partial = jnp.concatenate([partial_expanded, new_col], axis=1)

    def _eval_candidate(cand_idxs):
        sub_dist = pred_dist[cand_idxs][:, cand_idxs]
        mismatch = jnp.sum((sub_dist - gt_dist) ** 2)
        pair_eq = cand_idxs[:, None] == cand_idxs[None, :]
        n_dupes = (pair_eq.sum() - M) // 2
        return mismatch + n_dupes * 1e6

    mismatches = jax.vmap(_eval_candidate)(partial)
    best_idx = jnp.argmin(mismatches)
    best_positions = partial[best_idx]

    all_perms = jnp.array(list(permutations(range(M))))  # [M!, M]

    def _eval_perm(perm):
        reordered = best_positions[perm]
        sub_dist = pred_dist[reordered][:, reordered]
        return jnp.sum((sub_dist - gt_dist) ** 2)

    perm_mismatches = jax.vmap(_eval_perm)(all_perms)
    best_perm = all_perms[jnp.argmin(perm_mismatches)]
    return best_positions[best_perm]


class CollisionPenalty(LossTerm):
    """Penalizes multiple motifs assigned to the same design position.

    L_coll = Σ ReLU(row_sum - 1) where row_sum is the sum of assignment
    weights per design position. Differentiable w.r.t. assignment.
    """
    name: str = "collision_penalty"

    def __init__(self, name: str = "collision_penalty"):
        self.name = name

    @classmethod
    def from_scaffold(cls, scaffold: UnindexedScaffold, name: str = "collision_penalty"):
        return cls(name=name)

    def __call__(self, sequence, output, key, assignment, **kwargs):
        row_sums = assignment.sum(axis=1)
        collision = jnp.sum(jax.nn.relu(row_sums - 1.0))
        return collision, {self.name: collision}


class GeometricSearch(LossTerm):
    """Non-differentiable position search for unindexed scaffolding.

    Always returns loss=0. Puts geometric search results into aux
    for the optimizer to use as assignment votes.
    """
    _gt_distogram: jnp.ndarray
    _geo_top_k: int
    name: str = "geometric_search"

    def __init__(self, gt_distogram, geo_top_k=5, name="geometric_search"):
        self._gt_distogram = gt_distogram
        self._geo_top_k = geo_top_k
        self.name = name

    @classmethod
    def from_scaffold(cls, scaffold: UnindexedScaffold, geo_top_k=5,
                      name="geometric_search"):
        return cls(
            gt_distogram=scaffold.distogram(),
            geo_top_k=geo_top_k,
            name=name,
        )

    def __call__(self, sequence, output: StructureModelOutput, key,
                 assignment, **kwargs):
        N = assignment.shape[0]
        pred_cb = _compute_cb_coords(output.atom37_coords[:N], output.atom37_mask[:N])
        pred_cb_dist = _pairwise_distances(pred_cb)

        geo_idxs = jax.lax.stop_gradient(
            _find_best_positions(pred_cb_dist, self._gt_distogram, self._geo_top_k)
        )

        sub_dist = pred_cb_dist[geo_idxs][:, geo_idxs]
        geo_mismatch = jax.lax.stop_gradient(
            jnp.sqrt(jnp.mean((sub_dist - self._gt_distogram) ** 2))
        )

        loss = jnp.float32(0.0)
        return loss, {
            self.name: loss,
            "geo_motif_idxs": geo_idxs,
            "geo_distogram_mismatch": geo_mismatch,
        }
