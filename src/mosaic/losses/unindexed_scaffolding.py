import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx

from mosaic.common import restype_three_to_one, TOKENS, LossTerm
from mosaic.losses.atom37 import ATOM37_INDEX
from mosaic.util import gram_schmidt, kabsch
from mosaic.structure_prediction import TargetChain, StructureModelOutput
from jaxtyping import Float, Array, Bool


import biotite.structure as struct
from biotite.structure.io import pdb, pdbx
import gemmi


class Scaffold:
    def __init__(self, path_to_structure: str, keep_intervals: str | None = None,
                 loops: list[int] | None = None, order: list[int] | None = None,
                 length: int | None = None):
        self.structure_path = str(path_to_structure)

        if self.structure_path.endswith(".pdb"):
            pdb_file = pdb.PDBFile.read(self.structure_path)
            self.file_type = "pdb"
            structure = pdb.get_structure(pdb_file, model=1)
        elif self.structure_path.endswith(".cif"):
            cif_file = pdbx.CIFFile.read(self.structure_path)
            self.file_type = "cif"
            structure = pdbx.get_structure(cif_file, model=1)
        else:
            raise ValueError("File must be .pdb or .cif")

        structure = structure[struct.filter_amino_acids(structure)]

        if keep_intervals is not None:
            intervals = self._parse_intervals(keep_intervals)
            motif_res_ids = []
            motif_chain_ids = []
            ca_atoms = structure[structure.atom_name == "CA"]
            for start, end in intervals:
                frag_ca = ca_atoms[
                    (ca_atoms.res_id >= start) & (ca_atoms.res_id <= end)
                ]
                motif_res_ids.extend(frag_ca.res_id)
                motif_chain_ids.extend(frag_ca.chain_id)
            motif_res_ids = np.array(motif_res_ids)
            motif_chain_ids = np.array(motif_chain_ids)
        else:
            ca_atoms = structure[structure.atom_name == "CA"]
            motif_mask = ca_atoms.res_name != "ALA"
            motif_res_ids = ca_atoms.res_id[motif_mask]
            motif_chain_ids = ca_atoms.chain_id[motif_mask]

        n_motif = len(motif_res_ids)
        if n_motif == 0:
            raise ValueError(
                "No motif residues found"
                + (" in specified intervals" if keep_intervals else " (no non-ALA residues)")
            )

        def extract_residue(structure, cid, rid):
            res_atoms = structure[
                (structure.chain_id == cid) & (structure.res_id == rid)
            ]
            a37_coords = np.zeros((1, 37, 3), dtype=np.float32)
            a37_mask = np.zeros((1, 37), dtype=np.float32)
            for atom_name, coord in zip(res_atoms.atom_name, res_atoms.coord):
                idx = ATOM37_INDEX.get(atom_name)
                if idx is not None:
                    a37_coords[0, idx] = coord
                    a37_mask[0, idx] = 1.0
            seq = restype_three_to_one.get(res_atoms.res_name[0], "X")
            return a37_coords, a37_mask, seq

        def make_loop(n):
            a37 = np.zeros((n, 37, 3), dtype=np.float32)
            a37m = np.zeros((n, 37), dtype=np.float32)
            seq = "X" * n
            return a37, a37m, seq

        if loops is not None:
            # --- indexed mode: interleave motif fragments with loops ---
            # group contiguous residues into intervals
            motif_intervals = self._group_into_intervals(
                motif_res_ids, motif_chain_ids
            )
            n_intervals = len(motif_intervals)

            if order is None:
                order = list(range(n_intervals))

            assert len(loops) == len(order) + 1, \
                f"loops must have length len(order)+1={len(order)+1}, got {len(loops)}"
            assert len(order) == n_intervals and max(order) == n_intervals - 1, \
                f"order must be a permutation of 0..{n_intervals - 1}"

            if length is not None:
                loops = list(loops)
                motif_count = sum(
                    len(ids) for ids, _ in motif_intervals
                )
                adjusted_last = length - sum(loops[:-1]) - motif_count
                if adjusted_last < 0:
                    raise ValueError(
                        f"specified length={length} is too short to accommodate "
                        f"the scaffold (motifs={motif_count} + preceding loops={sum(loops[:-1])}). "
                        f"Increase length or reduce loops."
                    )
                if adjusted_last != loops[-1]:
                    print(f"[Scaffold] Adjusting last loop: {loops[-1]} -> {adjusted_last} "
                          f"to match specified length={length}")
                    loops[-1] = adjusted_last

            all_a37, all_a37m, all_seq, all_mask = [], [], [], []
            template_seqids: list[int | None] = []
            template_chain_ids: list[str | None] = []

            for i, frag_idx in enumerate(order):
                if loops[i] > 0:
                    a37, a37m, seq = make_loop(loops[i])
                    all_a37.append(a37)
                    all_a37m.append(a37m)
                    all_seq.append(seq)
                    all_mask.append(np.zeros(loops[i], dtype=bool))
                    template_seqids.extend([None] * loops[i])
                    template_chain_ids.extend([None] * loops[i])

                frag_rids, frag_cids = motif_intervals[frag_idx]
                for cid, rid in zip(frag_cids, frag_rids):
                    a37, a37m, seq = extract_residue(structure, cid, rid)
                    all_a37.append(a37)
                    all_a37m.append(a37m)
                    all_seq.append(seq)
                    all_mask.append(np.ones(1, dtype=bool))
                    template_seqids.append(int(rid))
                    template_chain_ids.append(str(cid))

            if loops[-1] > 0:
                a37, a37m, seq = make_loop(loops[-1])
                all_a37.append(a37)
                all_a37m.append(a37m)
                all_seq.append(seq)
                all_mask.append(np.zeros(loops[-1], dtype=bool))
                template_seqids.extend([None] * loops[-1])
                template_chain_ids.extend([None] * loops[-1])

            full_seq = "".join(all_seq)
            self._motif_sequence = "".join(c for c in full_seq if c != "X")
            self._sequence = full_seq
            self._atom37_coords = jnp.array(np.concatenate(all_a37, axis=0))
            self._atom37_mask = jnp.array(np.concatenate(all_a37m, axis=0))
            self.mask = jnp.array(np.concatenate(all_mask, axis=0))
            self._template_seqids = template_seqids
            self._template_chain_ids = template_chain_ids

        else:
            # --- unindexed mode: motif residues only ---
            if length is None:
                raise ValueError("length is required in unindexed mode (no loops)")

            all_a37 = []
            all_a37m = []
            motif_seq_parts = []
            template_seqids = []
            template_chain_ids = []

            for cid, rid in zip(motif_chain_ids, motif_res_ids):
                a37, a37m, seq = extract_residue(structure, cid, rid)
                all_a37.append(a37)
                all_a37m.append(a37m)
                motif_seq_parts.append(seq)
                template_seqids.append(int(rid))
                template_chain_ids.append(str(cid))

            self._motif_sequence = "".join(motif_seq_parts)
            self._sequence = "X" * length
            self._atom37_coords = jnp.array(np.concatenate(all_a37, axis=0))
            self._atom37_mask = jnp.array(np.concatenate(all_a37m, axis=0))
            self.mask = None
            self._template_seqids = template_seqids
            self._template_chain_ids = template_chain_ids

    @staticmethod
    def _group_into_intervals(res_ids, chain_ids):
        intervals = []
        current_rids = [res_ids[0]]
        current_cids = [chain_ids[0]]
        for rid, cid in zip(res_ids[1:], chain_ids[1:]):
            if cid == current_cids[-1] and rid == current_rids[-1] + 1:
                current_rids.append(rid)
                current_cids.append(cid)
            else:
                intervals.append((current_rids, current_cids))
                current_rids = [rid]
                current_cids = [cid]
        intervals.append((current_rids, current_cids))
        return intervals

    @staticmethod
    def _parse_intervals(intervals_str: str) -> list[tuple[int, int]]:
        intervals = []
        for part in intervals_str.split(","):
            part = part.strip()
            if "-" in part:
                split = part.split("-")
                if len(split) != 2:
                    raise ValueError(f"Expected exactly one '-' separator, got: '{part}'")
                start_str, end_str = split
                if not start_str.strip().isdigit() or not end_str.strip().isdigit():
                    raise ValueError(f"Expected integer values, got: '{part}'")
                start, end = int(start_str), int(end_str)
                if start > end:
                    raise ValueError(
                        f"Interval start must be <= end, got start:{start} end:{end}"
                    )
            else:
                if not part.isdigit():
                    raise ValueError(f"Expected integer value, got: '{part}'")
                start = end = int(part)
            intervals.append((start, end))
        sorted_intervals = sorted(intervals)
        for i in range(len(sorted_intervals) - 1):
            if sorted_intervals[i][1] >= sorted_intervals[i + 1][0]:
                raise ValueError(
                    f"Intervals must not overlap: {sorted_intervals[i]} and {sorted_intervals[i + 1]}"
                )
        return intervals

    @property
    def sequence(self) -> str:
        return self._sequence

    @property
    def motif_sequence(self) -> str:
        return self._motif_sequence

    def __len__(self) -> int:
        return len(self._sequence)

    def pssm(self) -> jnp.ndarray:
        if self.mask is not None:
            aa_indices = jnp.array([
                TOKENS.index(aa) if aa in TOKENS else 0
                for aa in self._sequence
            ])
            onehot = jax.nn.one_hot(aa_indices, num_classes=20)
            return jnp.where(self.mask[:, None], onehot, 0.0)
        else:
            aa_indices = jnp.array([
                TOKENS.index(aa) if aa in TOKENS else 0
                for aa in self._motif_sequence
            ])
            return jax.nn.one_hot(aa_indices, num_classes=20)

    def atom37_coordinates(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self._atom37_coords, self._atom37_mask

    def backbone_coordinates(self) -> jnp.ndarray:
        bb_idx = jnp.array([ATOM37_INDEX[a] for a in ("N", "CA", "C", "O")])
        return self._atom37_coords[:, bb_idx, :]

    def distogram(self) -> jnp.ndarray:
        ca = self._atom37_coords[:, ATOM37_INDEX["CA"], :]
        cb = self._atom37_coords[:, ATOM37_INDEX["CB"], :]
        has_cb = self._atom37_mask[:, ATOM37_INDEX["CB"]].astype(bool)
        pb = jnp.where(has_cb[:, None], cb, ca)
        diff = pb[:, None, :] - pb[None, :, :]
        dist = jnp.linalg.norm(diff, axis=-1)
        if self.mask is not None:
            pair_mask = self.mask[:, None] & self.mask[None, :]
            dist = jnp.where(pair_mask, dist, 0.0)
        return dist

    def backbone_frames(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        bb = self.backbone_coordinates()
        n_coords = bb[:, 0, :]
        ca_coords = bb[:, 1, :]
        c_coords = bb[:, 2, :]
        R = gram_schmidt(v1=c_coords - ca_coords, v2=n_coords - ca_coords)
        t = ca_coords
        if self.mask is not None:
            identity = jnp.broadcast_to(jnp.eye(3), R.shape)
            R = jnp.where(self.mask[:, None, None], R, identity)
            t = jnp.where(self.mask[:, None], ca_coords, 0.0)
        return t, R

    def build_chain(self, use_msa: bool = False, use_template: bool = False):
        template_chain = None
        template_mask = None

        if use_template and self.mask is not None:
            src_st = gemmi.read_structure(self.structure_path)
            src_st.remove_ligands_and_waters()
            src_by_key = {}
            for chain in src_st[0]:
                for r in chain:
                    src_by_key[(chain.name, r.seqid.num)] = r

            new_chain = gemmi.Chain("A")
            for new_idx, (cid, sid) in enumerate(
                zip(self._template_chain_ids, self._template_seqids), start=1
            ):
                if cid is None:
                    r = gemmi.Residue()
                    r.name = "UNK"
                else:
                    r = src_by_key[(cid, sid)].clone()
                r.seqid = gemmi.SeqId(new_idx, " ")
                new_chain.add_residue(r)

            template_chain = new_chain
            template_mask = self.mask

        self.chain = TargetChain(
            sequence=self.sequence,
            use_msa=use_msa,
            template_chain=template_chain,
            template_mask=template_mask,
        )
        return self.chain

# ===========================================================================
# Unindexed losses
#
# In indexed scaffolding, the motif positions in the designed protein are
# known ahead of time (_idx), so losses just index into the prediction.
#
# In unindexed scaffolding, the motif placement is unknown. The loss must:
#   1. Determine which M of the N predicted positions correspond to the
#      M motif residues (the "assignment problem").
#   2. Compare the predicted structure at those positions against the
#      ground truth stored in the Scaffold.
#   3. Return (loss_value, aux_dict) where aux_dict contains at least
#      {self.name: loss_value}.
#
# Ground truth is available from the Scaffold via:
#   scaffold.backbone_coordinates()  -> [M, 4, 3]  (N, CA, C, O)
#   scaffold.backbone_frames()       -> ([M, 3], [M, 3, 3])  (t, R)
#   scaffold.distogram()             -> [M, M]  (CB-CB distances)
#   scaffold.atom37_coordinates()    -> ([M, 37, 3], [M, 37])
#   scaffold.motif_sequence          -> str of length M
#
# The prediction (output) has shape [N, ...] where N = scaffold length:
#   output.backbone_coordinates      -> [N, 4, 3]
#
# The sequence (pssm) has shape [N, 20].
#
# The call signature must stay: __call__(self, sequence, output, key)
# to remain compatible with LossTerm composition via +.
# ===========================================================================

class UnindexedRMSD(LossTerm):
    gt_coords: Float[Array, "M 4 3"]
    motif_pssm: Float[Array, "M 20"]
    assignment: Float[Array, "N M"] | None
    _mode: str
    _gt_atom37_coords: Float[Array, "M 37 3"] | None
    _gt_atom37_mask: Float[Array, "M 37"] | None
    _atom_idx: Array | None
    _gt_flat: Float[Array, "n_atoms 3"] | None
    name: str = "unindexed_rmsd"

    def __init__(self, gt_coords, motif_pssm, assignment=None,
                 name: str = "unindexed_rmsd", mode: str = "ca_only",
                 gt_atom37_coords=None, gt_atom37_mask=None):
        self.name = name
        self.gt_coords = gt_coords
        self.motif_pssm = motif_pssm
        self.assignment = assignment
        self._mode = mode
        self._gt_atom37_coords = gt_atom37_coords
        self._gt_atom37_mask = gt_atom37_mask

        if mode == "all_atom":
            flat_mask = gt_atom37_mask.reshape(-1)
            n_atoms = int(flat_mask.sum())
            self._atom_idx = jnp.where(flat_mask, size=n_atoms)[0]
            self._gt_flat = gt_atom37_coords.reshape(-1, 3)[self._atom_idx]
        else:
            self._atom_idx = None
            self._gt_flat = None

    @classmethod
    def from_scaffold(cls, scaffold: Scaffold, name: str = "unindexed_rmsd",
                      mode: str = "ca_only"):
        assert mode in ("ca_only", "all_atom"), \
            f"Unknown UnindexedRMSD mode {mode}, available: ca_only, all_atom"
        if mode == "all_atom":
            a37_coords, a37_mask = scaffold.atom37_coordinates()
            return cls(
                gt_coords=scaffold.backbone_coordinates(),
                motif_pssm=scaffold.pssm(),
                name=name,
                mode=mode,
                gt_atom37_coords=a37_coords,
                gt_atom37_mask=a37_mask,
            )
        return cls(
            gt_coords=scaffold.backbone_coordinates(),
            motif_pssm=scaffold.pssm(),
            name=name,
            mode=mode,
        )

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        assignment = self.assignment

        # Cα RMSD — always valid, even with soft assignment
        pred_ca = jnp.einsum("nm,nd->md", assignment, output.backbone_coordinates[:, 1, :])
        gt_ca = self.gt_coords[:, 1, :]
        R, t = kabsch(pred_ca, gt_ca)
        aligned_ca = pred_ca @ R + t
        rmsd_ca = jnp.sqrt(jnp.mean(jnp.sum((aligned_ca - gt_ca) ** 2, axis=-1)))

        aux = {self.name: rmsd_ca, "motif_rmsd_ca": rmsd_ca}

        # All-atom RMSD — metric only (no gradient). Uses hard argmax so
        # atom37 slots correspond correctly. Only meaningful once assignment
        # is sharp and seq_ce has pushed the correct amino acid identity.
        if self._mode == "all_atom":
            motif_idxs = jnp.argmax(assignment, axis=0)
            pred_a37 = output.atom37_coords[motif_idxs]
            pred_flat = pred_a37.reshape(-1, 3)[self._atom_idx]
            R_aa, t_aa = kabsch(pred_flat, self._gt_flat)
            aligned_aa = pred_flat @ R_aa + t_aa
            rmsd_aa = jnp.sqrt(jnp.mean(jnp.sum((aligned_aa - self._gt_flat) ** 2, axis=-1)))
            aux["motif_rmsd_all_atom"] = jax.lax.stop_gradient(rmsd_aa)

        seq_probs = jax.nn.softmax(sequence, axis=-1)
        assigned_seq = jnp.einsum("nm,na->ma", assignment, seq_probs)
        seq_ce = -(self.motif_pssm * jnp.log(assigned_seq + 1e-10)).sum(axis=-1).mean()

        row_sums = assignment.sum(axis=1)
        collision = jnp.sum(jax.nn.relu(row_sums - 1.0))

        motif_idxs = jnp.argmax(assignment, axis=0)
        aux.update({
            "motif_seq_ce": seq_ce,
            "motif_collision": collision,
            "motif_idxs": motif_idxs,
        })
        return rmsd_ca + seq_ce + collision, aux


def _find_best_assignment_positions(assignment, pred_ca, gt_ca, top_k):
    """Find best motif positions by enumerating top-K from the assignment matrix.

    For each motif residue m, takes the top-K positions by assignment weight,
    enumerates all K^M combinations, computes Kabsch-aligned RMSD for each,
    and returns the best combination.

    Args:
        assignment: [N, M] soft assignment matrix (probabilities, axis=0 sums to 1).
        pred_ca: [N, 3] predicted CA coordinates.
        gt_ca: [M, 3] ground truth motif CA coordinates.
        top_k: number of candidates per motif residue.

    Returns:
        best_idxs: [M] indices into the N design positions.
        best_rmsd: scalar, RMSD of the best combination.
    """
    M = assignment.shape[1]

    topk_indices = []
    for m in range(M):
        _, idx = jax.lax.top_k(assignment[:, m], top_k)
        topk_indices.append(idx)
    topk_indices = jnp.stack(topk_indices, axis=1)  # [K, M]

    from itertools import product as _product
    combo_local = jnp.array(list(_product(*[range(top_k)] * M)))  # [K^M, M]
    candidates = jnp.stack(
        [topk_indices[combo_local[:, m], m] for m in range(M)], axis=-1
    )  # [K^M, M]

    def _eval_rmsd(idxs):
        sel = pred_ca[idxs]
        R, t = kabsch(sel, gt_ca)
        aligned = sel @ R + t
        sq_dev = jnp.sum((aligned - gt_ca) ** 2, axis=-1)
        pair_eq = idxs[:, None] == idxs[None, :]
        n_dupes = (pair_eq.sum() - M) // 2
        return jnp.sqrt(jnp.mean(sq_dev)) + n_dupes * 1e6

    rmsds = jax.vmap(_eval_rmsd)(candidates)
    best_idx = jnp.argmin(rmsds)
    return candidates[best_idx], rmsds[best_idx]


def _compute_cb_coords(atom37_coords, atom37_mask):
    """Extract CB coordinates with CA fallback for glycine."""
    ca = atom37_coords[:, ATOM37_INDEX["CA"], :]
    cb = atom37_coords[:, ATOM37_INDEX["CB"], :]
    has_cb = atom37_mask[:, ATOM37_INDEX["CB"]].astype(bool)
    return jnp.where(has_cb[:, None], cb, ca)


def _pairwise_distances(coords):
    """Compute pairwise Euclidean distance matrix [N, N] from coords [N, 3]."""
    diff = coords[:, None, :] - coords[None, :, :]
    return jnp.linalg.norm(diff + 1e-10, axis=-1)


def _find_best_positions(pred_dist, gt_dist, top_k):
    """Find M design positions whose internal geometry best matches the GT motif.

    Uses anchor-based sequential search: tries all N positions as motif 0,
    then for each, narrows candidates for motif 1 by distance matching, etc.
    Total candidates: N * K^(M-1), evaluated via the full [M, M] sub-distogram.

    Args:
        pred_dist: [N, N] predicted CB/CA pairwise distances.
        gt_dist: [M, M] ground truth motif distogram.
        top_k: candidates retained per motif at each expansion level.

    Returns:
        motif_idxs: [M] indices into the N design positions.
    """
    N = pred_dist.shape[0]
    M = gt_dist.shape[0]

    # Build candidates level by level.
    # Level 0: all N positions are candidates for motif 0.
    # Level m: for each partial assignment [n_cand, m], find top-K positions
    # for motif m by matching pred distances to already-assigned positions.
    partial = jnp.arange(N)[:, None]  # [N, 1]

    for m in range(1, M):
        n_cand = partial.shape[0]

        def _score_next(partial_row):
            # Score all N positions as candidates for motif m given partial_row.
            # Error = sum of |pred_dist[assigned_j, n] - gt_dist[j, m]| for j < m.
            assigned_dists = pred_dist[partial_row]  # [m, N]
            gt_dists_to_m = gt_dist[:m, m]  # [m]
            errors = jnp.abs(assigned_dists - gt_dists_to_m[:, None])  # [m, N]
            total_error = errors.sum(axis=0)  # [N]
            # Penalize already-assigned positions
            total_error = total_error.at[partial_row].set(1e6)
            return total_error

        all_errors = jax.vmap(_score_next)(partial)  # [n_cand, N]
        _, top_k_per_cand = jax.lax.top_k(-all_errors, top_k)  # [n_cand, K]

        # Expand: [n_cand, m] → [n_cand * K, m+1]
        partial_expanded = jnp.repeat(partial, top_k, axis=0)
        new_col = top_k_per_cand.reshape(-1, 1)
        partial = jnp.concatenate([partial_expanded, new_col], axis=1)

    # partial: [N * K^(M-1), M] — all candidate assignments
    # Evaluate each by full sub-distogram Frobenius norm
    def _eval_candidate(cand_idxs):
        sub_dist = pred_dist[cand_idxs][:, cand_idxs]
        mismatch = jnp.sum((sub_dist - gt_dist) ** 2)
        # Duplicate penalty
        pair_eq = cand_idxs[:, None] == cand_idxs[None, :]
        n_dupes = (pair_eq.sum() - M) // 2
        return mismatch + n_dupes * 1e6

    mismatches = jax.vmap(_eval_candidate)(partial)
    best_idx = jnp.argmin(mismatches)
    best_positions = partial[best_idx]

    # The sequential search may assign motifs in the wrong order.
    # Try all M! permutations to find the ordering that best matches GT.
    from itertools import permutations
    all_perms = jnp.array(list(permutations(range(M))))  # [M!, M]

    def _eval_perm(perm):
        reordered = best_positions[perm]
        sub_dist = pred_dist[reordered][:, reordered]
        return jnp.sum((sub_dist - gt_dist) ** 2)

    perm_mismatches = jax.vmap(_eval_perm)(all_perms)
    best_perm = all_perms[jnp.argmin(perm_mismatches)]
    return best_positions[best_perm]


class GeometricUnindexedRMSD(LossTerm):
    """Hybrid unindexed scaffolding loss (Approach 1+2).

    Combines geometric distogram matching (approach 2) with a persistent soft
    assignment matrix (approach 1). Each forward pass:
    1. Finds best geometric match via distogram → geo_motif_idxs (stop_gradient)
    2. If assignment is set: uses soft assignment for differentiable RMSD + seq_ce
       (approach 1), returns geo_motif_idxs in aux for the optimizer to use as a vote
    3. If assignment is None: falls back to pure geometric mode (hard indices)
    """
    gt_coords: Float[Array, "M 4 3"]
    gt_distogram: Float[Array, "M M"]
    motif_pssm: Float[Array, "M 20"]
    assignment: Float[Array, "N M"] | None
    _mode: str
    _gt_atom37_coords: Float[Array, "M 37 3"] | None
    _gt_atom37_mask: Float[Array, "M 37"] | None
    _atom_idx: Array | None
    _gt_flat: Float[Array, "n_atoms 3"] | None
    _top_k: int
    _seq_ce_weight: float
    _use_all_atom_loss: bool
    _assign_top_k: int
    name: str = "geometric_unindexed_rmsd"

    def __init__(self, gt_coords, gt_distogram, motif_pssm, assignment=None,
                 name: str = "geometric_unindexed_rmsd",
                 mode: str = "ca_only", top_k: int = 15,
                 seq_ce_weight: float = 10.0,
                 assign_top_k: int = 0,
                 gt_atom37_coords=None, gt_atom37_mask=None):
        self.name = name
        self.gt_coords = gt_coords
        self.gt_distogram = gt_distogram
        self.motif_pssm = motif_pssm
        self.assignment = assignment
        self._mode = mode
        self._gt_atom37_coords = gt_atom37_coords
        self._gt_atom37_mask = gt_atom37_mask
        self._top_k = top_k
        self._seq_ce_weight = seq_ce_weight
        self._use_all_atom_loss = False
        self._assign_top_k = assign_top_k

        if mode == "all_atom":
            flat_mask = gt_atom37_mask.reshape(-1)
            n_atoms = int(flat_mask.sum())
            self._atom_idx = jnp.where(flat_mask, size=n_atoms)[0]
            self._gt_flat = gt_atom37_coords.reshape(-1, 3)[self._atom_idx]
        else:
            self._atom_idx = None
            self._gt_flat = None

    @classmethod
    def from_scaffold(cls, scaffold: Scaffold, name: str = "geometric_unindexed_rmsd",
                      mode: str = "ca_only", top_k: int = 15, seq_ce_weight: float = 10.0,
                      assign_top_k: int = 0):
        assert mode in ("ca_only", "all_atom"), \
            f"Unknown GeometricUnindexedRMSD mode {mode}, available: ca_only, all_atom"

        gt_distogram = scaffold.distogram()

        if mode == "all_atom":
            a37_coords, a37_mask = scaffold.atom37_coordinates()
            return cls(
                gt_coords=scaffold.backbone_coordinates(),
                gt_distogram=gt_distogram,
                motif_pssm=scaffold.pssm(),
                name=name, mode=mode, top_k=top_k, seq_ce_weight=seq_ce_weight,
                assign_top_k=assign_top_k,
                gt_atom37_coords=a37_coords, gt_atom37_mask=a37_mask,
            )
        return cls(
            gt_coords=scaffold.backbone_coordinates(),
            gt_distogram=gt_distogram,
            motif_pssm=scaffold.pssm(),
            name=name, mode=mode, top_k=top_k, seq_ce_weight=seq_ce_weight,
            assign_top_k=assign_top_k,
        )

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        # Geometric position discovery (non-differentiable)
        pred_cb = _compute_cb_coords(output.atom37_coords, output.atom37_mask)
        pred_dist = _pairwise_distances(pred_cb)
        geo_motif_idxs = jax.lax.stop_gradient(
            _find_best_positions(pred_dist, self.gt_distogram, self._top_k)
        )

        if self.assignment is not None:
            # HYBRID MODE: soft assignment for differentiable losses
            assignment = self.assignment
            pred_ca = jnp.einsum("nm,nd->md", assignment, output.backbone_coordinates[:, 1, :])
            gt_ca = self.gt_coords[:, 1, :]
            R, t = kabsch(pred_ca, gt_ca)
            aligned_ca = pred_ca @ R + t
            rmsd_ca = jnp.sqrt(jnp.mean(jnp.sum((aligned_ca - gt_ca) ** 2, axis=-1)))

            if self._use_all_atom_loss and self._atom_idx is not None:
                pred_a37 = jnp.einsum("nm,nad->mad", assignment, output.atom37_coords)
                pred_flat = pred_a37.reshape(-1, 3)[self._atom_idx]
                R_aa, t_aa = kabsch(pred_flat, self._gt_flat)
                aligned_aa = pred_flat @ R_aa + t_aa
                rmsd_loss = jnp.sqrt(jnp.mean(jnp.sum((aligned_aa - self._gt_flat) ** 2, axis=-1)))
            else:
                rmsd_loss = rmsd_ca

            seq_probs = jax.nn.softmax(sequence, axis=-1)
            assigned_seq = jnp.einsum("nm,na->ma", assignment, seq_probs)
            seq_ce = -(self.motif_pssm * jnp.log(assigned_seq + 1e-10)).sum(axis=-1).mean()

            row_sums = assignment.sum(axis=1)
            collision = jnp.sum(jax.nn.relu(row_sums - 1.0))

            motif_idxs = jnp.argmax(assignment, axis=0)
            # Gate seq_ce by assignment sharpness: off during exploration
            # (uniform → max≈1/N), full strength once positions commit (max≈1)
            sharpness = jnp.max(assignment, axis=0).mean()
            seq_ce_gate = jnp.clip((sharpness - 0.1) / 0.4, 0.0, 1.0)
            loss = rmsd_loss + self._seq_ce_weight * seq_ce_gate * seq_ce + collision
        else:
            # PURE GEOMETRIC MODE: hard indices (for reprediction / standalone)
            motif_idxs = geo_motif_idxs
            pred_ca = output.backbone_coordinates[motif_idxs, 1, :]
            gt_ca = self.gt_coords[:, 1, :]
            R, t = kabsch(pred_ca, gt_ca)
            aligned_ca = pred_ca @ R + t
            rmsd_ca = jnp.sqrt(jnp.mean(jnp.sum((aligned_ca - gt_ca) ** 2, axis=-1)))

            seq_probs = jax.nn.softmax(sequence, axis=-1)
            pred_seq = seq_probs[motif_idxs]
            seq_ce = -(self.motif_pssm * jnp.log(pred_seq + 1e-10)).sum(axis=-1).mean()

            collision = jnp.float32(0.0)
            loss = rmsd_ca + self._seq_ce_weight * seq_ce

        aux = {self.name: rmsd_ca, "motif_rmsd_ca": rmsd_ca}

        # All-atom RMSD metric (stop_gradient, uses hard indices from assignment or geo)
        if self._mode == "all_atom":
            aa_idxs = motif_idxs
            pred_a37 = output.atom37_coords[aa_idxs]
            pred_flat = pred_a37.reshape(-1, 3)[self._atom_idx]
            R_aa, t_aa = kabsch(pred_flat, self._gt_flat)
            aligned_aa = pred_flat @ R_aa + t_aa
            rmsd_aa = jnp.sqrt(jnp.mean(jnp.sum((aligned_aa - self._gt_flat) ** 2, axis=-1)))
            aux["motif_rmsd_all_atom"] = jax.lax.stop_gradient(rmsd_aa)

        # Distogram mismatch at geometric match (for logging)
        geo_sub_dist = pred_dist[geo_motif_idxs][:, geo_motif_idxs]
        distogram_mismatch = jnp.sqrt(jnp.mean((geo_sub_dist - self.gt_distogram) ** 2))

        aux.update({
            "motif_seq_ce": seq_ce,
            "motif_collision": collision,
            "motif_distogram_mismatch": distogram_mismatch,
            "motif_idxs": motif_idxs,
            "geo_motif_idxs": geo_motif_idxs,
        })

        if self._assign_top_k > 0 and self.assignment is not None:
            assign_best_idxs, assign_best_rmsd = jax.lax.stop_gradient(
                _find_best_assignment_positions(
                    self.assignment,
                    output.backbone_coordinates[:, 1, :],
                    self.gt_coords[:, 1, :],
                    self._assign_top_k,
                )
            )
            aux["assign_motif_idxs"] = assign_best_idxs
            aux["assign_motif_rmsd"] = assign_best_rmsd

        return loss, aux


def set_assignment(loss, assignment):
    """Set assignment on all unindexed loss instances in a composed loss tree."""
    def _is_unindexed(x):
        return isinstance(x, (UnindexedRMSD, GeometricUnindexedRMSD))
    def update(leaf):
        if _is_unindexed(leaf):
            return eqx.tree_at(
                lambda l: l.assignment, leaf, assignment,
                is_leaf=lambda x: x is None,
            )
        return leaf
    return jax.tree.map(update, loss, is_leaf=_is_unindexed)


def set_seq_ce_weight(loss, seq_ce_weight):
    """Set seq_ce_weight on all GeometricUnindexedRMSD instances in a loss tree."""
    def _is_geo(x):
        return isinstance(x, GeometricUnindexedRMSD)
    def update(leaf):
        if _is_geo(leaf):
            return eqx.tree_at(lambda l: l._seq_ce_weight, leaf, seq_ce_weight)
        return leaf
    return jax.tree.map(update, loss, is_leaf=_is_geo)


def set_use_all_atom_loss(loss, use_all_atom_loss: bool):
    """Set _use_all_atom_loss on all GeometricUnindexedRMSD instances in a loss tree."""
    def _is_geo(x):
        return isinstance(x, GeometricUnindexedRMSD)
    def update(leaf):
        if _is_geo(leaf):
            return eqx.tree_at(lambda l: l._use_all_atom_loss, leaf, use_all_atom_loss)
        return leaf
    return jax.tree.map(update, loss, is_leaf=_is_geo)


if __name__ == "__main__":
    jax.config.update("jax_platforms", "cpu")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--theozyme_pdb", default="structures/theozyme.pdb")
    args = parser.parse_args()
    pdb_path = args.theozyme_pdb
    key = jax.random.key(42)

    def print_scaffold(scaffold, label):
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print("scaffold length:", len(scaffold))
        print("sequence:", scaffold.sequence)
        print("motif_sequence:", scaffold.motif_sequence)
        print("motif pssm shape:", scaffold.pssm().shape)
        if scaffold.mask is not None:
            print("mask:", scaffold.mask)
            print("motif residues:", int(scaffold.mask.sum()))
        print("distogram shape:", scaffold.distogram().shape)
        t, R = scaffold.backbone_frames()
        print("backbone_frames (t, R):", t.shape, R.shape)
        print("backbone_coords shape:", scaffold.backbone_coordinates().shape)
        a37_coords, a37_mask = scaffold.atom37_coordinates()
        print("atom37_coords shape:", a37_coords.shape)
        print("atom37_mask shape:", a37_mask.shape)
        print("atoms per motif residue:", a37_mask.sum(-1))
        result = scaffold.build_chain()
        print("build_chain():", result)

    # --- Mode 1: unindexed, auto-detect non-ALA ---
    s1 = Scaffold(pdb_path, length=100)
    print_scaffold(s1, "Unindexed — auto-detect non-ALA, length=100")

    # --- Mode 2: unindexed, explicit keep_intervals ---
    s2 = Scaffold(pdb_path, keep_intervals="56, 58, 84, 114", length=100)
    print_scaffold(s2, "Unindexed — explicit positions (56, 58, 84, 114), length=100")

    # --- Mode 3: unindexed, keep_intervals with ranges ---
    s3 = Scaffold(pdb_path, keep_intervals="56-58, 84, 114", length=100)
    print_scaffold(s3, "Unindexed — mixed ranges (56-58, 84, 114), length=100")

    # --- Mode 4: indexed, auto-detect motif + loops (sequential order) ---
    s4 = Scaffold(pdb_path, loops=[10, 5, 5, 5, 10])
    print_scaffold(s4, "Indexed — auto-detect + loops [10,5,5,5,10]")

    # --- Mode 5: indexed, explicit positions + loops + reorder ---
    s5 = Scaffold(pdb_path, keep_intervals="56, 58, 84, 114",
                  loops=[10, 5, 5, 5, 10], order=[3, 1, 2, 0])
    print_scaffold(s5, "Indexed — explicit + loops + order [3,1,2,0]")

    # --- Mode 6: indexed with target length ---
    s6 = Scaffold(pdb_path, loops=[10, 5, 5, 5, 10], length=50)
    print_scaffold(s6, "Indexed — auto-detect + loops + length=50")

    # --- RMSD loss test ---
    import jax
    from types import SimpleNamespace

    print(f"\n{'='*60}")
    print(f"  RMSD loss test (unindexed scaffold)")
    print(f"{'='*60}")

    scaffold = Scaffold(pdb_path, length=100)
    scaffold_length = len(scaffold)
    n_motif = len(scaffold.motif_sequence)

    np.random.seed(42)
    pred_backbone_coords = np.random.normal(size=(scaffold_length, 4, 3))
    pssm = np.random.normal(size=(scaffold_length, 20))

    output = SimpleNamespace(
        backbone_coordinates=pred_backbone_coords,
    )

    rmsd_loss = UnindexedRMSD.from_scaffold(scaffold=scaffold)
    v, aux = rmsd_loss(
        sequence=pssm,
        output=output,
        key=jax.random.key(42),
    )
    print(f"RMSD (random pred): {v}")
    print(f"aux: {aux}")

    # --- zero-loss verification: GT backbone as prediction ---
    print(f"\n{'='*60}")
    print(f"  RMSD zero-loss verification (GT as prediction)")
    print(f"{'='*60}")

    gt_bb = np.array(scaffold.backbone_coordinates())  # [M, 4, 3]

    gt_output = SimpleNamespace(
        backbone_coordinates=gt_bb,
    )
    gt_pssm = np.zeros((scaffold_length, 20))
    for i, aa in enumerate(scaffold.motif_sequence):
        if aa in TOKENS:
            gt_pssm[i, TOKENS.index(aa)] = 1.0

    v_gt, aux_gt = rmsd_loss(
        sequence=gt_pssm,
        output=gt_output,
        key=jax.random.key(42),
    )
    print(f"RMSD (GT pred): {v_gt}")
    print(f"aux: {aux_gt}")

