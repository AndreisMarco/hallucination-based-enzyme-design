import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Array, Bool

from mosaic.common import LossTerm, restype_three_to_one, TOKENS
from mosaic.losses.atom37 import ATOM37_INDEX
from mosaic.structure_prediction import StructureModelOutput
from mosaic.util import kabsch, gram_schmidt
from mosaic.structure_prediction import TargetChain

import biotite.structure as struct
from biotite.structure.io import pdb, pdbx
import gemmi


class Scaffold():
    def __init__(self,
                 path_to_structure: str,
                 keep_intervals: str,
                 loops: list[int],
                 order: list[int] | None = None,
                 length: int | None = None,
                 ):

        def str_to_intervals(intervals_str: str):
            parts = intervals_str.split(",")
            intervals = []
            for interval in parts:
                if "-" not in interval:
                    raise ValueError(f"Expected format 'start-end', got: '{interval}'")

                split = interval.split("-")
                if len(split) != 2:
                    raise ValueError(f"Expected exactly one '-' separator, got: '{interval}'")

                start_str, end_str = split
                if not start_str.strip().isdigit() or not end_str.strip().isdigit():
                    raise ValueError(f"Expected integer values, got: '{interval}'")

                intervals.append((int(start_str), int(end_str)))

            for interval in intervals:
                if interval[0] > interval[1]:
                    raise ValueError(
                        f"Interval start must be smaller or equal to end, got start:{interval[0]} end:{interval[1]}"
                    )

            # check for overlapping intervals
            sorted_intervals = sorted(intervals)
            for i in range(len(sorted_intervals) - 1):
                if sorted_intervals[i][1] >= sorted_intervals[i+1][0]:
                    raise ValueError(
                        f"Intervals must not overlap: {sorted_intervals[i]} and {sorted_intervals[i+1]}"
                    )
            return intervals

        def extract_fragment(structure, start, end):
            frag = structure[(structure.res_id >= start) & (structure.res_id <= end)]
            ca_atoms = frag[frag.atom_name == "CA"]
            n_res = len(ca_atoms)
            a37_coords = np.zeros((n_res, 37, 3), dtype=np.float32)
            a37_mask   = np.zeros((n_res, 37),    dtype=np.float32)
            for i, rid in enumerate(ca_atoms.res_id):
                res_atoms = frag[frag.res_id == rid]
                for atom_name, coord in zip(res_atoms.atom_name, res_atoms.coord):
                    idx = ATOM37_INDEX.get(atom_name)
                    if idx is not None:
                        a37_coords[i, idx] = coord
                        a37_mask[i, idx]   = 1.0
            sequence = "".join(restype_three_to_one.get(name, "X") for name in ca_atoms.res_name)
            return a37_coords, a37_mask, sequence

        def make_loop(n):
            a37  = np.zeros((n, 37, 3), dtype=np.float32)
            a37m = np.zeros((n, 37),    dtype=np.float32)
            seq  = "X" * n
            return a37, a37m, seq

        # convert string of intervals to list[tuple[int]]
        keep_intervals = str_to_intervals(keep_intervals)

        # adjust last loop so total scaffold length == length
        if length is not None:
            loops = list(loops)
            motif_residues = sum(end - start + 1 for start, end in keep_intervals)
            adjusted_last = length - sum(loops[:-1]) - motif_residues
            if adjusted_last < 0:
                raise ValueError(
                    f"specified length={length} is too short to accommodate "
                    f"the scaffold (motifs={motif_residues} + preceding loops={sum(loops[:-1])} residues). "
                    f"Increase length or reduce loops."
                )
            if adjusted_last != loops[-1]:
                print(f"[Scaffold] Adjusting last loop: {loops[-1]} -> {adjusted_last} to match specified length={length}")
                loops[-1] = adjusted_last

        # check intervals/loops/order consistency
        assert len(loops) == len(order) + 1, \
            f"loops must have length len(order)+1={len(order)+1}, got {len(loops)}"

        assert len(order) == len(keep_intervals) and max(order) == len(keep_intervals) - 1, \
            f"order must include an idx for each element of keep_intervals " \
            f"starting from 0 to len(keep_intervals)-1={len(keep_intervals)-1}"

        # load and filter to amino acids
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

        # assemble scaffold from fragments and loops
        all_a37, all_a37m, all_seq, all_mask = [], [], [], []
        # per-scaffold-position source residue id (None for loop positions)
        template_seqids: list[int | None] = []
        for i, frag_idx in enumerate(order):
            # create loop start/middle loop
            if loops[i] > 0:
                a37, a37m, seq = make_loop(loops[i])
                all_a37.append(a37)
                all_a37m.append(a37m)
                all_seq.append(seq)
                all_mask.append(np.zeros(loops[i], dtype=bool))
                template_seqids.extend([None] * loops[i])
            # create keep intervals
            start, end = keep_intervals[frag_idx]
            a37, a37m, seq = extract_fragment(structure, start, end)
            L = a37.shape[0]
            all_a37.append(a37)
            all_a37m.append(a37m)
            all_seq.append(seq)
            all_mask.append(np.ones(L, dtype=bool))
            template_seqids.extend(range(start, end + 1))
        # create last loop
        if loops[-1] > 0:
            a37, a37m, seq = make_loop(loops[-1])
            all_a37.append(a37)
            all_a37m.append(a37m)
            all_seq.append(seq)
            all_mask.append(np.zeros(loops[-1], dtype=bool))
            template_seqids.extend([None] * loops[-1])
        self._template_seqids = template_seqids

        # store coords and others
        a37_coords = np.concatenate(all_a37,  axis=0)  # [L_total, 37, 3]
        a37_mask   = np.concatenate(all_a37m, axis=0)  # [L_total, 37]
        valid      = np.concatenate(all_mask, axis=0)  # [L_total]

        self._sequence = "".join(all_seq)
        self.mask = jnp.array(valid)                     # [L_total] bool
        self._atom37_coords = jnp.array(a37_coords)      # [L_total, 37, 3]
        self._atom37_mask   = jnp.array(a37_mask)         # [L_total, 37]

    @property
    def sequence(self) -> str:
        return self._sequence

    def __len__(self) -> int:
        return len(self._sequence)

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
        pair_mask = self.mask[:, None] & self.mask[None, :]
        return jnp.where(pair_mask, dist, 0.0)

    def backbone_frames(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        bb = self.backbone_coordinates()
        n_coords  = bb[:, 0, :]
        ca_coords = bb[:, 1, :]
        c_coords  = bb[:, 2, :]

        R = gram_schmidt(v1=c_coords - ca_coords, v2=n_coords - ca_coords)
        identity = jnp.broadcast_to(jnp.eye(3), R.shape)
        R = jnp.where(self.mask[:, None, None], R, identity)
        t = jnp.where(self.mask[:, None], ca_coords, 0.0)
        return t, R

    def pssm(self, key) -> jnp.ndarray:
        aa_indices = jnp.array([
            TOKENS.index(aa) if aa in TOKENS else 0
            for aa in self._sequence
        ])
        pssm = jax.random.gumbel(key, shape=(len(self), 20))
        pssm = nn.softmax(0.5 * pssm, axis=-1)
        onehot = jax.nn.one_hot(aa_indices, num_classes=20)
        return jnp.where(self.mask[:, None], onehot, pssm)

    def build_chain(self, use_msa: bool = False, use_template: bool = False):
        template_chain = None
        template_mask = None

        if use_template:
            # source structure (full atoms) for the motif residues
            src_st = gemmi.read_structure(self.structure_path)
            src_st.remove_ligands_and_waters()
            src_chain = src_st[0][0]
            src_by_seqid = {r.seqid.num: r for r in src_chain}

            # build a new chain that follows the scaffold layout: motif residues
            # are cloned in their scaffold positions; loop slots are UNK
            # placeholders with no atoms (-> template_all_atom_mask=0 there).
            new_chain = gemmi.Chain("A")
            for new_idx, src_seqid in enumerate(self._template_seqids, start=1):
                if src_seqid is None:
                    r = gemmi.Residue()
                    r.name = "UNK"
                else:
                    r = src_by_seqid[src_seqid].clone()
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

class DistogramCCE(LossTerm):
    gt_distogram: Float[Array, "N N"]
    mask: Float[Array, "N"]
    _idx: Float[Array, "M"]
    name: str = "dgramm_cce"

    def __init__(self, gt_distogram, mask, name: str = "dgramm_cce"):
        self.gt_distogram = gt_distogram
        self.mask = mask
        self._idx = jnp.where(mask, size=int(mask.sum()))[0]
        self.name = name

    @classmethod
    def from_scaffold(cls, scaffold: Scaffold, name: str = "dgramm_cce"):
        return cls(gt_distogram=scaffold.distogram(), mask=scaffold.mask, name=name)

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        # create gt distogram for scaffolded positions
        pred_logits  = output.distogram_logits[self._idx][:, self._idx]
        gt_distogram = self.gt_distogram[self._idx][:, self._idx]
        num_bins   = pred_logits.shape[-1]
        # adapt bins to the values specified by the model
        bin_edges  = jnp.linspace(output.distogram_bins[0], output.distogram_bins[-1], num_bins - 1)
        gt_indices = (gt_distogram[..., None] > bin_edges).sum(-1)
        gt_one_hot = nn.one_hot(gt_indices, num_classes=num_bins)

        # compute CCE
        loss = -jnp.sum(gt_one_hot * nn.log_softmax(pred_logits, axis=-1), axis=-1)
        dgramm_cce = jnp.mean(loss)
        return dgramm_cce, {self.name: dgramm_cce}

class FAPE(LossTerm):
    gt_t: Float[Array, "N 3"]
    gt_R: Float[Array, "N 3 3"]
    mask: Float[Array, "N"]
    _idx: Float[Array, "M"]
    name: str = "fape"
    clamp: bool = False

    def __init__(self, gt_t, gt_R, mask, name: str = "fape", clamp: bool = False):
        self.gt_t = gt_t
        self.gt_R = gt_R
        self.mask = mask
        self._idx = jnp.where(mask, size=int(mask.sum()))[0]
        self.name = name
        self.clamp = clamp

    @classmethod
    def from_scaffold(cls, scaffold: Scaffold, name: str = "fape", clamp: bool = False):
        gt_t, gt_R = scaffold.backbone_frames()
        return cls(gt_t=gt_t, gt_R=gt_R, mask=scaffold.mask, name=name, clamp=clamp)

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        def robust_norm(x, eps=1e-8):
            return jnp.sqrt(jnp.square(x).sum(axis=-1) + eps)

        def get_ij(R, t):
            return jnp.einsum("rji, rsj -> rsi", R, t[None, :] - t[:, None])

        # only keep scaffold positions
        pred_bb = output.backbone_coordinates[self._idx]
        gt_R    = self.gt_R[self._idx]
        gt_t    = self.gt_t[self._idx]

        # compute rotation on prediction backbones
        pred_R = gram_schmidt(
            v1=pred_bb[:, 2, :] - pred_bb[:, 1, :],   # CA -> C
            v2=pred_bb[:, 0, :] - pred_bb[:, 1, :],   # CA -> N
        )
        pred_t = pred_bb[:, 1, :]
        pred_ij = get_ij(pred_R, pred_t)
        gt_ij = get_ij(gt_R, gt_t)

        # compute FAPE
        fape = robust_norm(pred_ij - gt_ij)
        if self.clamp:
            fape = jnp.clip(fape, 0.0, 10.0) / 10.0
        fape = fape.mean()

        return fape, {self.name: fape}

class RMSD(LossTerm):
    _gt: Float[Array, "K 3"]
    _weights: Float[Array, "K 1"] | None
    mask: Float[Array, "N"]
    _idx: Float[Array, "M"]
    _atom_idx: Float[Array, "K"] | None
    _mode: str = "backbone"
    _weighted: bool = False
    name: str = "rmsd"

    def __init__(self, gt_coords, mask, name: str = "rmsd", mode: str = "backbone",
                 weighted: bool = False, gt_atom37_mask=None):
        self.mask = mask
        self._idx = jnp.where(mask, size=int(mask.sum()))[0]
        self.name = name
        self._mode = mode
        self._weighted = weighted
        self._atom_idx = None
        self._weights = None

        gt_scaffold = gt_coords[self._idx]
        if mode in ("all_atom", "side_chain"):
            mask37 = gt_atom37_mask[self._idx]             # [M, 37]
            if mode == "side_chain":
                backbone_idx = jnp.array([0, 2, 4])          # N, C, O (keep CA)
                mask37 = mask37.at[:, backbone_idx].set(0.0)
            flat_mask = mask37.reshape(-1)
            # transform atom37 [M, 37, 3] to atom list [n_atoms, 3]
            n_atoms = int(flat_mask.sum())
            self._atom_idx = jnp.where(flat_mask, size=n_atoms)[0]
            self._gt = gt_scaffold.reshape(-1, 3)[self._atom_idx]
            if weighted:
                n_res = mask37.shape[0]
                atoms_per_res = mask37.sum(-1, keepdims=True)
                # all residues contribute the same
                per_atom_w = jnp.where(mask37, 1.0 / (n_res * atoms_per_res + 1e-8), 0.0)
                self._weights = per_atom_w.reshape(-1)[self._atom_idx][..., None]
        elif mode == "ca_only":
            self._gt = gt_scaffold[:, 1, :]
        else:
            self._gt = gt_scaffold.reshape(-1, 3)

        if weighted and self._weights is None:
            length = self._gt.shape[0]
            self._weights = (jnp.ones(length) / length)[..., None]

    @classmethod
    def from_scaffold(cls, scaffold: Scaffold, mode: str = "backbone",
                      weighted: bool = False, name: str = "rmsd"):
        assert mode in ("ca_only", "backbone", "all_atom", "side_chain"), \
            f"Unknown RMSD loss mode {mode}, available ca_only, backbone, all_atom, side_chain"
        if mode in ("all_atom", "side_chain"):
            coords, atom37_mask = scaffold.atom37_coordinates()
            return cls(gt_coords=coords, mask=scaffold.mask, name=name,
                       mode=mode, weighted=weighted, gt_atom37_mask=atom37_mask)
        return cls(gt_coords=scaffold.backbone_coordinates(), mask=scaffold.mask,
                   name=name, mode=mode, weighted=weighted)

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        if self._mode in ("all_atom", "side_chain"):
            pred = output.atom37_coords[self._idx].reshape(-1, 3)[self._atom_idx]
        elif self._mode == "ca_only":
            pred = output.backbone_coordinates[self._idx, 1, :]
        else:
            pred = output.backbone_coordinates[self._idx].reshape(-1, 3)
        # align prediction with gt
        R, t = kabsch(pred, self._gt, weights=self._weights)
        pred_aligned = pred @ R + t
        # compute rmsd
        if self._weighted:
            msd = (self._weights * jnp.square(pred_aligned - self._gt)).sum((-1, -2))
        else:
            msd = jnp.mean(jnp.sum((pred_aligned - self._gt) ** 2, axis=-1))
        rmsd = jnp.sqrt(msd)
        return rmsd, {self.name: rmsd}

class MaskedPLDDTLoss(LossTerm):
    _idx: Float[Array, "M"]
    name: str = "masked_plddt"

    def __init__(self, mask, name: str = "masked_plddt"):
        self._idx = jnp.where(mask, size=int(mask.sum()))[0]
        self.name = name

    @classmethod
    def from_scaffold(cls, scaffold: Scaffold, name: str = "masked_plddt", inverted: bool = False ):
        mask = scaffold.mask if not inverted else ~scaffold.mask
        return cls(mask=mask, name=name)


    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
        **kwargs,
    ):
        binder_len = sequence.shape[0]
        plddt = output.plddt[:binder_len][self._idx]
        return -plddt.mean(), {self.name: plddt.mean()}

if __name__ == "__main__":
    import os
    os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

    scaffold = Scaffold(
        path_to_structure="structures/1LNS.cif",
        keep_intervals="348-358,461-468,498-498",
        order=[0,2,1],
        loops=[5,4,6,6]
    )

    print("len scaffold:", len(scaffold))
    print("full sequence:", scaffold.sequence)
    print("mask shape:", scaffold.mask.shape)
    print("distogram shape:", scaffold.distogram().shape)
    print("backbone_coords shape:", scaffold.backbone_coordinates().shape)
    t, R = scaffold.backbone_frames()
    print("backbone_frames (t, R):", t.shape, R.shape)

    scaffold_length = len(scaffold)
    np.random.seed(42)
    pred_backbone_coords  = np.random.normal(size=(scaffold_length, 4, 3))
    pred_distogram_logits = np.random.normal(size=(scaffold_length, scaffold_length, 64))
    pred_atom37_coords    = np.random.normal(size=(scaffold_length, 37, 3))
    pred_atom37_mask      = np.ones((scaffold_length, 37))
    bins = np.linspace(2, 22, 64)

    a37_coords, a37_mask = scaffold.atom37_coordinates()
    print("atom37_coords shape:", a37_coords.shape)
    print("atom37_mask shape:", a37_mask.shape)
    print("atom37 atoms per motif residue:", a37_mask[scaffold.mask.astype(bool)].sum(-1))

    from types import SimpleNamespace
    test = SimpleNamespace(
        backbone_coordinates=pred_backbone_coords,
        distogram_logits=pred_distogram_logits,
        distogram_bins=bins,
        plddt=np.random.random(size=(len(scaffold))),
        atom37_coords=pred_atom37_coords,
        atom37_mask=pred_atom37_mask,
    )

    structure_loss = (
        RMSD.from_scaffold(scaffold=scaffold, name="rmsd_backbone") +
        RMSD.from_scaffold(scaffold=scaffold, mode="all_atom", name="rmsd_all_atom") +
        RMSD.from_scaffold(scaffold=scaffold, weighted=True, name="wrmsd_backbone") +
        RMSD.from_scaffold(scaffold=scaffold, mode="all_atom", weighted=True, name="wrmsd_all_atom") +
        FAPE.from_scaffold(scaffold=scaffold) +
        DistogramCCE.from_scaffold(scaffold=scaffold) +
        MaskedPLDDTLoss.from_scaffold(scaffold=scaffold, name="scaffold_plddt") +
        MaskedPLDDTLoss.from_scaffold(scaffold=scaffold, name="non_scaffold_plddt", inverted=True)
        )

    v, aux = structure_loss(
        sequence=np.random.normal(size=(scaffold_length, 20)),
        output=test,
        key=jax.random.key(42),
    )

    print(f"loss: {v}")
    for path, leaf in jax.tree.leaves_with_path(aux):
        print(f"{path}: {leaf}")

    # verify GT structure produces near zero loss
    print("\n--- zero-loss verification (GT as prediction) ---")
    gt_bb = np.array(scaffold.backbone_coordinates())
    gt_a37, gt_a37m = scaffold.atom37_coordinates()
    gt_a37, gt_a37m = np.array(gt_a37), np.array(gt_a37m)
    gt_distogram = np.array(scaffold.distogram())
    num_bins = 64
    bin_edges = np.linspace(bins[0], bins[-1], num_bins - 1)
    gt_bin_idx = (gt_distogram[..., None] > bin_edges).sum(-1)
    gt_logits = np.full((scaffold_length, scaffold_length, num_bins), -1e9)
    np.put_along_axis(gt_logits, gt_bin_idx[..., None], 1e9, axis=-1)

    gt_test = SimpleNamespace(
        backbone_coordinates=gt_bb,
        distogram_logits=gt_logits,
        distogram_bins=bins,
        plddt=np.ones(scaffold_length),
        atom37_coords=gt_a37,
        atom37_mask=gt_a37m,
    )

    v_gt, aux_gt = structure_loss(
        sequence=np.random.normal(size=(scaffold_length, 20)),
        output=gt_test,
        key=jax.random.key(42),
    )
    print(f"loss (GT): {v_gt}")
    for path, leaf in jax.tree.leaves_with_path(aux_gt):
        print(f"{path}: {leaf}")