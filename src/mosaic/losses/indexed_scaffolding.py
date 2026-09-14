# Assumes single-chain structures with integer-only residue numbering (no insertion codes).

import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np

import gemmi

from mosaic.common import restype_three_to_one, TOKENS, LossTerm
from mosaic.losses.atom37 import ATOM37_INDEX
from mosaic.util import gram_schmidt, kabsch
from mosaic.structure_prediction import TargetChain, StructureModelOutput


class IndexedScaffold:
    def __init__(self,
                 path_to_structure: str,
                 keep_intervals: str,
                 loops: list[int],
                 order: list[int] | None = None,
                 length: int | None = None):

        self._gemmi_structure = gemmi.read_structure(str(path_to_structure))
        intervals = self._parse_intervals(keep_intervals)

        motif_intervals = []
        for start, end in intervals:
            rids, cids = [], []
            for chain in self._gemmi_structure[0]:
                for res in chain.first_conformer():
                    if gemmi.find_tabulated_residue(res.name).is_amino_acid() \
                       and start <= res.seqid.num <= end:
                        rids.append(res.seqid.num)
                        cids.append(chain.name)
            if not rids:
                raise ValueError(f"No residues found in interval {start}-{end}")
            motif_intervals.append((rids, cids))
        n_intervals = len(motif_intervals)

        if order is None:
            order = list(range(n_intervals))

        assert len(loops) == len(order) + 1, \
            f"loops must have length len(order)+1={len(order)+1}, got {len(loops)}"
        assert len(order) == n_intervals and max(order) == n_intervals - 1, \
            f"order must be a permutation of 0..{n_intervals - 1}"

        if length is not None:
            loops = list(loops)
            motif_count = sum(len(ids) for ids, _ in motif_intervals)
            adjusted_last = length - sum(loops[:-1]) - motif_count
            if adjusted_last < 0:
                raise ValueError(
                    f"specified length={length} is too short to accommodate "
                    f"the scaffold (motifs={motif_count} + preceding loops={sum(loops[:-1])}). "
                    f"Increase length or reduce loops."
                )
            if adjusted_last != loops[-1]:
                print(f"[scaffold] Adjusting last loop: {loops[-1]} -> {adjusted_last} "
                        f"to match specified length={length}")
                loops[-1] = adjusted_last

        all_a37, all_a37m, all_seq, all_mask = [], [], [], []
        template_seqids: list[int | None] = []
        template_chain_ids: list[str | None] = []

        for i, frag_idx in enumerate(order):
            if loops[i] > 0:
                a37, a37m, seq = self._make_loop(loops[i])
                all_a37.append(a37)
                all_a37m.append(a37m)
                all_seq.append(seq)
                all_mask.append(np.zeros(loops[i], dtype=bool))
                template_seqids.extend([None] * loops[i])
                template_chain_ids.extend([None] * loops[i])

            frag_rids, frag_cids = motif_intervals[frag_idx]
            for cid, rid in zip(frag_cids, frag_rids):
                a37, a37m, seq = self._extract_residue(self._gemmi_structure, cid, rid)
                all_a37.append(a37)
                all_a37m.append(a37m)
                all_seq.append(seq)
                all_mask.append(np.ones(1, dtype=bool))
                template_seqids.append(int(rid))
                template_chain_ids.append(str(cid))

        if loops[-1] > 0:
            a37, a37m, seq = self._make_loop(loops[-1])
            all_a37.append(a37)
            all_a37m.append(a37m)
            all_seq.append(seq)
            all_mask.append(np.zeros(loops[-1], dtype=bool))
            template_seqids.extend([None] * loops[-1])
            template_chain_ids.extend([None] * loops[-1])

        self._sequence = "".join(all_seq)
        self._motif_sequence = "".join(c for c in self.sequence if c != "X")
        self._atom37_coords = jnp.array(np.concatenate(all_a37, axis=0))
        self._atom37_mask = jnp.array(np.concatenate(all_a37m, axis=0))
        self._mask = jnp.array(np.concatenate(all_mask, axis=0))
        self._template_seqids = template_seqids
        self._template_chain_ids = template_chain_ids
        self._residue_labels = [
            f"{c}{s}" for c, s, m in zip(template_chain_ids, template_seqids, self._mask)
            if m
        ]

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

    @staticmethod
    def _extract_residue(structure, cid, rid):
        res = structure[0].find_residue_group(cid, gemmi.SeqId(rid, " "))[0]
        a37_coords = np.zeros((1, 37, 3), dtype=np.float32)
        a37_mask = np.zeros((1, 37), dtype=np.float32)
        for atom in res:
            idx = ATOM37_INDEX.get(atom.name)
            if idx is not None:
                a37_coords[0, idx] = [atom.pos.x, atom.pos.y, atom.pos.z]
                a37_mask[0, idx] = 1.0
        seq = restype_three_to_one.get(res.name, "X")
        return a37_coords, a37_mask, seq

    @staticmethod
    def _make_loop(n):
        a37 = np.zeros((n, 37, 3), dtype=np.float32)
        a37m = np.zeros((n, 37), dtype=np.float32)
        seq = "X" * n
        return a37, a37m, seq

    @property
    def sequence(self) -> str:
        return self._sequence

    @property
    def motif_sequence(self) -> str:
        return self._motif_sequence

    @property
    def mask(self) -> jnp.ndarray:
        return self._mask

    @property
    def atom37_coordinates(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self._atom37_coords, self._atom37_mask

    def __len__(self) -> int:
        return len(self._sequence)

    def pssm(self) -> jnp.ndarray:
        aa_indices = jnp.array([TOKENS.index(aa) if aa in TOKENS else 0 for aa in self._sequence])
        onehot = jax.nn.one_hot(aa_indices, num_classes=20)
        return jnp.where(self.mask[:, None], onehot, 0.0)

    def backbone_coordinates(self) -> jnp.ndarray:
        bb_idx = jnp.array([ATOM37_INDEX[a] for a in ("N", "CA", "C", "O")])
        return self._atom37_coords[:, bb_idx, :]

    def distogram(self) -> jnp.ndarray:
        # the standard in AF2 is to have distogram computed on pseudo-cb
        ca = self._atom37_coords[:, ATOM37_INDEX["CA"], :]
        cb = self._atom37_coords[:, ATOM37_INDEX["CB"], :]
        has_cb = self._atom37_mask[:, ATOM37_INDEX["CB"]].astype(bool)
        pb = jnp.where(has_cb[:, None], cb, ca)

        diff = pb[:, None, :] - pb[None, :, :]
        dist = jnp.linalg.norm(diff, axis=-1)
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

        if use_template:
            src_by_key = {}
            for chain in self._gemmi_structure[0]:
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

class DistogramCCE(LossTerm):
    gt_distogram: jnp.ndarray
    mask: jnp.ndarray
    _idx: jnp.ndarray
    name: str = "dgram_cce"

    def __init__(self, gt_distogram, mask, name: str = "dgram_cce"):
        self.gt_distogram = gt_distogram
        self.mask = mask
        self._idx = jnp.where(mask, size=int(mask.sum()))[0]
        self.name = name

    @classmethod
    def from_scaffold(cls, scaffold: IndexedScaffold, name: str = "dgram_cce"):
        return cls(gt_distogram=scaffold.distogram(), mask=scaffold.mask, name=name)

    def __call__(self, sequence, output: StructureModelOutput, key, **kwargs):
        pred_logits  = output.distogram_logits[self._idx][:, self._idx]
        gt_distogram = self.gt_distogram[self._idx][:, self._idx]
        num_bins   = pred_logits.shape[-1]
        bin_edges  = jnp.linspace(output.distogram_bins[0], output.distogram_bins[-1], num_bins - 1)
        gt_indices = (gt_distogram[..., None] > bin_edges).sum(-1)
        gt_one_hot = nn.one_hot(gt_indices, num_classes=num_bins)
        loss = -jnp.sum(gt_one_hot * nn.log_softmax(pred_logits, axis=-1), axis=-1)
        dgram_cce = jnp.mean(loss)
        return dgram_cce, {self.name: dgram_cce}


class FAPE(LossTerm):
    gt_t: jnp.ndarray
    gt_R: jnp.ndarray
    mask: jnp.ndarray
    _idx: jnp.ndarray
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
    def from_scaffold(cls, scaffold: IndexedScaffold, name: str = "fape", clamp: bool = False):
        gt_t, gt_R = scaffold.backbone_frames()
        return cls(gt_t=gt_t, gt_R=gt_R, mask=scaffold.mask, name=name, clamp=clamp)

    def __call__(self, sequence, output: StructureModelOutput, key, **kwargs):
        def robust_norm(x, eps=1e-8):
            return jnp.sqrt(jnp.square(x).sum(axis=-1) + eps)

        def get_ij(R, t):
            return jnp.einsum("rji, rsj -> rsi", R, t[None, :] - t[:, None])

        pred_bb = output.backbone_coordinates[self._idx]
        gt_R    = self.gt_R[self._idx]
        gt_t    = self.gt_t[self._idx]

        pred_R = gram_schmidt(
            v1=pred_bb[:, 2, :] - pred_bb[:, 1, :],
            v2=pred_bb[:, 0, :] - pred_bb[:, 1, :],
        )
        pred_t = pred_bb[:, 1, :]
        pred_ij = get_ij(pred_R, pred_t)
        gt_ij = get_ij(gt_R, gt_t)

        fape = robust_norm(pred_ij - gt_ij)
        if self.clamp:
            fape = jnp.clip(fape, 0.0, 10.0) / 10.0
        fape = fape.mean()
        return fape, {self.name: fape}


class RMSD(LossTerm):
    _gt: jnp.ndarray
    _weights: jnp.ndarray | None
    mask: jnp.ndarray
    _idx: jnp.ndarray
    _atom_idx: jnp.ndarray | None
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
            mask37 = gt_atom37_mask[self._idx]
            if mode == "side_chain":
                backbone_idx = jnp.array([0, 2, 4])
                mask37 = mask37.at[:, backbone_idx].set(0.0)
            flat_mask = mask37.reshape(-1)
            n_atoms = int(flat_mask.sum())
            self._atom_idx = jnp.where(flat_mask, size=n_atoms)[0]
            self._gt = gt_scaffold.reshape(-1, 3)[self._atom_idx]
            if weighted:
                n_res = mask37.shape[0]
                atoms_per_res = mask37.sum(-1, keepdims=True)
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
    def from_scaffold(cls, scaffold: IndexedScaffold, mode: str = "backbone",
                      weighted: bool = False, name: str = "rmsd"):
        assert mode in ("ca_only", "backbone", "all_atom", "side_chain"), \
            f"Unknown RMSD loss mode {mode}, available ca_only, backbone, all_atom, side_chain"
        if mode in ("all_atom", "side_chain"):
            coords, atom37_mask = scaffold.atom37_coordinates
            return cls(gt_coords=coords, mask=scaffold.mask, name=name,
                       mode=mode, weighted=weighted, gt_atom37_mask=atom37_mask)
        return cls(gt_coords=scaffold.backbone_coordinates(), mask=scaffold.mask,
                   name=name, mode=mode, weighted=weighted)

    def __call__(self, sequence, output: StructureModelOutput, key, **kwargs):
        if self._mode in ("all_atom", "side_chain"):
            pred = output.atom37_coords[self._idx].reshape(-1, 3)[self._atom_idx]
        elif self._mode == "ca_only":
            pred = output.backbone_coordinates[self._idx, 1, :]
        else:
            pred = output.backbone_coordinates[self._idx].reshape(-1, 3)
        R, t = kabsch(pred, self._gt, weights=self._weights)
        pred_aligned = pred @ R + t
        if self._weighted:
            msd = (self._weights * jnp.square(pred_aligned - self._gt)).sum((-1, -2))
        else:
            msd = jnp.mean(jnp.sum((pred_aligned - self._gt) ** 2, axis=-1))
        rmsd = jnp.sqrt(msd)
        return rmsd, {self.name: rmsd}


class ScaffoldPLDDT(LossTerm):
    _idx: jnp.ndarray
    name: str = "scaffold_plddt"

    def __init__(self, mask, name: str = "scaffold_plddt"):
        self._idx = jnp.where(mask, size=int(mask.sum()))[0]
        self.name = name

    @classmethod
    def from_scaffold(cls, scaffold: IndexedScaffold, name: str = "scaffold_plddt"):
        return cls(mask=scaffold.mask, name=name)

    def __call__(self, sequence, output: StructureModelOutput, key, **kwargs):
        binder_len = sequence.shape[0]
        plddt = output.plddt[:binder_len][self._idx]
        return -plddt.mean(), {self.name: plddt.mean()}


if __name__ == "__main__":
    jax.config.update("jax_platforms", "cpu")

    scaffold = IndexedScaffold(
        path_to_structure="structures/theozyme.pdb",
        keep_intervals="56, 58, 84, 114",
        loops=[35, 1, 26, 30, 24],
        order=[0, 1, 2, 3],
    )

    print(f"length:         {len(scaffold)}")
    print(f"sequence:       {scaffold.sequence}")
    print(f"motif_sequence: {scaffold.motif_sequence}")
    print(f"mask:           {scaffold.mask}")
    print(f"pssm shape:     {scaffold.pssm().shape}")
    print(f"distogram shape:{scaffold.distogram().shape}")

    coords, a37mask = scaffold.atom37_coordinates
    print(f"atom37 coords:  {coords.shape}")
    print(f"atom37 mask:    {a37mask.shape}")
    print(f"atom37 atoms per motif residue: {a37mask[scaffold.mask.astype(bool)].sum(-1)}")

    bb_coords = scaffold.backbone_coordinates()
    print(f"bb coords:      {bb_coords.shape}")

    t, R = scaffold.backbone_frames()
    print(f"frames t:       {t.shape}")
    print(f"frames R:       {R.shape}")

    # --- loss computation test ---
    from types import SimpleNamespace
    scaffold_length = len(scaffold)
    np.random.seed(42)
    bins = np.linspace(2, 22, 64)

    test_output = SimpleNamespace(
        backbone_coordinates=np.random.normal(size=(scaffold_length, 4, 3)),
        distogram_logits=np.random.normal(size=(scaffold_length, scaffold_length, 64)),
        distogram_bins=bins,
        plddt=np.random.random(size=(scaffold_length,)),
        atom37_coords=np.random.normal(size=(scaffold_length, 37, 3)),
        atom37_mask=np.ones((scaffold_length, 37)),
    )

    structure_loss = (
        RMSD.from_scaffold(scaffold=scaffold, name="rmsd_backbone") +
        RMSD.from_scaffold(scaffold=scaffold, mode="ca_only", name="rmsd_ca") +
        RMSD.from_scaffold(scaffold=scaffold, mode="all_atom", name="rmsd_all_atom") +
        RMSD.from_scaffold(scaffold=scaffold, mode="side_chain", name="rmsd_side_chain") +
        RMSD.from_scaffold(scaffold=scaffold, mode="all_atom", weighted=True, name="wrmsd_all_atom") +
        FAPE.from_scaffold(scaffold=scaffold) +
        DistogramCCE.from_scaffold(scaffold=scaffold) +
        ScaffoldPLDDT.from_scaffold(scaffold=scaffold)
    )

    v, aux = structure_loss(
        sequence=np.random.normal(size=(scaffold_length, 20)),
        output=test_output,
        key=jax.random.key(42),
    )
    print(f"\nloss (random pred): {v}")
    for path, leaf in jax.tree.leaves_with_path(aux):
        print(f"  {path}: {leaf}")

    # --- zero-loss verification: GT as prediction ---
    gt_bb = np.array(scaffold.backbone_coordinates())
    gt_a37, gt_a37m = scaffold.atom37_coordinates
    gt_a37, gt_a37m = np.array(gt_a37), np.array(gt_a37m)
    gt_distogram = np.array(scaffold.distogram())
    num_bins = 64
    bin_edges = np.linspace(bins[0], bins[-1], num_bins - 1)
    gt_bin_idx = (gt_distogram[..., None] > bin_edges).sum(-1)
    gt_logits = np.full((scaffold_length, scaffold_length, num_bins), -1e9)
    np.put_along_axis(gt_logits, gt_bin_idx[..., None], 1e9, axis=-1)

    gt_output = SimpleNamespace(
        backbone_coordinates=gt_bb,
        distogram_logits=gt_logits,
        distogram_bins=bins,
        plddt=np.ones(scaffold_length),
        atom37_coords=gt_a37,
        atom37_mask=gt_a37m,
    )

    v_gt, aux_gt = structure_loss(
        sequence=np.random.normal(size=(scaffold_length, 20)),
        output=gt_output,
        key=jax.random.key(42),
    )
    print(f"\nloss (GT, expect ~-1 from plddt): {v_gt}")
    for path, leaf in jax.tree.leaves_with_path(aux_gt):
        print(f"  {path}: {leaf}")

