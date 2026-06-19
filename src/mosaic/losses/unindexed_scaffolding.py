import jax.numpy as jnp
import numpy as np

from mosaic.common import restype_three_to_one, LossTerm
from mosaic.losses.atom37 import ATOM37_INDEX
from mosaic.util import gram_schmidt, kabsch
from mosaic.structure_prediction import TargetChain, StructureModelOutput
from jaxtyping import Float, Array, Bool


import biotite.structure as struct
from biotite.structure.io import pdb, pdbx
import gemmi


class Scaffold:
    def __init__(self, path_to_structure: str, scaffold_pos: str | None = None,
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

        if scaffold_pos is not None:
            intervals = self._parse_intervals(scaffold_pos)
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
                + (" in specified intervals" if scaffold_pos else " (no non-ALA residues)")
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

class RMSD(LossTerm):
    name: str = "rmsd"
    gt_: Float[Array, "K 3"]

    def __init__(self):
        raise NotImplementedError

    @classmethod
    def from_scaffold(cls, scaffold: Scaffold, name: str="rmsd"):
        raise NotImplementedError

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--theozyme_pdb", default="structures/theozyme.pdb")
    args = parser.parse_args()
    pdb_path = args.theozyme_pdb

    def print_scaffold(scaffold, label):
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print("scaffold length:", len(scaffold))
        print("sequence:", scaffold.sequence)
        print("motif_sequence:", scaffold.motif_sequence)
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

    # --- Mode 2: unindexed, explicit scaffold_pos ---
    s2 = Scaffold(pdb_path, scaffold_pos="56, 58, 84, 114", length=100)
    print_scaffold(s2, "Unindexed — explicit positions (56, 58, 84, 114), length=100")

    # --- Mode 3: unindexed, scaffold_pos with ranges ---
    s3 = Scaffold(pdb_path, scaffold_pos="56-58, 84, 114", length=100)
    print_scaffold(s3, "Unindexed — mixed ranges (56-58, 84, 114), length=100")

    # --- Mode 4: indexed, auto-detect motif + loops (sequential order) ---
    s4 = Scaffold(pdb_path, loops=[10, 5, 5, 5, 10])
    print_scaffold(s4, "Indexed — auto-detect + loops [10,5,5,5,10]")

    # --- Mode 5: indexed, explicit positions + loops + reorder ---
    s5 = Scaffold(pdb_path, scaffold_pos="56, 58, 84, 114",
                  loops=[10, 5, 5, 5, 10], order=[3, 1, 2, 0])
    print_scaffold(s5, "Indexed — explicit + loops + order [3,1,2,0]")

    # --- Mode 6: indexed with target length ---
    s6 = Scaffold(pdb_path, loops=[10, 5, 5, 5, 10], length=50)
    print_scaffold(s6, "Indexed — auto-detect + loops + length=50")

