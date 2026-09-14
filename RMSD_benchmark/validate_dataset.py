#!/usr/bin/env python3
"""
Validate downloaded structures for the diffusion steps benchmark.

Checks each CIF/PDB file for:
  - parsability
  - single protein chain
  - missing CA atoms / backbone gaps
  - non-standard residues
  - sequence length within expected range

Usage:
    python validate_dataset.py [dataset_dir]
"""

import sys
from pathlib import Path

import gemmi

STANDARD_AA = set(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)

# Selenomethionine is a common isomorphous replacement, structurally equivalent to MET
SAFE_NONSTANDARD = {"MSE"}


def validate_structure(path: Path) -> list[str]:
    issues = []

    try:
        st = gemmi.read_structure(str(path))
    except Exception as e:
        return [f"PARSE ERROR: {e}"]

    st.remove_ligands_and_waters()

    if len(st) == 0 or len(st[0]) == 0:
        return ["No models/chains after removing ligands and waters"]

    protein_chains = []
    for chain in st[0]:
        residues = [r for r in chain if r.entity_type == gemmi.EntityType.Polymer]
        if len(residues) > 0:
            protein_chains.append(chain)

    if len(protein_chains) == 0:
        return ["No polymer chains found"]
    if len(protein_chains) > 1:
        issues.append(f"Multiple polymer chains: {[c.name for c in protein_chains]}")

    chain = protein_chains[0]
    polymer = chain.get_polymer()
    residues = [r for r in polymer if r.entity_type == gemmi.EntityType.Polymer]
    seq_len = len(residues)

    if seq_len < 50 or seq_len > 250:
        issues.append(f"Sequence length {seq_len} outside expected range 50-250")

    missing_ca = 0
    nonstandard = []
    for res in residues:
        ca = res.find_atom("CA", "\0")
        if ca is None:
            missing_ca += 1

        if res.name not in STANDARD_AA and res.name not in SAFE_NONSTANDARD:
            nonstandard.append(f"{res.name}{res.seqid}")

    if missing_ca > 0:
        pct = 100 * missing_ca / seq_len
        issues.append(f"Missing CA atoms: {missing_ca}/{seq_len} ({pct:.0f}%)")

    if nonstandard:
        if len(nonstandard) <= 5:
            issues.append(f"Non-standard residues: {', '.join(nonstandard)}")
        else:
            issues.append(f"Non-standard residues: {len(nonstandard)} total ({', '.join(nonstandard[:5])}, ...)")

    altloc_count = sum(1 for res in residues if any(a.altloc != "\0" for a in res))
    if altloc_count > 0:
        issues.append(f"Residues with altlocs: {altloc_count}")

    if not issues:
        issues.append(f"OK (len={seq_len})")

    return issues


def main():
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "dataset_post_cutoff"

    files = sorted(list(dataset_dir.glob("*.cif")) + list(dataset_dir.glob("*.pdb")))
    if not files:
        print(f"No structure files found in {dataset_dir}")
        sys.exit(1)

    print(f"Validating {len(files)} structures in {dataset_dir}\n")

    problems = []
    for f in files:
        issues = validate_structure(f)
        status = "OK" if issues[0].startswith("OK") else "WARN"
        print(f"  {f.name:20s}  {status:4s}  {'; '.join(issues)}")
        if status == "WARN":
            problems.append((f.name, issues))

    print(f"\n{len(files) - len(problems)}/{len(files)} passed, {len(problems)} with warnings")
    if problems:
        print("\nStructures to review:")
        for name, issues in problems:
            print(f"  {name}: {'; '.join(issues)}")


if __name__ == "__main__":
    main()
