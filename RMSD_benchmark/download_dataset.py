#!/usr/bin/env python3
"""
Download a post-training-cutoff monomer dataset from RCSB PDB for the
diffusion steps benchmark.

Queries for single-chain monomer X-ray structures deposited after a cutoff
date, stratified by sequence length into bins, then downloads CIF files.

Usage:
    python download_dataset.py [--output-dir dataset_post_cutoff] [--cutoff-date 2025-06-30]
"""

import argparse
import json
import random
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"

LENGTH_BINS = [
    (50, 100),
    (100, 150),
    (150, 200),
    (200, 250),
]


def build_query(cutoff_date: str, len_min: int, len_max: int):
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.deposit_date",
                        "operator": "greater",
                        "value": cutoff_date,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.0,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 1,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "equals",
                        "value": 1,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "range",
                        "value": {"from": len_min, "to": len_max, "include_lower": True, "include_upper": False},
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 500},
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }


def run_query(query: dict) -> list[str]:
    body = json.dumps(query).encode()
    req = Request(RCSB_SEARCH_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req) as resp:
            data = json.loads(resp.read())
    except HTTPError as e:
        if e.code == 204:
            return []
        raise
    return [hit["identifier"] for hit in data.get("result_set", [])]


def download_cif(pdb_id: str, output_dir: Path) -> Path:
    url = f"{RCSB_DOWNLOAD_URL}/{pdb_id}.cif"
    out_path = output_dir / f"{pdb_id}.cif"
    if out_path.exists():
        return out_path
    req = Request(url)
    with urlopen(req) as resp:
        out_path.write_bytes(resp.read())
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "dataset_post_cutoff")
    parser.add_argument("--cutoff-date", default="2025-06-30",
                        help="Only structures deposited after this date (YYYY-MM-DD)")
    parser.add_argument("--per-bin", type=int, default=7,
                        help="Target number of structures per length bin")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = {}
    for lo, hi in LENGTH_BINS:
        label = f"{lo}-{hi}"
        print(f"Querying RCSB for length bin {label} (deposited after {args.cutoff_date})...")
        query = build_query(args.cutoff_date, lo, hi)
        hits = run_query(query)
        print(f"  Found {len(hits)} entries")

        if len(hits) > args.per_bin:
            picks = random.sample(hits, args.per_bin)
        else:
            picks = hits
        selected[label] = picks
        print(f"  Selected {len(picks)}: {picks}")

    all_ids = []
    manifest = []
    for label, ids in selected.items():
        for pdb_id in ids:
            all_ids.append(pdb_id)
            manifest.append({"pdb_id": pdb_id, "length_bin": label})

    print(f"\nDownloading {len(all_ids)} structures...")
    for i, entry in enumerate(manifest):
        pdb_id = entry["pdb_id"]
        print(f"  [{i+1}/{len(all_ids)}] {pdb_id} (bin {entry['length_bin']})...", end=" ", flush=True)
        try:
            path = download_cif(pdb_id, args.output_dir)
            entry["file"] = path.name
            entry["status"] = "ok"
            print("ok")
        except Exception as e:
            entry["file"] = ""
            entry["status"] = f"FAILED: {e}"
            print(f"FAILED: {e}")
        time.sleep(0.25)

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    ok = sum(1 for e in manifest if e["status"] == "ok")
    print(f"\nDone: {ok}/{len(all_ids)} downloaded to {args.output_dir}")
    print(f"Manifest: {manifest_path}")

    print("\nPer-bin counts:")
    for label, ids in selected.items():
        print(f"  {label}: {len(ids)}")


if __name__ == "__main__":
    main()
