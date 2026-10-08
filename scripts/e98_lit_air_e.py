#!/usr/bin/env python3
"""Transfer the frozen E74-E acquisition rule to E15-matched LIT-PCBA.

E74-E parameters (negative=0.15, diversity=0.10) were fixed on TDC-HTS;
they are not selected or modified using LIT-PCBA outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import DataStructs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e74_tdc_local_risk_candidates import acquire  # noqa: E402
from e90_lit_classic_mechanism import sha, write_new_json  # noqa: E402
from run_e15_matched100k import ASSAYS, load_matched_pool, official_initial  # noqa: E402

METHOD = "AIR-E-transfer"
NEGATIVE_PENALTY = 0.15
DIVERSITY_PENALTY = 0.10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assay", choices=ASSAYS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--inject", choices=("none", "crash", "sleep"), default="none")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("AIR-E-transfer must be CPU-only")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "4":
            raise RuntimeError(f"{key} must be exactly 4")
    if args.seed not in range(47001, 47011):
        raise ValueError("Seed outside E15 candidate range")
    ids, packed, y, audit = load_matched_pool(args.assay)
    if audit["status"] != "pass" or args.seed not in audit["selected_seeds"]:
        raise RuntimeError("E15 common-pool audit failed")
    init = official_initial(y, args.seed)
    if len(init) != 64 or len(np.unique(init)) != 64:
        raise RuntimeError("E15 initial set mismatch")
    source = ROOT / "external_baselines/official_code/repos/glare/data" / args.assay / "original/screen.csv"
    source_hash = sha(source)
    init_hash = hashlib.sha256(init.tobytes()).hexdigest()
    prefix = f"e98_lit_air_e_{args.assay}_seed{args.seed}_q192"
    out = ROOT / "tables" / f"{prefix}.csv"
    query = ROOT / "tables" / f"{prefix}_query.csv"
    marker = ROOT / "tables" / f"{prefix}_manifest.json"
    if marker.exists():
        info = json.loads(marker.read_text(encoding="utf-8"))
        if (info.get("status") != "complete" or info.get("source_sha256") != source_hash
                or info.get("initial_sha256") != init_hash
                or info.get("params") != [NEGATIVE_PENALTY, DIVERSITY_PENALTY]
                or not out.is_file() or not query.is_file()
                or sha(out) != info.get("trajectory_sha256")
                or sha(query) != info.get("query_sha256")):
            raise RuntimeError("Completed result failed integrity check")
        print(json.dumps({"status": "skip_verified", "assay": args.assay,
                          "seed": args.seed, "hits": info["final_hits"]}), flush=True)
        return 0
    if out.exists() or query.exists():
        raise RuntimeError("Partial final output retained; inspect before retry")
    print(json.dumps({"status": "preflight_pass", "method": METHOD, "assay": args.assay,
                      "seed": args.seed, "n": len(y), "positive": int(y.sum()),
                      "initial_hits": int(y[init].sum()), "screen_sha256": source_hash,
                      "initial_sha256": init_hash, "params": [NEGATIVE_PENALTY, DIVERSITY_PENALTY],
                      "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}), flush=True)
    if args.preflight_only:
        return 0
    if packed.shape != (len(y), 256):
        raise RuntimeError(f"Unexpected ECFP geometry: {packed.shape}")
    fps = [DataStructs.CreateFromBinaryText(bytes(row)) for row in packed]
    queried = init.astype(int).tolist()
    rows = [{"budget": 64, "total_hits": int(y[init].sum()), "new_hits": 0}]
    for budget in (128, 192):
        checkpoint = ROOT / "tables" / f"{prefix}_q{budget}.json"
        if checkpoint.exists():
            info = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (info.get("source_sha256") != source_hash or info.get("initial_sha256") != init_hash
                    or info.get("params") != [NEGATIVE_PENALTY, DIVERSITY_PENALTY]
                    or info.get("budget") != budget
                    or info.get("query_indices", [])[:len(queried)] != queried
                    or len(info.get("query_indices", [])) != budget
                    or len(set(info["query_indices"])) != budget
                    or info.get("total_hits") != int(y[info["query_indices"]].sum())):
                raise RuntimeError("Checkpoint audit failed")
            queried = info["query_indices"]
            print(json.dumps({"status": "resume_verified", "budget": budget}), flush=True)
        else:
            choose = acquire(fps, y, queried, NEGATIVE_PENALTY, DIVERSITY_PENALTY)
            selected = set(queried)
            if len(choose) != 64 or len(set(map(int, choose))) != 64 or any(int(i) in selected for i in choose):
                raise RuntimeError("Invalid acquisition batch")
            queried.extend(choose.astype(int).tolist())
            write_new_json(checkpoint, {
                "status": "round_complete", "method": METHOD, "assay": args.assay,
                "seed": args.seed, "source_sha256": source_hash,
                "initial_sha256": init_hash,
                "params": [NEGATIVE_PENALTY, DIVERSITY_PENALTY],
                "budget": budget, "query_indices": queried,
                "total_hits": int(y[queried].sum()),
            })
            print(json.dumps({"status": "round_checkpoint", "budget": budget,
                              "total_hits": int(y[queried].sum())}), flush=True)
            if budget == 128 and args.inject == "crash":
                return 86
            if budget == 128 and args.inject == "sleep":
                time.sleep(30)
        total = int(y[queried].sum())
        rows.append({"budget": budget, "total_hits": total,
                     "new_hits": total - rows[0]["total_hits"]})
    if len(queried) != 192 or len(set(queried)) != 192:
        raise RuntimeError("Incomplete Q192 trajectory")
    tmp_out = Path(str(out) + f".stage_pid{os.getpid()}")
    tmp_query = Path(str(query) + f".stage_pid{os.getpid()}")
    with tmp_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("budget", "total_hits", "new_hits"))
        writer.writeheader()
        writer.writerows(rows)
    with tmp_query.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("order", "index", "smiles", "y"))
        writer.writeheader()
        for order, idx in enumerate(queried, start=1):
            writer.writerow({"order": order, "index": idx,
                             "smiles": str(ids.canonical_smiles.iloc[idx]), "y": int(y[idx])})
    tmp_out.replace(out)
    tmp_query.replace(query)
    write_new_json(marker, {
        "status": "complete", "protocol": "E98_E15_matched_Q192",
        "method": METHOD, "assay": args.assay, "seed": args.seed,
        "source_sha256": source_hash, "initial_sha256": init_hash,
        "q0": 64, "Q": 192, "batch_size": 64,
        "params": [NEGATIVE_PENALTY, DIVERSITY_PENALTY],
        "trajectory_sha256": sha(out), "query_sha256": sha(query),
        "initial_hits": rows[0]["total_hits"], "final_hits": rows[-1]["total_hits"],
        "implementation": "frozen E74-E rule transferred without LIT-PCBA parameter tuning",
    })
    print(json.dumps({"status": "complete", "assay": args.assay,
                      "seed": args.seed, "final_hits": rows[-1]["total_hits"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
