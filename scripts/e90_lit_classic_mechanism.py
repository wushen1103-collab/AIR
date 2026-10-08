#!/usr/bin/env python3
"""Four external algorithm baselines on the frozen E15 LIT-PCBA common pool."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_e15_matched100k import ASSAYS, load_matched_pool, official_initial  # noqa: E402

METHODS = ("random", "rf_ecfp", "positive_tanimoto", "positive_tversky_t1")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_new_json(path: Path, obj):
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    tmp = Path(str(path) + f".stage_pid{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    if path.exists():
        raise RuntimeError(f"Concurrent checkpoint creation: {path}")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assay", choices=ASSAYS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--inject", choices=("none", "crash"), default="none")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    if args.seed not in range(47001, 47011):
        raise ValueError("Seed outside E15 candidate range")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("This CPU-only baseline requires CUDA_VISIBLE_DEVICES=-1")
    import torch
    torch.set_num_threads(4)
    if torch.cuda.device_count() != 0:
        raise RuntimeError("CUDA must be invisible")
    ids, packed, y, audit = load_matched_pool(args.assay)
    if audit["status"] != "pass" or args.seed not in audit["selected_seeds"]:
        raise RuntimeError("E15 common-pool audit failed")
    init = official_initial(y, args.seed)
    if len(init) != 64 or len(np.unique(init)) != 64:
        raise RuntimeError("E15 initial set mismatch")
    source = ROOT / "external_baselines/official_code/repos/glare/data" / args.assay / "original/screen.csv"
    source_sha = sha(source)
    prefix = f"e90_lit_{args.method}_{args.assay}_seed{args.seed}_q192"
    out = ROOT / "tables" / f"{prefix}.csv"
    query = ROOT / "tables" / f"{prefix}_query.csv"
    marker = ROOT / "tables" / f"{prefix}_manifest.json"
    if marker.exists():
        m = json.loads(marker.read_text(encoding="utf-8"))
        if (m.get("status") != "complete" or m.get("source_sha256") != source_sha
                or not out.is_file() or not query.is_file()
                or sha(out) != m.get("trajectory_sha256") or sha(query) != m.get("query_sha256")):
            raise RuntimeError("Completed result failed integrity check")
        print(json.dumps({"status": "skip_verified", "method": args.method, "assay": args.assay,
                          "seed": args.seed, "hits": m["final_hits"]}), flush=True)
        return 0
    if out.exists() or query.exists():
        raise RuntimeError("Partial final output retained; inspect before retry")
    initial_hash = hashlib.sha256(init.tobytes()).hexdigest()
    print(json.dumps({"status": "preflight_pass", "method": args.method, "assay": args.assay,
                      "seed": args.seed, "n": len(y), "positive": int(y.sum()),
                      "initial_hits": int(y[init].sum()), "screen_sha256": source_sha,
                      "initial_sha256": initial_hash, "cuda_device_count": 0}), flush=True)
    if args.preflight_only:
        return 0

    fps = None
    X = None
    if args.method in ("positive_tanimoto", "positive_tversky_t1"):
        if packed.shape != (len(y), 256):
            raise RuntimeError(f"Unexpected 2048-bit fingerprint geometry: {packed.shape}")
        fps = [DataStructs.CreateFromBinaryText(bytes(row)) for row in packed]
    elif args.method == "rf_ecfp":
        X = np.unpackbits(packed, axis=1).astype(np.uint8, copy=False)
        if X.shape != (len(y), 2048):
            raise RuntimeError("RF/ECFP fingerprint geometry mismatch")

    queried = init.astype(int).tolist()
    rows = [{"budget": 64, "total_hits": int(y[init].sum()), "new_hits": 0}]
    did_inject = False
    for round_number, budget in enumerate((128, 192), start=1):
        checkpoint = ROOT / "tables" / f"{prefix}_q{budget}.json"
        if checkpoint.exists():
            c = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (c.get("source_sha256") != source_sha or c.get("method") != args.method
                    or c.get("seed") != args.seed or c.get("budget") != budget
                    or c.get("query_indices", [])[:len(queried)] != queried
                    or len(c.get("query_indices", [])) != budget
                    or len(set(c["query_indices"])) != budget
                    or c.get("total_hits") != int(y[c["query_indices"]].sum())):
                raise RuntimeError("Checkpoint audit failed")
            queried = c["query_indices"]
            print(json.dumps({"status": "resume_verified", "budget": budget}), flush=True)
        else:
            selected = np.zeros(len(y), dtype=bool)
            selected[queried] = True
            remaining = np.flatnonzero(~selected)
            if args.method == "random":
                rng = np.random.RandomState(args.seed + round_number)
                choose = rng.choice(remaining, size=64, replace=False)
            elif args.method == "rf_ecfp":
                model = RandomForestClassifier(
                    n_estimators=256, max_features="sqrt", min_samples_leaf=1,
                    class_weight="balanced_subsample", n_jobs=4,
                    random_state=args.seed + round_number)
                model.fit(X[queried], y[queried])
                if model.classes_.tolist() != [0, 1]:
                    raise RuntimeError("RF training lost a class")
                score = model.predict_proba(X[remaining])[:, 1]
                choose = remaining[np.argsort(-score, kind="stable")[:64]]
            else:
                positive_refs = np.asarray(queried, dtype=np.int64)[y[queried] == 1]
                if len(positive_refs) == 0:
                    raise RuntimeError("No positive reference in E15 initial set")
                candidates = [fps[int(i)] for i in remaining]
                score = np.zeros(len(remaining), dtype=np.float32)
                for ref in positive_refs:
                    if args.method == "positive_tanimoto":
                        sim = DataStructs.BulkTanimotoSimilarity(fps[int(ref)], candidates)
                    else:
                        sim = DataStructs.BulkTverskySimilarity(fps[int(ref)], candidates, 1.0, 0.25)
                    np.maximum(score, np.asarray(sim, dtype=np.float32), out=score)
                choose = remaining[np.argsort(-score, kind="stable")[:64]]
            if len(choose) != 64 or len(np.unique(choose)) != 64 or selected[choose].any():
                raise RuntimeError("Invalid acquisition batch")
            queried.extend(choose.astype(int).tolist())
            write_new_json(checkpoint, {
                "status": "round_complete", "method": args.method, "assay": args.assay,
                "seed": args.seed, "source_sha256": source_sha, "initial_sha256": initial_hash,
                "budget": budget, "query_indices": queried,
                "total_hits": int(y[queried].sum()),
            })
            print(json.dumps({"status": "round_checkpoint", "budget": budget,
                              "total_hits": int(y[queried].sum())}), flush=True)
            if not did_inject and args.inject == "crash":
                did_inject = True
                return 86
        total = int(y[queried].sum())
        rows.append({"budget": budget, "total_hits": total,
                     "new_hits": total - rows[0]["total_hits"]})
    if len(queried) != 192 or len(set(queried)) != 192:
        raise RuntimeError("Incomplete or duplicate Q192 trajectory")
    tmp_out = Path(str(out) + f".stage_pid{os.getpid()}")
    tmp_query = Path(str(query) + f".stage_pid{os.getpid()}")
    with tmp_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("budget", "total_hits", "new_hits"))
        writer.writeheader()
        writer.writerows(rows)
    with tmp_query.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("order", "index", "smiles", "y"))
        writer.writeheader()
        for order, idx in enumerate(queried, start=1):
            writer.writerow({"order": order, "index": idx,
                             "smiles": str(ids.canonical_smiles.iloc[idx]), "y": int(y[idx])})
    m = {
        "status": "complete", "protocol": "E90_E15_matched_Q192", "method": args.method,
        "assay": args.assay, "seed": args.seed, "source_sha256": source_sha,
        "initial_sha256": initial_hash, "q0": 64, "Q": 192, "batch_size": 64,
        "trajectory_sha256": sha(tmp_out), "query_sha256": sha(tmp_query),
        "initial_hits": rows[0]["total_hits"], "final_hits": rows[-1]["total_hits"],
        "implementation": "project rerun of literature algorithm; not official author code",
    }
    tmp_out.replace(out)
    tmp_query.replace(query)
    write_new_json(marker, m)
    print(json.dumps({"status": "complete", "method": args.method,
                      "assay": args.assay, "seed": args.seed, "final_hits": m["final_hits"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
