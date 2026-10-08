#!/usr/bin/env python3
"""Port the unchanged AIR-Legacy method to frozen E69 TDC-HTS assay pools."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_e10_dual_head_pilot as air  # noqa: E402
from e72_tdc_neighbor_baseline import fingerprints, initial, sha  # noqa: E402

METHOD = "AIR_p1_legacy_unc_neighbor_gate"
DEV_AIDS = ("1798", "463087", "485290")


def args_for_air():
    old = sys.argv
    try:
        sys.argv = [old[0]]
        return air.parse_args()
    finally:
        sys.argv = old


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aid", choices=DEV_AIDS, required=True)
    p.add_argument("--seed", type=int, choices=range(47001, 47006), required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--fault-after-preflight", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.gpu < 0:
        raise ValueError("GPU index must be non-negative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    x = torch.randn((8, 8), device="cuda")
    if not bool(torch.isfinite(x @ x).all()):
        raise RuntimeError("CUDA kernel witness failed")
    gpu_name = torch.cuda.get_device_name(0)

    alias = f"TDC_E69_AID{args.aid}"
    manifest = json.loads((ROOT / "tables" / f"e69_tdc_dev_{args.aid}_manifest.json").read_text())
    screen_path = ROOT / "tables" / f"e69_tdc_dev_{args.aid}_screen.csv"
    if manifest["status"] != "frozen" or sha(screen_path) != manifest["screen_sha256"]:
        raise RuntimeError("E69 TDC pool not frozen or checksum mismatch")
    frame = pd.read_csv(screen_path)
    if list(frame.columns) != ["smiles", "y"] or frame.smiles.duplicated().any():
        raise RuntimeError("Frozen E69 screen rows malformed")
    smiles = frame.smiles.astype(str).tolist()
    y = frame.y.to_numpy(dtype=np.int8)
    if len(y) != manifest["screen_size"] or int(y.sum()) != manifest["screen_positives"]:
        raise RuntimeError("Frozen E69 pool geometry mismatch")
    init = initial(y, args.seed)
    seed_manifest = next(v for v in manifest["seeds"] if v["seed"] == args.seed)
    initial_hash = hashlib.sha256(init.tobytes()).hexdigest()
    if initial_hash != seed_manifest["initial_index_sha256"]:
        raise RuntimeError("Frozen E69 initial set changed")
    protocol = {
        "status": "preflight_pass", "alias": alias, "seed": args.seed,
        "method": METHOD, "source_sha256": manifest["screen_sha256"],
        "initial_sha256": initial_hash, "initial_hits": int(y[init].sum()),
        "screen_size": len(y), "positive_screen": int(y.sum()),
        "q0": 64, "Q": 384, "batch_size": 64, "rounds": 5,
        "air_epochs": 2, "b1_budget": 384,
        "gpu_physical_id": args.gpu, "gpu_name": gpu_name,
    }
    print(json.dumps(protocol), flush=True)
    if args.preflight_only:
        return 0
    prefix = f"e90_tdc_legacy_{args.aid}_seed{args.seed}"
    out_dir = ROOT / "results" / prefix / f"{alias}_seed{args.seed}_{METHOD}"
    if args.fault_after_preflight:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "fault_injected.json").write_text(json.dumps({"status": "injected_fault", **protocol}), encoding="utf-8")
        return 86
    packed = fingerprints(args.aid, smiles, manifest["screen_sha256"], "none")
    if packed.shape != (len(y), 256):
        raise RuntimeError("Fingerprint cache geometry mismatch")
    ids = pd.DataFrame({"molecule_id": [f"{alias}:{i}" for i in range(len(y))],
                        "canonical_smiles": smiles})

    def matched_loader(assay):
        if assay != alias:
            raise RuntimeError("Unexpected assay in AIR method")
        return ids, packed, y

    def matched_initial(labels, assay, seed, q0):
        if assay != alias or seed != args.seed or q0 != 64 or not np.array_equal(labels, y):
            raise RuntimeError("Unexpected initial-set context")
        return init

    air.load_assay = matched_loader
    air.make_initial = matched_initial
    air.screening_budgets = lambda n: (384, 384, 64)
    conf = args_for_air()
    conf.assays = [alias]
    conf.seeds = [args.seed]
    conf.methods = [METHOD]
    conf.prefix = prefix
    conf.q_total_cap = 384
    conf.b1_total_cap = 384
    conf.q0_total_cap = 64
    conf.rounds = 5
    conf.epochs = 2
    conf.batch_size = 128
    conf.accelerator = "gpu"
    conf.gpu_physical_id = args.gpu
    conf.cpu_threads = 4
    conf.lgbm_jobs = 4
    conf.timeout_sec = 2400
    conf.resume = args.resume
    result = air.run_method(conf, alias, args.seed, METHOD)
    air.summarize([result], prefix)
    summary = ROOT / "tables" / f"{prefix}.csv"
    d = pd.read_csv(summary)
    row = d.iloc[-1]
    if row.status != "pass" or int(row.q_spent) != 384 or int(row.q0) != 64:
        raise RuntimeError("AIR-Legacy TDC trajectory failed completion audit")
    print(json.dumps({"status": "complete", "alias": alias, "seed": args.seed,
                      "hits": int(row.hits), "result": str(summary),
                      "at": time.time()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
