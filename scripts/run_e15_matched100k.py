#!/usr/bin/env python3
"""Run AIR or ChemScreener on the existing GLARE/AC-Active screen pool.

No acquisition implementation is changed.  The adapter substitutes the
candidate rows and reconstructs the official initial set from its published
dataset code, then checks that set against both completed official runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_e10_dual_head_pilot as air
import run_official_chemscreener_adapter as chem

ASSAYS = ("ADRB2", "ALDH1", "FEN1", "GBA", "IDH1", "KAT2A", "MAPK1", "MTORC1", "OPRK1", "PKM2", "VDR")
SEED_CANDIDATES = tuple(range(47001, 47011))
METHOD = "AIR_p1_legacy_unc_neighbor_gate"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def official_initial(y: np.ndarray, seed: int, q0: int = 64) -> np.ndarray:
    """Match both official dataset.py:get_start_idx(mode='a', ptal=False)."""
    rng = np.random.RandomState(seed)
    hit_idx = np.flatnonzero(y == 1)
    chosen_hit = hit_idx[rng.choice(len(hit_idx), size=1, replace=False)]
    remaining = np.array([i for i in range(len(y)) if i not in chosen_hit], dtype=np.int64)
    # Intentionally preserve the authors' default replace=True here.  If it
    # produces duplicates, the audit fails and that cell must be re-run with
    # a documented common-initial-set protocol instead of silently fixing it.
    chosen_other = rng.choice(remaining, size=q0 - 1)
    return rng.permutation(np.concatenate((chosen_hit, chosen_other))).astype(np.int64)


def load_matched_pool(assay: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    full_ids, full_packed, full_y = air.load_assay(assay)
    glare_csv = ROOT / "external_baselines/official_code/repos/glare/data" / assay / "original/screen.csv"
    ac_csv = ROOT / "external_baselines/official_code/repos/ac_active/data" / assay / "original/screen.csv"
    if not glare_csv.is_file() or not ac_csv.is_file():
        raise FileNotFoundError(f"Missing official screen.csv for {assay}")
    glare_hash, ac_hash = sha256(glare_csv), sha256(ac_csv)
    if glare_hash != ac_hash:
        raise RuntimeError(f"GLARE and AC-Active candidate pools differ on {assay}")
    screen = pd.read_csv(glare_csv)
    if list(screen.columns) != ["smiles", "y"] or screen.smiles.isna().any() or screen.smiles.duplicated().any():
        raise RuntimeError(f"Malformed/duplicate official screen rows on {assay}")
    full_smiles = pd.Index(full_ids["canonical_smiles"].astype(str))
    if not full_smiles.is_unique:
        raise RuntimeError(f"Original AIR canonical SMILES are not unique on {assay}")
    indices = full_smiles.get_indexer(screen.smiles.astype(str))
    if (indices < 0).any():
        raise RuntimeError(f"Unmapped official screen SMILES on {assay}: {(indices < 0).sum()}")
    ids = full_ids.iloc[indices].reset_index(drop=True)
    y = np.asarray(full_y[indices], dtype=np.int8)
    if not np.array_equal(y, screen.y.astype(np.int8).to_numpy()):
        raise RuntimeError(f"Label mismatch on {assay}")
    packed = np.asarray(full_packed[indices])
    ids_hash = hashlib.sha256("\n".join(ids.molecule_id.astype(str)).encode()).hexdigest()
    report = {
        "assay": assay,
        "status": "pool_verified",
        "screen_sha256": glare_hash,
        "molecule_id_order_sha256": ids_hash,
        "n_screen": int(len(y)),
        "n_active_screen": int(y.sum()),
        "n_removed_from_air_pool": int(len(full_y) - len(y)),
        "official_initial_audit": [],
    }
    discarded_duplicate_seeds = []
    selected_seeds = []
    missing_official = []
    for seed in SEED_CANDIDATES:
        if len(selected_seeds) == 5:
            break
        init = official_initial(y, seed)
        n_unique = int(len(np.unique(init)))
        if n_unique != 64:
            discarded_duplicate_seeds.append(seed)
            continue
        selected_seeds.append(seed)
        initial_hits = int(y[init].sum())
        row = {"seed": seed, "n_initial": int(len(init)), "n_unique": n_unique, "initial_hits": initial_hits,
               "initial_indices_sha256": hashlib.sha256(init.tobytes()).hexdigest()}
        for family in ("glare", "acactive"):
            path = ROOT / "tables" / (
                f"official_{family}_{assay}_air100k_q192_{assay}_seed{seed}_Q192_b64_e2_ens2.csv"
            )
            if not path.is_file():
                row[f"{family}_recorded_initial_hits"] = None
                missing_official.append(str(path.relative_to(ROOT)))
                continue
            official = pd.read_csv(path, nrows=1)
            observed = int(official.total_hit_discover.iloc[0])
            row[f"{family}_recorded_initial_hits"] = observed
            if observed != initial_hits:
                raise RuntimeError(f"Official initial set mismatch: {assay} {seed} {family}: expected {initial_hits}, got {observed}")
        report["official_initial_audit"].append(row)
    report["seed_selection_rule"] = "first five seeds in 47001..47010 with 64 distinct official initial indices; collision check ignores downstream hit outcomes"
    report["selected_seeds"] = selected_seeds
    report["discarded_duplicate_seeds"] = discarded_duplicate_seeds
    report["missing_official_results"] = missing_official
    report["status"] = "pass" if len(selected_seeds) == 5 and not missing_official else "pending_official"
    return ids, packed, y, report


def default_args(module, argv: list[str]):
    old = sys.argv
    try:
        sys.argv = [old[0]] + argv
        return module.parse_args()
    finally:
        sys.argv = old


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("air", "chemscreener"), required=True)
    p.add_argument("--assay", choices=ASSAYS, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--gpu-physical-id", type=int, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--timeout-sec", type=int, default=1800)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--fault-after-preflight", action="store_true")
    p.add_argument("--self-test-guards", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.prefix.startswith("e15_"):
        raise SystemExit("E15 prefix must start with e15_")
    if args.seeds is not None and (not set(args.seeds).issubset(SEED_CANDIDATES) or len(set(args.seeds)) != len(args.seeds)):
        raise SystemExit("Seeds must be distinct members of 47001..47010")
    if args.cpu_threads > max(1, (air.os.cpu_count() or 1) - 30):
        raise SystemExit("CPU request violates leave-30-core rule")
    air.validate_gpu_id(args.gpu_physical_id, "gpu")
    ids, packed, y, report = load_matched_pool(args.assay)
    if args.seeds is None:
        args.seeds = report["selected_seeds"]
    if not set(args.seeds).issubset(report["selected_seeds"]):
        raise RuntimeError(f"Requested seeds are not in collision-free selected set: {report['selected_seeds']}")
    audit_path = ROOT / "tables" / f"{args.prefix}_protocol_audit.json"
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "assay": args.assay, "n_screen": len(y),
                      "n_active": int(y.sum()), "audit": str(audit_path)}, ensure_ascii=False), flush=True)
    if args.preflight_only:
        return 0
    if report["status"] != "pass":
        raise RuntimeError(f"Protocol audit did not pass; missing official records: {report['missing_official_results']}")
    if args.fault_after_preflight:
        marker = ROOT / "results" / args.prefix / "injected_fault.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"status": "fault_injected", "stage": "after_preflight",
                                      "at": time.time()}), encoding="utf-8")
        print(json.dumps({"status": "fault_injected", "marker": str(marker)}), flush=True)
        return 86

    def subset_loader(assay: str):
        if assay != args.assay:
            raise ValueError("Matched adapter is single-assay only")
        return ids, packed, y

    def subset_initial(labels: np.ndarray, assay: str, seed: int, q0: int):
        if assay != args.assay or q0 != 64 or not np.array_equal(labels, y):
            raise RuntimeError("Unexpected AIR initial-set context")
        return official_initial(y, seed, q0)

    module = air if args.family == "air" else chem
    module.load_assay = subset_loader
    module.make_initial = subset_initial
    original_budgets = module.screening_budgets
    def matched_budgets(n: int):
        q_base, b1_base, _ = original_budgets(n)
        return q_base, b1_base, 64
    module.screening_budgets = matched_budgets
    if args.family == "air":
        conf = default_args(air, [])
        conf.methods = [METHOD]
        conf.epochs = 2
        conf.batch_size = 128
        conf.q_total_cap = 192
        conf.b1_total_cap = 192
    else:
        conf = default_args(chem, [])
        conf.chem_mode = "Balanced_Ranking"
        conf.chem_ensemble = 1
        conf.chem_epochs = 1
        conf.chem_batch_size = 128
        conf.q_total_cap = 192
        conf.b1_total_cap = 192
    conf.assays = [args.assay]
    conf.seeds = args.seeds
    conf.prefix = args.prefix
    conf.accelerator = "gpu"
    conf.gpu_physical_id = args.gpu_physical_id
    conf.cpu_threads = args.cpu_threads
    conf.lgbm_jobs = min(4, args.cpu_threads)
    conf.timeout_sec = args.timeout_sec
    conf.resume = args.resume
    conf.self_test_guards = args.self_test_guards
    if args.self_test_guards:
        module.self_test_guards(conf)
        return 0
    rows = []
    for seed in args.seeds:
        if args.family == "air":
            rows.append(module.run_method(conf, args.assay, int(seed), METHOD))
        else:
            rows.append(module.run_one(conf, args.assay, int(seed)))
        module.summarize(rows, args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
