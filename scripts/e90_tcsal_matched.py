#!/usr/bin/env python3
"""Matched-pool adapter for the unmodified TcsAL screening/model/acquisition code.

Only the dataset and the initial-set handler are replaced.  A smoke run may
override training epochs, but a strong run uses the author's defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--assay", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--mode", choices=("smoke", "strong"), required=True)
    p.add_argument("--acquisition", choices=("bald", "exploitation"), default="bald")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--fault-after-preflight", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    repo = root / "external_baselines/official_code/repos/traversing_chem_space"
    if not repo.is_dir():
        raise RuntimeError("Official TcsAL checkout not found")
    if args.assay not in ("ADRB2", "ALDH1", "FEN1", "GBA", "IDH1", "KAT2A", "MAPK1", "MTORC1", "OPRK1", "PKM2", "VDR"):
        raise ValueError("Assay outside locked E15 roster")
    if args.seed not in range(47001, 47011):
        raise ValueError("Seed outside locked E15 candidate range")
    if args.gpu < 0:
        raise ValueError("Invalid physical GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(root / "scripts"))

    import numpy as np
    import pandas as pd
    import torch
    import run_e15_matched100k as e15
    from active_learning import nn, screening
    from active_learning.utils import smiles_to_ecfp

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    devname = torch.cuda.get_device_name(0)
    witness = torch.randn((8, 8), device="cuda")
    if not bool(torch.isfinite(witness @ witness).all()):
        raise RuntimeError("CUDA kernel witness failed")
    ids, packed, y, audit = e15.load_matched_pool(args.assay)
    del ids, packed
    if audit["status"] != "pass" or args.seed not in audit["selected_seeds"]:
        raise RuntimeError("E15 common-pool/common-initial-set audit failed")
    screen_csv = repo.parent / "glare/data" / args.assay / "original/screen.csv"
    test_csv = repo.parent / "glare/data" / args.assay / "original/test.csv"
    screen_df = pd.read_csv(screen_csv)
    test_df = pd.read_csv(test_csv)
    if len(screen_df) != len(y) or not np.array_equal(screen_df.y.to_numpy(), y):
        raise RuntimeError("Matched screen labels differ from E15 audit")
    initial = e15.official_initial(y, args.seed)
    if len(np.unique(initial)) != 64:
        raise RuntimeError("Initial set is not 64 unique molecules")
    if args.acquisition == "bald":
        name = f"e90_tcsal_{args.mode}_{args.assay}_seed{args.seed}_q192"
        marker_root = root / "results/e90_tcsal"
    else:
        name = f"e93_tcsal_exploitation_{args.mode}_{args.assay}_seed{args.seed}_q192"
        marker_root = root / "results/e93_tcsal_exploitation"
    out = root / "tables" / f"{name}.csv"
    marker = marker_root / f"{name}.json"
    provenance = {
        "method": f"TcsAL-MLP-{args.acquisition}",
        "implementation": "official author model, ensemble, acquisition, and screening loop; dataset/initial-index adapter",
        "author_commit": "36a1ed85ddf607e3a0e080adb6097add61c8c235",
        "screen_sha256": sha256(screen_csv),
        "test_sha256": sha256(test_csv),
        "initial_indices_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "initial_hits": int(y[initial].sum()),
        "screen_size": int(len(y)),
        "positive_screen": int(y.sum()),
        "gpu_physical_id": args.gpu,
        "gpu_name": devname,
        "seed": args.seed,
        "assay": args.assay,
        "mode": args.mode,
        "acquisition_method": args.acquisition,
        "q0": 64,
        "Q": 192,
        "batch_size": 64,
        "epochs": 2 if args.mode == "smoke" else 50,
        "ensemble_size": 2 if args.mode == "smoke" else 10,
    }
    print(json.dumps({"status": "preflight_pass", **provenance}), flush=True)
    if args.preflight_only:
        return 0
    if out.exists() and marker.exists() and args.resume:
        old = json.loads(marker.read_text(encoding="utf-8"))
        rows = pd.read_csv(out)
        if old.get("status") == "done" and rows.total_mols_screened.astype(int).tolist() == [64, 128, 192]:
            print(json.dumps({"status": "skip_verified", "result": str(out)}), flush=True)
            return 0
    elif out.exists():
        raise RuntimeError("Result exists; use --resume for checked reuse")
    atomic_json(marker, {"status": "started", "at": time.time(), **provenance})
    if args.fault_after_preflight:
        atomic_json(marker, {"status": "injected_fault", "at": time.time(), **provenance})
        return 86

    cache = root / "results/e90_tcsal/features" / args.assay
    cache.mkdir(parents=True, exist_ok=True)

    class MatchedDataset:
        def __init__(self, part):
            df = screen_df if part == "screen" else test_df
            source = screen_csv if part == "screen" else test_csv
            feature_file = cache / f"{part}_{sha256(source)[:12]}_ecfp1024_float32.npy"
            if not feature_file.is_file():
                features = np.asarray(smiles_to_ecfp(df.smiles.astype(str).tolist(), radius=2, nbits=1024), dtype=np.float32)
                if features.shape != (len(df), 1024):
                    raise RuntimeError("Author ECFP fingerprint shape mismatch")
                tmp = feature_file.with_suffix(".tmp")
                with tmp.open("wb") as f:
                    np.save(f, features)
                os.replace(tmp, feature_file)
            self.x = np.load(feature_file, mmap_mode="r")
            self.y = torch.tensor(df.y.to_numpy(dtype=np.int64))
            self.smiles = df.smiles.astype(str).to_numpy()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            if isinstance(idx, int):
                idx = [idx]
            return np.asarray(self.x[idx], dtype=np.float32), self.y[idx], self.smiles[idx]

        def all(self):
            return self[range(len(self.y))]

    class MatchedHandler:
        def __init__(self, n_start=64, seed=0, bias="random", dataset=None):
            if n_start != 64 or seed != args.seed or bias != "random" or dataset != args.assay:
                raise RuntimeError("Unexpected official screening-loop context")
            self.train_idx = initial.copy()
            self.screen_idx = np.setdiff1d(np.arange(len(y)), self.train_idx)
            self.smiles_index = {s: i for i, s in enumerate(screen_df.smiles.astype(str))}

        def __call__(self):
            return self.train_idx, self.screen_idx

        def add(self, picks):
            added = np.array([self.smiles_index[str(s)] for s in picks], dtype=np.int64)
            if len(added) != 64 or len(np.unique(added)) != 64 or np.isin(added, self.train_idx).any():
                raise RuntimeError("Author acquisition returned duplicate or already selected molecules")
            self.train_idx = np.concatenate((self.train_idx, added))
            self.screen_idx = np.setdiff1d(self.screen_idx, added)

    screening.MasterDataset = lambda part, **kwargs: MatchedDataset(part)
    screening.Handler = MatchedHandler
    if args.mode == "smoke":
        author_mlp = nn.MLP

        class SmokeMLP(author_mlp):
            def __init__(self, *a, **kw):
                kw["epochs"] = 2
                super().__init__(*a, **kw)

        nn.MLP = SmokeMLP

    result = screening.active_learning(
        n_start=64,
        acquisition_method=args.acquisition,
        max_screen_size=192,
        batch_size=64,
        architecture="mlp",
        seed=args.seed,
        bias="random",
        ensemble_size=provenance["ensemble_size"],
        retrain=True,
        anchored=True,
        dataset=args.assay,
        optimize_hyperparameters=False,
    )
    if result.total_mols_screened.astype(int).tolist() != [64, 128, 192]:
        raise RuntimeError("Query budget trajectory is not [64,128,192]")
    if int(result.hits_discovered.iloc[0]) != provenance["initial_hits"]:
        raise RuntimeError("Initial-hit witness failed")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".csv.tmp")
    result.to_csv(tmp, index=False)
    os.replace(tmp, out)
    atomic_json(marker, {"status": "done", "at": time.time(), "final_hits": int(result.hits_discovered.iloc[-1]), **provenance})
    print(json.dumps({"status": "done", "result": str(out), "final_hits": int(result.hits_discovered.iloc[-1])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
