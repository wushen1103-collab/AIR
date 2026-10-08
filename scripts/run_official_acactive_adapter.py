#!/usr/bin/env python3
"""Controlled wrapper for AC-Active official active-learning code."""

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--dataset", default="ALDH1")
    parser.add_argument("--seed", type=int, default=47001)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--mode", default="a")
    parser.add_argument("--architecture", default="ginl")
    parser.add_argument("--strategy", default="cliff")
    parser.add_argument("--task", default="cla")
    parser.add_argument("--ptal", action="store_true")
    parser.add_argument("--start-active-num", type=int, default=1)
    parser.add_argument("--start-num", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-screen-size", type=int, default=192)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--infer-batch-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--simulate-crash-after-start", action="store_true")
    parser.add_argument("--deterministic-algorithms", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    tables_dir = project_root / "tables"
    runs_dir = project_root / "runs" / "official_acactive"
    tables_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"{args.output_prefix}_{args.dataset}_seed{args.seed}_"
        f"Q{args.max_screen_size}_b{args.batch_size}_e{args.epochs}_ens{args.ensemble_size}"
    )
    marker_path = runs_dir / f"{run_id}.json"
    result_csv = tables_dir / f"{run_id}.csv"
    official_outdir = repo_root / "experiments" / "result" / run_id
    official_csv = official_outdir / f"{args.architecture}_{args.strategy}_{args.dataset}_{args.seed}_results.csv"

    params = vars(args).copy()
    params.update(
        {
            "run_id": run_id,
            "result_csv": str(result_csv),
            "official_csv": str(official_csv),
            "repo_root": str(repo_root),
        }
    )

    if result_csv.exists() and marker_path.exists() and not args.force:
        marker = json.loads(marker_path.read_text())
        if marker.get("status") == "done":
            print(json.dumps({"status": "skipped_completed", "run_id": run_id, "result_csv": str(result_csv)}))
            return 0

    if official_outdir.exists() and args.force:
        shutil.rmtree(official_outdir)
    official_outdir.mkdir(parents=True, exist_ok=True)

    marker_path.write_text(
        json.dumps(
            {
                "status": "started",
                "params": params,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.simulate_crash_after_start:
        print(json.dumps({"status": "simulated_crash_after_start", "marker": str(marker_path)}))
        return 86

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    if args.deterministic_algorithms:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    set_seed(args.seed)

    import torch
    if args.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available() and torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU after masking, got {torch.cuda.device_count()}"
        )

    sys.argv = [
        "main.py",
        "-cuda",
        "0",
        "-output_folder",
        run_id,
        "-log_file",
        str(official_csv),
        "-mode",
        args.mode,
        "-architecture",
        args.architecture,
        "-strategy",
        args.strategy,
        "-dataset",
        args.dataset,
        "-task",
        args.task,
        "-seed",
        str(args.seed),
        "-start_active_num",
        str(args.start_active_num),
        "-start_num",
        str(args.start_num),
        "-batch_size",
        str(args.batch_size),
        "-max_screen_size",
        str(args.max_screen_size),
        "-ensemble_size",
        str(args.ensemble_size),
        "-epochs",
        str(args.epochs),
        "-train_batch_size",
        str(args.train_batch_size),
        "-infer_batch_size",
        str(args.infer_batch_size),
    ]
    if args.ptal:
        sys.argv.extend(["-ptal", "True"])

    from config import config
    import main as ac_main

    ac_args = config()
    ac_args.log_file = str(official_csv)
    if hasattr(ac_main, "set_seed"):
        ac_main.set_seed(args.seed)

    started = time.time()
    results = ac_main.active_learning(ac_args)
    seconds = round(time.time() - started, 3)

    results.to_csv(result_csv, index=False)
    results.to_csv(official_csv, index=False)

    final_hits = int(results["total_hit_discover"].iloc[-1])
    final_budget = int(results["total_mol_screen"].iloc[-1])
    final_ef = float(results["enrichment_factor"].iloc[-1])
    marker = {
        "status": "done",
        "params": params,
        "started_at": json.loads(marker_path.read_text()).get("started_at"),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": seconds,
        "final_hits": final_hits,
        "final_budget": final_budget,
        "final_enrichment_factor": final_ef,
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2))
    print(json.dumps(marker, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
