#!/usr/bin/env python3
"""Prepare AIR assay data in GLARE/AC-Active official repository format.

This script intentionally reuses GLARE's official `MasterDataset` and
`mol_to_graph_data_obj_simple_3D` preprocessing definitions.  It creates
`data/<ASSAY>/original/{screen.csv,test.csv,actives.smi,inactives.smi}`,
then builds `screen/` and `test/` tensors plus `graphs2`.

AC-Active uses the same serialized data format in the current official code
checkout, so we mirror GLARE-prepared directories via hard links when possible.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    project = Path(__file__).resolve().parents[1]
    repos = project / "external_baselines" / "official_code" / "repos"
    p.add_argument("--project-root", default=str(project))
    p.add_argument("--glare-repo", default=str(repos / "glare"))
    p.add_argument("--acactive-repo", default=str(repos / "ac_active"))
    p.add_argument("--assays", nargs="+", required=True)
    p.add_argument("--screen-size", type=int, default=100000)
    p.add_argument("--test-size", type=int, default=20000)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--progress-every", type=int, default=20000)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--mirror-acactive", action="store_true")
    return p.parse_args()


def adaptive_split(df: pd.DataFrame, screen_size: int, test_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n >= screen_size + test_size:
        train_size = screen_size
        test_n = test_size
    else:
        test_n = min(test_size, max(1, int(round(0.20 * n))))
        train_size = n - test_n
    stratify = df["y"] if df["y"].nunique() == 2 and df["y"].value_counts().min() >= 2 else None
    screen, test = train_test_split(
        df,
        train_size=train_size,
        test_size=test_n,
        random_state=seed,
        stratify=stratify,
    )
    return screen.reset_index(drop=True), test.reset_index(drop=True)


def load_air_assay(project: Path, assay: str) -> pd.DataFrame:
    sys.path.insert(0, str(project / "src"))
    sys.path.insert(0, str(project / "scripts"))
    from run_e03_headroom import load_assay

    ids_df, _packed, y = load_assay(assay)
    return pd.DataFrame(
        {
            "smiles": ids_df["canonical_smiles"].astype(str).to_numpy(),
            "y": y.astype(int),
            "molecule_id": ids_df["molecule_id"].astype(str).to_numpy(),
        }
    )


def hardlink_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (dst / rel / d).mkdir(exist_ok=True)
        for f in files:
            sp = Path(root) / f
            dp = dst / rel / f
            try:
                os.link(sp, dp)
            except OSError:
                shutil.copy2(sp, dp)


def main() -> int:
    args = parse_args()
    project = Path(args.project_root).resolve()
    glare_repo = Path(args.glare_repo).resolve()
    ac_repo = Path(args.acactive_repo).resolve()
    tables = project / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    # Import official GLARE preprocessing code.
    os.chdir(glare_repo)
    sys.path.insert(0, str(glare_repo / "utils"))
    sys.path.insert(0, str(glare_repo))
    from rdkit import Chem, RDLogger
    import torch
    import data_prep
    from data_prep import MasterDataset

    RDLogger.DisableLog("rdApp.*")
    data_prep.ROOT_DIR = "."

    src = (glare_repo / "utils" / "preprocess_data.py").read_text()
    prefix = src.split("if __name__ == '__main__':", 1)[0]
    namespace = {"__name__": "glare_preprocess_defs"}
    exec(prefix, namespace)
    mol_to_graph_data_obj_simple_3D = namespace["mol_to_graph_data_obj_simple_3D"]

    for assay in args.assays:
        started = time.time()
        assay_dir = glare_repo / "data" / assay
        done_marker = assay_dir / "air_prepare_done.json"
        if args.skip_existing and done_marker.exists():
            marker = json.loads(done_marker.read_text())
            manifest_rows.append(marker)
            print(json.dumps({"status": "skip_existing", "assay": assay, "marker": str(done_marker)}), flush=True)
            if args.mirror_acactive:
                hardlink_tree(assay_dir, ac_repo / "data" / assay)
            continue

        original = assay_dir / "original"
        screen_dir = assay_dir / "screen"
        test_dir = assay_dir / "test"
        original.mkdir(parents=True, exist_ok=True)

        df = load_air_assay(project, assay)
        # GLARE's official preprocessing assumes every SMILES can be parsed by
        # RDKit and crashes otherwise.  AIR's raw assay loaders may keep a small
        # number of molecules that are unusable for graph baselines, so we filter
        # them here and record the count in the manifest.  This keeps the
        # official model logic unchanged while making the adapter robust.
        def official_featurizer_ok(s: str) -> bool:
            s = str(s)
            if Chem.MolFromSmiles(s, sanitize=True) is None:
                return False
            try:
                return bool(data_prep.check_featurizability(s))
            except Exception:
                return False

        mol_ok = df["smiles"].map(official_featurizer_ok)
        n_invalid_smiles = int((~mol_ok).sum())
        if n_invalid_smiles:
            print(
                json.dumps(
                    {
                        "status": "filter_invalid_smiles",
                        "assay": assay,
                        "n_invalid_smiles": n_invalid_smiles,
                        "n_before": int(len(df)),
                    }
                ),
                flush=True,
            )
        df = df.loc[mol_ok].reset_index(drop=True)
        screen, test = adaptive_split(df[["smiles", "y"]], args.screen_size, args.test_size, args.split_seed)
        screen.to_csv(original / "screen.csv", index=False)
        test.to_csv(original / "test.csv", index=False)
        df[df.y == 1][["smiles"]].assign(id=np.arange(int(df.y.sum()))).to_csv(
            original / "actives.smi", sep=" ", header=False, index=False
        )
        df[df.y == 0][["smiles"]].assign(id=np.arange(int((df.y == 0).sum()))).to_csv(
            original / "inactives.smi", sep=" ", header=False, index=False
        )

        for d in [screen_dir, test_dir]:
            if d.exists():
                shutil.rmtree(d)

        MasterDataset(name="screen", df=screen, overwrite=True, dataset=assay)
        MasterDataset(name="test", df=test, overwrite=True, dataset=assay)

        usage_summary = {}
        for usage in ["screen", "test"]:
            base = assay_dir / usage
            fp = torch.load(base / "x", map_location="cpu", weights_only=False)
            graph_list = torch.load(base / "graphs", map_location="cpu", weights_only=False)
            graph2_list = []
            t0 = time.time()
            for i, graph in enumerate(graph_list):
                graph.fp = torch.tensor([fp[i]], dtype=torch.float32)
                mol = Chem.MolFromSmiles(graph.smiles, sanitize=True)
                if mol is None:
                    raise RuntimeError(f"bad smiles at {assay}/{usage}/{i}: {graph.smiles}")
                xp, edgep_index, edgep_attr = mol_to_graph_data_obj_simple_3D(mol)
                graph.xp = xp
                graph.edgep_index = edgep_index
                graph.edgep_attr = edgep_attr
                graph2_list.append(graph)
                if args.progress_every and (i + 1) % args.progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "assay": assay,
                                "usage": usage,
                                "done": i + 1,
                                "total": len(graph_list),
                                "elapsed_sec": round(time.time() - t0, 1),
                            }
                        ),
                        flush=True,
                    )
            torch.save(graph2_list, base / "graphs2", pickle_protocol=4)
            usage_summary[usage] = {
                "n": len(graph2_list),
                "active": int((screen if usage == "screen" else test)["y"].sum()),
                "graphs2_bytes": int((base / "graphs2").stat().st_size),
                "seconds": round(time.time() - t0, 3),
            }

        marker = {
            "status": "done",
            "assay": assay,
            "split_seed": args.split_seed,
            "n_after_invalid_smiles_filter": int(len(df)),
            "n_invalid_smiles_filtered": n_invalid_smiles,
            "screen_size_requested": args.screen_size,
            "test_size_requested": args.test_size,
            "usage": usage_summary,
            "seconds_total": round(time.time() - started, 3),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        done_marker.write_text(json.dumps(marker, ensure_ascii=False, indent=2))
        manifest_rows.append(marker)
        print(json.dumps(marker, ensure_ascii=False, indent=2), flush=True)

        if args.mirror_acactive:
            hardlink_tree(assay_dir, ac_repo / "data" / assay)

    out = tables / "e13_glare_acactive_data_prepare_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(out, index=False)
    print(json.dumps({"manifest": str(out), "n_assays": len(manifest_rows)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
