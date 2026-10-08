#!/usr/bin/env python3
"""Read-only numerical and trajectory audit for the compact release tables."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "source_tables"
TRANSFER = TABLES / "e98_air_e_transfer"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_vector(row: dict[str, str], *, values: str, mean: str, sd: str,
                 separator: str, tolerance: float) -> list[float]:
    vector = [float(x) for x in row[values].split(separator)]
    assert len(vector) == int(row["n_seeds"]) == 5, row
    assert math.isclose(statistics.mean(vector), float(row[mean]), abs_tol=tolerance), row
    assert math.isclose(statistics.stdev(vector), float(row[sd]), abs_tol=tolerance), row
    return vector


def check_original_air() -> None:
    main = rows(TABLES / "e13_q192_main_with_official_external.csv")
    budget = rows(TABLES / "e13_multi_budget_seed_summary.csv")
    assert len(main) == 156 and len(budget) == 176
    assert len({r["assay"] for r in main}) == 11
    assert len(rows(TABLES / "e11_leave_one_assay_out_gate_selection.csv")) == 11
    for row in main + budget:
        check_vector(row, values="values", mean="mean_hits", sd="std_hits",
                     separator=" ", tolerance=1e-7)
        assert row["complete_5seed"] == "True" and row["all_budget_ok"] == "True", row


def check_transfer() -> None:
    per_assay = rows(TRANSFER / "e98_matched_lit_per_assay.csv")
    macro = rows(TRANSFER / "e98_matched_lit_macro.csv")
    baseline = rows(TABLES / "e96_matched_lit_final_status.csv")
    assert len(per_assay) == 121 and len(macro) == 11 and len(baseline) == 110
    observed: dict[tuple[str, str], dict[int, int]] = {}
    for row in per_assay:
        values = check_vector(row, values="hits_by_seed", mean="mean_hits",
                              sd="sd_hits", separator=";", tolerance=5e-5)
        seeds = [int(x) for x in row["seeds"].split(";")]
        assert len(seeds) == len(set(seeds)) == 5
        key = (row["assay"], row["method"])
        assert key not in observed
        observed[key] = dict(zip(seeds, map(int, values), strict=True))
    assays = {assay for assay, _ in observed}
    methods = {method for _, method in observed}
    assert len(assays) == len(methods) == 11
    assert set(observed) == {(a, m) for a in assays for m in methods}
    for row in macro:
        values = [float(next(r["mean_hits"] for r in per_assay
                             if r["assay"] == assay and r["method"] == row["method"]))
                  for assay in assays]
        assert math.isclose(statistics.mean(values), float(row["macro_mean_hits"]),
                            abs_tol=6e-5), row
    for row in baseline:
        key = (row["assay"], row["method"])
        values = [int(x) for x in row["hits_by_seed"].split(";")]
        seeds = [int(x) for x in row["seeds"].split(";")]
        assert observed[key] == dict(zip(seeds, values, strict=True)), key

    manifests = sorted(TRANSFER.glob("e98_lit_air_e_*_manifest.json"))
    assert len(manifests) == 55
    seen: set[tuple[str, int]] = set()
    for manifest in manifests:
        info = json.loads(manifest.read_text(encoding="utf-8"))
        assay, seed = info["assay"], int(info["seed"])
        assert (assay, seed) not in seen
        seen.add((assay, seed))
        assert info["status"] == "complete" and info["method"] == "AIR-E-transfer"
        assert info["q0"] == 64 and info["Q"] == 192 and info["batch_size"] == 64
        assert info["params"] == [0.15, 0.1]
        stem = manifest.name.removesuffix("_manifest.json")
        trajectory = TRANSFER / f"{stem}.csv"
        query = TRANSFER / f"{stem}_query.csv"
        assert digest(trajectory) == info["trajectory_sha256"]
        assert digest(query) == info["query_sha256"]
        rounds, queries = rows(trajectory), rows(query)
        assert [int(r["budget"]) for r in rounds] == [64, 128, 192]
        assert len(queries) == 192
        assert [int(r["order"]) for r in queries] == list(range(1, 193))
        assert len({int(r["index"]) for r in queries}) == 192
        labels = [int(r["y"]) for r in queries]
        assert set(labels) <= {0, 1}
        assert sum(labels[:64]) == int(rounds[0]["total_hits"]) == info["initial_hits"]
        assert sum(labels[:128]) == int(rounds[1]["total_hits"])
        assert sum(labels) == int(rounds[2]["total_hits"]) == info["final_hits"]
        assert observed[(assay, "AIR-E-transfer")][seed] == info["final_hits"]
    assert len(seen) == 11 * 5


def check_tdc() -> None:
    status = rows(TABLES / "e91_tdc_authorstart_aligned_status.csv")
    paper = rows(TABLES / "e92_fcs_tdc_q384_paper_table.csv")
    assert len(status) == 21 and len(paper) == 7
    assert {row["aid"] for row in status} == {"1798", "463087", "485290"}
    by_method: dict[str, list[float]] = defaultdict(list)
    for row in status:
        check_vector(row, values="hits_by_seed", mean="mean_hits", sd="sd_hits",
                     separator=";", tolerance=1e-7)
        assert row["status"] == "complete" and row["q0"] == "64" and row["Q"] == "384"
        by_method[row["method"]].append(float(row["mean_hits"]))
    for row in paper:
        values = by_method[row["method"]]
        assert len(values) == 3
        assert math.isclose(statistics.mean(values), float(row["macro_mean_hits"]),
                            abs_tol=5e-4), row


if __name__ == "__main__":
    check_original_air()
    check_transfer()
    check_tdc()
    print("PASS: original AIR, 55 AIR-E trajectories, matched LIT-PCBA and TDC-HTS tables")
