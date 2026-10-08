#!/usr/bin/env python3
"""Audit E98 transfer cells and summarize the fixed E15 comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "e96_matched_lit_final_status.csv"
METHOD = "AIR-E-transfer"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    baseline = read_csv(BASE)
    expected: dict[str, list[int]] = {}
    hits: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
    for row in baseline:
        assay, method = row["assay"], row["method"]
        seeds = [int(v) for v in row["seeds"].split(";")]
        values = [int(v) for v in row["hits_by_seed"].split(";")]
        assert len(seeds) == len(values) == 5
        if assay in expected:
            assert expected[assay] == seeds, (assay, method)
        else:
            expected[assay] = seeds
        assert (assay, method) not in hits
        hits[assay, method] = dict(zip(seeds, values, strict=True))

    manifests = sorted(HERE.glob("e98_lit_air_e_*_manifest.json"))
    assert len(expected) == 11 and len(manifests) == 55
    observed: set[tuple[str, int]] = set()
    for path in manifests:
        item = json.loads(path.read_text(encoding="utf-8"))
        assay, seed = item["assay"], int(item["seed"])
        assert item["status"] == "complete" and item["method"] == METHOD
        assert item["q0"] == 64 and item["Q"] == 192 and item["batch_size"] == 64
        assert item["params"] == [0.15, 0.1]
        assert assay in expected and seed in expected[assay]
        assert (assay, seed) not in observed
        observed.add((assay, seed))
        stem = path.name.removesuffix("_manifest.json")
        trajectory = HERE / f"{stem}.csv"
        query = HERE / f"{stem}_query.csv"
        assert trajectory.is_file() and query.is_file()
        assert digest(trajectory) == item["trajectory_sha256"]
        assert digest(query) == item["query_sha256"]
        rounds = read_csv(trajectory)
        query_rows = read_csv(query)
        assert [int(r["budget"]) for r in rounds] == [64, 128, 192]
        assert len(query_rows) == 192
        assert [int(r["order"]) for r in query_rows] == list(range(1, 193))
        assert len({int(r["index"]) for r in query_rows}) == 192
        assert all(int(r["y"]) in (0, 1) for r in query_rows)
        initial = sum(int(r["y"]) for r in query_rows[:64])
        final = sum(int(r["y"]) for r in query_rows)
        assert initial == item["initial_hits"] == int(rounds[0]["total_hits"])
        assert final == item["final_hits"] == int(rounds[-1]["total_hits"])
        assert sum(int(r["y"]) for r in query_rows[:128]) == int(rounds[1]["total_hits"])
        hits[assay, METHOD][seed] = final
    assert observed == {(assay, seed) for assay, seeds in expected.items() for seed in seeds}

    summary: list[dict] = []
    for assay in sorted(expected):
        for method in sorted({m for a, m in hits if a == assay}):
            values = [hits[assay, method][seed] for seed in expected[assay]]
            summary.append({
                "assay": assay, "method": method, "n_seeds": 5,
                "mean_hits": f"{statistics.mean(values):.4f}",
                "sd_hits": f"{statistics.stdev(values):.4f}",
                "seeds": ";".join(map(str, expected[assay])),
                "hits_by_seed": ";".join(map(str, values)),
                "provenance": ("project rerun of frozen TDC E74-E rule on E15 pool"
                               if method == METHOD else "see e96_matched_lit_final_status.csv"),
            })
    write_csv(HERE / "e98_matched_lit_per_assay.csv", summary,
              list(summary[0]))

    methods = sorted({method for _, method in hits})
    macro: list[dict] = []
    for method in methods:
        assay_means = [statistics.mean(hits[assay, method].values()) for assay in sorted(expected)]
        macro.append({"method": method, "n_assays": 11, "n_seeds_each": 5,
                      "macro_mean_hits": f"{statistics.mean(assay_means):.4f}"})
    macro.sort(key=lambda x: float(x["macro_mean_hits"]), reverse=True)
    for rank, row in enumerate(macro, 1):
        row["rank"] = rank
    write_csv(HERE / "e98_matched_lit_macro.csv", macro,
              ["rank", "method", "n_assays", "n_seeds_each", "macro_mean_hits"])
    paired: list[dict] = []
    for comparator in ("RF/ECFP", "ACActive-50e10", "GLARE-50e10",
                       "Positive-Tversky-T1", "AIR-Legacy-2e"):
        differences = [
            statistics.mean(hits[assay, METHOD].values())
            - statistics.mean(hits[assay, comparator].values())
            for assay in sorted(expected)
        ]
        rng = random.Random(98005)
        draws = sorted(statistics.mean(rng.choices(differences, k=11))
                       for _ in range(10000))
        paired.append({
            "method": METHOD, "comparator": comparator,
            "mean_assay_paired_delta_hits": f"{statistics.mean(differences):.4f}",
            "bootstrap_assay_95ci_low": f"{draws[249]:.4f}",
            "bootstrap_assay_95ci_high": f"{draws[9749]:.4f}",
            "assay_win_tie_loss": (
                f"{sum(d > 1e-9 for d in differences)}/"
                f"{sum(abs(d) <= 1e-9 for d in differences)}/"
                f"{sum(d < -1e-9 for d in differences)}"
            ),
        })
    write_csv(HERE / "e98_matched_lit_paired.csv", paired, list(paired[0]))
    print(f"AUDIT PASS: {len(observed)} completed cells, 11 assays, 5 matched seeds, 192 queries")
    for row in macro:
        print(f"{row['rank']:2} {row['method']:30} {row['macro_mean_hits']}")
    for row in paired:
        print(f"versus {row['comparator']:28} {row['mean_assay_paired_delta_hits']} "
              f"CI [{row['bootstrap_assay_95ci_low']}, "
              f"{row['bootstrap_assay_95ci_high']}] "
              f"W/T/L {row['assay_win_tie_loss']}")


if __name__ == "__main__":
    main()
