from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from airvs.data.registry import LIT_ASSAYS, SEEDS
from airvs.utils.determinism import make_rng

ROOT = Path(__file__).resolve().parents[1]
MAIN_ASSAYS = {a.name for a in LIT_ASSAYS if a.is_main}


def budgets(n: int) -> tuple[int, int, int]:
    """Return total wet-query budget Q, first-stage candidate pool B1, and seeded warm start q0."""
    q = min(1000, math.ceil(0.01 * n))
    b1 = min(5000, math.ceil(0.05 * n))
    q0 = min(64, max(32, math.floor(0.10 * q)))
    return q, b1, q0


def load_assay(assay: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ids = pd.read_parquet(ROOT / 'data' / 'processed' / 'fingerprints' / f'{assay}_ids.parquet')
    packed = np.load(ROOT / 'data' / 'processed' / 'fingerprints' / f'{assay}_morgan_r2_2048_chiral.packbits.npy', mmap_mode='r')
    labels = pd.read_parquet(ROOT / 'data' / 'secure' / 'lit_pcba_labels.secure.parquet')
    labels = labels[labels['assay'] == assay][['molecule_id', 'y']]
    merged = ids.merge(labels, on='molecule_id', how='left', validate='one_to_one')
    if merged['y'].isna().any():
        raise RuntimeError(f'missing labels for {assay}')
    y = merged['y'].astype(np.int8).to_numpy()
    return merged, packed, y


def unpack_rows(packed: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    arr = packed if rows is None else packed[rows]
    bits = np.unpackbits(np.asarray(arr), axis=1, bitorder='little')
    return bits[:, :2048].astype(np.uint8, copy=False)


def train_predict_ensemble(x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray, seed: int, n_jobs: int) -> np.ndarray:
    preds = []
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, int((y_train == 0).sum()))
    for offset in [0, 100, 200, 300, 400]:
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            reg_lambda=1.0,
            scale_pos_weight=n_neg / n_pos,
            random_state=seed + offset,
            n_jobs=n_jobs,
            verbosity=-1,
        )
        model.fit(x_train, y_train)
        preds.append(model.predict_proba(x_all)[:, 1])
    return np.mean(preds, axis=0)


def q_limited_oracle_metrics(y: np.ndarray, normal_idx: np.ndarray, discarded_idx: np.ndarray, init_hits: int, q: int, q0: int) -> dict[str, int | float]:
    """Conservative same-Q oracle: re-entry can replace only tail normal queries inside the fixed Q budget."""
    query_slots = max(0, q - q0)
    base_query_idx = normal_idx[: min(query_slots, len(normal_idx))]
    base_hits = init_hits + int(y[base_query_idx].sum())

    # Keep re-entry to 20% of post-seed wet-query slots. The same number of
    # lowest-ranked normal queries are displaced, so this is a net same-budget upper bound.
    reentry_slots = min(math.floor(0.20 * query_slots), len(discarded_idx), len(base_query_idx))
    if reentry_slots > 0:
        displaced_idx = base_query_idx[-reentry_slots:]
        displaced_active = int(y[displaced_idx].sum())
    else:
        displaced_active = 0
    active_discarded = int(y[discarded_idx].sum())
    oracle_add = min(reentry_slots, active_discarded)
    oracle_hits = base_hits - displaced_active + oracle_add
    net_gain = oracle_hits - base_hits
    return {
        'q_limited_post_seed_query_slots': int(query_slots),
        'q_limited_reentry_slots': int(reentry_slots),
        'q_limited_base_hits': int(base_hits),
        'q_limited_displaced_active': int(displaced_active),
        'q_limited_oracle_add_hits': int(oracle_add),
        'q_limited_oracle_hit_capacity': int(oracle_hits),
        'q_limited_oracle_net_gain': int(net_gain),
        'q_limited_oracle_relative_gain': float(net_gain / max(base_hits, 1)),
        'q_limited_base_hit_rate': float(base_hits / max(q, 1)),
        'q_limited_oracle_hit_rate': float(oracle_hits / max(q, 1)),
    }


def run_one(assay: str, seed: int, n_jobs: int) -> dict[str, object]:
    ids, packed, y = load_assay(assay)
    n = len(ids)
    q, b1, q0 = budgets(n)
    active_idx = np.flatnonzero(y == 1)
    inactive_idx = np.flatnonzero(y == 0)
    if len(active_idx) == 0 or len(inactive_idx) < q0 - 1:
        raise RuntimeError(f'bad label distribution for {assay}')
    rng = make_rng(seed, assay, 'seeded_post_hit')
    init_active = rng.choice(active_idx, size=1, replace=False)
    init_inactive = rng.choice(inactive_idx, size=q0 - 1, replace=False)
    train_idx = np.concatenate([init_active, init_inactive]).astype(np.int64)
    x_train = unpack_rows(packed, train_idx)
    # Headroom analysis uses p0 from the cheap model; labels outside train_idx
    # are accessed only after the fixed funnel is frozen.
    x_all = unpack_rows(packed)
    p0 = train_predict_ensemble(x_train, y[train_idx].astype(int), x_all, seed, n_jobs)
    unqueried_mask = np.ones(n, dtype=bool)
    unqueried_mask[train_idx] = False
    unqueried_idx = np.flatnonzero(unqueried_mask)
    order = unqueried_idx[np.argsort(-p0[unqueried_idx], kind='mergesort')]
    normal_idx = order[: min(b1, len(order))]
    discarded_idx = order[min(b1, len(order)):]
    active_total = int(y.sum())
    active_normal = int(y[normal_idx].sum())
    active_discarded = int(y[discarded_idx].sum())
    init_hits = int(y[train_idx].sum())

    # Capacity upper bound: if a second-stage auditor could append up to 20% of
    # the first-stage pool from discarded candidates, how many labels are there to recover?
    reentry_slots = min(math.floor(0.20 * b1), len(discarded_idx))
    oracle_add = min(reentry_slots, active_discarded)
    base_hit_capacity = init_hits + active_normal
    oracle_hit_capacity = base_hit_capacity + oracle_add
    discarded_active_fraction = active_discarded / active_total if active_total else 0.0
    oracle_relative_gain = oracle_add / max(base_hit_capacity, 1)

    q_metrics = q_limited_oracle_metrics(y, normal_idx, discarded_idx, init_hits, q, q0)
    row = {
        'assay': assay,
        'seed': seed,
        'n': n,
        'total_active': active_total,
        'Q': q,
        'B1': b1,
        'q0': q0,
        'init_hits': init_hits,
        'normal_size': int(len(normal_idx)),
        'discarded_size': int(len(discarded_idx)),
        'active_normal': active_normal,
        'active_discarded': active_discarded,
        'discarded_active_fraction': discarded_active_fraction,
        'capacity_oracle_reentry_slots': int(reentry_slots),
        'capacity_oracle_add_hits': int(oracle_add),
        'capacity_base_hit_count': int(base_hit_capacity),
        'capacity_oracle_hit_count': int(oracle_hit_capacity),
        'capacity_oracle_relative_gain': float(oracle_relative_gain),
        # Backward-compatible aliases retained for pilot scripts/tables already produced.
        'oracle_reentry_slots': int(reentry_slots),
        'oracle_add_hits': int(oracle_add),
        'base_hit_capacity': int(base_hit_capacity),
        'oracle_hit_capacity': int(oracle_hit_capacity),
        'oracle_relative_gain': float(oracle_relative_gain),
    }
    row.update(q_metrics)
    return row


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for assay, sub in df.groupby('assay'):
        mean_discarded = sub['discarded_active_fraction'].mean()
        mean_q_rel = sub['q_limited_oracle_relative_gain'].mean()
        mean_q_net = sub['q_limited_oracle_net_gain'].mean()
        rows.append({
            'assay': assay,
            'n_seeds': len(sub),
            'n': int(sub['n'].iloc[0]),
            'total_active': int(sub['total_active'].iloc[0]),
            'mean_discarded_active_fraction': mean_discarded,
            'mean_capacity_oracle_relative_gain': sub['capacity_oracle_relative_gain'].mean(),
            'mean_q_limited_oracle_relative_gain': mean_q_rel,
            'mean_q_limited_oracle_net_gain': mean_q_net,
            'mean_q_limited_base_hits': sub['q_limited_base_hits'].mean(),
            'mean_q_limited_oracle_hit_capacity': sub['q_limited_oracle_hit_capacity'].mean(),
            'mean_active_discarded': sub['active_discarded'].mean(),
            'mean_capacity_base_hit_count': sub['capacity_base_hit_count'].mean(),
            'passes_g1_component': bool((mean_discarded >= 0.20) and (mean_q_rel >= 0.10) and (mean_q_net >= 5.0)),
        })
    out = pd.DataFrame(rows).sort_values('assay')
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--assays', nargs='*', default=sorted(MAIN_ASSAYS))
    parser.add_argument('--seeds', nargs='*', type=int, default=list(SEEDS))
    parser.add_argument('--lgbm-jobs', type=int, default=8)
    parser.add_argument('--output-prefix', default='e03_headroom')
    args = parser.parse_args()
    assays = [a for a in args.assays if a in MAIN_ASSAYS]
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'tables').mkdir(exist_ok=True)
    prefix = args.output_prefix
    rows = []
    for assay in assays:
        for seed in tqdm(args.seeds, desc=assay):
            row = run_one(assay, seed, n_jobs=args.lgbm_jobs)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            pd.DataFrame(rows).to_parquet(ROOT / 'results' / f'{prefix}_runs.partial.parquet', index=False)
    df = pd.DataFrame(rows)
    df.to_parquet(ROOT / 'results' / f'{prefix}_runs.parquet', index=False)
    summary = summarize(df)
    summary.to_csv(ROOT / 'tables' / f'{prefix}_by_target.csv', index=False)
    pass_count = int(summary['passes_g1_component'].sum())
    gate = {
        'assays': assays,
        'seeds': args.seeds,
        'passing_targets': pass_count,
        'target_total': len(summary),
        'g1_pass_if_full_lit': bool(len(summary) == 11 and pass_count >= 8),
        'definition': 'cheap-model fixed funnel; report both capacity upper bound and conservative same-Q oracle; G1 component requires discarded_active_fraction>=0.20, q_limited_relative_gain>=0.10, q_limited_net_gain>=5',
    }
    (ROOT / 'results' / f'{prefix}_gate_status.json').write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
