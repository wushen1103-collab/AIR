from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import beta
from tqdm import tqdm

from airvs.audit.sampling import assign_equal_size_strata, stratified_sample
from airvs.data.registry import SEEDS
from airvs.metrics.core import hits, nef, recall
from airvs.simulator.evaluator import BudgetLedger, SpyEvaluator
from airvs.utils.determinism import make_rng
from run_e03_headroom import load_assay, unpack_rows

ROOT = Path(__file__).resolve().parents[1]
DEV_ASSAYS = ['ALDH1', 'PKM2', 'VDR']
METHODS = ['B0_random', 'B1_fixed_cascade', 'B2_no_hard_delete_static', 'B3_epsilon_greedy', 'B4_ensemble_ucb', 'B10_audit_rule']


def screening_budgets(n: int) -> tuple[int, int, int]:
    if n >= 20000:
        q = min(1000, math.ceil(0.01 * n))
        b1 = min(5000, math.ceil(0.05 * n))
    else:
        q = min(400, math.ceil(0.05 * n))
        b1 = min(1000, math.ceil(0.20 * n))
    q0 = min(64, max(32, math.floor(0.10 * q)))
    return q, b1, q0


def train_predict_ensemble_stats(x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray, seed: int, n_jobs: int) -> tuple[np.ndarray, np.ndarray]:
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
    stacked = np.vstack(preds)
    return stacked.mean(axis=0), stacked.std(axis=0)


def order_by(score: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    # Candidate arrays are molecule_id sorted, so stable mergesort gives deterministic tie handling.
    return candidates[np.argsort(-score[candidates], kind='mergesort')]


def choose_random(candidates: np.ndarray, k: int, seed: int, *parts: object) -> np.ndarray:
    k = min(k, len(candidates))
    if k <= 0:
        return np.array([], dtype=np.int64)
    rng = make_rng(seed, *parts)
    return rng.choice(candidates, size=k, replace=False).astype(np.int64)


def choose_stratified_audit(candidates: np.ndarray, ids: np.ndarray, smiles: np.ndarray, p0: np.ndarray, k: int, seed: int, assay: str, method: str) -> np.ndarray:
    k = min(k, len(candidates))
    if k <= 0:
        return np.array([], dtype=np.int64)
    frame = pd.DataFrame({
        'row_idx': candidates.astype(np.int64),
        'molecule_id': ids[candidates].astype(str),
        'canonical_smiles': smiles[candidates].astype(str),
        'score': p0[candidates].astype(float),
    })
    frame = assign_equal_size_strata(frame, 'score', smiles_col='canonical_smiles', n_strata=5)
    audit_seed = int(make_rng(seed, assay, method, 'audit_base').integers(0, 2**31 - 1))
    sampled = stratified_sample(frame, q=k, seed=audit_seed, stratum_col='stratum', id_col='molecule_id')
    return sampled['row_idx'].astype(np.int64).to_numpy()


def make_initial(y: np.ndarray, assay: str, seed: int, q0: int) -> np.ndarray:
    active_idx = np.flatnonzero(y == 1)
    inactive_idx = np.flatnonzero(y == 0)
    rng = make_rng(seed, assay, 'seeded_post_hit')
    init_active = rng.choice(active_idx, size=1, replace=False)
    init_inactive = rng.choice(inactive_idx, size=q0 - 1, replace=False)
    return np.concatenate([init_active, init_inactive]).astype(np.int64)


def query_indices(indices: Iterable[int], ids: np.ndarray, evaluator: SpyEvaluator, ledger: BudgetLedger, discovered: dict[str, int]) -> dict[str, int]:
    selected_ids = [str(ids[int(i)]) for i in indices]
    labels = evaluator.query_labels(selected_ids, ledger)
    discovered.update(labels)
    return labels


def spend_expensive(indices: Iterable[int], ids: np.ndarray, ledger: BudgetLedger) -> None:
    selected_ids = [str(ids[int(i)]) for i in indices]
    ledger.spend_b1(selected_ids)


def audit_rule_select(
    p0: np.ndarray,
    normal_pool: np.ndarray,
    discarded_pool: np.ndarray,
    audit_labels: dict[int, int],
    audit_indices: np.ndarray,
    reentry_pool_size: int,
    h: int = 5,
) -> np.ndarray:
    if reentry_pool_size <= 0 or len(discarded_pool) == 0:
        return np.array([], dtype=np.int64)
    low_to_high = discarded_pool[np.argsort(p0[discarded_pool], kind='mergesort')]
    strata = np.array_split(low_to_high, h)
    theta = {}
    local_rate = {}
    for sid, stratum in enumerate(strata):
        if len(stratum) == 0:
            theta[sid] = 0.0
            local_rate[sid] = 0.0
            continue
        stratum_set = set(map(int, stratum.tolist()))
        obs = [audit_labels[i] for i in audit_indices.tolist() if int(i) in stratum_set]
        n = len(obs)
        k = int(sum(obs))
        local_rate[sid] = (k / n) if n else 0.0
        # Bayesian-smoothed 90% upper rate; used as rule score only, not a certificate.
        theta[sid] = float(beta.ppf(0.90, k + 1, n - k + 1)) if n else 1.0
    p_rank = pd.Series(p0[discarded_pool]).rank(method='average', pct=True).to_numpy()
    sid_for = {}
    for sid, stratum in enumerate(strata):
        for idx in stratum.tolist():
            sid_for[int(idx)] = sid
    scores = []
    for j, idx in enumerate(discarded_pool.tolist()):
        sid = sid_for[int(idx)]
        scores.append(0.50 * float(p_rank[j]) + 0.25 * theta[sid] + 0.25 * local_rate[sid])
    score_arr = np.asarray(scores, dtype=float)
    order = np.argsort(-score_arr, kind='mergesort')
    return discarded_pool[order[: min(reentry_pool_size, len(discarded_pool))]]


def run_method(assay: str, seed: int, method: str, ids: np.ndarray, smiles: np.ndarray, y: np.ndarray, p0_mean: np.ndarray, p0_std: np.ndarray, init_idx: np.ndarray) -> dict[str, object]:
    n = len(ids)
    q, b1, q0 = screening_budgets(n)
    id_to_label = {str(ids[i]): int(y[i]) for i in range(n)}
    evaluator = SpyEvaluator(id_to_label)
    ledger = BudgetLedger(q_budget=q, b1_budget=b1)
    discovered: dict[str, int] = {}
    query_indices(init_idx, ids, evaluator, ledger, discovered)
    already = np.zeros(n, dtype=bool)
    already[init_idx] = True
    available = np.flatnonzero(~already)
    q_remaining = q - q0
    source_counts = {'seed': q0, 'audit': 0, 'normal': 0, 'reentry': 0, 'random': 0}
    source_hits = {'seed': hits(discovered), 'audit': 0, 'normal': 0, 'reentry': 0, 'random': 0}

    selected_final: np.ndarray
    score = p0_mean.copy()
    expensive_pool = np.array([], dtype=np.int64)

    if method == 'B0_random':
        expensive_pool = choose_random(available, min(b1, len(available)), seed, assay, method, 'b1_pool')
        spend_expensive(expensive_pool, ids, ledger)
        selected_final = choose_random(expensive_pool, q_remaining, seed, assay, method, 'q_query')
        labels = query_indices(selected_final, ids, evaluator, ledger, discovered)
        source_counts['random'] = len(selected_final)
        source_hits['random'] = hits(labels)
    elif method in {'B1_fixed_cascade', 'B2_no_hard_delete_static', 'B3_epsilon_greedy'}:
        ordered = order_by(score, available)
        expensive_pool = ordered[: min(b1, len(ordered))]
        spend_expensive(expensive_pool, ids, ledger)
        if method == 'B3_epsilon_greedy':
            greedy_k = min(math.floor(0.90 * q_remaining), len(expensive_pool))
            greedy = expensive_pool[:greedy_k]
            pool_mask = np.ones(len(expensive_pool), dtype=bool)
            pool_mask[:greedy_k] = False
            random_part = choose_random(expensive_pool[pool_mask], q_remaining - greedy_k, seed, assay, method, 'epsilon')
            selected_final = np.concatenate([greedy, random_part]).astype(np.int64)
        else:
            selected_final = expensive_pool[: min(q_remaining, len(expensive_pool))]
        labels = query_indices(selected_final, ids, evaluator, ledger, discovered)
        source_counts['normal'] = len(selected_final)
        source_hits['normal'] = hits(labels)
    elif method == 'B4_ensemble_ucb':
        ucb = p0_mean + p0_std
        ordered = order_by(ucb, available)
        expensive_pool = ordered[: min(b1, len(ordered))]
        spend_expensive(expensive_pool, ids, ledger)
        selected_final = expensive_pool[: min(q_remaining, len(expensive_pool))]
        labels = query_indices(selected_final, ids, evaluator, ledger, discovered)
        source_counts['normal'] = len(selected_final)
        source_hits['normal'] = hits(labels)
    elif method == 'B10_audit_rule':
        ordered = order_by(score, available)
        normal_pool_size = min(math.floor(0.80 * b1), len(ordered))
        normal_pool = ordered[:normal_pool_size]
        discarded_pool = ordered[normal_pool_size:]
        audit_k = min(math.floor(0.20 * q_remaining), len(discarded_pool))
        audit_idx = choose_stratified_audit(discarded_pool, ids, smiles, score, audit_k, seed, assay, method)
        audit_label_by_id = query_indices(audit_idx, ids, evaluator, ledger, discovered)
        audit_labels = {int(i): int(audit_label_by_id[str(ids[int(i)])]) for i in audit_idx.tolist()}
        source_counts['audit'] = len(audit_idx)
        source_hits['audit'] = hits(audit_label_by_id)
        remaining_after_audit = q - ledger.q_spent
        reentry_query_k = min(math.floor(0.20 * remaining_after_audit), remaining_after_audit)
        normal_query_k = remaining_after_audit - reentry_query_k
        discarded_after_audit = np.array([i for i in discarded_pool.tolist() if i not in set(audit_idx.tolist())], dtype=np.int64)
        reentry_pool_size = min(b1 - normal_pool_size, len(discarded_after_audit))
        reentry_pool = audit_rule_select(score, normal_pool, discarded_after_audit, audit_labels, audit_idx, reentry_pool_size)
        expensive_pool = np.concatenate([normal_pool, reentry_pool]).astype(np.int64)
        spend_expensive(expensive_pool, ids, ledger)
        normal_query = normal_pool[: min(normal_query_k, len(normal_pool))]
        reentry_query = reentry_pool[: min(reentry_query_k, len(reentry_pool))]
        selected_final = np.concatenate([normal_query, reentry_query]).astype(np.int64)
        labels_normal = query_indices(normal_query, ids, evaluator, ledger, discovered)
        labels_reentry = query_indices(reentry_query, ids, evaluator, ledger, discovered)
        source_counts['normal'] = len(normal_query)
        source_counts['reentry'] = len(reentry_query)
        source_hits['normal'] = hits(labels_normal)
        source_hits['reentry'] = hits(labels_reentry)
    else:
        raise ValueError(method)

    evaluator.unlock_metrics()
    total_active = evaluator.total_active
    discovered_hits = hits(discovered)
    return {
        'protocol': 'cheap_only_v0',
        'assay': assay,
        'seed': seed,
        'method': method,
        'n': n,
        'total_active': total_active,
        'Q': q,
        'B1': b1,
        'q0': q0,
        'q_spent': ledger.q_spent,
        'b1_spent': ledger.b1_spent,
        'hits': discovered_hits,
        'recall': recall(discovered, total_active),
        'nef': nef(discovered, ledger.q_spent, total_active, n),
        'seed_hits': source_hits['seed'],
        'audit_hits': source_hits['audit'],
        'normal_hits': source_hits['normal'],
        'reentry_hits': source_hits['reentry'],
        'random_hits': source_hits['random'],
        'seed_queries': source_counts['seed'],
        'audit_queries': source_counts['audit'],
        'normal_queries': source_counts['normal'],
        'reentry_queries': source_counts['reentry'],
        'random_queries': source_counts['random'],
        'budget_ok': ledger.q_spent <= q and ledger.b1_spent <= b1 and len(ledger.queried_ids) == ledger.q_spent,
    }


def run_assay_seed(assay: str, seed: int, methods: list[str], lgbm_jobs: int) -> list[dict[str, object]]:
    ids_df, packed, y = load_assay(assay)
    ids = ids_df['molecule_id'].astype(str).to_numpy()
    q, b1, q0 = screening_budgets(len(ids))
    init_idx = make_initial(y, assay, seed, q0)
    x_train = unpack_rows(packed, init_idx)
    x_all = unpack_rows(packed)
    p0_mean, p0_std = train_predict_ensemble_stats(x_train, y[init_idx].astype(int), x_all, seed, lgbm_jobs)
    smiles = ids_df['canonical_smiles'].astype(str).to_numpy()
    return [run_method(assay, seed, method, ids, smiles, y, p0_mean, p0_std, init_idx) for method in methods]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(['protocol', 'method', 'assay'], as_index=False)
        .agg(
            n_seeds=('seed', 'nunique'),
            mean_hits=('hits', 'mean'),
            mean_recall=('recall', 'mean'),
            mean_nef=('nef', 'mean'),
            mean_q_spent=('q_spent', 'mean'),
            mean_b1_spent=('b1_spent', 'mean'),
            mean_audit_hits=('audit_hits', 'mean'),
            mean_reentry_hits=('reentry_hits', 'mean'),
            all_budget_ok=('budget_ok', 'all'),
        )
        .sort_values(['assay', 'method'])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--assays', nargs='*', default=DEV_ASSAYS)
    parser.add_argument('--seeds', nargs='*', type=int, default=list(SEEDS[:5]))
    parser.add_argument('--methods', nargs='*', default=METHODS)
    parser.add_argument('--lgbm-jobs', type=int, default=8)
    parser.add_argument('--prefix', default='e05_baseline_smoke')
    args = parser.parse_args()
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'tables').mkdir(exist_ok=True)
    rows = []
    for assay in args.assays:
        for seed in tqdm(args.seeds, desc=assay):
            batch = run_assay_seed(assay, seed, args.methods, args.lgbm_jobs)
            rows.extend(batch)
            pd.DataFrame(rows).to_parquet(ROOT / 'results' / f'{args.prefix}.partial.parquet', index=False)
            print(json.dumps({'assay': assay, 'seed': seed, 'rows': len(batch)}, ensure_ascii=False), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(ROOT / 'results' / f'{args.prefix}.parquet', index=False)
    df.to_csv(ROOT / 'tables' / f'{args.prefix}.csv', index=False)
    summary = summarize(df)
    summary.to_csv(ROOT / 'tables' / f'{args.prefix}_summary.csv', index=False)
    random_rows = df[df['method'] == 'B0_random'].copy()
    expected = []
    for _, row in random_rows.iterrows():
        n_unq = row['n'] - row['q0']
        act_unq = row['total_active'] - row['seed_hits']
        expected_hits = row['seed_hits'] + (row['Q'] - row['q0']) * (act_unq / n_unq)
        expected.append(expected_hits)
    random_rows['expected_random_hits_seeded'] = expected
    random_check = random_rows.groupby('assay', as_index=False).agg(
        observed_mean_hits=('hits', 'mean'),
        expected_mean_hits=('expected_random_hits_seeded', 'mean'),
        n_seeds=('seed', 'nunique'),
    )
    random_check['abs_gap'] = (random_check['observed_mean_hits'] - random_check['expected_mean_hits']).abs()
    random_check.to_csv(ROOT / 'tables' / f'{args.prefix}_random_check.csv', index=False)
    report = {
        'protocol': 'cheap_only_v0',
        'assays': args.assays,
        'seeds': args.seeds,
        'methods': args.methods,
        'n_rows': int(len(df)),
        'all_budget_ok': bool(df['budget_ok'].all()),
        'outputs': {
            'runs': f'tables/{args.prefix}.csv',
            'summary': f'tables/{args.prefix}_summary.csv',
            'random_check': f'tables/{args.prefix}_random_check.csv',
        },
        'warning': 'cheap-only smoke uses p0 as final score because chemprop/p1 is not yet available; do not report as formal E07 primary result',
    }
    (ROOT / 'results' / f'{args.prefix}_budget_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (ROOT / 'tables' / f'{args.prefix}_budget_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
