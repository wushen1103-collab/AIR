
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta
from tqdm import tqdm

from airvs.audit.sampling import assign_equal_size_strata
from airvs.data.registry import LIT_ASSAYS, SEEDS
from airvs.utils.determinism import make_rng
from run_e03_headroom import load_assay, unpack_rows
from run_e05_baseline_smoke import make_initial, screening_budgets, train_predict_ensemble_stats

ROOT = Path(__file__).resolve().parents[1]
K_ROUNDS = 4
N_STRATA = 5
ALL_ASSAYS = [a.name for a in LIT_ASSAYS]

FEATURE_COLUMNS = [
    'feat_p0', 'feat_p0_logit', 'feat_p0_percentile', 'feat_stratum_frac',
    'feat_p0_std', 'feat_entropy', 'feat_nn_queried_active_tanimoto',
    'feat_nn_queried_inactive_tanimoto', 'feat_nn_audit_active_tanimoto',
    'feat_nn_audit_inactive_tanimoto', 'feat_audit_neighbor_tanimoto2_active_rate',
    'feat_audit_neighbor_effective_count_frac', 'feat_theta_mean', 'feat_theta_u90',
    'feat_stratum_audit_frac', 'feat_round_frac', 'feat_remaining_b1_frac',
    'feat_remaining_q_frac', 'mask_no_queried_active', 'mask_no_audit_active',
    'mask_no_audit_neighbor',
]


def split_budget(total: int, k: int = K_ROUNDS) -> list[int]:
    if total < 0:
        raise ValueError('budget must be non-negative')
    base, rem = divmod(int(total), int(k))
    return [base + (1 if i < rem else 0) for i in range(k)]


def percentile_ranks(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(len(scores), dtype=np.float32)
    return ranks / max(len(scores) - 1, 1)


def choose_by_score(scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    return candidates[np.argsort(-scores[candidates], kind='mergesort')]


def allocate_counts(sizes: dict[int, int], total: int, min_per_nonempty: int = 2) -> dict[int, int]:
    total = int(total)
    nonempty = {int(k): int(v) for k, v in sizes.items() if int(v) > 0}
    counts = {k: 0 for k in sorted(nonempty)}
    if total <= 0 or not nonempty:
        return counts
    total = min(total, sum(nonempty.values()))
    if total < min_per_nonempty * len(nonempty):
        order = sorted(nonempty, key=lambda k: (nonempty[k], -k), reverse=True)
        for k in order[:total]:
            counts[k] = 1
        return counts
    for k, size in nonempty.items():
        counts[k] = min(min_per_nonempty, size)
    remaining = total - sum(counts.values())
    if remaining <= 0:
        return counts
    capacity = {k: max(0, nonempty[k] - counts[k]) for k in nonempty}
    cap_sum = sum(capacity.values())
    if cap_sum <= 0:
        return counts
    exact = {k: remaining * capacity[k] / cap_sum for k in nonempty}
    base = {k: min(capacity[k], int(math.floor(exact[k]))) for k in nonempty}
    for k in nonempty:
        counts[k] += base[k]
    need = remaining - sum(base.values())
    order = sorted(nonempty, key=lambda k: (exact[k] - base[k], capacity[k], -k), reverse=True)
    for k in order:
        if need <= 0:
            break
        if counts[k] < nonempty[k]:
            counts[k] += 1
            need -= 1
    if need != 0:
        raise RuntimeError('allocation failed')
    return counts


def stratified_sample_indices(frame: pd.DataFrame, q: int, seed: int, purpose: str) -> np.ndarray:
    if q <= 0 or len(frame) == 0:
        return np.array([], dtype=np.int64)
    sizes = frame.groupby('stratum', observed=True).size().to_dict()
    counts = allocate_counts(sizes, min(q, len(frame)), min_per_nonempty=2)
    pieces: list[np.ndarray] = []
    for stratum, count in counts.items():
        if count <= 0:
            continue
        subset = frame[frame['stratum'] == stratum]
        rng = make_rng(seed, purpose, stratum, len(subset), count)
        chosen_index = rng.choice(subset.index.to_numpy(), size=count, replace=False)
        pieces.append(subset.loc[chosen_index, 'row_idx'].astype(np.int64).to_numpy())
    if not pieces:
        return np.array([], dtype=np.int64)
    out = np.concatenate(pieces).astype(np.int64)
    if len(np.unique(out)) != len(out):
        raise RuntimeError(f'duplicate sampled indices for {purpose}')
    return out


def sample_candidates_by_stratum(frame: pd.DataFrame, max_per_stratum: int, seed: int) -> np.ndarray:
    if max_per_stratum <= 0 or len(frame) == 0:
        return np.array([], dtype=np.int64)
    pieces: list[np.ndarray] = []
    for stratum, subset in frame.groupby('stratum', observed=True, sort=True):
        k = min(int(max_per_stratum), len(subset))
        if k <= 0:
            continue
        rng = make_rng(seed, 'e06_candidate', int(stratum), len(subset), k)
        chosen_index = rng.choice(subset.index.to_numpy(), size=k, replace=False)
        pieces.append(subset.loc[chosen_index, 'row_idx'].astype(np.int64).to_numpy())
    if not pieces:
        return np.array([], dtype=np.int64)
    out = np.concatenate(pieces).astype(np.int64)
    if len(np.unique(out)) != len(out):
        raise RuntimeError('duplicate counterfactual candidate indices')
    return out


def max_tanimoto(cand_bits: np.ndarray, ref_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = cand_bits.shape[0]
    if ref_bits.shape[0] == 0:
        return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.int8)
    c = cand_bits.astype(np.float32, copy=False)
    r = ref_bits.astype(np.float32, copy=False)
    inter = c @ r.T
    denom = c.sum(axis=1, keepdims=True) + r.sum(axis=1)[None, :] - inter
    sims = np.divide(inter, denom, out=np.zeros_like(inter, dtype=np.float32), where=denom > 0)
    return sims.max(axis=1).astype(np.float32), np.zeros(n, dtype=np.int8)


def audit_neighbor_rate(cand_bits: np.ndarray, audit_bits: np.ndarray, audit_y: np.ndarray, k: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = cand_bits.shape[0]
    if audit_bits.shape[0] == 0:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.int8)
    c = cand_bits.astype(np.float32, copy=False)
    r = audit_bits.astype(np.float32, copy=False)
    inter = c @ r.T
    denom = c.sum(axis=1, keepdims=True) + r.sum(axis=1)[None, :] - inter
    sims = np.divide(inter, denom, out=np.zeros_like(inter, dtype=np.float32), where=denom > 0)
    top_k = min(k, sims.shape[1])
    if top_k <= 0:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.int8)
    idx = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]
    top_sims = np.take_along_axis(sims, idx, axis=1)
    top_labels = audit_y[idx].astype(np.float32, copy=False)
    weights = top_sims ** 2
    denom_w = weights.sum(axis=1)
    rate = np.divide((weights * top_labels).sum(axis=1), denom_w, out=np.zeros(n, dtype=np.float32), where=denom_w > 0)
    eff = (top_sims > 0).sum(axis=1).astype(np.float32) / float(k)
    mask = (denom_w <= 0).astype(np.int8)
    return rate.astype(np.float32), eff.astype(np.float32), mask


def unpack_ref(packed: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    arr = np.array(list(indices), dtype=np.int64)
    if len(arr) == 0:
        return np.zeros((0, 2048), dtype=np.uint8)
    return unpack_rows(packed, arr)


def build_feature_rows(assay: str, seed: int, round_idx: int, ids: np.ndarray, y: np.ndarray, packed: np.ndarray,
                       p0_mean: np.ndarray, p0_std: np.ndarray, p0_percentile: np.ndarray,
                       candidate_idx: np.ndarray, normal_ref_idx: int, stratum_by_idx: dict[int, int],
                       stratum_sizes: dict[int, int], current_audit_idx: np.ndarray, audit_idx_all: list[int],
                       queried_idx: set[int], q_remaining_after_audit: int, b1_remaining_before_round: int,
                       q_budget: int, b1_budget: int) -> list[dict[str, object]]:
    if len(candidate_idx) == 0:
        return []
    cand_bits = unpack_rows(packed, candidate_idx)
    queried_active = [i for i in queried_idx if int(y[i]) == 1]
    queried_inactive = [i for i in queried_idx if int(y[i]) == 0]
    audit_active = [i for i in audit_idx_all if int(y[i]) == 1]
    audit_inactive = [i for i in audit_idx_all if int(y[i]) == 0]
    qa_sim, qa_mask = max_tanimoto(cand_bits, unpack_ref(packed, queried_active))
    qi_sim, _ = max_tanimoto(cand_bits, unpack_ref(packed, queried_inactive))
    aa_sim, aa_mask = max_tanimoto(cand_bits, unpack_ref(packed, audit_active))
    ai_sim, _ = max_tanimoto(cand_bits, unpack_ref(packed, audit_inactive))
    audit_all_arr = np.array(audit_idx_all, dtype=np.int64)
    audit_bits = unpack_ref(packed, audit_all_arr)
    audit_labels = y[audit_all_arr].astype(np.int8) if len(audit_all_arr) else np.zeros(0, dtype=np.int8)
    neigh_rate, neigh_eff, neigh_mask = audit_neighbor_rate(cand_bits, audit_bits, audit_labels)
    current_audit_by_stratum: dict[int, list[int]] = {h: [] for h in range(N_STRATA)}
    for idx in current_audit_idx.tolist():
        sid = stratum_by_idx.get(int(idx))
        if sid is not None:
            current_audit_by_stratum[int(sid)].append(int(idx))
    theta_mean: dict[int, float] = {}
    theta_u90: dict[int, float] = {}
    audit_frac: dict[int, float] = {}
    for sid in range(N_STRATA):
        obs = current_audit_by_stratum.get(sid, [])
        n_obs = len(obs)
        k_obs = int(y[np.array(obs, dtype=np.int64)].sum()) if n_obs else 0
        theta_mean[sid] = float((1 + k_obs) / (2 + n_obs))
        theta_u90[sid] = float(beta.ppf(0.90, 1 + k_obs, 1 + n_obs - k_obs)) if n_obs else 1.0
        audit_frac[sid] = float(n_obs / max(stratum_sizes.get(sid, 0), 1))
    p = np.clip(p0_mean[candidate_idx].astype(np.float64), 1e-6, 1 - 1e-6)
    entropy = (-(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)).astype(np.float32)
    logit = np.log(p / (1 - p)).astype(np.float32)
    y_ref = int(y[int(normal_ref_idx)])
    rows: list[dict[str, object]] = []
    for j, idx in enumerate(candidate_idx.tolist()):
        idx = int(idx)
        sid = int(stratum_by_idx[idx])
        delta = int(y[idx]) - y_ref
        rows.append({
            'protocol': 'counterfactual_proxy_v0',
            'proxy_warning': 'one_slot_label_delta_without_chemprop_p1_or_continuations; not a formal E07 primary result',
            'assay': assay, 'seed': int(seed), 'round': int(round_idx),
            'snapshot_id': f'{assay}|{seed}|r{round_idx}',
            'group_id': f'{assay}|{seed}|r{round_idx}|x0={int(normal_ref_idx)}',
            'row_idx': idx, 'molecule_id': str(ids[idx]), 'stratum': sid,
            'normal_reference_row_idx': int(normal_ref_idx),
            'normal_reference_molecule_id': str(ids[int(normal_ref_idx)]),
            'candidate_y': int(y[idx]), 'normal_reference_y': y_ref,
            'delta_h': float(delta), 'label_grade_proxy': int({-1: 0, 0: 2, 1: 4}[delta]),
            'positive_rescue': int(delta > 0),
            'feat_p0': float(p0_mean[idx]), 'feat_p0_logit': float(logit[j]),
            'feat_p0_percentile': float(p0_percentile[idx]), 'feat_stratum_frac': float(sid / max(N_STRATA - 1, 1)),
            'feat_p0_std': float(p0_std[idx]), 'feat_entropy': float(entropy[j]),
            'feat_nn_queried_active_tanimoto': float(qa_sim[j]),
            'feat_nn_queried_inactive_tanimoto': float(qi_sim[j]),
            'feat_nn_audit_active_tanimoto': float(aa_sim[j]),
            'feat_nn_audit_inactive_tanimoto': float(ai_sim[j]),
            'feat_audit_neighbor_tanimoto2_active_rate': float(neigh_rate[j]),
            'feat_audit_neighbor_effective_count_frac': float(neigh_eff[j]),
            'feat_theta_mean': float(theta_mean[sid]), 'feat_theta_u90': float(theta_u90[sid]),
            'feat_stratum_audit_frac': float(audit_frac[sid]), 'feat_round_frac': float(round_idx / K_ROUNDS),
            'feat_remaining_b1_frac': float(b1_remaining_before_round / max(b1_budget, 1)),
            'feat_remaining_q_frac': float(q_remaining_after_audit / max(q_budget, 1)),
            'mask_no_queried_active': int(qa_mask[j]), 'mask_no_audit_active': int(aa_mask[j]),
            'mask_no_audit_neighbor': int(neigh_mask[j]),
        })
    return rows


def run_assay_seed(assay: str, seed: int, max_candidates_per_stratum: int, lgbm_jobs: int) -> list[dict[str, object]]:
    ids_df, packed, y = load_assay(assay)
    ids = ids_df['molecule_id'].astype(str).to_numpy()
    smiles = ids_df['canonical_smiles'].astype(str).to_numpy()
    n = len(ids)
    q, b1, q0 = screening_budgets(n)
    init_idx = make_initial(y, assay, seed, q0)
    queried = set(map(int, init_idx.tolist()))
    audit_all: list[int] = []
    expensive_scored: set[int] = set()
    q_audit_total = int(math.floor(0.20 * (q - q0)))
    q_exploit_total = int(q - q0 - q_audit_total)
    q_audit_rounds = split_budget(q_audit_total, K_ROUNDS)
    q_exploit_rounds = split_budget(q_exploit_total, K_ROUNDS)
    b1_rounds = split_budget(b1, K_ROUNDS)
    rows: list[dict[str, object]] = []
    for zero_round in range(K_ROUNDS):
        round_idx = zero_round + 1
        train_idx = np.array(sorted(queried), dtype=np.int64)
        x_train = unpack_rows(packed, train_idx)
        x_all = unpack_rows(packed)
        p0_mean, p0_std = train_predict_ensemble_stats(x_train, y[train_idx].astype(int), x_all, seed + 1000 * round_idx, lgbm_jobs)
        p0_pct = percentile_ranks(p0_mean)
        unavailable = np.zeros(n, dtype=bool)
        unavailable[list(queried)] = True
        if expensive_scored:
            unavailable[list(expensive_scored)] = True
        available = np.flatnonzero(~unavailable)
        if len(available) == 0:
            break
        ordered = choose_by_score(p0_mean, available)
        b1_this_round = min(int(b1_rounds[zero_round]), len(ordered))
        normal_b1 = min(max(1, int(math.floor(0.80 * b1_this_round))), len(ordered))
        normal_pool = ordered[:normal_b1]
        discarded_pool = ordered[normal_b1:]
        if len(normal_pool) == 0 or len(discarded_pool) == 0:
            continue
        frame = pd.DataFrame({
            'row_idx': discarded_pool.astype(np.int64),
            'molecule_id': ids[discarded_pool].astype(str),
            'canonical_smiles': smiles[discarded_pool].astype(str),
            'score': p0_mean[discarded_pool].astype(float),
        })
        frame = assign_equal_size_strata(frame, 'score', smiles_col='canonical_smiles', n_strata=N_STRATA)
        stratum_by_idx = {int(r.row_idx): int(r.stratum) for r in frame[['row_idx', 'stratum']].itertuples(index=False)}
        stratum_sizes = {int(k): int(v) for k, v in frame.groupby('stratum', observed=True).size().to_dict().items()}
        audit_k = min(int(q_audit_rounds[zero_round]), len(frame), q - len(queried))
        audit_idx = stratified_sample_indices(frame, audit_k, int(seed + 17 * round_idx), f'e06_audit_{assay}_{round_idx}')
        for idx in audit_idx.tolist():
            queried.add(int(idx)); audit_all.append(int(idx))
        q_remaining_after_audit = max(0, q - len(queried))
        frame_after_audit = frame[~frame['row_idx'].isin(set(map(int, audit_idx.tolist())))]
        candidate_idx = sample_candidates_by_stratum(frame_after_audit, max_candidates_per_stratum, int(seed + 31 * round_idx))
        rows.extend(build_feature_rows(
            assay=assay, seed=int(seed), round_idx=round_idx, ids=ids, y=y, packed=packed,
            p0_mean=p0_mean, p0_std=p0_std, p0_percentile=p0_pct,
            candidate_idx=candidate_idx, normal_ref_idx=int(normal_pool[0]),
            stratum_by_idx=stratum_by_idx, stratum_sizes=stratum_sizes,
            current_audit_idx=audit_idx, audit_idx_all=audit_all, queried_idx=queried,
            q_remaining_after_audit=q_remaining_after_audit,
            b1_remaining_before_round=max(0, b1 - len(expensive_scored)),
            q_budget=q, b1_budget=b1,
        ))
        expensive_scored.update(map(int, normal_pool.tolist()))
        exploit_k = min(int(q_exploit_rounds[zero_round]), len(normal_pool), max(0, q - len(queried)))
        if exploit_k > 0:
            for idx in normal_pool[:exploit_k].tolist():
                queried.add(int(idx))
    return rows


def summarize(df: pd.DataFrame, args: argparse.Namespace, failures: list[dict[str, object]]) -> dict[str, object]:
    by_stratum = df.groupby('stratum').size().to_dict() if len(df) else {}
    positive_by_stratum = df.groupby('stratum')['positive_rescue'].sum().to_dict() if len(df) else {}
    return {
        'protocol': 'counterfactual_proxy_v0',
        'warning': 'Proxy label uses y[x]-y[x0] one-slot delta and no Chemprop p1; use only for E06 mechanism screening.',
        'assays': args.assays, 'seeds': args.seeds, 'rounds': K_ROUNDS,
        'max_candidates_per_stratum': int(args.max_candidates_per_stratum),
        'n_rows': int(len(df)), 'n_groups': int(df['group_id'].nunique()) if len(df) else 0,
        'n_assays': int(df['assay'].nunique()) if len(df) else 0,
        'positive_rescue_rows': int(df['positive_rescue'].sum()) if len(df) else 0,
        'negative_rescue_rows': int((df['delta_h'] < 0).sum()) if len(df) else 0,
        'zero_delta_rows': int((df['delta_h'] == 0).sum()) if len(df) else 0,
        'pairs_by_stratum': {str(k): int(v) for k, v in sorted(by_stratum.items())},
        'positive_by_stratum': {str(k): int(v) for k, v in sorted(positive_by_stratum.items())},
        'feature_columns': FEATURE_COLUMNS, 'failures': failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate E06 AIR counterfactual proxy pair bank.')
    parser.add_argument('--assays', nargs='*', default=ALL_ASSAYS)
    parser.add_argument('--seeds', nargs='*', type=int, default=list(SEEDS))
    parser.add_argument('--max-candidates-per-stratum', type=int, default=32)
    parser.add_argument('--max-workers', type=int, default=16)
    parser.add_argument('--lgbm-jobs', type=int, default=1)
    parser.add_argument('--prefix', default='e06_pair_bank_proxy')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / 'results' / 'e06_proxy'; table_dir = ROOT / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True); table_dir.mkdir(exist_ok=True)
    tasks = [(a, int(s), int(args.max_candidates_per_stratum), int(args.lgbm_jobs)) for a in args.assays for s in args.seeds]
    cpu_total = os.cpu_count() or 64
    workers = max(1, min(int(args.max_workers), max(1, cpu_total - 30), len(tasks) or 1))
    print(json.dumps({'prefix': args.prefix, 'tasks': len(tasks), 'max_workers': workers,
                      'lgbm_jobs_per_worker': int(args.lgbm_jobs), 'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                      'protocol': 'counterfactual_proxy_v0'}, ensure_ascii=False), flush=True)
    rows: list[dict[str, object]] = []; failures: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_assay_seed, *task): task for task in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=args.prefix):
            assay, seed, *_ = futs[fut]
            try:
                batch = fut.result()
            except Exception as exc:
                failure = {'assay': assay, 'seed': int(seed), 'error': repr(exc), 'traceback': ''.join(traceback.format_exception(exc))}
                failures.append(failure); print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True); continue
            rows.extend(batch)
            print(json.dumps({'assay': assay, 'seed': int(seed), 'rows': len(batch), 'total_rows': len(rows)}, ensure_ascii=False), flush=True)
            if rows:
                pd.DataFrame(rows).to_parquet(out_dir / f'{args.prefix}.partial.parquet', index=False)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(['assay', 'seed', 'round', 'group_id', 'stratum', 'molecule_id'], kind='mergesort').reset_index(drop=True)
    df.to_parquet(out_dir / f'{args.prefix}.parquet', index=False)
    summary = summarize(df, args, failures)
    (out_dir / f'{args.prefix}_audit.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (table_dir / f'{args.prefix}_audit.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    if len(df):
        table = df.groupby(['assay', 'stratum'], as_index=False).agg(n_pairs=('delta_h', 'size'), positive_rescue=('positive_rescue', 'sum'), mean_delta_h=('delta_h', 'mean'))
        table.to_csv(table_dir / f'{args.prefix}_summary.csv', index=False)
    else:
        pd.DataFrame().to_csv(table_dir / f'{args.prefix}_summary.csv', index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} E06 proxy pair tasks failed')


if __name__ == '__main__':
    main()
