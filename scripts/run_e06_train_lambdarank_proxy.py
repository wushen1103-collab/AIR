
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from airvs.data.registry import FOLDS
from airvs.utils.determinism import make_rng
from run_e06_pair_bank_proxy import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
FOLD_ORDER = ['F1', 'F2', 'F3', 'F4', 'F5']


def fold_split(fold: str) -> tuple[list[str], list[str], list[str]]:
    idx = FOLD_ORDER.index(fold)
    test_fold = fold
    val_fold = FOLD_ORDER[(idx + 1) % len(FOLD_ORDER)]
    test = list(FOLDS[test_fold])
    val = list(FOLDS[val_fold])
    train: list[str] = []
    for f in FOLD_ORDER:
        if f not in {test_fold, val_fold}:
            train.extend(FOLDS[f])
    return train, val, test


def group_arrays(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    out = df.sort_values(['group_id', 'stratum', 'molecule_id'], kind='mergesort').reset_index(drop=True)
    group = out.groupby('group_id', sort=False).size().to_numpy(dtype=np.int32)
    return out, group


def dcg(relevance: np.ndarray, k: int) -> float:
    rel = np.asarray(relevance, dtype=np.float64)[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2, dtype=np.float64))
    return float(((2.0 ** rel - 1.0) * discounts).sum())


def mean_ndcg_at_k(df: pd.DataFrame, scores: np.ndarray, k: int = 100, rel_col: str = 'positive_rescue') -> float:
    vals: list[float] = []
    tmp = df[['group_id', rel_col]].copy()
    tmp['_score'] = scores
    for _gid, sub in tmp.groupby('group_id', sort=False):
        rel = sub[rel_col].to_numpy(dtype=np.float64)
        ideal = dcg(np.sort(rel)[::-1], k)
        if ideal <= 0:
            continue
        order = np.argsort(-sub['_score'].to_numpy(dtype=np.float64), kind='mergesort')
        vals.append(dcg(rel[order], k) / ideal)
    return float(np.mean(vals)) if vals else 0.0


def random_ndcg_at_k(df: pd.DataFrame, k: int = 100, repeats: int = 20, seed: int = 47001) -> float:
    vals: list[float] = []
    for rep in range(repeats):
        rng = make_rng(seed, 'random_ndcg', rep, len(df))
        vals.append(mean_ndcg_at_k(df, rng.random(len(df)), k=k, rel_col='positive_rescue'))
    return float(np.mean(vals)) if vals else 0.0


def pairwise_accuracy(df: pd.DataFrame, scores: np.ndarray, max_pairs_per_group: int = 2048, seed: int = 47001) -> float:
    tmp = df[['group_id', 'delta_h']].copy()
    tmp['_score'] = scores
    correct = 0.0
    total = 0
    for gid, sub in tmp.groupby('group_id', sort=False):
        y = sub['delta_h'].to_numpy(dtype=np.float64)
        s = sub['_score'].to_numpy(dtype=np.float64)
        pos = np.flatnonzero(y > 0)
        neg = np.flatnonzero(y <= 0)
        if len(pos) == 0 or len(neg) == 0:
            continue
        all_pairs = len(pos) * len(neg)
        if all_pairs <= max_pairs_per_group:
            pp, nn = np.meshgrid(pos, neg, indexing='ij')
            pidx = pp.ravel(); nidx = nn.ravel()
        else:
            rng = make_rng(seed, 'pairwise', gid, len(pos), len(neg))
            pidx = rng.choice(pos, size=max_pairs_per_group, replace=True)
            nidx = rng.choice(neg, size=max_pairs_per_group, replace=True)
        correct += float((s[pidx] > s[nidx]).sum()) + 0.5 * float((s[pidx] == s[nidx]).sum())
        total += len(pidx)
    return float(correct / total) if total else 0.0


def train_one_fold(df: pd.DataFrame, fold: str, args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    train_assays, val_assays, test_assays = fold_split(fold)
    train_df = df[df['assay'].isin(train_assays)].copy()
    val_df = df[df['assay'].isin(val_assays)].copy()
    if args.train_informative_groups_only:
        if args.label_mode == 'positive':
            keep = train_df.groupby('group_id')['positive_rescue'].transform('sum') > 0
        else:
            keep = train_df.groupby('group_id')['label_grade_proxy'].transform('nunique') > 1
        train_df = train_df[keep].copy()
    train_df, train_group = group_arrays(train_df)
    val_df, val_group = group_arrays(val_df)
    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError(f'empty train/val for {fold}')
    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    X_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if args.label_mode == 'positive':
        y_train = train_df['positive_rescue'].to_numpy(dtype=np.int32)
        y_val = val_df['positive_rescue'].to_numpy(dtype=np.int32)
        label_gain = [0, 1]
    else:
        y_train = train_df['label_grade_proxy'].to_numpy(dtype=np.int32)
        y_val = val_df['label_grade_proxy'].to_numpy(dtype=np.int32)
        label_gain = [0, 1, 2, 3, 4]
    model = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', n_estimators=int(args.n_estimators),
        learning_rate=0.03, num_leaves=31, min_child_samples=50,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
        label_gain=label_gain, random_state=47000 + FOLD_ORDER.index(fold),
        n_jobs=int(args.lgbm_jobs), verbosity=-1,
    )
    evals_result: dict[str, object] = {}
    model.fit(
        X_train, y_train, group=train_group,
        eval_set=[(X_val, y_val)], eval_group=[val_group], eval_at=[100],
        callbacks=[lgb.early_stopping(int(args.early_stopping_rounds), verbose=False), lgb.record_evaluation(evals_result)],
    )
    model_dir = ROOT / 'models' / 'e06_proxy'; model_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = str(args.prefix).replace('/', '_').replace(' ', '_')
    model_path = model_dir / f'{safe_prefix}_{fold}_proxy.txt'
    model.booster_.save_model(model_path)
    val_score = model.predict(X_val, num_iteration=model.best_iteration_)
    ndcg100 = mean_ndcg_at_k(val_df, val_score, k=100, rel_col='positive_rescue')
    rnd = random_ndcg_at_k(val_df, k=100, repeats=int(args.random_repeats), seed=47000 + FOLD_ORDER.index(fold))
    pacc = pairwise_accuracy(val_df, val_score, seed=48000 + FOLD_ORDER.index(fold))
    shuffled_ndcg = None
    if args.shuffle_sanity:
        rng = make_rng(49000, 'shuffle_sanity', fold, len(y_train))
        y_shuffle = y_train.copy(); rng.shuffle(y_shuffle)
        shuf = lgb.LGBMRanker(
            objective='lambdarank', metric='ndcg', n_estimators=min(150, int(args.n_estimators)),
            learning_rate=0.03, num_leaves=31, min_child_samples=50,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
            label_gain=label_gain, random_state=50000 + FOLD_ORDER.index(fold),
            n_jobs=int(args.lgbm_jobs), verbosity=-1,
        )
        shuf.fit(X_train, y_shuffle, group=train_group, eval_set=[(X_val, y_val)], eval_group=[val_group], eval_at=[100], callbacks=[lgb.log_evaluation(0)])
        shuffled_ndcg = mean_ndcg_at_k(val_df, shuf.predict(X_val), k=100, rel_col='positive_rescue')
    train_counts = train_df.groupby('stratum').size().to_dict()
    val_counts = val_df.groupby('stratum').size().to_dict()
    row = {
        'protocol': 'counterfactual_proxy_v0', 'label_mode': args.label_mode, 'fold': fold,
        'train_assays': ','.join(train_assays), 'validation_assays': ','.join(val_assays),
        'heldout_test_assays_not_used': ','.join(test_assays),
        'train_informative_groups_only': bool(args.train_informative_groups_only),
        'train_rows': int(len(train_df)), 'validation_rows': int(len(val_df)),
        'train_groups': int(len(train_group)), 'validation_groups': int(len(val_group)),
        'train_positive_rescue': int(train_df['positive_rescue'].sum()),
        'validation_positive_rescue': int(val_df['positive_rescue'].sum()),
        'min_train_pairs_per_stratum': int(min(train_counts.values())) if train_counts else 0,
        'train_pairs_by_stratum': json.dumps({str(k): int(v) for k, v in sorted(train_counts.items())}, ensure_ascii=False),
        'validation_pairs_by_stratum': json.dumps({str(k): int(v) for k, v in sorted(val_counts.items())}, ensure_ascii=False),
        'best_iteration': int(model.best_iteration_ or int(args.n_estimators)),
        'validation_ndcg_at_100': float(ndcg100), 'random_ndcg_at_100': float(rnd),
        'ndcg_lift_over_random': float(ndcg100 - rnd), 'pairwise_accuracy': float(pacc),
        'shuffle_sanity_ndcg_at_100': None if shuffled_ndcg is None else float(shuffled_ndcg),
        'passes_proxy_gate': bool((ndcg100 - rnd) > 0.05 and pacc > 0.55),
        'model_path': str(model_path.relative_to(ROOT)),
        'warning': 'Proxy E06 only: no Chemprop p1, no two-batch continuation; do not use as primary performance claim.',
    }
    curve_rows: list[dict[str, object]] = []
    valid_result = evals_result.get('valid_0', {}) if isinstance(evals_result, dict) else {}
    for metric, values in valid_result.items():
        for i, value in enumerate(values, start=1):
            curve_rows.append({'fold': fold, 'iteration': i, 'metric': metric, 'value': float(value)})
    return row, curve_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train E06 LambdaRank proxy AIR value models.')
    parser.add_argument('--pair-bank', default='results/e06_proxy/e06_pair_bank_proxy.parquet')
    parser.add_argument('--label-mode', choices=['delta_grade', 'positive'], default='delta_grade')
    parser.add_argument('--folds', nargs='*', default=FOLD_ORDER)
    parser.add_argument('--n-estimators', type=int, default=500)
    parser.add_argument('--early-stopping-rounds', type=int, default=50)
    parser.add_argument('--lgbm-jobs', type=int, default=32)
    parser.add_argument('--random-repeats', type=int, default=20)
    parser.add_argument('--train-informative-groups-only', action='store_true')
    parser.add_argument('--no-shuffle-sanity', dest='shuffle_sanity', action='store_false')
    parser.set_defaults(shuffle_sanity=True)
    parser.add_argument('--prefix', default='e06_proxy')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(ROOT / args.pair_bank)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f'missing feature columns: {missing}')
    table_dir = ROOT / 'tables'; table_dir.mkdir(exist_ok=True)
    metrics: list[dict[str, object]] = []; curves: list[dict[str, object]] = []
    for fold in args.folds:
        row, curve_rows = train_one_fold(df, fold, args)
        metrics.append(row); curves.extend(curve_rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    mdf = pd.DataFrame(metrics); cdf = pd.DataFrame(curves)
    mdf.to_csv(table_dir / f'{args.prefix}_fold_metrics.csv', index=False)
    cdf.to_csv(table_dir / f'{args.prefix}_training_curves.csv', index=False)
    pass_count = int(mdf['passes_proxy_gate'].sum()) if len(mdf) else 0
    gate = {
        'protocol': 'counterfactual_proxy_v0', 'label_mode': args.label_mode, 'pair_bank': args.pair_bank, 'folds': args.folds,
        'passing_folds': pass_count, 'fold_total': int(len(mdf)),
        'proxy_gate_pass_if_at_least_4_of_5': bool(len(mdf) == 5 and pass_count >= 4),
        'formal_e06_gate_status': 'not_applicable_proxy_missing_chemprop_p1_and_two_batch_continuations',
        'train_informative_groups_only': bool(args.train_informative_groups_only),
        'warning': 'Use this gate only to decide whether mechanism signal merits formal Chemprop p1 experiments.',
    }
    (table_dir / f'{args.prefix}_gate_status.json').write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
