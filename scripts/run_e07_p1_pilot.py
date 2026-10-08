
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.warning')

from airvs.audit.sampling import assign_equal_size_strata
from airvs.data.registry import SEEDS
from airvs.metrics.core import hits, nef, recall
from airvs.simulator.evaluator import BudgetLedger, SpyEvaluator
from airvs.utils.determinism import make_rng
from run_e03_headroom import load_assay, unpack_rows
from run_e05_baseline_smoke import make_initial, screening_budgets, train_predict_ensemble_stats
from run_e06_pair_bank_proxy import percentile_ranks, split_budget, stratified_sample_indices

ROOT = Path(__file__).resolve().parents[1]
AIR311 = ROOT / 'envs' / 'air311'
AIR311_CHEMPROP = AIR311 / 'bin' / 'chemprop'
AIR311_PYTHON = AIR311 / 'bin' / 'python'
METHODS = [
    'B1_p1_fixed',
    'B10_p1_audit_rule',
    'B10_p1_audit_safe',
    'B10_p1_reentry_safe',
    'AIR_p1_rescue_heuristic',
    'AIR_p1_adaptive_quota',
    'AIR_p1_hit_ucb',
    'AIR_p1_hit_ucb_mix',
    'AIR_p1_risk_adaptive_mix',
    'AIR_p1_safe_cluster_adaptive_mix',
    'AIR_p1_similarity_portfolio_gate',
    'AIR_p1_probe_hndiv_gate',
    'AIR_p1_soft_neighbor_mix',
    'AIR_p1_neighbor_probe_gate',
    'AIR_p1_hit_neighbor_diverse_mix',
    # External-route common-protocol baselines.  These are intentionally named
    # X* rather than AIR* because they are not proposed-method variants: they
    # instantiate the acquisition families reviewers expect to see under the
    # same Q/B1/Chemprop protocol before investing in heavyweight official
    # code reproduction (MolPAL / ChemScreener / ACActive / ALBF / GLARE).
    'X0_p1_random',
    'X1_p1_ucb',
    'X2_p1_diverse',
    'X3_p1_balanced_rank',
    'X4_p1_hit_neighbor',
]
NO_AUDIT_METHODS = {'B1_p1_fixed', 'X0_p1_random', 'X1_p1_ucb', 'X2_p1_diverse', 'X3_p1_balanced_rank', 'X4_p1_hit_neighbor', 'AIR_p1_soft_neighbor_mix', 'AIR_p1_hit_neighbor_diverse_mix'}
EXTERNAL_ROUTE_METHODS = {'X0_p1_random', 'X1_p1_ucb', 'X2_p1_diverse', 'X3_p1_balanced_rank', 'X4_p1_hit_neighbor'}
N_STRATA = 5


def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def stage_passed(path: Path) -> bool:
    data = read_json(path)
    return bool(data and data.get('status') == 'pass')


def archive_nonpass_manifest(path: Path) -> Path | None:
    data = read_json(path)
    if not data or data.get('status') == 'pass':
        return None
    stamp = time.strftime('%Y%m%dT%H%M%S')
    archive = path.with_name(f'{path.stem}.retry_{stamp}{path.suffix}')
    n = 1
    while archive.exists():
        archive = path.with_name(f'{path.stem}.retry_{stamp}_{n}{path.suffix}')
        n += 1
    archive.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    return archive


def run_text(cmd: list[str], timeout_sec: int | None = None, env: dict[str, str] | None = None) -> tuple[int, str, bool, float]:
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout_sec)
        return int(proc.returncode), out or '', timed_out, time.time() - started
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            out, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, _ = proc.communicate()
        return 124, (out or '') + f'\n[TIMEOUT after {timeout_sec} sec]\n', timed_out, time.time() - started


def nvidia_snapshot() -> list[dict[str, object]]:
    cmd = ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits']
    code, out, _to, _elapsed = run_text(cmd, timeout_sec=20)
    rows: list[dict[str, object]] = []
    if code != 0:
        return [{'error': out.strip(), 'returncode': code}]
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            rows.append({'index': int(parts[0]), 'name': parts[1], 'memory_used_mib': int(parts[2]), 'memory_total_mib': int(parts[3]), 'utilization_gpu_pct': int(parts[4])})
    return rows


def compute_apps_snapshot() -> list[dict[str, object]]:
    cmd = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits']
    code, out, _to, _elapsed = run_text(cmd, timeout_sec=20)
    rows: list[dict[str, object]] = []
    if code != 0:
        return [{'error': out.strip(), 'returncode': code}]
    for line in out.splitlines():
        if not line.strip() or 'No running processes' in line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 4:
            rows.append({'gpu_uuid': parts[0], 'pid': parts[1], 'process_name': parts[2], 'used_memory_mib': parts[3]})
    return rows


def git_commit() -> str:
    code, out, _to, _elapsed = run_text(['git', 'rev-parse', '--short', 'HEAD'], timeout_sec=10)
    return out.strip() if code == 0 else 'unknown'


def validate_gpu_id(gpu_physical_id: int | None, accelerator: str) -> list[int]:
    snapshot = nvidia_snapshot()
    ids = sorted(int(row['index']) for row in snapshot if 'index' in row)
    if accelerator == 'gpu':
        if gpu_physical_id is None:
            raise ValueError('gpu accelerator requires --gpu-physical-id')
        if int(gpu_physical_id) not in ids:
            raise ValueError(f'gpu id {gpu_physical_id} is not in physical GPU list {ids}')
    return ids


def guarded_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    env['OMP_NUM_THREADS'] = str(args.cpu_threads)
    env['MKL_NUM_THREADS'] = str(args.cpu_threads)
    env['OPENBLAS_NUM_THREADS'] = str(args.cpu_threads)
    env['NUMEXPR_NUM_THREADS'] = str(args.cpu_threads)
    env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.accelerator == 'gpu':
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_physical_id)
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def torch_witness(args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    code_snippet = (
        "import json, os, torch; "
        "payload={'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES',''), "
        "'torch_cuda_available': bool(torch.cuda.is_available()), "
        "'torch_cuda_device_count': int(torch.cuda.device_count())}; "
        "payload['device_names']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; "
        "print(json.dumps(payload, ensure_ascii=False))"
    )
    env = guarded_env(args)
    code, out, timed_out, elapsed = run_text([str(AIR311_PYTHON), '-c', code_snippet], timeout_sec=60, env=env)
    report: dict[str, object] = {'returncode': code, 'timed_out': timed_out, 'elapsed_sec': elapsed, 'stdout': out.strip()}
    try:
        payload = json.loads(out.strip().splitlines()[-1])
        report.update(payload)
    except Exception:
        report['parse_error'] = True
    if args.accelerator == 'gpu':
        report['status'] = 'pass' if code == 0 and report.get('torch_cuda_device_count') == 1 else 'fail'
    else:
        report['status'] = 'pass' if code == 0 and report.get('torch_cuda_device_count') == 0 else 'fail'
    write_json(out_dir / 'torch_witness.json', report)
    return report


def resource_audit(args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    validate_gpu_id(args.gpu_physical_id, args.accelerator)
    cpu_total = os.cpu_count() or 1
    audit = {
        'status': 'pending', 'created_at': now_iso(), 'git_commit': git_commit(),
        'accelerator': args.accelerator, 'gpu_physical_id': args.gpu_physical_id,
        'cpu_total': cpu_total, 'cpu_threads_per_task': args.cpu_threads,
        'cpu_reserved_minimum': 30, 'cpu_threads_ok': bool(args.cpu_threads <= max(1, cpu_total - 30)),
        'timeout_sec': args.timeout_sec, 'nvidia_before': nvidia_snapshot(), 'compute_apps_before': compute_apps_snapshot(),
        'air311_chemprop_exists': AIR311_CHEMPROP.exists(), 'air311_python_exists': AIR311_PYTHON.exists(),
    }
    audit['torch_witness'] = torch_witness(args, out_dir)
    audit['status'] = 'pass' if audit['cpu_threads_ok'] and audit['air311_chemprop_exists'] and audit['air311_python_exists'] and audit['torch_witness'].get('status') == 'pass' else 'fail'
    audit['nvidia_after_witness'] = nvidia_snapshot()
    audit['compute_apps_after_witness'] = compute_apps_snapshot()
    write_json(out_dir / 'resource_audit.json', audit)
    if audit['status'] != 'pass':
        raise RuntimeError(f'resource audit failed; see {out_dir / "resource_audit.json"}')
    return audit


def self_test_guards(args: argparse.Namespace) -> dict[str, object]:
    out_dir = ROOT / 'results' / args.prefix / 'guard_selftest'
    if out_dir.exists() and not args.resume:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {'protocol': 'e07_guard_selftest_v0', 'created_at': now_iso(), 'prefix': args.prefix, 'checks': {}}
    report['checks']['resource_audit'] = resource_audit(args, out_dir)
    invalid_rejected = False
    invalid_error = ''
    try:
        validate_gpu_id(999999, 'gpu')
    except Exception as exc:
        invalid_rejected = True
        invalid_error = repr(exc)
    report['checks']['invalid_gpu_rejected_before_launch'] = {'status': 'pass' if invalid_rejected else 'fail', 'error': invalid_error}
    code, out, timed_out, elapsed = run_text([sys.executable, '-c', 'import time; time.sleep(60)'], timeout_sec=1, env=os.environ.copy())
    report['checks']['timeout_kills_process_group'] = {'status': 'pass' if timed_out and code == 124 else 'fail', 'returncode': code, 'timed_out': timed_out, 'elapsed_sec': elapsed, 'stdout_tail': out[-500:]}
    resume_dir = out_dir / 'resume_probe'
    resume_dir.mkdir(exist_ok=True)
    marker = resume_dir / 'stage_done.json'
    first_action = 'skipped' if stage_passed(marker) else 'created'
    if first_action == 'created':
        write_json(marker, {'status': 'pass', 'created_at': now_iso(), 'stage': 'resume_probe'})
    second_action = 'skipped' if stage_passed(marker) else 'would_run'
    report['checks']['resume_marker_probe'] = {'status': 'pass' if first_action in {'created', 'skipped'} and second_action == 'skipped' else 'fail', 'first_action': first_action, 'second_action': second_action, 'marker': str(marker.relative_to(ROOT))}
    report['status'] = 'pass' if all(v.get('status') == 'pass' for v in report['checks'].values()) else 'fail'
    table_dir = ROOT / 'tables'
    table_dir.mkdir(exist_ok=True)
    write_json(out_dir / 'guard_selftest_report.json', report)
    write_json(table_dir / f'{args.prefix}_guard_selftest_report.json', report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report['status'] != 'pass':
        raise SystemExit(2)
    return report


def safe_splits(labels: np.ndarray, seed: int) -> list[str]:
    labels = np.asarray(labels, dtype=int)
    splits = np.array(['train'] * len(labels), dtype=object)
    rng = make_rng(seed, 'e07_safe_splits', len(labels), int(labels.sum()))
    for label in [0, 1]:
        idx = np.flatnonzero(labels == label).astype(np.int64)
        if len(idx) == 0:
            continue
        idx = idx.copy(); rng.shuffle(idx)
        if len(idx) >= 10:
            n_val = max(1, int(round(0.10 * len(idx))))
            n_test = max(1, int(round(0.10 * len(idx))))
            splits[idx[:n_val]] = 'val'; splits[idx[n_val:n_val+n_test]] = 'test'
        elif len(idx) >= 3:
            splits[idx[0]] = 'val'; splits[idx[1]] = 'test'
        elif len(idx) == 2:
            splits[idx[0]] = 'val'
        else:
            splits[idx[0]] = 'train'
    if not np.any(splits == 'val') and len(labels) > 1:
        candidates = np.flatnonzero(splits == 'train'); splits[candidates[-1]] = 'val'
    if not np.any(splits == 'test') and len(labels) > 2:
        candidates = np.flatnonzero(splits == 'train')
        if len(candidates): splits[candidates[-1]] = 'test'
    return [str(x) for x in splits.tolist()]


def choose_by_score(scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) == 0:
        return candidates
    return candidates[np.argsort(-scores[candidates], kind='mergesort')]


def query_indices(indices: Iterable[int], ids: np.ndarray, evaluator: SpyEvaluator, ledger: BudgetLedger, discovered: dict[str, int]) -> dict[str, int]:
    selected = [str(ids[int(i)]) for i in indices]
    if not selected:
        return {}
    labels = evaluator.query_labels(selected, ledger)
    discovered.update(labels)
    return labels


def spend_expensive(indices: Iterable[int], ids: np.ndarray, ledger: BudgetLedger) -> None:
    selected = [str(ids[int(i)]) for i in indices]
    if selected:
        ledger.spend_b1(selected)


def chemprop_valid_smiles_mask(values: Iterable[object]) -> np.ndarray:
    """Return which SMILES Chemprop/RDKit can featurize without producing None molecules."""
    keep: list[bool] = []
    for value in values:
        try:
            keep.append(Chem.MolFromSmiles(str(value)) is not None)
        except Exception:
            keep.append(False)
    return np.asarray(keep, dtype=bool)


def prepare_chemprop_csvs(out_dir: Path, ids: np.ndarray, smiles: np.ndarray, y: np.ndarray, train_idx: np.ndarray, predict_idx: np.ndarray, seed: int) -> dict[str, object]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    predict_idx = np.asarray(predict_idx, dtype=np.int64)
    train_valid = chemprop_valid_smiles_mask(smiles[train_idx].astype(str))
    predict_valid = chemprop_valid_smiles_mask(smiles[predict_idx].astype(str))
    invalid_train_idx = train_idx[~train_valid].astype(int)
    invalid_predict_idx = predict_idx[~predict_valid].astype(int)
    valid_train_idx = train_idx[train_valid].astype(np.int64)
    valid_predict_idx = predict_idx[predict_valid].astype(np.int64)
    if len(valid_train_idx) == 0:
        raise ValueError('all Chemprop training SMILES are invalid after RDKit filtering')
    labels = y[valid_train_idx].astype(int)
    train_df = pd.DataFrame({'smiles': smiles[valid_train_idx].astype(str), 'y': labels, 'split': safe_splits(labels, seed), 'molecule_id': ids[valid_train_idx].astype(str), 'row_idx': valid_train_idx.astype(int)})
    predict_df = pd.DataFrame({'smiles': smiles[valid_predict_idx].astype(str), 'molecule_id': ids[valid_predict_idx].astype(str), 'row_idx': valid_predict_idx.astype(int)})
    train_csv = out_dir / 'train.csv'; predict_csv = out_dir / 'predict.csv'
    train_df.to_csv(train_csv, index=False); predict_df.to_csv(predict_csv, index=False)
    train_split = train_df.loc[train_df['split'] == 'train', 'y'].astype(int)
    return {
        'train_csv': str(train_csv.relative_to(ROOT)),
        'predict_csv': str(predict_csv.relative_to(ROOT)),
        'n_train_rows': int(len(train_df)),
        'n_predict_rows': int(len(predict_df)),
        'n_train_rows_raw': int(len(train_idx)),
        'n_predict_rows_raw': int(len(predict_idx)),
        'invalid_train_rows': int(len(invalid_train_idx)),
        'invalid_predict_rows': int(len(invalid_predict_idx)),
        'invalid_train_row_idx': [int(x) for x in invalid_train_idx.tolist()],
        'invalid_predict_row_idx': [int(x) for x in invalid_predict_idx.tolist()],
        'train_positive': int(train_df['y'].sum()),
        'train_negative': int((train_df['y'] == 0).sum()),
        'train_split_positive': int(train_split.sum()),
        'train_split_negative': int((train_split == 0).sum()),
        'split_counts': {str(k): int(v) for k, v in train_df['split'].value_counts().sort_index().items()},
    }


def chemprop_class_balance_is_safe(train_csv: Path, batch_size: int) -> tuple[bool, dict[str, int]]:
    """Return whether Chemprop class-balanced sampling can form at least one full batch.

    Chemprop/lightning can produce zero train batches with --class-balance when the
    minority class in the train split is too small relative to batch_size. In that
    regime disabling class balancing is the stable and fair fallback because both
    baseline and rescue methods use the same rule.
    """
    df = pd.read_csv(train_csv, usecols=['split', 'y'])
    train_y = df.loc[df['split'] == 'train', 'y'].astype(int)
    pos = int(train_y.sum())
    neg = int((train_y == 0).sum())
    minority = min(pos, neg)
    stats = {'train_split_positive': pos, 'train_split_negative': neg, 'minority_count': int(minority), 'batch_size': int(batch_size)}
    return bool(pos > 0 and neg > 0 and 2 * minority >= int(batch_size)), stats


def run_chemprop_round(args: argparse.Namespace, round_dir: Path, ids: np.ndarray, smiles: np.ndarray, y: np.ndarray, train_idx: np.ndarray, predict_idx: np.ndarray, seed: int) -> dict[int, float]:
    round_dir.mkdir(parents=True, exist_ok=True)
    data_manifest = round_dir / 'data_manifest.json'; train_manifest = round_dir / 'train_manifest.json'; pred_manifest = round_dir / 'predict_manifest.json'
    model_dir = round_dir / 'model'; pred_out = round_dir / 'preds.csv'
    data = read_json(data_manifest)
    data_is_current = bool(data and data.get('status') == 'pass' and 'invalid_predict_row_idx' in data)
    if args.resume and stage_passed(pred_manifest) and pred_out.exists() and data_is_current:
        preds = pd.read_csv(pred_out); score_col = [c for c in preds.columns if c not in {'smiles', 'molecule_id', 'row_idx'}][0]
        score_map = {int(r.row_idx): float(getattr(r, score_col)) for r in preds.itertuples(index=False)}
        score_map.update({int(i): -1.0e9 for i in data.get('invalid_predict_row_idx', [])})
        return score_map
    regenerated_data = False
    if not data_is_current:
        archive_nonpass_manifest(data_manifest)
        data = prepare_chemprop_csvs(round_dir, ids, smiles, y, train_idx, predict_idx, seed); data.update({'status': 'pass', 'created_at': now_iso()}); write_json(data_manifest, data)
        regenerated_data = True
    data = read_json(data_manifest) or data or {}
    invalid_predict_scores = {int(i): -1.0e9 for i in data.get('invalid_predict_row_idx', [])}
    if int(data.get('n_predict_rows', len(predict_idx))) == 0:
        manifest = {'status': 'pass', 'returncode': 0, 'timed_out': False, 'elapsed_sec': 0.0, 'prediction_rows': 0, 'prediction_nan': 0, 'prediction_output': str(pred_out.relative_to(ROOT)), 'invalid_predict_rows': int(data.get('invalid_predict_rows', 0)), 'skipped_predict_all_invalid': True, 'created_at': now_iso()}
        write_json(pred_manifest, manifest)
        return invalid_predict_scores
    force_retrain = bool(regenerated_data and int(data.get('invalid_train_rows', 0)) > 0)
    if not (args.resume and stage_passed(train_manifest) and model_dir.exists() and not force_retrain):
        archive_nonpass_manifest(train_manifest)
        if model_dir.exists(): shutil.rmtree(model_dir)
        env = guarded_env(args)
        env['PYTHONHASHSEED'] = str(seed)
        class_balance, class_balance_stats = chemprop_class_balance_is_safe(round_dir / 'train.csv', args.batch_size)
        train_cmd = [str(AIR311_CHEMPROP), 'train', '-i', str(round_dir / 'train.csv'), '-o', str(model_dir), '-s', 'smiles', '--target-columns', 'y', '--ignore-columns', 'molecule_id', 'row_idx', 'split', '--splits-column', 'split', '-t', 'classification', '-l', 'bce', '--metrics', 'accuracy', '--tracking-metric', 'accuracy', '--ensemble-size', '1', '--message-hidden-dim', str(args.message_hidden_dim), '--depth', str(args.depth), '--dropout', str(args.dropout), '--ffn-num-layers', '2', '--warmup-epochs', '1', '--init-lr', '1e-4', '--max-lr', '1e-3', '--final-lr', '1e-4', '--epochs', str(args.epochs), '--patience', '1', '--batch-size', str(args.batch_size), '--num-workers', '0', '--data-seed', str(seed), '--pytorch-seed', str(seed), '--accelerator', args.accelerator]
        if class_balance:
            train_cmd.append('--class-balance')
        if args.accelerator == 'gpu': train_cmd.extend(['--devices', '1'])
        code, log, timed_out, elapsed = run_text(train_cmd, timeout_sec=args.timeout_sec, env=env)
        (round_dir / 'train_stdout.log').write_text(log, encoding='utf-8')
        manifest = {'status': 'pass' if code == 0 and not timed_out else 'fail', 'returncode': int(code), 'timed_out': bool(timed_out), 'elapsed_sec': float(elapsed), 'command': train_cmd, 'class_balance_used': bool(class_balance), 'class_balance_stats': class_balance_stats, 'created_at': now_iso()}
        write_json(train_manifest, manifest)
        if manifest['status'] != 'pass': raise RuntimeError(f'Chemprop train failed or timed out; see {train_manifest}')
    if not (args.resume and stage_passed(pred_manifest) and pred_out.exists()):
        archive_nonpass_manifest(pred_manifest)
        env = guarded_env(args)
        env['PYTHONHASHSEED'] = str(seed)
        pred_cmd = [str(AIR311_CHEMPROP), 'predict', '-i', str(round_dir / 'predict.csv'), '-o', str(pred_out), '-s', 'smiles', '--model-paths', str(model_dir), '--batch-size', str(args.batch_size), '--num-workers', '0', '--accelerator', args.accelerator]
        if args.accelerator == 'gpu': pred_cmd.extend(['--devices', '1'])
        code, log, timed_out, elapsed = run_text(pred_cmd, timeout_sec=args.timeout_sec, env=env)
        (round_dir / 'predict_stdout.log').write_text(log, encoding='utf-8')
        preds = pd.read_csv(pred_out) if pred_out.exists() else pd.DataFrame(); score_cols = [c for c in preds.columns if c not in {'smiles', 'molecule_id', 'row_idx'}]
        nan_count = int(pd.to_numeric(preds[score_cols[0]], errors='coerce').isna().sum()) if score_cols else -1
        expected_pred_rows = int((read_json(data_manifest) or {}).get('n_predict_rows', len(predict_idx)))
        invalid_predict_rows = int((read_json(data_manifest) or {}).get('invalid_predict_rows', 0))
        manifest = {'status': 'pass' if code == 0 and not timed_out and len(preds) == expected_pred_rows and nan_count == 0 else 'fail', 'returncode': int(code), 'timed_out': bool(timed_out), 'elapsed_sec': float(elapsed), 'prediction_rows': int(len(preds)), 'prediction_nan': nan_count, 'prediction_output': str(pred_out.relative_to(ROOT)), 'expected_prediction_rows': int(expected_pred_rows), 'invalid_predict_rows': int(invalid_predict_rows), 'command': pred_cmd, 'created_at': now_iso()}
        write_json(pred_manifest, manifest)
        if manifest['status'] != 'pass': raise RuntimeError(f'Chemprop predict failed or timed out; see {pred_manifest}')
    data = read_json(data_manifest) or {}
    preds = pd.read_csv(pred_out); score_col = [c for c in preds.columns if c not in {'smiles', 'molecule_id', 'row_idx'}][0]
    score_map = {int(r.row_idx): float(getattr(r, score_col)) for r in preds.itertuples(index=False)}
    score_map.update({int(i): -1.0e9 for i in data.get('invalid_predict_row_idx', [])})
    return score_map


def audit_frame(discarded_pool: np.ndarray, ids: np.ndarray, smiles: np.ndarray, p0: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({'row_idx': discarded_pool.astype(np.int64), 'molecule_id': ids[discarded_pool].astype(str), 'canonical_smiles': smiles[discarded_pool].astype(str), 'score': p0[discarded_pool].astype(float)})
    return assign_equal_size_strata(frame, 'score', smiles_col='canonical_smiles', n_strata=N_STRATA)


def max_tanimoto_to_refs(x_all: np.ndarray, candidate_idx: np.ndarray, ref_idx: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    ref_idx = np.asarray(ref_idx, dtype=np.int64)
    n = int(len(candidate_idx))
    if n == 0 or len(ref_idx) == 0:
        return np.zeros(n, dtype=np.float32)
    ref_bits = x_all[ref_idx].astype(np.float32, copy=False)
    ref_sum = ref_bits.sum(axis=1, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        cand_bits = x_all[candidate_idx[start:end]].astype(np.float32, copy=False)
        inter = cand_bits @ ref_bits.T
        denom = cand_bits.sum(axis=1, dtype=np.float32)[:, None] + ref_sum[None, :] - inter
        sims = np.divide(inter, denom, out=np.zeros_like(inter, dtype=np.float32), where=denom > 0)
        out[start:end] = sims.max(axis=1).astype(np.float32)
    return out


def percentile_ranks_subset(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.zeros(0, dtype=np.float32)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks / max(len(values) - 1, 1)


def recent_inactive_refs(queried_idx: set[int], y: np.ndarray, max_refs: int = 64) -> np.ndarray:
    refs = [int(i) for i in sorted(queried_idx) if int(y[int(i)]) == 0]
    if len(refs) > int(max_refs):
        refs = refs[-int(max_refs):]
    return np.asarray(refs, dtype=np.int64)


def queried_active_refs(queried_idx: set[int], y: np.ndarray) -> np.ndarray:
    return np.asarray([int(i) for i in sorted(queried_idx) if int(y[int(i)]) == 1], dtype=np.int64)


def hit_ucb_order_candidates(p0_mean: np.ndarray, p0_pct: np.ndarray, p0_std_pct: np.ndarray, x_all: np.ndarray,
                             candidate_idx: np.ndarray, active_idx: np.ndarray, inactive_idx: np.ndarray) -> np.ndarray:
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if len(candidate_idx) == 0:
        return candidate_idx
    if len(active_idx) == 0:
        score = 0.75 * p0_pct[candidate_idx].astype(np.float64) + 0.25 * p0_std_pct[candidate_idx].astype(np.float64)
    else:
        active_sim = max_tanimoto_to_refs(x_all, candidate_idx, active_idx).astype(np.float64)
        inactive_sim = max_tanimoto_to_refs(x_all, candidate_idx, inactive_idx).astype(np.float64) if len(inactive_idx) else np.zeros(len(candidate_idx), dtype=np.float64)
        cliff_edge = np.maximum(0.0, inactive_sim - active_sim)
        score = (
            0.58 * p0_pct[candidate_idx].astype(np.float64)
            + 0.25 * active_sim
            + 0.12 * p0_std_pct[candidate_idx].astype(np.float64)
            + 0.05 * cliff_edge
        )
    order = np.argsort(-score, kind='mergesort')
    return candidate_idx[order].astype(np.int64)


def seeded_random_order(candidate_idx: np.ndarray, seed: int, assay: str, method: str, round_idx: int) -> np.ndarray:
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64).copy()
    if len(candidate_idx) <= 1:
        return candidate_idx
    rng = make_rng(int(seed), 'e07_external_random_order', str(assay), str(method), int(round_idx), int(len(candidate_idx)))
    rng.shuffle(candidate_idx)
    return candidate_idx.astype(np.int64)


def ucb_order_candidates(p0_pct: np.ndarray, p0_std_pct: np.ndarray, candidate_idx: np.ndarray) -> np.ndarray:
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if len(candidate_idx) == 0:
        return candidate_idx
    score = 0.70 * p0_pct[candidate_idx].astype(np.float64) + 0.30 * p0_std_pct[candidate_idx].astype(np.float64)
    return candidate_idx[np.argsort(-score, kind='mergesort')].astype(np.int64)


def diverse_order_candidates(p0_pct: np.ndarray, p0_std_pct: np.ndarray, x_all: np.ndarray, candidate_idx: np.ndarray,
                             active_idx: np.ndarray, request_k: int) -> np.ndarray:
    """Greedy diversity/coreset-style order on a high-score head.

    This approximates diversity-aware active-learning baselines under the same
    simulator: keep the p0/UCB head competitive, then penalize redundancy to
    already known hits and to the in-batch selections.  It is deterministic and
    label-safe because it only uses queried active references.
    """
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if len(candidate_idx) == 0:
        return candidate_idx
    base = 0.75 * p0_pct[candidate_idx].astype(np.float64) + 0.25 * p0_std_pct[candidate_idx].astype(np.float64)
    head_k = min(len(candidate_idx), max(512, int(request_k) * 80))
    head_pos = np.argsort(-base, kind='mergesort')[:head_k]
    head = candidate_idx[head_pos]
    head_base = base[head_pos]
    known_sim = max_tanimoto_to_refs(x_all, head, active_idx).astype(np.float64) if len(active_idx) else np.zeros(len(head), dtype=np.float64)
    remaining = list(range(len(head)))
    selected_pos: list[int] = []
    selected_idx: list[int] = []
    selected_target = min(int(request_k), len(head))
    while remaining and len(selected_idx) < selected_target:
        if selected_idx:
            batch_sim = max_tanimoto_to_refs(x_all, head[np.asarray(remaining, dtype=np.int64)], np.asarray(selected_idx, dtype=np.int64)).astype(np.float64)
        else:
            batch_sim = np.zeros(len(remaining), dtype=np.float64)
        rem_base = head_base[np.asarray(remaining, dtype=np.int64)]
        rem_known = known_sim[np.asarray(remaining, dtype=np.int64)]
        # Maximize score while discouraging redundant near-duplicates.
        score = rem_base - 0.20 * rem_known - 0.30 * batch_sim
        best_j = int(np.argmax(score))
        best_pos = int(remaining.pop(best_j))
        selected_pos.append(best_pos)
        selected_idx.append(int(head[best_pos]))
    selected = head[np.asarray(selected_pos, dtype=np.int64)] if selected_pos else np.array([], dtype=np.int64)
    return unique_fill(selected, candidate_idx, len(candidate_idx)).astype(np.int64)


def balanced_rank_order_candidates(p0_pct: np.ndarray, p0_std_pct: np.ndarray, x_all: np.ndarray, candidate_idx: np.ndarray,
                                   active_idx: np.ndarray, inactive_idx: np.ndarray) -> np.ndarray:
    """ChemScreener/SMAL-style balanced ranking: activity + uncertainty + nonredundancy."""
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if len(candidate_idx) == 0:
        return candidate_idx
    if len(active_idx):
        active_sim = max_tanimoto_to_refs(x_all, candidate_idx, active_idx).astype(np.float64)
        inactive_sim = max_tanimoto_to_refs(x_all, candidate_idx, inactive_idx).astype(np.float64) if len(inactive_idx) else np.zeros(len(candidate_idx), dtype=np.float64)
        novelty = 1.0 - active_sim
        cliff_edge = np.maximum(0.0, inactive_sim - active_sim)
        score = 0.52 * p0_pct[candidate_idx].astype(np.float64) + 0.25 * p0_std_pct[candidate_idx].astype(np.float64) + 0.15 * novelty + 0.08 * cliff_edge
    else:
        score = 0.60 * p0_pct[candidate_idx].astype(np.float64) + 0.35 * p0_std_pct[candidate_idx].astype(np.float64)
    return candidate_idx[np.argsort(-score, kind='mergesort')].astype(np.int64)


def hit_neighbor_order_candidates(p0_pct: np.ndarray, p0_std_pct: np.ndarray, x_all: np.ndarray, candidate_idx: np.ndarray,
                                  active_idx: np.ndarray, inactive_idx: np.ndarray) -> np.ndarray:
    """ACActive/ALBF-style local hit propagation/acitivity-cliff competitor."""
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if len(candidate_idx) == 0:
        return candidate_idx
    if len(active_idx):
        active_sim = max_tanimoto_to_refs(x_all, candidate_idx, active_idx).astype(np.float64)
        inactive_sim = max_tanimoto_to_refs(x_all, candidate_idx, inactive_idx).astype(np.float64) if len(inactive_idx) else np.zeros(len(candidate_idx), dtype=np.float64)
        cliff_edge = np.maximum(0.0, inactive_sim - active_sim)
        score = 0.42 * p0_pct[candidate_idx].astype(np.float64) + 0.40 * active_sim + 0.10 * p0_std_pct[candidate_idx].astype(np.float64) + 0.08 * cliff_edge
    else:
        score = 0.70 * p0_pct[candidate_idx].astype(np.float64) + 0.30 * p0_std_pct[candidate_idx].astype(np.float64)
    return candidate_idx[np.argsort(-score, kind='mergesort')].astype(np.int64)


def hit_ucb_select_reentry(p0_mean: np.ndarray, p0_pct: np.ndarray, p0_std_pct: np.ndarray, x_all: np.ndarray,
                           frame_all: pd.DataFrame, frame_after_audit: pd.DataFrame, audit_labels: dict[int, int],
                           reentry_pool_size: int, active_idx: np.ndarray, inactive_idx: np.ndarray) -> np.ndarray:
    if reentry_pool_size <= 0 or len(frame_after_audit) == 0:
        return np.array([], dtype=np.int64)
    stratum_by_idx = {int(r.row_idx): int(r.stratum) for r in frame_all[['row_idx', 'stratum']].itertuples(index=False)}
    theta: dict[int, float] = {}
    local_rate: dict[int, float] = {}
    for sid in range(N_STRATA):
        obs = [int(v) for idx, v in audit_labels.items() if stratum_by_idx.get(int(idx)) == sid]
        n_obs = len(obs)
        k_obs = int(sum(obs))
        local_rate[sid] = (k_obs / n_obs) if n_obs else 0.0
        theta[sid] = float(beta.ppf(0.90, k_obs + 1, n_obs - k_obs + 1)) if n_obs else 1.0
    idx_all = frame_after_audit['row_idx'].astype(np.int64).to_numpy()
    sid_all = frame_after_audit['stratum'].astype(int).to_numpy()
    theta_arr = np.array([theta[int(s)] for s in sid_all], dtype=np.float64)
    local_arr = np.array([local_rate[int(s)] for s in sid_all], dtype=np.float64)
    base = 0.50 * p0_pct[idx_all].astype(np.float64) + 0.30 * theta_arr + 0.20 * p0_std_pct[idx_all].astype(np.float64)
    pre_k = min(len(idx_all), max(4096, int(reentry_pool_size) * 200))
    if len(idx_all) > pre_k:
        pre_pos = np.argpartition(-base, kth=pre_k - 1)[:pre_k]
    else:
        pre_pos = np.arange(len(idx_all), dtype=np.int64)
    idx = idx_all[pre_pos]
    theta_pre = theta_arr[pre_pos]
    local_pre = local_arr[pre_pos]
    if len(active_idx):
        active_sim = max_tanimoto_to_refs(x_all, idx, active_idx).astype(np.float64)
        inactive_sim = max_tanimoto_to_refs(x_all, idx, inactive_idx).astype(np.float64) if len(inactive_idx) else np.zeros(len(idx), dtype=np.float64)
        cliff_edge = np.maximum(0.0, inactive_sim - active_sim)
        score = (
            0.35 * p0_pct[idx].astype(np.float64)
            + 0.25 * active_sim
            + 0.15 * p0_std_pct[idx].astype(np.float64)
            + 0.20 * theta_pre
            + 0.05 * np.maximum(local_pre, cliff_edge)
        )
    else:
        score = 0.55 * p0_pct[idx].astype(np.float64) + 0.25 * theta_pre + 0.20 * p0_std_pct[idx].astype(np.float64)
    order = np.argsort(-score, kind='mergesort')
    return idx[order[: min(int(reentry_pool_size), len(idx))]].astype(np.int64)


def select_hybrid_from_pool(score_map: dict[int, float], pool: np.ndarray, k: int, p0_pct: np.ndarray, x_all: np.ndarray, active_idx: np.ndarray) -> np.ndarray:
    k = min(int(k), len(pool))
    if k <= 0:
        return np.array([], dtype=np.int64)
    pool = np.asarray(pool, dtype=np.int64)
    p1_scores = p1_score_array(score_map, pool)
    p1_rank = percentile_ranks_subset(p1_scores).astype(np.float64)
    if len(active_idx):
        active_sim = max_tanimoto_to_refs(x_all, pool, active_idx).astype(np.float64)
        score = 0.70 * p1_rank + 0.20 * active_sim + 0.10 * p0_pct[pool].astype(np.float64)
    else:
        score = 0.85 * p1_rank + 0.15 * p0_pct[pool].astype(np.float64)
    order = np.argsort(-score, kind='mergesort')
    return pool[order[:k]].astype(np.int64)


def select_hybrid_diverse_from_pool(score_map: dict[int, float], pool: np.ndarray, k: int, p0_pct: np.ndarray,
                                    x_all: np.ndarray, active_idx: np.ndarray) -> np.ndarray:
    """Hybrid p1/hit-neighbor query selection with in-batch redundancy control."""
    k = min(int(k), len(pool))
    if k <= 0:
        return np.array([], dtype=np.int64)
    pool = np.asarray(pool, dtype=np.int64)
    p1_scores = p1_score_array(score_map, pool)
    p1_rank = percentile_ranks_subset(p1_scores).astype(np.float64)
    if len(active_idx):
        active_sim = max_tanimoto_to_refs(x_all, pool, active_idx).astype(np.float64)
        base = 0.62 * p1_rank + 0.25 * active_sim + 0.13 * p0_pct[pool].astype(np.float64)
        known_sim = active_sim
    else:
        base = 0.82 * p1_rank + 0.18 * p0_pct[pool].astype(np.float64)
        known_sim = np.zeros(len(pool), dtype=np.float64)
    head_k = min(len(pool), max(k * 8, k))
    head_pos = np.argsort(-base, kind='mergesort')[:head_k]
    head = pool[head_pos]
    head_base = base[head_pos]
    head_known = known_sim[head_pos]
    remaining = list(range(len(head)))
    selected_pos: list[int] = []
    selected_idx: list[int] = []
    while remaining and len(selected_idx) < k:
        if selected_idx:
            batch_sim = max_tanimoto_to_refs(x_all, head[np.asarray(remaining, dtype=np.int64)], np.asarray(selected_idx, dtype=np.int64)).astype(np.float64)
        else:
            batch_sim = np.zeros(len(remaining), dtype=np.float64)
        rem = np.asarray(remaining, dtype=np.int64)
        score = head_base[rem] - 0.18 * head_known[rem] - 0.25 * batch_sim
        best_j = int(np.argmax(score))
        best_pos = int(remaining.pop(best_j))
        selected_pos.append(best_pos)
        selected_idx.append(int(head[best_pos]))
    selected = head[np.asarray(selected_pos, dtype=np.int64)] if selected_pos else np.array([], dtype=np.int64)
    return unique_fill(selected, pool, k)[:k].astype(np.int64)


def unique_fill(primary: np.ndarray, secondary: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k <= 0:
        return np.array([], dtype=np.int64)
    out: list[int] = []
    seen: set[int] = set()
    for arr in (np.asarray(primary, dtype=np.int64), np.asarray(secondary, dtype=np.int64)):
        for value in arr.tolist():
            iv = int(value)
            if iv in seen:
                continue
            seen.add(iv)
            out.append(iv)
            if len(out) >= k:
                return np.asarray(out, dtype=np.int64)
    return np.asarray(out, dtype=np.int64)


def audit_rule_select_with_frame(p0: np.ndarray, frame_all: pd.DataFrame, frame_after_audit: pd.DataFrame, audit_labels: dict[int, int], reentry_pool_size: int) -> np.ndarray:
    if reentry_pool_size <= 0 or len(frame_after_audit) == 0:
        return np.array([], dtype=np.int64)
    stratum_by_idx = {int(r.row_idx): int(r.stratum) for r in frame_all[['row_idx', 'stratum']].itertuples(index=False)}
    theta: dict[int, float] = {}
    local_rate: dict[int, float] = {}
    for sid in range(N_STRATA):
        obs = [int(v) for idx, v in audit_labels.items() if stratum_by_idx.get(int(idx)) == sid]
        n_obs = len(obs)
        k_obs = int(sum(obs))
        local_rate[sid] = (k_obs / n_obs) if n_obs else 0.0
        theta[sid] = float(beta.ppf(0.90, k_obs + 1, n_obs - k_obs + 1)) if n_obs else 1.0
    frame = frame_after_audit.copy()
    idx = frame['row_idx'].astype(int).to_numpy()
    p_rank = pd.Series(p0[idx]).rank(method='average', pct=True).to_numpy(dtype=np.float64)
    sid = frame['stratum'].astype(int).to_numpy()
    score = np.array([0.50 * p_rank[j] + 0.25 * theta[int(sid[j])] + 0.25 * local_rate[int(sid[j])] for j in range(len(idx))], dtype=np.float64)
    order = np.argsort(-score, kind='mergesort')
    return idx[order[: min(reentry_pool_size, len(idx))]].astype(np.int64)


def air_rescue_select(p0: np.ndarray, p0_pct: np.ndarray, frame_all: pd.DataFrame, frame_after_audit: pd.DataFrame, audit_idx: np.ndarray, audit_labels: dict[int, int], reentry_pool_size: int) -> np.ndarray:
    if reentry_pool_size <= 0 or len(frame_after_audit) == 0:
        return np.array([], dtype=np.int64)
    stratum_by_idx = {int(r.row_idx): int(r.stratum) for r in frame_all[['row_idx', 'stratum']].itertuples(index=False)}
    prior_mean = 0.5
    theta: dict[int, float] = {}
    for sid in range(N_STRATA):
        obs = [int(v) for idx, v in audit_labels.items() if stratum_by_idx.get(int(idx)) == sid]
        theta[sid] = (sum(obs) + 1.0) / (len(obs) + 2.0) if obs else prior_mean
    frame = frame_after_audit.copy()
    idx = frame['row_idx'].astype(int).to_numpy()
    sid = frame['stratum'].astype(int).to_numpy()
    stratum_score = np.array([theta[int(s)] for s in sid], dtype=np.float64)
    score = 0.45 * p0_pct[idx].astype(np.float64) + 0.55 * stratum_score
    return idx[np.argsort(-score, kind='mergesort')[: min(reentry_pool_size, len(idx))]].astype(np.int64)

def p1_score_array(score_map: dict[int, float], candidate_idx: np.ndarray) -> np.ndarray:
    vals = np.array([score_map.get(int(i), np.nan) for i in candidate_idx], dtype=np.float64)
    if np.isnan(vals).any(): raise RuntimeError('missing Chemprop predictions for candidate pool')
    return vals


def select_top_from_pool(score_map: dict[int, float], pool: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), len(pool))
    if k <= 0: return np.array([], dtype=np.int64)
    scores = p1_score_array(score_map, pool)
    return pool[np.argsort(-scores, kind='mergesort')[:k]].astype(np.int64)


def exploit_budget_after_audit(exploit_planned: int, audit_planned: int, audit_actual: int, q_remaining_after_audit: int, pool_size: int) -> int:
    """Exploit labels for a round, backfilling unused audit quota when possible."""
    audit_shortfall = max(0, int(audit_planned) - int(audit_actual))
    target = int(exploit_planned) + audit_shortfall
    return max(0, min(target, int(q_remaining_after_audit), int(pool_size)))


def audit_sampling_key(method: str) -> str:
    """Use a shared audit RNG key for B10 variants so ablations isolate design changes."""
    if method in {'B10_p1_audit_safe', 'B10_p1_reentry_safe', 'B10_p1_reentry_throttle'}:
        return 'B10_p1_audit_rule'
    if method in {'AIR_p1_risk_adaptive_mix', 'AIR_p1_safe_cluster_adaptive_mix', 'AIR_p1_neighbor_probe_gate'}:
        return 'AIR_p1_hit_ucb_mix'
    return method


def audit_evidence_gate_open(cumulative_audit_queries: int, cumulative_audit_hits: int, min_queries: int) -> bool:
    """Return whether rescue/audit should remain active under the online evidence gate.

    The gate is deliberately conservative: before enough sentinel audit labels have
    accumulated, Audit-UCB behaves normally; once the sentinel sample reaches the
    threshold, a zero-hit audit channel is disabled and its planned budget is
    backfilled into the ordinary p1 exploit path.
    """
    if int(min_queries) <= 0:
        return True
    if int(cumulative_audit_queries) < int(min_queries):
        return True
    return int(cumulative_audit_hits) > 0


def throttled_reentry_query_count(exploit_k: int, reentry_pool_size: int, throttle_fraction: float) -> int:
    """Small post-gate re-entry floor: reduce rescue pressure without hard-closing it."""
    exploit_k = int(exploit_k)
    reentry_pool_size = int(reentry_pool_size)
    throttle_fraction = float(throttle_fraction)
    if exploit_k <= 0 or reentry_pool_size <= 0 or throttle_fraction <= 0:
        return 0
    count = int(math.floor(throttle_fraction * exploit_k))
    count = max(1, count)
    return min(count, reentry_pool_size, exploit_k)


def adaptive_reentry_query_count(exploit_k: int, reentry_pool_size: int, audit_labels: dict[int, int], base_fraction: float) -> int:
    exploit_k = int(exploit_k)
    reentry_pool_size = int(reentry_pool_size)
    if exploit_k <= 0 or reentry_pool_size <= 0:
        return 0
    n_obs = len(audit_labels)
    k_obs = int(sum(int(v) for v in audit_labels.values()))
    if n_obs == 0:
        frac = float(base_fraction)
    elif k_obs == 0:
        # No observed evidence that the discarded region is productive: keep a small sentinel rescue slice.
        frac = min(float(base_fraction), 0.10)
    else:
        # Positive audit evidence expands rescue pressure, but keeps at least half the exploit batch for p1-normal picks.
        post_mean = (k_obs + 1.0) / (n_obs + 2.0)
        frac = min(0.45, max(float(base_fraction), float(base_fraction) + 0.50 * post_mean))
    count = int(math.floor(frac * exploit_k))
    if k_obs > 0:
        count = max(1, count)
    return min(count, reentry_pool_size, exploit_k)


def risk_adaptive_mix_fraction(args: argparse.Namespace, round_idx: int, last_round_new_hits: int, cumulative_new_hits: int,
                               initial_mean_score: float, initial_mean_sim: float, p0_mean: np.ndarray, p0_std_pct: np.ndarray, x_all: np.ndarray,
                               ordered: np.ndarray, active_refs: np.ndarray, b1_this: int) -> tuple[float, dict[str, object]]:
    """Choose a HitUCB mix strength using only already observable online signals.

    The first round is deliberately conservative to protect strong p0 baselines.
    Later rounds keep the conservative p0-protected arm only when the previous
    round produced hit momentum, otherwise they switch to an aggressive or
    pure HitUCB recovery arm for sparse cluster rescue. No unqueried labels are
    used; all features are p0 scores/uncertainty and Tanimoto to discovered hits.
    """
    ordered = np.asarray(ordered, dtype=np.int64)
    top_k = min(len(ordered), max(1, int(b1_this)))
    top_idx = ordered[:top_k]
    mean_score = float(np.mean(p0_mean[top_idx])) if len(top_idx) else 0.0
    mean_unc = float(np.mean(p0_std_pct[top_idx])) if len(top_idx) else 0.0
    if len(top_idx) and len(active_refs):
        sim = max_tanimoto_to_refs(x_all, top_idx, active_refs)
        mean_sim = float(np.mean(sim))
        p90_sim = float(np.quantile(sim, 0.90))
    else:
        mean_sim = 0.0
        p90_sim = 0.0
    if int(round_idx) <= 1:
        mode = 'p0_probe'
        frac = float(args.adaptive_conservative_mix_fraction)
    elif int(cumulative_new_hits) >= int(args.adaptive_momentum_hit_threshold):
        mode = 'p0_cumulative_momentum_keep'
        frac = float(args.adaptive_conservative_mix_fraction)
    elif int(last_round_new_hits) >= 1 and float(initial_mean_score) >= float(args.adaptive_initial_keep_score_threshold) and float(initial_mean_sim) >= float(args.adaptive_initial_keep_sim_threshold):
        mode = 'initial_p0_similarity_keep'
        frac = float(args.adaptive_conservative_mix_fraction)
    elif int(last_round_new_hits) >= 1 and mean_score >= float(args.adaptive_keep_score_threshold) and mean_sim >= float(args.adaptive_keep_sim_threshold):
        mode = 'p0_similarity_keep'
        frac = float(args.adaptive_conservative_mix_fraction)
    elif (mean_sim >= float(args.adaptive_pure_sim_threshold)
          and mean_score >= float(getattr(args, 'adaptive_pure_score_lower', 0.0))
          and mean_score <= float(args.adaptive_pure_score_upper)):
        mode = 'pure_hitucb_recovery'
        frac = float(args.adaptive_pure_mix_fraction)
    else:
        mode = 'aggressive_hitucb_recovery'
        frac = float(args.adaptive_aggressive_mix_fraction)
    frac = min(1.0, max(0.0, frac))
    return frac, {
        'adaptive_mix_fraction': float(frac),
        'adaptive_mix_mode': mode,
        'adaptive_last_round_new_hits': int(last_round_new_hits),
        'adaptive_cumulative_new_hits': int(cumulative_new_hits),
        'adaptive_p0_top_mean_score': mean_score,
        'adaptive_p0_top_mean_unc_pct': mean_unc,
        'adaptive_p0_top_mean_active_sim': mean_sim,
        'adaptive_p0_top_p90_active_sim': p90_sim,
        'adaptive_initial_p0_top_mean_score': float(initial_mean_score) if not math.isnan(float(initial_mean_score)) else np.nan,
        'adaptive_initial_p0_top_mean_active_sim': float(initial_mean_sim) if not math.isnan(float(initial_mean_sim)) else np.nan,
    }


def run_method(args: argparse.Namespace, assay: str, seed: int, method: str) -> dict[str, object]:
    ids_df, packed, y = load_assay(assay); ids = ids_df['molecule_id'].astype(str).to_numpy(); smiles = ids_df['canonical_smiles'].astype(str).to_numpy(); n = len(ids)
    q_base, b1_base, q0_base = screening_budgets(n); q = min(q_base, args.q_total_cap) if args.q_total_cap else q_base; b1 = min(b1_base, args.b1_total_cap) if args.b1_total_cap else b1_base; q0 = min(q0_base, max(2, q - 1))
    if q <= q0: raise ValueError(f'Q={q} must exceed q0={q0}')
    if method == 'AIR_p1_safe_cluster_adaptive_mix':
        # Formal no-regret operating point discovered in E07: keep the strong
        # NoAudit-Mix025 backbone, and only allow a high-similarity cluster
        # recovery arm when p0 confidence is also high enough. This prevents
        # sparse false-positive similarity traps while preserving MTORC1 gain.
        args = argparse.Namespace(**vars(args))
        args.audit_fraction = 0.0
        args.reentry_fraction = 0.0
        args.reentry_query_fraction = 0.0
        args.adaptive_conservative_mix_fraction = 0.25
        args.adaptive_aggressive_mix_fraction = 0.25
        args.adaptive_pure_mix_fraction = 0.75
        args.adaptive_pure_sim_threshold = 0.24
        args.adaptive_pure_score_lower = 0.20
        args.adaptive_pure_score_upper = 0.30
        args.adaptive_keep_score_threshold = 1.10
        args.adaptive_initial_keep_score_threshold = 1.10
    if method == 'AIR_p1_similarity_portfolio_gate':
        # Label-free structural portfolio gate.  The initial p0/similarity
        # geometry decides whether to run the hit-neighbor arm from round 1;
        # otherwise the method falls back to the safe-cluster backbone.
        args = argparse.Namespace(**vars(args))
        args.audit_fraction = 0.0
        args.reentry_fraction = 0.0
        args.reentry_query_fraction = 0.0
        args.adaptive_conservative_mix_fraction = 0.25
        args.adaptive_aggressive_mix_fraction = 0.25
        args.adaptive_pure_mix_fraction = 0.75
        args.adaptive_pure_sim_threshold = 0.24
        args.adaptive_pure_score_lower = 0.20
        args.adaptive_pure_score_upper = 0.30
        args.adaptive_keep_score_threshold = 1.10
        args.adaptive_initial_keep_score_threshold = 1.10
    if method == 'AIR_p1_probe_hndiv_gate':
        # One-round non-oracle HNDiv probe. Round 1 uses the hit-neighbor +
        # diversity rescue arm. Only if that first probe returns enough real
        # queried hits do later rounds continue the rescue arm; otherwise the
        # method falls back to the same safe-cluster backbone used by SimGate.
        args = argparse.Namespace(**vars(args))
        args.audit_fraction = 0.0
        args.reentry_fraction = 0.0
        args.reentry_query_fraction = 0.0
        args.adaptive_conservative_mix_fraction = 0.25
        args.adaptive_aggressive_mix_fraction = 0.25
        args.adaptive_pure_mix_fraction = 0.75
        args.adaptive_pure_sim_threshold = 0.24
        args.adaptive_pure_score_lower = 0.20
        args.adaptive_pure_score_upper = 0.30
        args.adaptive_keep_score_threshold = 1.10
        args.adaptive_initial_keep_score_threshold = 1.10
    if method == 'AIR_p1_soft_neighbor_mix':
        # Soft non-destructive neighbor portfolio: reserve only a small slice of
        # the expensive pool for local hit-neighbor candidates, while keeping
        # the safe p0/HitUCB backbone intact. Small-q0 assays disable the slice
        # because one seed hit is too little evidence for reliable propagation.
        args = argparse.Namespace(**vars(args))
        args.audit_fraction = 0.0
        args.reentry_fraction = 0.0
        args.reentry_query_fraction = 0.0
    if method == 'AIR_p1_neighbor_probe_gate':
        # One-round online neighbor probe. Round 1 uses the hit-neighbor arm;
        # if it immediately returns enough hits, we commit to that arm,
        # otherwise subsequent rounds fall back to the safe-cluster backbone.
        args = argparse.Namespace(**vars(args))
        args.audit_fraction = 0.0
        args.reentry_fraction = 0.0
        args.reentry_query_fraction = 0.0
        args.adaptive_conservative_mix_fraction = 0.25
        args.adaptive_aggressive_mix_fraction = 0.25
        args.adaptive_pure_mix_fraction = 0.75
        args.adaptive_pure_sim_threshold = 0.24
        args.adaptive_pure_score_lower = 0.20
        args.adaptive_pure_score_upper = 0.30
        args.adaptive_keep_score_threshold = 1.10
        args.adaptive_initial_keep_score_threshold = 1.10
    out_dir = ROOT / 'results' / args.prefix / f'{assay}_seed{seed}_{method}'; final_manifest_path = out_dir / 'method_manifest.json'
    if args.resume and stage_passed(final_manifest_path):
        print(json.dumps({'status': 'skipped_resume', 'assay': assay, 'seed': seed, 'method': method, 'manifest': str(final_manifest_path.relative_to(ROOT))}, ensure_ascii=False), flush=True)
        return read_json(final_manifest_path) or {}
    if out_dir.exists() and not args.resume: shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True); resource_audit(args, out_dir)
    evaluator = SpyEvaluator({str(ids[i]): int(y[i]) for i in range(n)}); ledger = BudgetLedger(q_budget=q, b1_budget=b1); discovered: dict[str, int] = {}
    init_idx = make_initial(y, assay, seed, q0); query_indices(init_idx, ids, evaluator, ledger, discovered); queried: set[int] = set(map(int, init_idx.tolist())); expensive_scored: set[int] = set()
    source_counts = {'seed': int(q0), 'audit': 0, 'normal': 0, 'reentry': 0}; source_hits = {'seed': int(y[init_idx].sum()), 'audit': 0, 'normal': 0, 'reentry': 0}; round_rows: list[dict[str, object]] = []
    cumulative_audit_queries = 0; cumulative_audit_hits = 0; audit_gate_disabled_rounds = 0; reentry_gate_disabled_rounds = 0; reentry_gate_throttled_rounds = 0; adaptive_last_round_new_hits = 0; adaptive_cumulative_new_hits = 0; adaptive_initial_p0_top_mean_score = float('nan'); adaptive_initial_p0_top_mean_active_sim = float('nan'); neighbor_probe_active = False; neighbor_probe_allowed = not (method == 'AIR_p1_neighbor_probe_gate' and int(q0) <= int(args.neighbor_probe_min_q0)); probe_hndiv_active = False; probe_hndiv_allowed = not (method == 'AIR_p1_probe_hndiv_gate' and int(q0) <= int(args.neighbor_probe_min_q0)); similarity_gate_active = False; similarity_gate_decided = False; similarity_gate_info: dict[str, object] = {}
    q_after_seed = q - q0; audit_total = 0 if method in NO_AUDIT_METHODS else int(math.floor(args.audit_fraction * q_after_seed)); exploit_total = q_after_seed - audit_total
    audit_rounds = split_budget(audit_total, args.rounds); exploit_rounds = split_budget(exploit_total, args.rounds); b1_rounds = split_budget(b1, args.rounds)
    for zero_round in range(args.rounds):
        round_idx = zero_round + 1
        if ledger.q_spent >= q or ledger.b1_spent >= b1: break
        train_idx = np.array(sorted(queried), dtype=np.int64); x_train = unpack_rows(packed, train_idx); x_all = unpack_rows(packed)
        p0_mean, p0_std = train_predict_ensemble_stats(x_train, y[train_idx].astype(int), x_all, seed + 1000 * round_idx, args.lgbm_jobs); p0_pct = percentile_ranks(p0_mean); p0_std_pct = percentile_ranks(p0_std)
        unavailable = np.zeros(n, dtype=bool); unavailable[list(queried)] = True
        if expensive_scored: unavailable[list(expensive_scored)] = True
        available = np.flatnonzero(~unavailable); ordered = choose_by_score(p0_mean, available); b1_this = min(int(b1_rounds[zero_round]), len(ordered), b1 - ledger.b1_spent)
        active_refs_pre = queried_active_refs(queried, y)
        inactive_refs_pre = recent_inactive_refs(queried, y)
        soft_neighbor_mix_used = float('nan')
        soft_neighbor_mode = ''
        if method == 'AIR_p1_similarity_portfolio_gate' and not similarity_gate_decided and b1_this > 0:
            safe_order_gate = ordered
            x4_order_gate = hit_neighbor_order_candidates(p0_pct, p0_std_pct, x_all, available, active_refs_pre, inactive_refs_pre)
            safe_top_gate = safe_order_gate[:min(len(safe_order_gate), int(b1_this))]
            x4_top_gate = x4_order_gate[:min(len(x4_order_gate), int(b1_this))]
            if len(active_refs_pre) and len(safe_top_gate):
                safe_sim_gate = max_tanimoto_to_refs(x_all, safe_top_gate, active_refs_pre)
                safe_mean_gate = float(np.mean(safe_sim_gate))
            else:
                safe_mean_gate = 0.0
            if len(active_refs_pre) and len(x4_top_gate):
                x4_sim_gate = max_tanimoto_to_refs(x_all, x4_top_gate, active_refs_pre)
                x4_mean_gate = float(np.mean(x4_sim_gate))
                x4_p90_gate = float(np.quantile(x4_sim_gate, 0.90))
            else:
                x4_mean_gate = 0.0
                x4_p90_gate = 0.0
            similarity_gate_active = bool(
                int(q0) > int(args.sim_gate_min_q0)
                and x4_p90_gate >= float(args.sim_gate_x4_p90_threshold)
                and x4_mean_gate >= float(args.sim_gate_x4_mean_threshold)
                and safe_mean_gate <= float(args.sim_gate_safe_mean_upper)
            )
            similarity_gate_decided = True
            similarity_gate_info = {
                'similarity_gate_active': bool(similarity_gate_active),
                'similarity_gate_safe_sim_mean': safe_mean_gate,
                'similarity_gate_x4_sim_mean': x4_mean_gate,
                'similarity_gate_x4_sim_p90': x4_p90_gate,
                'similarity_gate_min_q0': int(args.sim_gate_min_q0),
                'similarity_gate_x4_p90_threshold': float(args.sim_gate_x4_p90_threshold),
                'similarity_gate_x4_mean_threshold': float(args.sim_gate_x4_mean_threshold),
                'similarity_gate_safe_mean_upper': float(args.sim_gate_safe_mean_upper),
            }
        if method == 'X0_p1_random' and b1_this > 0:
            ordered = seeded_random_order(available, seed, assay, method, round_idx)
        elif method == 'X1_p1_ucb' and b1_this > 0:
            ordered = ucb_order_candidates(p0_pct, p0_std_pct, available)
        elif method == 'X2_p1_diverse' and b1_this > 0:
            ordered = diverse_order_candidates(p0_pct, p0_std_pct, x_all, ordered, active_refs_pre, b1_this)
        elif method == 'X3_p1_balanced_rank' and b1_this > 0:
            ordered = balanced_rank_order_candidates(p0_pct, p0_std_pct, x_all, available, active_refs_pre, inactive_refs_pre)
        elif method == 'X4_p1_hit_neighbor' and b1_this > 0:
            ordered = hit_neighbor_order_candidates(p0_pct, p0_std_pct, x_all, available, active_refs_pre, inactive_refs_pre)
        elif method == 'AIR_p1_neighbor_probe_gate' and b1_this > 0 and neighbor_probe_allowed and (round_idx == 1 or neighbor_probe_active):
            ordered = hit_neighbor_order_candidates(p0_pct, p0_std_pct, x_all, available, active_refs_pre, inactive_refs_pre)
        elif method == 'AIR_p1_similarity_portfolio_gate' and b1_this > 0 and similarity_gate_active:
            ordered = hit_neighbor_order_candidates(p0_pct, p0_std_pct, x_all, available, active_refs_pre, inactive_refs_pre)
        elif method == 'AIR_p1_probe_hndiv_gate' and b1_this > 0 and probe_hndiv_allowed and (round_idx == 1 or probe_hndiv_active):
            hit_order = hit_neighbor_order_candidates(p0_pct, p0_std_pct, x_all, available, active_refs_pre, inactive_refs_pre)
            div_order = diverse_order_candidates(p0_pct, p0_std_pct, x_all, choose_by_score(p0_mean, available), active_refs_pre, b1_this)
            hit_k = max(1, int(math.ceil(0.70 * b1_this)))
            div_k = max(0, int(b1_this) - hit_k)
            ordered = unique_fill(hit_order[:max(hit_k * 6, hit_k)], div_order[:max(div_k * 8, div_k)], len(available))
        elif method == 'AIR_p1_soft_neighbor_mix' and b1_this > 0:
            safe_order = ordered
            if int(q0) <= int(args.soft_neighbor_min_q0):
                soft_neighbor_mix_used = 0.0
                soft_neighbor_mode = 'soft_neighbor_disabled_small_q0'
                ordered = safe_order
            else:
                soft_neighbor_mix_used = min(1.0, max(0.0, float(args.soft_neighbor_mix_fraction)))
                soft_neighbor_mode = 'soft_neighbor_mix'
                neighbor_order = hit_neighbor_order_candidates(p0_pct, p0_std_pct, x_all, available, active_refs_pre, inactive_refs_pre)
                neighbor_k = min(int(b1_this), max(0, int(round(soft_neighbor_mix_used * int(b1_this)))))
                p0_k = max(0, int(b1_this) - neighbor_k)
                mixed_head = unique_fill(safe_order[:p0_k], neighbor_order[:max(neighbor_k * 6, neighbor_k)], int(b1_this))
                ordered = unique_fill(mixed_head, safe_order, len(available))
        elif method == 'AIR_p1_hit_neighbor_diverse_mix' and b1_this > 0:
            hit_order = hit_neighbor_order_candidates(p0_pct, p0_std_pct, x_all, available, active_refs_pre, inactive_refs_pre)
            div_order = diverse_order_candidates(p0_pct, p0_std_pct, x_all, choose_by_score(p0_mean, available), active_refs_pre, b1_this)
            # Preserve the high-yield hit-neighbor head, but force part of the
            # expensive pool through diversity-aware fill so a single active
            # chemotype cannot monopolize the whole batch.
            hit_k = max(1, int(math.ceil(0.70 * b1_this)))
            div_k = max(0, int(b1_this) - hit_k)
            ordered = unique_fill(hit_order[:max(hit_k * 6, hit_k)], div_order[:max(div_k * 8, div_k)], len(available))
        if method == 'AIR_p1_hit_ucb' and b1_this > 0:
            head_k = min(len(ordered), max(2048, int(b1_this) * 50))
            ordered_head = hit_ucb_order_candidates(p0_mean, p0_pct, p0_std_pct, x_all, ordered[:head_k], active_refs_pre, inactive_refs_pre)
            ordered = np.concatenate([ordered_head, ordered[head_k:]]).astype(np.int64)
        if b1_this <= 0: break
        hit_ucb_mixed_normal_pool: np.ndarray | None = None
        hit_ucb_mixed_discarded_pool: np.ndarray | None = None
        adaptive_mix_info: dict[str, object] = {
            'adaptive_mix_fraction': np.nan, 'adaptive_mix_mode': '',
            'adaptive_last_round_new_hits': int(adaptive_last_round_new_hits), 'adaptive_cumulative_new_hits': int(adaptive_cumulative_new_hits),
            'adaptive_p0_top_mean_score': np.nan, 'adaptive_p0_top_mean_unc_pct': np.nan,
            'adaptive_p0_top_mean_active_sim': np.nan, 'adaptive_p0_top_p90_active_sim': np.nan,
            'adaptive_initial_p0_top_mean_score': adaptive_initial_p0_top_mean_score,
            'adaptive_initial_p0_top_mean_active_sim': adaptive_initial_p0_top_mean_active_sim,
        }
        if method == 'AIR_p1_similarity_portfolio_gate':
            adaptive_mix_info.update(similarity_gate_info)
            adaptive_mix_info['adaptive_mix_fraction'] = 1.0 if similarity_gate_active else float(args.adaptive_conservative_mix_fraction)
            adaptive_mix_info['adaptive_mix_mode'] = 'similarity_gate_x4' if similarity_gate_active else 'similarity_gate_safe'
        if method == 'AIR_p1_soft_neighbor_mix':
            top_k_soft = min(len(ordered), max(1, int(b1_this)))
            top_idx_soft = ordered[:top_k_soft]
            mean_score_soft = float(np.mean(p0_mean[top_idx_soft])) if len(top_idx_soft) else 0.0
            mean_unc_soft = float(np.mean(p0_std_pct[top_idx_soft])) if len(top_idx_soft) else 0.0
            if len(top_idx_soft) and len(active_refs_pre):
                sim_soft = max_tanimoto_to_refs(x_all, top_idx_soft, active_refs_pre)
                mean_sim_soft = float(np.mean(sim_soft))
                p90_sim_soft = float(np.quantile(sim_soft, 0.90))
            else:
                mean_sim_soft = 0.0
                p90_sim_soft = 0.0
            adaptive_mix_info.update({
                'adaptive_mix_fraction': float(soft_neighbor_mix_used),
                'adaptive_mix_mode': soft_neighbor_mode,
                'adaptive_p0_top_mean_score': mean_score_soft,
                'adaptive_p0_top_mean_unc_pct': mean_unc_soft,
                'adaptive_p0_top_mean_active_sim': mean_sim_soft,
                'adaptive_p0_top_p90_active_sim': p90_sim_soft,
                'adaptive_initial_p0_top_mean_score': adaptive_initial_p0_top_mean_score,
                'adaptive_initial_p0_top_mean_active_sim': adaptive_initial_p0_top_mean_active_sim,
            })
        if method == 'AIR_p1_probe_hndiv_gate':
            adaptive_mix_info['adaptive_mix_fraction'] = 1.0 if (probe_hndiv_allowed and (round_idx == 1 or probe_hndiv_active)) else float(args.adaptive_conservative_mix_fraction)
            adaptive_mix_info['adaptive_mix_mode'] = 'probe_hndiv_arm' if (probe_hndiv_allowed and (round_idx == 1 or probe_hndiv_active)) else 'probe_hndiv_safe_fallback'
        if method in {'AIR_p1_hit_ucb_mix', 'AIR_p1_risk_adaptive_mix', 'AIR_p1_safe_cluster_adaptive_mix', 'AIR_p1_neighbor_probe_gate', 'AIR_p1_similarity_portfolio_gate', 'AIR_p1_probe_hndiv_gate'} and not (method == 'AIR_p1_neighbor_probe_gate' and neighbor_probe_allowed and (round_idx == 1 or neighbor_probe_active)) and not (method == 'AIR_p1_probe_hndiv_gate' and probe_hndiv_allowed and (round_idx == 1 or probe_hndiv_active)) and not (method == 'AIR_p1_similarity_portfolio_gate' and similarity_gate_active):
            mix_normal_size = min(max(1, int(math.floor((1.0 - args.reentry_fraction) * b1_this))), len(ordered))
            head_k = min(len(ordered), max(2048, int(b1_this) * 50))
            active_refs_pre = queried_active_refs(queried, y)
            inactive_refs_pre = recent_inactive_refs(queried, y)
            hit_order_head = hit_ucb_order_candidates(p0_mean, p0_pct, p0_std_pct, x_all, ordered[:head_k], active_refs_pre, inactive_refs_pre)
            hit_order = np.concatenate([hit_order_head, ordered[head_k:]]).astype(np.int64)
            mix_fraction_this = float(args.hit_ucb_mix_fraction)
            if method in {'AIR_p1_risk_adaptive_mix', 'AIR_p1_safe_cluster_adaptive_mix', 'AIR_p1_neighbor_probe_gate', 'AIR_p1_similarity_portfolio_gate', 'AIR_p1_probe_hndiv_gate'}:
                mix_fraction_this, adaptive_mix_info = risk_adaptive_mix_fraction(args, round_idx, adaptive_last_round_new_hits, adaptive_cumulative_new_hits, adaptive_initial_p0_top_mean_score, adaptive_initial_p0_top_mean_active_sim, p0_mean, p0_std_pct, x_all, ordered, active_refs_pre, b1_this)
            if method in {'AIR_p1_risk_adaptive_mix', 'AIR_p1_safe_cluster_adaptive_mix', 'AIR_p1_neighbor_probe_gate', 'AIR_p1_similarity_portfolio_gate', 'AIR_p1_probe_hndiv_gate'} and int(round_idx) == 1:
                adaptive_initial_p0_top_mean_score = float(adaptive_mix_info.get('adaptive_p0_top_mean_score', np.nan))
                adaptive_initial_p0_top_mean_active_sim = float(adaptive_mix_info.get('adaptive_p0_top_mean_active_sim', np.nan))
                adaptive_mix_info['adaptive_initial_p0_top_mean_score'] = adaptive_initial_p0_top_mean_score
                adaptive_mix_info['adaptive_initial_p0_top_mean_active_sim'] = adaptive_initial_p0_top_mean_active_sim
            p0_k = min(mix_normal_size, int(math.ceil((1.0 - mix_fraction_this) * mix_normal_size)))
            hit_k = max(0, mix_normal_size - p0_k)
            mixed_primary = unique_fill(ordered[:p0_k], hit_order[:max(hit_k * 4, hit_k)], mix_normal_size)
            hit_ucb_mixed_normal_pool = unique_fill(mixed_primary, ordered, mix_normal_size)
            normal_set = set(map(int, hit_ucb_mixed_normal_pool.tolist()))
            hit_ucb_mixed_discarded_pool = np.array([int(i) for i in ordered.tolist() if int(i) not in normal_set], dtype=np.int64)
        audit_idx = np.array([], dtype=np.int64); reentry_pool = np.array([], dtype=np.int64); audit_label_by_idx: dict[int, int] = {}
        audit_gate_open_pre = True
        audit_gate_open_post = True
        if method in NO_AUDIT_METHODS or (method == 'AIR_p1_neighbor_probe_gate' and neighbor_probe_allowed and (round_idx == 1 or neighbor_probe_active)) or (method == 'AIR_p1_probe_hndiv_gate' and probe_hndiv_allowed and (round_idx == 1 or probe_hndiv_active)) or (method == 'AIR_p1_similarity_portfolio_gate' and similarity_gate_active):
            normal_pool = ordered[:b1_this]
        else:
            audit_gate_open_pre = True
            if method == 'B10_p1_audit_safe':
                audit_gate_open_pre = audit_evidence_gate_open(cumulative_audit_queries, cumulative_audit_hits, args.audit_gate_min_queries)
            if method == 'B10_p1_audit_safe' and not audit_gate_open_pre:
                normal_pool = ordered[:b1_this]
                frame = pd.DataFrame()
                frame_after = pd.DataFrame()
                audit_gate_open_post = False
                audit_gate_disabled_rounds += 1
            else:
                if method in {'AIR_p1_hit_ucb_mix', 'AIR_p1_risk_adaptive_mix', 'AIR_p1_safe_cluster_adaptive_mix', 'AIR_p1_neighbor_probe_gate', 'AIR_p1_similarity_portfolio_gate', 'AIR_p1_probe_hndiv_gate'} and hit_ucb_mixed_normal_pool is not None and hit_ucb_mixed_discarded_pool is not None:
                    normal_pool = hit_ucb_mixed_normal_pool
                    discarded_pool = hit_ucb_mixed_discarded_pool
                else:
                    normal_size = min(max(1, int(math.floor((1.0 - args.reentry_fraction) * b1_this))), len(ordered)); normal_pool = ordered[:normal_size]; discarded_pool = ordered[normal_size:]
                frame = audit_frame(discarded_pool, ids, smiles, p0_mean) if len(discarded_pool) else pd.DataFrame(); audit_k = min(int(audit_rounds[zero_round]), len(frame), q - ledger.q_spent)
                if audit_k > 0 and len(frame):
                    audit_idx = stratified_sample_indices(frame, audit_k, int(seed + 17 * round_idx), f'e07_audit_{assay}_{audit_sampling_key(method)}_{round_idx}')
                    labels = query_indices(audit_idx, ids, evaluator, ledger, discovered); audit_label_by_idx = {int(i): int(labels[str(ids[int(i)])]) for i in audit_idx.tolist()}; queried.update(map(int, audit_idx.tolist())); source_counts['audit'] += len(audit_idx); source_hits['audit'] += hits(labels)
                    cumulative_audit_queries += len(audit_idx); cumulative_audit_hits += int(hits(labels))
                frame_after = frame[~frame['row_idx'].isin(set(map(int, audit_idx.tolist())))] if len(frame) else pd.DataFrame(); reentry_size = min(b1_this - len(normal_pool), len(frame_after))
                audit_gate_open_post = audit_gate_open_pre
                if method in {'B10_p1_audit_safe', 'B10_p1_reentry_safe', 'B10_p1_reentry_throttle'}:
                    audit_gate_open_post = audit_evidence_gate_open(cumulative_audit_queries, cumulative_audit_hits, args.audit_gate_min_queries)
                if method in {'B10_p1_audit_rule', 'B10_p1_audit_safe', 'B10_p1_reentry_safe', 'B10_p1_reentry_throttle', 'AIR_p1_adaptive_quota'}:
                    discarded_after = frame_after['row_idx'].astype(np.int64).to_numpy() if len(frame_after) else np.array([], dtype=np.int64); reentry_pool = audit_rule_select_with_frame(p0_mean, frame, frame_after, audit_label_by_idx, reentry_size)
                elif method == 'AIR_p1_rescue_heuristic':
                    reentry_pool = air_rescue_select(p0_mean, p0_pct, frame, frame_after, audit_idx, audit_label_by_idx, reentry_size)
                elif method in {'AIR_p1_hit_ucb', 'AIR_p1_hit_ucb_mix', 'AIR_p1_risk_adaptive_mix', 'AIR_p1_safe_cluster_adaptive_mix', 'AIR_p1_neighbor_probe_gate', 'AIR_p1_similarity_portfolio_gate', 'AIR_p1_probe_hndiv_gate'}:
                    active_refs_post = queried_active_refs(queried, y)
                    inactive_refs_post = recent_inactive_refs(queried, y)
                    reentry_pool = hit_ucb_select_reentry(p0_mean, p0_pct, p0_std_pct, x_all, frame, frame_after, audit_label_by_idx, reentry_size, active_refs_post, inactive_refs_post)
                else: raise ValueError(method)
        expensive_pool = np.concatenate([normal_pool, reentry_pool]).astype(np.int64)
        if len(expensive_pool) == 0: continue
        spend_expensive(expensive_pool, ids, ledger); expensive_scored.update(map(int, expensive_pool.tolist()))
        score_map = run_chemprop_round(args, out_dir / f'round{round_idx:02d}', ids, smiles, y, np.array(sorted(queried), dtype=np.int64), expensive_pool, seed + 100 * round_idx)
        exploit_k = exploit_budget_after_audit(int(exploit_rounds[zero_round]), int(audit_rounds[zero_round]), len(audit_idx), q - ledger.q_spent, len(expensive_pool))
        if exploit_k <= 0: continue
        if method == 'X0_p1_random':
            selected = normal_pool[:exploit_k].astype(np.int64)
            normal_query = selected
            reentry_query = np.array([], dtype=np.int64)
        elif method in {'B1_p1_fixed', 'X1_p1_ucb', 'X2_p1_diverse', 'X3_p1_balanced_rank', 'X4_p1_hit_neighbor', 'AIR_p1_neighbor_probe_gate', 'AIR_p1_similarity_portfolio_gate', 'AIR_p1_probe_hndiv_gate', 'AIR_p1_soft_neighbor_mix', 'AIR_p1_hit_neighbor_diverse_mix', 'AIR_p1_rescue_heuristic'}:
            if method in {'X3_p1_balanced_rank', 'X4_p1_hit_neighbor', 'AIR_p1_neighbor_probe_gate', 'AIR_p1_similarity_portfolio_gate', 'AIR_p1_soft_neighbor_mix'} or (method == 'AIR_p1_probe_hndiv_gate' and not (probe_hndiv_allowed and (round_idx == 1 or probe_hndiv_active))):
                active_refs_query = queried_active_refs(queried, y)
                selected = select_hybrid_from_pool(score_map, expensive_pool, exploit_k, p0_pct, x_all, active_refs_query)
            elif method == 'AIR_p1_hit_neighbor_diverse_mix' or (method == 'AIR_p1_probe_hndiv_gate' and probe_hndiv_allowed and (round_idx == 1 or probe_hndiv_active)):
                active_refs_query = queried_active_refs(queried, y)
                selected = select_hybrid_diverse_from_pool(score_map, expensive_pool, exploit_k, p0_pct, x_all, active_refs_query)
            elif method == 'X2_p1_diverse':
                preselected = select_top_from_pool(score_map, expensive_pool, min(len(expensive_pool), max(exploit_k * 4, exploit_k)))
                selected = diverse_order_candidates(p0_pct, p0_std_pct, x_all, preselected, queried_active_refs(queried, y), exploit_k)[:exploit_k]
            else:
                selected = select_top_from_pool(score_map, expensive_pool, exploit_k)
            reentry_set = set(map(int, reentry_pool.tolist())); normal_query = np.array([i for i in selected.tolist() if int(i) not in reentry_set], dtype=np.int64); reentry_query = np.array([i for i in selected.tolist() if int(i) in reentry_set], dtype=np.int64)
        else:
            if method in {'AIR_p1_hit_ucb', 'AIR_p1_hit_ucb_mix', 'AIR_p1_risk_adaptive_mix', 'AIR_p1_safe_cluster_adaptive_mix', 'AIR_p1_neighbor_probe_gate', 'AIR_p1_similarity_portfolio_gate', 'AIR_p1_probe_hndiv_gate'}:
                re_k = adaptive_reentry_query_count(exploit_k, len(reentry_pool), audit_label_by_idx, args.reentry_query_fraction)
            elif method == 'AIR_p1_adaptive_quota':
                re_k = adaptive_reentry_query_count(exploit_k, len(reentry_pool), audit_label_by_idx, args.reentry_query_fraction)
            elif method == 'B10_p1_reentry_throttle' and not audit_gate_open_post:
                re_k = throttled_reentry_query_count(exploit_k, len(reentry_pool), args.reentry_throttle_fraction)
                reentry_gate_throttled_rounds += 1
            elif method in {'B10_p1_audit_safe', 'B10_p1_reentry_safe'} and not audit_gate_open_post:
                re_k = 0
                reentry_gate_disabled_rounds += 1
            else:
                re_k = min(int(math.floor(args.reentry_query_fraction * exploit_k)), len(reentry_pool))
            no_k = min(exploit_k - re_k, len(normal_pool))
            if method in {'AIR_p1_hit_ucb', 'AIR_p1_hit_ucb_mix', 'AIR_p1_risk_adaptive_mix', 'AIR_p1_safe_cluster_adaptive_mix', 'AIR_p1_neighbor_probe_gate', 'AIR_p1_similarity_portfolio_gate', 'AIR_p1_probe_hndiv_gate'}:
                active_refs_query = queried_active_refs(queried, y)
                normal_query = select_hybrid_from_pool(score_map, normal_pool, no_k, p0_pct, x_all, active_refs_query)
                reentry_query = select_hybrid_from_pool(score_map, reentry_pool, re_k, p0_pct, x_all, active_refs_query)
            else:
                normal_query = select_top_from_pool(score_map, normal_pool, no_k)
                reentry_query = select_top_from_pool(score_map, reentry_pool, re_k)
            selected = np.concatenate([normal_query, reentry_query]).astype(np.int64)
        labels_n = query_indices(normal_query, ids, evaluator, ledger, discovered); labels_r = query_indices(reentry_query, ids, evaluator, ledger, discovered); queried.update(map(int, selected.tolist()))
        round_new_hits = int(hits(labels_n) + hits(labels_r) + (sum(audit_label_by_idx.values()) if audit_label_by_idx else 0))
        source_counts['normal'] += len(normal_query); source_counts['reentry'] += len(reentry_query); source_hits['normal'] += hits(labels_n); source_hits['reentry'] += hits(labels_r)
        if method == 'AIR_p1_neighbor_probe_gate' and neighbor_probe_allowed and int(round_idx) == 1:
            neighbor_probe_active = int(round_new_hits) >= int(args.neighbor_probe_hit_threshold)
        if method == 'AIR_p1_probe_hndiv_gate' and probe_hndiv_allowed and int(round_idx) == 1:
            probe_hndiv_active = int(round_new_hits) >= int(args.probe_hndiv_hit_threshold)
        row = {'assay': assay, 'seed': int(seed), 'method': method, 'round': int(round_idx), 'train_size': int(len(train_idx)), 'train_pos': int(y[train_idx].sum()), 'audit_queries': int(len(audit_idx)), 'audit_planned': int(audit_rounds[zero_round]), 'exploit_planned': int(exploit_rounds[zero_round]), 'exploit_actual': int(len(selected)), 'audit_backfill': int(max(0, int(audit_rounds[zero_round]) - len(audit_idx))), 'normal_pool': int(len(normal_pool)), 'reentry_pool': int(len(reentry_pool)), 'expensive_this_round': int(len(expensive_pool)), 'queried_this_round': int(len(selected) + len(audit_idx)), 'normal_hits_round': int(hits(labels_n)), 'reentry_hits_round': int(hits(labels_r)), 'audit_hits_round': int(sum(audit_label_by_idx.values())) if audit_label_by_idx else 0, 'round_new_hits': int(round_new_hits), 'audit_cumulative_queries': int(cumulative_audit_queries), 'audit_cumulative_hits': int(cumulative_audit_hits), 'audit_gate_open_pre': bool(audit_gate_open_pre), 'audit_gate_open_post': bool(audit_gate_open_post), 'audit_gate_min_queries': int(args.audit_gate_min_queries), 'reentry_gate_disabled_rounds_after': int(reentry_gate_disabled_rounds), 'reentry_gate_throttled_rounds_after': int(reentry_gate_throttled_rounds), 'neighbor_probe_active_after': bool(neighbor_probe_active), 'neighbor_probe_allowed': bool(neighbor_probe_allowed), 'neighbor_probe_hit_threshold': int(args.neighbor_probe_hit_threshold), 'q_spent_after': int(ledger.q_spent), 'b1_spent_after': int(ledger.b1_spent)}
        row.update(adaptive_mix_info)
        round_rows.append(row)
        adaptive_last_round_new_hits = int(round_new_hits)
        adaptive_cumulative_new_hits += int(round_new_hits)
        pd.DataFrame(round_rows).to_csv(out_dir / 'round_trace.csv', index=False)
    evaluator.unlock_metrics(); total_active = evaluator.total_active
    result = {'status': 'pass', 'protocol': 'p1_guarded_pilot_v0', 'warning': 'single-task guarded p1 pilot; capped budgets/short epochs are not formal E07 primary results unless caps are unset and preregistered settings are used', 'assay': assay, 'seed': int(seed), 'method': method, 'n': int(n), 'total_active': int(total_active), 'Q_base': int(q_base), 'B1_base': int(b1_base), 'q0_base': int(q0_base), 'Q': int(q), 'B1': int(b1), 'q0': int(q0), 'rounds': int(args.rounds), 'q_spent': int(ledger.q_spent), 'b1_spent': int(ledger.b1_spent), 'hits': int(hits(discovered)), 'recall': float(recall(discovered, total_active)), 'nef': float(nef(discovered, ledger.q_spent, total_active, n)), 'seed_hits': int(source_hits['seed']), 'audit_hits': int(source_hits['audit']), 'normal_hits': int(source_hits['normal']), 'reentry_hits': int(source_hits['reentry']), 'seed_queries': int(source_counts['seed']), 'audit_queries': int(source_counts['audit']), 'normal_queries': int(source_counts['normal']), 'reentry_queries': int(source_counts['reentry']), 'audit_gate_min_queries': int(args.audit_gate_min_queries), 'audit_gate_disabled_rounds': int(audit_gate_disabled_rounds), 'reentry_gate_disabled_rounds': int(reentry_gate_disabled_rounds), 'reentry_gate_throttled_rounds': int(reentry_gate_throttled_rounds), 'budget_ok': bool(ledger.q_spent <= q and ledger.b1_spent <= b1 and len(ledger.queried_ids) == ledger.q_spent and len(ledger.expensive_ids) == ledger.b1_spent), 'accelerator': args.accelerator, 'gpu_physical_id': args.gpu_physical_id, 'cpu_threads': args.cpu_threads, 'timeout_sec': args.timeout_sec, 'created_at': now_iso(), 'round_trace': str((out_dir / 'round_trace.csv').relative_to(ROOT)), 'manifest': str(final_manifest_path.relative_to(ROOT))}
    write_json(final_manifest_path, result); return result


def summarize(rows: list[dict[str, object]], prefix: str) -> None:
    table_dir = ROOT / 'tables'; table_dir.mkdir(exist_ok=True); df = pd.DataFrame(rows); df.to_csv(table_dir / f'{prefix}.csv', index=False)
    if len(df):
        summary = df.groupby(['protocol', 'method'], as_index=False).agg(n_tasks=('assay', 'size'), mean_hits=('hits', 'mean'), mean_recall=('recall', 'mean'), mean_nef=('nef', 'mean'), all_budget_ok=('budget_ok', 'all')).sort_values(['mean_recall', 'method'], ascending=[False, True]); summary.to_csv(table_dir / f'{prefix}_summary.csv', index=False)
    audit = {'status': 'pass' if len(df) and bool(df['budget_ok'].all()) else 'fail', 'n_rows': int(len(df)), 'outputs': {'runs': f'tables/{prefix}.csv', 'summary': f'tables/{prefix}_summary.csv'}}; write_json(table_dir / f'{prefix}_audit.json', audit); print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    if len(df): print(df[['assay', 'seed', 'method', 'hits', 'recall', 'nef', 'q_spent', 'b1_spent', 'budget_ok']].to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Guarded Chemprop p1 single-task pilot for E07 before any batch parallel run.')
    parser.add_argument('--assays', nargs='*', default=['ALDH1']); parser.add_argument('--seeds', nargs='*', type=int, default=[SEEDS[0]]); parser.add_argument('--methods', nargs='*', default=METHODS); parser.add_argument('--prefix', default='e07_p1_pilot'); parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--q-total-cap', type=int, default=0); parser.add_argument('--b1-total-cap', type=int, default=0); parser.add_argument('--audit-fraction', type=float, default=0.20); parser.add_argument('--reentry-fraction', type=float, default=0.20); parser.add_argument('--reentry-query-fraction', type=float, default=0.20); parser.add_argument('--reentry-throttle-fraction', type=float, default=0.10); parser.add_argument('--audit-gate-min-queries', type=int, default=10); parser.add_argument('--hit-ucb-mix-fraction', type=float, default=0.50); parser.add_argument('--adaptive-conservative-mix-fraction', type=float, default=0.25); parser.add_argument('--adaptive-aggressive-mix-fraction', type=float, default=0.75); parser.add_argument('--adaptive-pure-mix-fraction', type=float, default=1.00); parser.add_argument('--adaptive-momentum-hit-threshold', type=int, default=2); parser.add_argument('--adaptive-keep-score-threshold', type=float, default=0.05); parser.add_argument('--adaptive-keep-sim-threshold', type=float, default=0.150); parser.add_argument('--adaptive-pure-sim-threshold', type=float, default=0.145); parser.add_argument('--adaptive-pure-score-lower', type=float, default=0.0); parser.add_argument('--adaptive-pure-score-upper', type=float, default=0.055); parser.add_argument('--adaptive-initial-keep-score-threshold', type=float, default=0.055); parser.add_argument('--adaptive-initial-keep-sim-threshold', type=float, default=0.160)
    parser.add_argument('--lgbm-jobs', type=int, default=4); parser.add_argument('--epochs', type=int, default=2); parser.add_argument('--batch-size', type=int, default=128); parser.add_argument('--message-hidden-dim', type=int, default=128); parser.add_argument('--depth', type=int, default=3); parser.add_argument('--dropout', type=float, default=0.10)
    parser.add_argument('--accelerator', choices=['cpu', 'gpu'], default='cpu'); parser.add_argument('--gpu-physical-id', type=int, default=None); parser.add_argument('--cpu-threads', type=int, default=4); parser.add_argument('--timeout-sec', type=int, default=1800); parser.add_argument('--neighbor-probe-hit-threshold', type=int, default=2); parser.add_argument('--probe-hndiv-hit-threshold', type=int, default=3); parser.add_argument('--neighbor-probe-min-q0', type=int, default=40); parser.add_argument('--soft-neighbor-mix-fraction', type=float, default=0.25); parser.add_argument('--soft-neighbor-min-q0', type=int, default=40); parser.add_argument('--sim-gate-x4-p90-threshold', type=float, default=0.59); parser.add_argument('--sim-gate-x4-mean-threshold', type=float, default=0.42); parser.add_argument('--sim-gate-safe-mean-upper', type=float, default=0.25); parser.add_argument('--sim-gate-min-q0', type=int, default=40); parser.add_argument('--resume', action='store_true'); parser.add_argument('--self-test-guards', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_threads > max(1, (os.cpu_count() or 1) - 30): raise SystemExit(f'--cpu-threads={args.cpu_threads} would violate leave-30-core rule on this host')
    if args.epochs <= 1: raise SystemExit('--epochs must be >=2 because warmup-epochs is fixed at 1')
    if args.self_test_guards:
        self_test_guards(args); return
    rows: list[dict[str, object]] = []
    for assay in args.assays:
        for seed in args.seeds:
            for method in args.methods:
                rows.append(run_method(args, assay, int(seed), method)); summarize(rows, args.prefix)
    summarize(rows, args.prefix)


if __name__ == '__main__':
    main()
