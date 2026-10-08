
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT_BOOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_BOOT / 'src'))
sys.path.insert(0, str(ROOT_BOOT / 'scripts'))
from typing import Iterable

import numpy as np
import pandas as pd

from airvs.data.registry import SEEDS
from airvs.metrics.core import hits, nef, recall
from airvs.simulator.evaluator import BudgetLedger, SpyEvaluator
from run_e03_headroom import load_assay, unpack_rows
from run_e05_baseline_smoke import make_initial, screening_budgets, train_predict_ensemble_stats
from run_e06_pair_bank_proxy import split_budget
from run_e07_p1_pilot import (
    compute_apps_snapshot,
    guarded_env,
    nvidia_snapshot,
    stage_passed,
    torch_witness,
    validate_gpu_id,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
AIR311_PYTHON = ROOT / 'envs' / 'air311' / 'bin' / 'python'
CHEMSCREENER_ROOT = ROOT / 'external_baselines' / 'official_code' / 'repos' / 'chemscreener'
OFFICIAL_MAIN = CHEMSCREENER_ROOT / 'main.py'
METHOD = 'Official-ChemScreener-BalancedRanking'


def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def run_text_cwd(cmd: list[str], cwd: Path, timeout_sec: int | None, env: dict[str, str]) -> tuple[int, str, bool, float]:
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
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
            out, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, _ = proc.communicate()
        return 124, (out or '') + f'\n[TIMEOUT after {timeout_sec} sec]\n', timed_out, time.time() - started


def stable_desc(score: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    return candidates[np.argsort(-score[candidates], kind='mergesort')]


def selected_smiles_to_indices(selected_csv: Path, pool_df: pd.DataFrame, k: int) -> np.ndarray:
    if not selected_csv.exists():
        raise RuntimeError(f'ChemScreener did not produce selected csv: {selected_csv}')
    sel = pd.read_csv(selected_csv)
    if len(sel) == 0:
        return np.array([], dtype=np.int64)
    if 'smiles' not in sel.columns:
        raise RuntimeError(f'Expected smiles column in {selected_csv}, got {sel.columns.tolist()}')
    # ChemScreener writes selections in ranked order. Use first occurrence in B1 pool.
    smi_to_idx = {}
    for row in pool_df[['row_idx', 'smiles']].itertuples(index=False):
        smi_to_idx.setdefault(str(row.smiles), int(row.row_idx))
    out = []
    missing = []
    for smi in sel['smiles'].astype(str).tolist():
        if smi in smi_to_idx:
            out.append(smi_to_idx[smi])
        else:
            missing.append(smi)
        if len(out) >= int(k):
            break
    if missing:
        raise RuntimeError(f'{len(missing)} selected SMILES not in pool; first={missing[:3]}')
    return np.asarray(out, dtype=np.int64)


def query_indices(indices: Iterable[int], ids: np.ndarray, evaluator: SpyEvaluator, ledger: BudgetLedger, discovered: dict[str, int]) -> dict[str, int]:
    selected_ids = [str(ids[int(i)]) for i in indices]
    labels = evaluator.query_labels(selected_ids, ledger)
    discovered.update(labels)
    return labels


def spend_b1(indices: Iterable[int], ids: np.ndarray, ledger: BudgetLedger) -> None:
    selected_ids = [str(ids[int(i)]) for i in indices]
    ledger.spend_b1(selected_ids)


def class_balance_ok(y_train: np.ndarray) -> bool:
    return int(np.sum(y_train == 1)) > 0 and int(np.sum(y_train == 0)) > 0


def self_test_guards(args: argparse.Namespace) -> None:
    out_dir = ROOT / 'results' / args.prefix / '_guard_selftest'
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_gpu_id(args.gpu_physical_id, args.accelerator)
    env = guarded_env(args)
    witness = torch_witness(args, out_dir)
    code, out, timed_out, elapsed = run_text_cwd(
        [str(AIR311_PYTHON), '-c', 'import time; time.sleep(30); print("should_not_print")'],
        cwd=ROOT,
        timeout_sec=2,
        env=env,
    )
    resume_marker = out_dir / 'resume_marker.json'
    if not resume_marker.exists():
        write_json(resume_marker, {'status': 'pass', 'created_at': now_iso()})
    resume_ok = stage_passed(resume_marker)
    report = {
        'status': 'pass' if witness.get('status') == 'pass' and code == 124 and timed_out and resume_ok else 'fail',
        'tool': 'official_chemscreener_adapter',
        'gpu_snapshot': nvidia_snapshot(),
        'compute_apps_snapshot': compute_apps_snapshot(),
        'torch_witness': witness,
        'timeout_fault_injection': {'returncode': code, 'timed_out': timed_out, 'elapsed_sec': elapsed, 'stdout_tail': out[-300:]},
        'resume_marker_ok': resume_ok,
        'cpu_total': os.cpu_count(),
        'cpu_threads_per_task': int(args.cpu_threads),
        'leave_30_core_rule_ok': bool(args.cpu_threads <= max(1, (os.cpu_count() or 1) - 30)),
        'created_at': now_iso(),
    }
    out_path = ROOT / 'tables' / f'{args.prefix}_guard_selftest_report.json'
    write_json(out_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report['status'] != 'pass':
        raise SystemExit('guard self-test failed')


def run_one(args: argparse.Namespace, assay: str, seed: int) -> dict[str, object]:
    ids_df, packed, y = load_assay(assay)
    ids = ids_df['molecule_id'].astype(str).to_numpy()
    smiles = ids_df['canonical_smiles'].astype(str).to_numpy()
    n = len(ids)
    q_base, b1_base, q0_base = screening_budgets(n)
    q = min(q_base, args.q_total_cap) if args.q_total_cap else q_base
    b1 = min(b1_base, args.b1_total_cap) if args.b1_total_cap else b1_base
    q0 = min(q0_base, max(2, q - 1))
    if q <= q0:
        raise ValueError(f'Q={q} must exceed q0={q0}')
    if not OFFICIAL_MAIN.exists():
        raise FileNotFoundError(OFFICIAL_MAIN)
    out_dir = ROOT / 'results' / args.prefix / f'{assay}_seed{seed}_official_chemscreener_balanced'
    final_manifest = out_dir / 'method_manifest.json'
    if args.resume and stage_passed(final_manifest):
        data = json.loads(final_manifest.read_text(encoding='utf-8'))
        data['status'] = 'pass_resumed'
        return data
    out_dir.mkdir(parents=True, exist_ok=True)
    env = guarded_env(args)
    env['PYTHONPATH'] = str(CHEMSCREENER_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    if args.accelerator == 'gpu':
        chem_accel = 'gpu'
    else:
        chem_accel = 'cpu'
    x_all = unpack_rows(packed, np.arange(n, dtype=np.int64))
    evaluator = SpyEvaluator({str(ids[i]): int(y[i]) for i in range(n)})
    ledger = BudgetLedger(q_budget=q, b1_budget=b1)
    discovered: dict[str, int] = {}
    init_idx = make_initial(y, assay, seed, q0)
    query_indices(init_idx, ids, evaluator, ledger, discovered)
    queried: set[int] = set(map(int, init_idx.tolist()))
    expensive_scored: set[int] = set()
    q_after_seed = q - q0
    exploit_rounds = split_budget(q_after_seed, args.rounds)
    b1_rounds = split_budget(b1, args.rounds)
    round_rows = []
    for zero_round in range(args.rounds):
        round_idx = zero_round + 1
        round_dir = out_dir / f'round{round_idx:02d}'
        round_manifest = round_dir / 'round_manifest.json'
        if args.resume and stage_passed(round_manifest):
            payload = json.loads(round_manifest.read_text(encoding='utf-8'))
            selected = np.asarray(payload['selected_row_idx'], dtype=np.int64)
            b1_saved = payload.get('b1_row_idx')
            if b1_saved is None and (round_dir / 'library_b1.csv').exists():
                b1_saved = pd.read_csv(round_dir / 'library_b1.csv')['row_idx'].astype(int).tolist()
            if b1_saved is not None:
                b1_saved_arr = np.asarray(b1_saved, dtype=np.int64)
                spend_b1(b1_saved_arr, ids, ledger)
                expensive_scored.update(map(int, b1_saved_arr.tolist()))
            query_indices(selected, ids, evaluator, ledger, discovered)
            queried.update(map(int, selected.tolist()))
            round_rows.append(payload['round_row'])
            continue
        round_dir.mkdir(parents=True, exist_ok=True)
        train_idx = np.asarray(sorted(queried), dtype=np.int64)
        if not class_balance_ok(y[train_idx]):
            raise RuntimeError(f'train set lacks both classes: train_pos={int(y[train_idx].sum())}, train_n={len(train_idx)}')
        p0_mean, _p0_std = train_predict_ensemble_stats(x_all[train_idx], y[train_idx].astype(int), x_all, seed + 1000 * round_idx, args.lgbm_jobs)
        unavailable = np.zeros(n, dtype=bool)
        unavailable[list(queried)] = True
        if expensive_scored:
            unavailable[list(expensive_scored)] = True
        available = np.flatnonzero(~unavailable)
        ordered = stable_desc(p0_mean, available)
        b1_this = min(int(b1_rounds[zero_round]), len(ordered), b1 - ledger.b1_spent)
        b1_pool = ordered[:b1_this].astype(np.int64)
        spend_b1(b1_pool, ids, ledger)
        expensive_scored.update(map(int, b1_pool.tolist()))
        acquire_k = min(int(exploit_rounds[zero_round]), q - ledger.q_spent, len(b1_pool))
        assay_csv = round_dir / 'assay_labeled.csv'
        test_csv = round_dir / 'library_b1.csv'
        pd.DataFrame({
            'smiles': smiles[train_idx].astype(str),
            'y': y[train_idx].astype(int),
            'molecule_id': ids[train_idx].astype(str),
            'row_idx': train_idx.astype(int),
        }).to_csv(assay_csv, index=False)
        pool_df = pd.DataFrame({
            'smiles': smiles[b1_pool].astype(str),
            'molecule_id': ids[b1_pool].astype(str),
            'row_idx': b1_pool.astype(int),
        })
        pool_df.to_csv(test_csv, index=False)
        cmd = [
            str(AIR311_PYTHON), str(OFFICIAL_MAIN),
            '--assay_csv', str(assay_csv),
            '--test_csv', str(test_csv),
            '--smiles_column', 'smiles',
            '--target_columns', 'y',
            '--output_dir', str(round_dir / 'chemscreener_out'),
            '--n_ensemble', str(args.chem_ensemble),
            '--batch_size', str(args.chem_batch_size),
            '--max_epochs', str(args.chem_epochs),
            '--mode', args.chem_mode,
            '--n_acquire', str(acquire_k),
            '--accelerator', chem_accel,
            '--devices', '1',
            '--seed', str(seed + 100 * round_idx),
            '--run_id', str(round_idx),
        ]
        code, log, timed_out, elapsed = run_text_cwd(cmd, cwd=CHEMSCREENER_ROOT, timeout_sec=args.timeout_sec, env=env)
        (round_dir / 'official_command.json').write_text(json.dumps({'cmd': cmd, 'cwd': str(CHEMSCREENER_ROOT), 'env_cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES',''), 'timeout_sec': args.timeout_sec}, indent=2), encoding='utf-8')
        (round_dir / 'official_stdout.log').write_text(log, encoding='utf-8')
        if code != 0:
            write_json(round_manifest, {'status': 'fail', 'returncode': code, 'timed_out': timed_out, 'elapsed_sec': elapsed, 'stdout_tail': log[-4000:]})
            raise RuntimeError(f'ChemScreener official command failed rc={code}; see {round_dir / "official_stdout.log"}')
        selected_csv = round_dir / 'chemscreener_out' / f'Chemprop_{args.chem_mode}_y_run{round_idx}.csv'
        selected = selected_smiles_to_indices(selected_csv, pool_df, acquire_k)
        labels = query_indices(selected, ids, evaluator, ledger, discovered)
        queried.update(map(int, selected.tolist()))
        round_row = {
            'assay': assay,
            'seed': int(seed),
            'method': METHOD,
            'round': int(round_idx),
            'train_size': int(len(train_idx)),
            'train_pos': int(y[train_idx].sum()),
            'b1_candidates': int(len(b1_pool)),
            'queried': int(len(selected)),
            'round_hits': int(hits(labels)),
            'q_spent': int(ledger.q_spent),
            'b1_spent': int(ledger.b1_spent),
            'official_elapsed_sec': float(elapsed),
            'official_timed_out': bool(timed_out),
        }
        round_rows.append(round_row)
        write_json(round_manifest, {
            'status': 'pass',
            'b1_row_idx': b1_pool.astype(int).tolist(),
            'selected_row_idx': selected.astype(int).tolist(),
            'selected_labels': {str(ids[int(i)]): int(y[int(i)]) for i in selected.tolist()},
            'round_row': round_row,
            'created_at': now_iso(),
        })
        pd.DataFrame(round_rows).to_csv(out_dir / 'round_trace.csv', index=False)
    evaluator.unlock_metrics()
    labels_all = {str(k): int(v) for k, v in discovered.items()}
    h = int(hits(labels_all))
    result = {
        'status': 'pass',
        'protocol': 'official_author_code_chemscreener_adapter_v1',
        'method': METHOD,
        'official_repo': 'https://github.com/Novartis/ChemScreener',
        'official_commit': subprocess.check_output(['git','-C',str(CHEMSCREENER_ROOT),'rev-parse','HEAD'], text=True).strip(),
        'assay': assay,
        'seed': int(seed),
        'hits': h,
        'recall': float(recall(labels_all, evaluator.total_active)),
        'nef': float(nef(labels_all, q, evaluator.total_active, n)),
        'q_spent': int(ledger.q_spent),
        'b1_spent': int(ledger.b1_spent),
        'q_budget': int(q),
        'b1_budget': int(b1),
        'q0': int(q0),
        'rounds': int(args.rounds),
        'chem_mode': args.chem_mode,
        'chem_ensemble': int(args.chem_ensemble),
        'chem_epochs': int(args.chem_epochs),
        'budget_ok': bool(ledger.q_spent <= q and ledger.b1_spent <= b1),
        'created_at': now_iso(),
    }
    write_json(final_manifest, result)
    return result


def summarize(rows: list[dict[str, object]], prefix: str) -> None:
    table_dir = ROOT / 'tables'
    table_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out = table_dir / f'{prefix}.csv'
    df.to_csv(out, index=False)
    if len(df):
        summary = df.groupby(['protocol', 'method'], as_index=False).agg(
            n_tasks=('assay', 'size'),
            mean_hits=('hits', 'mean'),
            std_hits=('hits', 'std'),
            mean_recall=('recall', 'mean'),
            mean_nef=('nef', 'mean'),
            all_budget_ok=('budget_ok', 'all'),
        )
        summary.to_csv(table_dir / f'{prefix}_summary.csv', index=False)
        print(df[['assay','seed','method','hits','recall','nef','q_spent','b1_spent','budget_ok']].to_string(index=False), flush=True)
        print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Official author-code ChemScreener adapter under AIR Q/B1 protocol.')
    p.add_argument('--assays', nargs='*', default=['ALDH1'])
    p.add_argument('--seeds', nargs='*', type=int, default=[SEEDS[0]])
    p.add_argument('--prefix', default='official_chemscreener_smoke')
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--q-total-cap', type=int, default=192)
    p.add_argument('--b1-total-cap', type=int, default=192)
    p.add_argument('--lgbm-jobs', type=int, default=4)
    p.add_argument('--chem-ensemble', type=int, default=1)
    p.add_argument('--chem-epochs', type=int, default=1)
    p.add_argument('--chem-batch-size', type=int, default=128)
    p.add_argument('--chem-mode', choices=['MI','Exploitative','Balanced_Ranking'], default='Balanced_Ranking')
    p.add_argument('--accelerator', choices=['cpu','gpu'], default='cpu')
    p.add_argument('--gpu-physical-id', type=int, default=None)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--timeout-sec', type=int, default=900)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--self-test-guards', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_threads > max(1, (os.cpu_count() or 1) - 30):
        raise SystemExit(f'--cpu-threads={args.cpu_threads} violates leave-30-core rule')
    validate_gpu_id(args.gpu_physical_id, args.accelerator)
    if args.self_test_guards:
        self_test_guards(args)
        return
    rows = []
    for assay in args.assays:
        for seed in args.seeds:
            rows.append(run_one(args, assay, int(seed)))
            summarize(rows, args.prefix)
    summarize(rows, args.prefix)


if __name__ == '__main__':
    main()
