
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_e07_p1_pilot import adaptive_reentry_query_count, air_rescue_select, audit_evidence_gate_open, audit_sampling_key, chemprop_class_balance_is_safe, chemprop_valid_smiles_mask, exploit_budget_after_audit, safe_splits, select_top_from_pool, throttled_reentry_query_count


def test_safe_splits_keeps_single_positive_train_and_adds_val():
    labels = np.array([1] + [0] * 31)
    splits = safe_splits(labels, seed=47001)
    assert splits[0] == 'train'
    assert 'val' in splits
    assert len(splits) == len(labels)


def test_select_top_from_pool_orders_by_p1_score():
    score_map = {10: 0.1, 11: 0.9, 12: 0.4}
    selected = select_top_from_pool(score_map, np.array([10, 11, 12]), 2)
    assert selected.tolist() == [11, 12]



def test_air_rescue_uses_audit_stratum_signal():
    import pandas as pd
    p0 = np.zeros(10, dtype=float)
    p0_pct = np.linspace(0.0, 0.9, 10)
    frame_all = pd.DataFrame({
        'row_idx': [0, 1, 2, 3, 4, 5],
        'stratum': [0, 0, 1, 1, 2, 2],
    })
    frame_after = pd.DataFrame({
        'row_idx': [1, 3, 5],
        'stratum': [0, 1, 2],
    })
    selected = air_rescue_select(p0, p0_pct, frame_all, frame_after, np.array([2]), {2: 1}, 1)
    assert selected.tolist() == [3]







def test_b10_safe_shares_audit_sampling_key_with_b10_default():
    assert audit_sampling_key('B10_p1_audit_safe') == 'B10_p1_audit_rule'
    assert audit_sampling_key('B10_p1_reentry_safe') == 'B10_p1_audit_rule'
    assert audit_sampling_key('B10_p1_reentry_throttle') == 'B10_p1_audit_rule'
    assert audit_sampling_key('AIR_p1_rescue_heuristic') == 'AIR_p1_rescue_heuristic'

def test_audit_evidence_gate_waits_then_closes_only_on_zero_hits():
    assert audit_evidence_gate_open(0, 0, 10) is True
    assert audit_evidence_gate_open(9, 0, 10) is True
    assert audit_evidence_gate_open(10, 0, 10) is False
    assert audit_evidence_gate_open(10, 1, 10) is True
    assert audit_evidence_gate_open(100, 0, 0) is True

def test_adaptive_reentry_query_count_contract():
    assert adaptive_reentry_query_count(13, 10, {1: 0, 2: 0, 3: 0}, 0.20) == 1
    assert adaptive_reentry_query_count(13, 10, {1: 1, 2: 0}, 0.20) == 5
    assert adaptive_reentry_query_count(13, 0, {1: 1}, 0.20) == 0

def test_throttled_reentry_query_count_keeps_small_floor():
    assert throttled_reentry_query_count(31, 10, 0.10) == 3
    assert throttled_reentry_query_count(31, 10, 0.05) == 1
    assert throttled_reentry_query_count(31, 10, 0.0) == 0
    assert throttled_reentry_query_count(31, 0, 0.10) == 0



def test_chemprop_valid_smiles_mask_filters_unparseable_smiles():
    mask = chemprop_valid_smiles_mask(['c1ccccc1', 'not_a_smiles'])
    assert mask.tolist() == [True, False]

def test_chemprop_class_balance_guard_disables_tiny_minority(tmp_path):
    import pandas as pd
    train_csv = tmp_path / 'train.csv'
    pd.DataFrame({'split': ['train'] * 129 + ['val'], 'y': [1] + [0] * 129}).to_csv(train_csv, index=False)
    ok, stats = chemprop_class_balance_is_safe(train_csv, batch_size=64)
    assert ok is False
    assert stats['train_split_positive'] == 1


def test_chemprop_class_balance_guard_enables_full_balanced_batch(tmp_path):
    import pandas as pd
    train_csv = tmp_path / 'train.csv'
    pd.DataFrame({'split': ['train'] * 100, 'y': [1] * 32 + [0] * 68}).to_csv(train_csv, index=False)
    ok, stats = chemprop_class_balance_is_safe(train_csv, batch_size=64)
    assert ok is True
    assert stats['minority_count'] == 32

def test_exploit_budget_backfills_unused_audit_quota():
    assert exploit_budget_after_audit(exploit_planned=26, audit_planned=7, audit_actual=5, q_remaining_after_audit=99, pool_size=48) == 28
    assert exploit_budget_after_audit(exploit_planned=26, audit_planned=7, audit_actual=5, q_remaining_after_audit=20, pool_size=48) == 20
    assert exploit_budget_after_audit(exploit_planned=26, audit_planned=7, audit_actual=5, q_remaining_after_audit=99, pool_size=10) == 10
