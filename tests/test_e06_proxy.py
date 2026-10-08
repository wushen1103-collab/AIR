
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_e06_pair_bank_proxy import allocate_counts, split_budget
from run_e06_train_lambdarank_proxy import fold_split


def test_split_budget_exact_sum():
    parts = split_budget(187, 4)
    assert parts == [47, 47, 47, 46]
    assert sum(parts) == 187


def test_allocate_counts_respects_total_and_caps():
    sizes = {0: 100, 1: 50, 2: 1, 3: 20, 4: 10}
    counts = allocate_counts(sizes, 17, min_per_nonempty=2)
    assert sum(counts.values()) == 17
    assert counts[2] <= 1
    assert all(counts[k] <= sizes[k] for k in sizes)


def test_fold_split_excludes_test_and_validation():
    train, val, test = fold_split('F1')
    assert set(test) == {'ADRB2', 'OPRK1'}
    assert set(val) == {'ESR_ago', 'ESR_antago', 'PPARG', 'VDR'}
    assert not (set(train) & set(test))
    assert not (set(train) & set(val))
