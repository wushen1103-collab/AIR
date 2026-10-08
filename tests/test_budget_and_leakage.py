import pytest
from airvs.simulator.evaluator import BudgetError, BudgetLedger, LeakageError, SpyEvaluator

def test_budget_counts_q_and_rejects_requery():
    ledger = BudgetLedger(q_budget=2, b1_budget=3)
    ev = SpyEvaluator({'a': 1, 'b': 0, 'c': 1})
    assert ev.query_labels(['a', 'b'], ledger) == {'a': 1, 'b': 0}
    with pytest.raises(BudgetError):
        ev.query_labels(['a'], ledger)

def test_budget_rejects_expensive_overrun():
    ledger = BudgetLedger(q_budget=10, b1_budget=1)
    ev = SpyEvaluator({'a': 1, 'b': 0}, {'a': 0.1, 'b': 0.2})
    ev.query_expensive(['a'], ledger)
    with pytest.raises(BudgetError):
        ev.query_expensive(['b'], ledger)

def test_total_active_hidden_until_metrics_phase():
    ev = SpyEvaluator({'a': 1, 'b': 0})
    with pytest.raises(LeakageError):
        _ = ev.total_active
    ev.unlock_metrics()
    assert ev.total_active == 1
