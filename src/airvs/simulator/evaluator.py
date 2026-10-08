from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Sequence

class LeakageError(RuntimeError):
    pass

class BudgetError(RuntimeError):
    pass

@dataclass
class BudgetLedger:
    q_budget: int
    b1_budget: int
    q_spent: int = 0
    b1_spent: int = 0
    queried_ids: set[str] = field(default_factory=set)
    expensive_ids: set[str] = field(default_factory=set)

    def spend_q(self, ids: Sequence[str]) -> None:
        unique = list(dict.fromkeys(map(str, ids)))
        if len(unique) != len(ids):
            raise BudgetError('duplicate ids inside one query batch')
        if self.queried_ids.intersection(unique):
            raise BudgetError('id queried more than once')
        if self.q_spent + len(unique) > self.q_budget:
            raise BudgetError('Q budget exceeded')
        self.queried_ids.update(unique)
        self.q_spent += len(unique)

    def spend_b1(self, ids: Sequence[str], allow_repeat: bool = False) -> None:
        unique = list(dict.fromkeys(map(str, ids)))
        if len(unique) != len(ids):
            raise BudgetError('duplicate ids inside one expensive batch')
        if not allow_repeat and self.expensive_ids.intersection(unique):
            raise BudgetError('expensive score repeated without explicit budget mode')
        if self.b1_spent + len(unique) > self.b1_budget:
            raise BudgetError('B1 budget exceeded')
        self.expensive_ids.update(unique)
        self.b1_spent += len(unique)

class SpyEvaluator:
    def __init__(self, labels: Mapping[str, int], expensive_scores: Mapping[str, float] | None = None):
        self._labels = {str(k): int(v) for k, v in labels.items()}
        self._scores = None if expensive_scores is None else {str(k): float(v) for k, v in expensive_scores.items()}
        self._metric_unlocked = False

    def query_labels(self, ids: Sequence[str], ledger: BudgetLedger) -> dict[str, int]:
        ids = list(map(str, ids))
        ledger.spend_q(ids)
        missing = [i for i in ids if i not in self._labels]
        if missing:
            raise KeyError(f'unknown label ids: {missing[:5]}')
        return {i: self._labels[i] for i in ids}

    def query_expensive(self, ids: Sequence[str], ledger: BudgetLedger) -> dict[str, float]:
        if self._scores is None:
            raise LeakageError('expensive score table is not available')
        ids = list(map(str, ids))
        ledger.spend_b1(ids)
        missing = [i for i in ids if i not in self._scores]
        if missing:
            raise KeyError(f'unknown expensive ids: {missing[:5]}')
        return {i: self._scores[i] for i in ids}

    @property
    def total_active(self) -> int:
        if not self._metric_unlocked:
            raise LeakageError('total_active is hidden from policies; unlock only in metrics phase')
        return sum(self._labels.values())

    def unlock_metrics(self) -> None:
        self._metric_unlocked = True

    def lock_metrics(self) -> None:
        self._metric_unlocked = False
