from __future__ import annotations
from collections.abc import Mapping

def hits(discovered_labels: Mapping[str, int]) -> int:
    return int(sum(int(v) for v in discovered_labels.values()))

def recall(discovered_labels: Mapping[str, int], total_active: int) -> float:
    if total_active < 0:
        raise ValueError('total_active must be non-negative')
    if total_active == 0:
        return 0.0
    return hits(discovered_labels) / total_active

def nef(discovered_labels: Mapping[str, int], q_spent: int, total_active: int, population: int) -> float:
    if q_spent <= 0 or population <= 0:
        raise ValueError('q_spent and population must be positive')
    prevalence = total_active / population
    if prevalence == 0:
        return 0.0
    return (hits(discovered_labels) / q_spent) / prevalence
