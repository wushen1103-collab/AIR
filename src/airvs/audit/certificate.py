from __future__ import annotations
from dataclasses import dataclass
from scipy.stats import hypergeom

@dataclass(frozen=True)
class StratumAudit:
    population: int
    sample: int
    observed_active: int

@dataclass(frozen=True)
class StratumBound:
    population: int
    sample: int
    observed_active: int
    lower_active: int

def hypergeom_lower_bound(population: int, sample: int, observed_active: int, alpha: float = 0.05) -> int:
    if population < 0 or sample < 0 or observed_active < 0:
        raise ValueError('population, sample, and observed_active must be non-negative')
    if sample > population:
        raise ValueError('sample cannot exceed population')
    if observed_active > sample:
        raise ValueError('observed_active cannot exceed sample')
    if not (0.0 < alpha < 1.0):
        raise ValueError('alpha must be in (0,1)')
    if observed_active == 0 or population == 0:
        return 0
    for total_active in range(observed_active, population + 1):
        p_tail = hypergeom.sf(observed_active - 1, population, total_active, sample)
        if p_tail >= alpha:
            return total_active
    return population

def stratified_lower_bound(strata: list[StratumAudit], alpha: float = 0.05) -> tuple[int, list[StratumBound]]:
    if not strata:
        return 0, []
    alpha_local = alpha / len(strata)
    bounds = []
    for s in strata:
        lower = hypergeom_lower_bound(s.population, s.sample, s.observed_active, alpha_local)
        bounds.append(StratumBound(s.population, s.sample, s.observed_active, lower))
    return sum(b.lower_active for b in bounds), bounds
