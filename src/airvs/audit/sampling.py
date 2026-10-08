from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from airvs.utils.determinism import make_rng, stable_int

@dataclass(frozen=True)
class StratumAllocation:
    stratum: int
    size: int
    n_sample: int

def assign_equal_size_strata(frame: pd.DataFrame, score_col: str, smiles_col: str = 'canonical_smiles', n_strata: int = 5) -> pd.DataFrame:
    required = {score_col, smiles_col}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f'missing columns: {sorted(missing)}')
    if n_strata <= 0:
        raise ValueError('n_strata must be positive')
    out = frame.copy()
    out['_tie'] = out[smiles_col].astype(str).map(lambda s: stable_int(s, bits=64))
    out = out.sort_values([score_col, '_tie'], ascending=[False, True], kind='mergesort')
    n = len(out)
    if n == 0:
        out['stratum'] = pd.Series(dtype='int64')
        return out.drop(columns=['_tie'])
    out['stratum'] = np.floor(np.arange(n) * n_strata / n).astype(int).clip(0, n_strata - 1)
    return out.drop(columns=['_tie'])

def largest_remainder_allocation(sizes: dict[int, int], total: int, min_per_nonempty: int = 2) -> list[StratumAllocation]:
    if total < 0:
        raise ValueError('total must be non-negative')
    nonempty = {int(k): int(v) for k, v in sizes.items() if int(v) > 0}
    if not nonempty:
        return []
    if total == 0:
        return [StratumAllocation(k, v, 0) for k, v in sorted(nonempty.items())]
    min_needed = min_per_nonempty * len(nonempty)
    if total < min_needed:
        raise ValueError(f'total={total} cannot give {min_per_nonempty} to each of {len(nonempty)} nonempty strata')
    capped_min = {k: min(min_per_nonempty, v) for k, v in nonempty.items()}
    remaining_total = total - sum(capped_min.values())
    remaining_capacity = {k: max(0, v - capped_min[k]) for k, v in nonempty.items()}
    cap_sum = sum(remaining_capacity.values())
    if remaining_total > cap_sum:
        raise ValueError('sample total exceeds population size')
    if remaining_total == 0 or cap_sum == 0:
        return [StratumAllocation(k, nonempty[k], capped_min[k]) for k in sorted(nonempty)]
    exact = {k: remaining_total * cap / cap_sum for k, cap in remaining_capacity.items()}
    base = {k: min(remaining_capacity[k], int(np.floor(v))) for k, v in exact.items()}
    need = remaining_total - sum(base.values())
    order = sorted(nonempty, key=lambda k: (exact[k] - base[k], remaining_capacity[k], -k), reverse=True)
    extra = {k: 0 for k in nonempty}
    for k in order:
        if need <= 0:
            break
        if base[k] + extra[k] < remaining_capacity[k]:
            extra[k] += 1
            need -= 1
    if need != 0:
        raise RuntimeError('largest-remainder allocation failed')
    return [StratumAllocation(k, nonempty[k], capped_min[k] + base[k] + extra[k]) for k in sorted(nonempty)]

def stratified_sample(frame: pd.DataFrame, q: int, seed: int, stratum_col: str = 'stratum', id_col: str = 'molecule_id') -> pd.DataFrame:
    if q == 0:
        return frame.iloc[[]].copy()
    if stratum_col not in frame.columns or id_col not in frame.columns:
        raise KeyError('frame must contain stratum and molecule id columns')
    sizes = frame.groupby(stratum_col, observed=True).size().to_dict()
    allocation = largest_remainder_allocation(sizes, q)
    pieces = []
    for item in allocation:
        subset = frame[frame[stratum_col] == item.stratum]
        if item.n_sample == 0:
            continue
        rng = make_rng(seed, 'audit', item.stratum, len(subset))
        idx = rng.choice(subset.index.to_numpy(), size=item.n_sample, replace=False)
        pieces.append(frame.loc[idx].copy())
    sampled = pd.concat(pieces, axis=0).sort_values([stratum_col, id_col], kind='mergesort') if pieces else frame.iloc[[]].copy()
    if sampled[id_col].duplicated().any():
        raise RuntimeError('duplicate audited molecule id')
    return sampled
