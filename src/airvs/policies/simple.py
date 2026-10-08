from __future__ import annotations
import pandas as pd
from airvs.utils.determinism import make_rng, stable_int

def random_select(frame: pd.DataFrame, k: int, seed: int, id_col: str = 'molecule_id') -> list[str]:
    if k < 0:
        raise ValueError('k must be non-negative')
    if k > len(frame):
        raise ValueError('k exceeds pool size')
    rng = make_rng(seed, 'random_select', len(frame), k)
    ids = frame[id_col].astype(str).to_numpy()
    return list(rng.choice(ids, size=k, replace=False))

def topk_select(frame: pd.DataFrame, k: int, score_col: str, id_col: str = 'molecule_id') -> list[str]:
    if k < 0:
        raise ValueError('k must be non-negative')
    if k > len(frame):
        raise ValueError('k exceeds pool size')
    out = frame.copy()
    out['_tie'] = out[id_col].astype(str).map(lambda s: stable_int(s, bits=64))
    out = out.sort_values([score_col, '_tie'], ascending=[False, True], kind='mergesort')
    return list(out[id_col].astype(str).head(k))
