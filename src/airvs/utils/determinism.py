from __future__ import annotations
import hashlib
from typing import Any
import numpy as np

def stable_sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def stable_int(text: str, bits: int = 64) -> int:
    if bits <= 0 or bits > 256:
        raise ValueError('bits must be in 1..256')
    digest = stable_sha256(text)
    hex_chars = (bits + 3) // 4
    return int(digest[:hex_chars], 16) & ((1 << bits) - 1)

def make_rng(seed: int, *parts: Any) -> np.random.Generator:
    joined = '::'.join([str(seed), *map(str, parts)])
    return np.random.default_rng(stable_int(joined, bits=63))
