"""Freeze E68 development pools from verified E67 SQLite registries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'tables'
VALID_AIDS = ('1798', '463087', '485290')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def initial(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    positives = np.flatnonzero(y == 1)
    one = positives[rng.choice(len(positives), size=1, replace=False)]
    remaining = np.array([i for i in range(len(y)) if i not in one], dtype=np.int64)
    others = rng.choice(remaining, size=63, replace=False)
    return rng.permutation(np.concatenate((one, others))).astype(np.int64)


def verified_complete(aid: str, meta_path: Path, screen: Path, test: Path) -> dict | None:
    if not meta_path.exists():
        return None
    info = json.loads(meta_path.read_text())
    if (info.get('status') != 'frozen' or info.get('aid') != aid
            or not screen.is_file() or not test.is_file()
            or sha(screen) != info['screen_sha256']
            or sha(test) != info['test_sha256']):
        raise RuntimeError('Existing E69 pool or manifest mismatch')
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--aid', required=True, choices=VALID_AIDS)
    parser.add_argument('--inject-crash', action='store_true')
    parser.add_argument('--inject-sleep', action='store_true')
    args = parser.parse_args()
    base = TABLES / f'e69_tdc_dev_{args.aid}'
    screen_path = base.with_name(base.name + '_screen.csv')
    test_path = base.with_name(base.name + '_test.csv')
    meta_path = base.with_name(base.name + '_manifest.json')
    existing = verified_complete(args.aid, meta_path, screen_path, test_path)
    if existing:
        print(json.dumps({'status': 'skip_verified', 'aid': args.aid,
                          'screen': existing['screen_size']}))
        return
    audit_path = TABLES / f'e67_tdc_dev_{args.aid}_audit.json'
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'complete' or audit['aid'] != args.aid:
        raise RuntimeError('E67 audit not complete')
    db_path = TABLES / f'e67_tdc_dev_{args.aid}_audit.sqlite'
    if sha(db_path) != audit['database_sha256']:
        raise RuntimeError('E67 registry database changed')
    screen_stage = Path(str(screen_path) + '.stage')
    test_stage = Path(str(test_path) + '.stage')
    meta_stage = Path(str(meta_path) + '.stage')
    if not meta_stage.exists():
        if any(p.exists() for p in (screen_stage, test_stage, screen_path, test_path)):
            raise RuntimeError('Orphan stage/final pool without staged manifest')
        db = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        rows = db.execute('SELECT smiles, mask FROM mol WHERE mask IN (1,2) ORDER BY smiles').fetchall()
        db.close()
        if len(rows) != audit['retained_candidates']:
            raise RuntimeError('Retained pool count changed')
        frame = pd.DataFrame({'smiles': [row[0] for row in rows],
                              'y': [int(row[1] == 2) for row in rows]})
        if int(frame.y.sum()) != audit['retained_positives'] or frame.smiles.duplicated().any():
            raise RuntimeError('Registry content mismatch')
        frame = frame.sample(frac=1, random_state=42).reset_index(drop=True)
        screen, test = train_test_split(frame, test_size=2,
                                        stratify=frame.y.tolist(), random_state=42)
        y = screen.y.to_numpy(dtype=np.int8)
        seeds = []
        for seed in range(47001, 47006):
            indices = initial(y, seed)
            if len(set(indices)) != 64 or int(y[indices].sum()) < 1:
                raise RuntimeError(f'Invalid common initialization: {seed}')
            seeds.append({'seed': seed, 'initial_hits': int(y[indices].sum()),
                          'initial_index_sha256': hashlib.sha256(indices.tobytes()).hexdigest()})
        screen_stage.write_bytes(screen.to_csv(index=False, lineterminator='\n').encode())
        test_stage.write_bytes(test.to_csv(index=False, lineterminator='\n').encode())
        info = {
            'status': 'frozen', 'protocol': 'E68', 'aid': args.aid,
            'e67_audit_sha256': sha(audit_path), 'e67_registry_sha256': sha(db_path),
            'screen_size': len(screen), 'screen_positives': int(screen.y.sum()),
            'test_size': len(test), 'test_positives': int(test.y.sum()),
            'screen_sha256': sha(screen_stage), 'test_sha256': sha(test_stage),
            'screen_molecule_order_sha256': hashlib.sha256(
                '\n'.join(screen.smiles.astype(str)).encode()).hexdigest(),
            'seeds': seeds, 'budgets': [64,128,192,256,320,384],
            'initialization': 'AC-Active mode=a; one active plus 63 others',
        }
        meta_stage.write_text(json.dumps(info, indent=2, sort_keys=True) + '\n')
    else:
        info = json.loads(meta_stage.read_text())
        if info['aid'] != args.aid or info['e67_audit_sha256'] != sha(audit_path):
            raise RuntimeError('Staged manifest source mismatch')
    for stage, final, key in ((screen_stage, screen_path, 'screen_sha256'),
                               (test_stage, test_path, 'test_sha256')):
        selected = final if final.exists() else stage
        if not selected.exists() or sha(selected) != info[key]:
            raise RuntimeError('Staged/final pool file hash mismatch')
    print(json.dumps({'status': 'stage_verified', 'aid': args.aid,
                      'screen_size': info['screen_size']}), flush=True)
    if args.inject_crash:
        raise SystemExit(86)
    if args.inject_sleep:
        time.sleep(30)
    if not screen_path.exists():
        screen_stage.replace(screen_path)
    if not test_path.exists():
        test_stage.replace(test_path)
    if meta_path.exists():
        raise RuntimeError('Concurrent final manifest appeared')
    meta_stage.replace(meta_path)
    verified_complete(args.aid, meta_path, screen_path, test_path)
    print(json.dumps({'status': 'frozen', 'aid': args.aid,
                      'screen_size': info['screen_size'],
                      'screen_positives': info['screen_positives'],
                      'seeds': info['seeds']}), flush=True)


if __name__ == '__main__':
    main()
