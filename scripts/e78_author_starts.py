"""Freeze exact AC-Active/GLARE author starts over unchanged E69 pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'tables'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def author_start(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    hits = np.flatnonzero(y == 1)
    first = hits[rng.choice(len(hits), size=1, replace=False)]
    remainder = np.array([i for i in range(len(y)) if i not in first], dtype=np.int64)
    other = rng.choice(remainder, size=63)  # Author default: replace=True.
    return rng.permutation(np.concatenate((first, other))).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--aid', choices=('1798','463087','485290'), required=True)
    parser.add_argument('--inject', choices=('none','crash','sleep'), default='none')
    args = parser.parse_args()
    base = TABLES / f'e69_tdc_dev_{args.aid}'
    source = base.with_name(base.name+'_screen.csv')
    source_manifest = base.with_name(base.name+'_manifest.json')
    pool = json.loads(source_manifest.read_text())
    if pool['status']!='frozen' or sha(source)!=pool['screen_sha256']:
        raise RuntimeError('E69 source pool changed')
    target = TABLES / f'e78_tdc_dev_{args.aid}_author_starts.json'
    stage = TABLES / f'e78_tdc_dev_{args.aid}_author_starts.stage.json'
    if target.exists():
        info = json.loads(target.read_text())
        if (info['status']!='frozen' or info['aid']!=args.aid
                or info['screen_sha256']!=pool['screen_sha256']
                or len(info['seeds'])!=5):
            raise RuntimeError('E78 final manifest mismatch')
        print(json.dumps({'status':'skip_verified','aid':args.aid,
                          'all_unique':info['all_unique']}))
        return
    if not stage.exists():
        y = pd.read_csv(source,usecols=['y']).y.to_numpy(dtype=np.int8)
        seeds = []
        for seed in range(47001,47006):
            arr = author_start(y,seed)
            seeds.append({'seed':seed,'initial_hits':int(y[arr].sum()),
                          'unique_count':len(set(arr.tolist())),
                          'index_sha256':hashlib.sha256(arr.tobytes()).hexdigest(),
                          'indices':arr.tolist()})
        info = {'status':'frozen','protocol':'E78','aid':args.aid,
                'screen_sha256':pool['screen_sha256'],
                'E69_manifest_sha256':sha(source_manifest),
                'initialization':'exact author np.random.choice default replace=True',
                'all_unique':all(s['unique_count']==64 for s in seeds),
                'seeds':seeds}
        stage.write_text(json.dumps(info,indent=2)+'\n')
    else:
        info = json.loads(stage.read_text())
        if (info['aid']!=args.aid or info['screen_sha256']!=pool['screen_sha256']
                or info['E69_manifest_sha256']!=sha(source_manifest)):
            raise RuntimeError('E78 stage source mismatch')
    print(json.dumps({'status':'stage_verified','aid':args.aid,
                      'all_unique':info['all_unique']}),flush=True)
    if args.inject=='crash':
        raise SystemExit(86)
    if args.inject=='sleep':
        time.sleep(30)
    if not info['all_unique']:
        raise RuntimeError('Author start has duplicate indices; freeze an explicit repair')
    if target.exists():
        raise RuntimeError('Concurrent E78 final manifest appeared')
    stage.replace(target)
    print(json.dumps({'status':'frozen','aid':args.aid,
                      'initial_hits':[s['initial_hits'] for s in info['seeds']],
                      'hashes':[s['index_sha256'] for s in info['seeds']]}),flush=True)


if __name__=='__main__':
    main()
