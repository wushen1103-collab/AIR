"""Prospective fixed-parameter Tversky hit-neighbor development baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'tables'
sys.path.insert(0,str(ROOT/'scripts'))
from e72_tdc_neighbor_baseline import fingerprints, initial, sha  # noqa: E402

PARAMS = {'T1':(1.0,0.25),'T2':(0.25,1.0),
          'T3':(0.5,0.5),'T4':(1.5,1.5)}


def acquire(fps: list, y: np.ndarray, queried: list[int], a: float, b: float):
    selected = np.zeros(len(y),dtype=bool)
    selected[queried] = True
    remaining = np.flatnonzero(~selected)
    candidate_fps = [fps[int(i)] for i in remaining]
    known = np.asarray(queried,dtype=np.int64)
    positives = known[y[known]==1]
    if not len(positives):
        raise RuntimeError('No acquired positive')
    scores = np.zeros(len(remaining),dtype=np.float32)
    for ref in positives:
        similarity = np.asarray(DataStructs.BulkTverskySimilarity(
            fps[int(ref)],candidate_fps,a,b),dtype=np.float32)
        np.maximum(scores,similarity,out=scores)
    choose = remaining[np.argsort(-scores,kind='stable')[:64]]
    if len(choose)!=64 or selected[choose].any():
        raise RuntimeError('Invalid Tversky batch')
    return choose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--aid',choices=('1798','463087','485290'),required=True)
    parser.add_argument('--seed',type=int,choices=range(47001,47006),required=True)
    parser.add_argument('--variant',choices=tuple(PARAMS),required=True)
    parser.add_argument('--inject',choices=('none','crash','sleep'),default='none')
    args = parser.parse_args()
    base = TABLES/f'e69_tdc_dev_{args.aid}'
    pool = json.loads(base.with_name(base.name+'_manifest.json').read_text())
    screen_path = base.with_name(base.name+'_screen.csv')
    if pool['status']!='frozen' or sha(screen_path)!=pool['screen_sha256']:
        raise RuntimeError('E69 source mismatch')
    if args.seed not in [s['seed'] for s in pool['seeds']]:
        raise RuntimeError('Unfrozen seed')
    prefix = f'e76_tdc_{args.aid}_{args.variant}_seed{args.seed}'
    output = TABLES/f'{prefix}_hits.csv'
    query = TABLES/f'{prefix}_query.csv'
    marker = TABLES/f'{prefix}_manifest.json'
    if marker.exists():
        info = json.loads(marker.read_text())
        if (info['status']!='complete' or info['source_sha256']!=pool['screen_sha256']
                or sha(output)!=info['hits_sha256'] or sha(query)!=info['query_sha256']):
            raise RuntimeError('Completed E76 output changed')
        print(json.dumps({'status':'skip_verified','aid':args.aid,
                          'variant':args.variant,'seed':args.seed,
                          'final_hits':info['final_hits']}),flush=True)
        return
    if output.exists() or query.exists():
        raise RuntimeError('Partial E76 output retained')
    frame = pd.read_csv(screen_path)
    smiles = frame.smiles.astype(str).tolist()
    y = frame.y.to_numpy(dtype=np.int8)
    packed = fingerprints(args.aid,smiles,pool['screen_sha256'],'none')
    fps = [DataStructs.CreateFromBinaryText(bytes(row)) for row in packed]
    init = initial(y,args.seed)
    expected = next(s for s in pool['seeds'] if s['seed']==args.seed)
    if hashlib.sha256(init.tobytes()).hexdigest()!=expected['initial_index_sha256']:
        raise RuntimeError('Common initial indices changed')
    queried = init.astype(int).tolist()
    init_hits = int(y[init].sum())
    rows = [{'budget':64,'total_hits':init_hits,'new_hits':0}]
    a,b = PARAMS[args.variant]
    injected = False
    for budget in (128,192,256,320,384):
        checkpoint = TABLES/f'{prefix}_q{budget}.json'
        if checkpoint.exists():
            info = json.loads(checkpoint.read_text())
            if (info['source_sha256']!=pool['screen_sha256']
                    or info['budget']!=budget or info['variant']!=args.variant
                    or info['query_indices'][:len(queried)]!=queried
                    or len(info['query_indices'])!=budget
                    or len(set(info['query_indices']))!=budget):
                raise RuntimeError('E76 checkpoint mismatch')
            queried = info['query_indices']
            if info['total_hits']!=int(y[queried].sum()):
                raise RuntimeError('E76 checkpoint label count mismatch')
            print(json.dumps({'status':'resume_verified','budget':budget}),flush=True)
        else:
            choose = acquire(fps,y,queried,a,b)
            queried.extend(choose.astype(int).tolist())
            info = {'protocol':'E76','source_sha256':pool['screen_sha256'],
                    'aid':args.aid,'seed':args.seed,'variant':args.variant,
                    'params':[a,b],'budget':budget,'query_indices':queried,
                    'total_hits':int(y[queried].sum())}
            stage = Path(str(checkpoint)+f'.stage_pid{os.getpid()}')
            stage.write_text(json.dumps(info)+'\n')
            if checkpoint.exists():
                raise RuntimeError('Concurrent E76 checkpoint creation')
            stage.replace(checkpoint)
            print(json.dumps({'status':'round_checkpoint','budget':budget,
                              'total_hits':info['total_hits']}),flush=True)
            if not injected:
                injected = True
                if args.inject=='crash':
                    raise SystemExit(86)
                if args.inject=='sleep':
                    time.sleep(30)
        total = int(y[queried].sum())
        rows.append({'budget':budget,'total_hits':total,'new_hits':total-init_hits})
    stage_out = Path(str(output)+f'.stage_pid{os.getpid()}')
    stage_query = Path(str(query)+f'.stage_pid{os.getpid()}')
    stage_marker = Path(str(marker)+f'.stage_pid{os.getpid()}')
    with stage_out.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=('budget','total_hits','new_hits'))
        writer.writeheader();writer.writerows(rows)
    with stage_query.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=('order','index','smiles','y'))
        writer.writeheader()
        for order,index in enumerate(queried,start=1):
            writer.writerow({'order':order,'index':index,
                             'smiles':smiles[index],'y':int(y[index])})
    info = {'status':'complete','protocol':'E76','aid':args.aid,'seed':args.seed,
            'variant':args.variant,'params':[a,b],
            'source_sha256':pool['screen_sha256'],
            'hits_sha256':sha(stage_out),'query_sha256':sha(stage_query),
            'initial_hits':init_hits,'final_hits':rows[-1]['total_hits'],
            'final_new_hits':rows[-1]['new_hits']}
    stage_marker.write_text(json.dumps(info,indent=2)+'\n')
    stage_out.replace(output);stage_query.replace(query);stage_marker.replace(marker)
    print(json.dumps({'status':'complete','aid':args.aid,'seed':args.seed,
                      'variant':args.variant,'final_hits':info['final_hits']}),flush=True)


if __name__=='__main__':
    main()
