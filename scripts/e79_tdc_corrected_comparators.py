"""Run fixed E74-E, E76-T1, or E73-RF with exact E78 author starts."""

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
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'tables'
sys.path.insert(0,str(ROOT/'scripts'))
from e72_tdc_neighbor_baseline import fingerprints,sha  # noqa: E402
from e74_tdc_local_risk_candidates import acquire as local_risk_acquire  # noqa: E402
from e76_tdc_tversky import acquire as tversky_acquire  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--aid',choices=('1798','463087','485290'),required=True)
    parser.add_argument('--seed',type=int,choices=range(47001,47006),required=True)
    parser.add_argument('--method',choices=('E','T1','RF'),required=True)
    parser.add_argument('--inject',choices=('none','crash','sleep'),default='none')
    args = parser.parse_args()
    base = TABLES/f'e69_tdc_dev_{args.aid}'
    pool = json.loads(base.with_name(base.name+'_manifest.json').read_text())
    screen_path = base.with_name(base.name+'_screen.csv')
    starts = json.loads((TABLES/f'e78_tdc_dev_{args.aid}_author_starts.json').read_text())
    if (pool['status']!='frozen' or sha(screen_path)!=pool['screen_sha256']
            or starts['status']!='frozen' or starts['screen_sha256']!=pool['screen_sha256']):
        raise RuntimeError('E69/E78 source changed')
    record = next(s for s in starts['seeds'] if s['seed']==args.seed)
    initial = np.asarray(record['indices'],dtype=np.int64)
    if (len(initial)!=64 or len(set(initial.tolist()))!=64
            or hashlib.sha256(initial.tobytes()).hexdigest()!=record['index_sha256']):
        raise RuntimeError('E78 author initial invalid')
    prefix = f'e79_tdc_{args.aid}_seed{args.seed}_{args.method}'
    output = TABLES/f'{prefix}_hits.csv'
    query = TABLES/f'{prefix}_query.csv'
    marker = TABLES/f'{prefix}_manifest.json'
    if marker.exists():
        info = json.loads(marker.read_text())
        if (info['status']!='complete' or info['source_sha256']!=pool['screen_sha256']
                or info['start_hash']!=record['index_sha256']
                or sha(output)!=info['hits_sha256'] or sha(query)!=info['query_sha256']):
            raise RuntimeError('Completed E79 output changed')
        print(json.dumps({'status':'skip_verified','aid':args.aid,
                          'seed':args.seed,'method':args.method,
                          'final_hits':info['final_hits']}),flush=True)
        return
    if output.exists() or query.exists():
        raise RuntimeError('Partial E79 output retained')
    frame = pd.read_csv(screen_path)
    smiles = frame.smiles.astype(str).tolist()
    y = frame.y.to_numpy(dtype=np.int8)
    if int(y[initial].sum())!=record['initial_hits']:
        raise RuntimeError('E78 initial label count changed')
    packed = fingerprints(args.aid,smiles,pool['screen_sha256'],'none')
    if args.method=='RF':
        X = np.unpackbits(packed,axis=1).astype(np.uint8,copy=False)
        if X.shape!=(len(y),2048):
            raise RuntimeError('RF feature shape mismatch')
        fps = None
    else:
        fps = [DataStructs.CreateFromBinaryText(bytes(row)) for row in packed]
        X = None
    queried = initial.astype(int).tolist()
    init_hits = int(y[initial].sum())
    rows = [{'budget':64,'total_hits':init_hits,'new_hits':0}]
    injected = False
    for round_no,budget in enumerate((128,192,256,320,384),start=1):
        checkpoint = TABLES/f'{prefix}_q{budget}.json'
        if checkpoint.exists():
            info = json.loads(checkpoint.read_text())
            if (info['source_sha256']!=pool['screen_sha256']
                    or info['start_hash']!=record['index_sha256']
                    or info['method']!=args.method or info['budget']!=budget
                    or info['query_indices'][:len(queried)]!=queried
                    or len(info['query_indices'])!=budget
                    or len(set(info['query_indices']))!=budget):
                raise RuntimeError('E79 checkpoint mismatch')
            queried = info['query_indices']
            if info['total_hits']!=int(y[queried].sum()):
                raise RuntimeError('E79 checkpoint label count changed')
            print(json.dumps({'status':'resume_verified','budget':budget}),flush=True)
        else:
            if args.method=='E':
                choose = local_risk_acquire(fps,y,queried,0.15,0.10)
            elif args.method=='T1':
                choose = tversky_acquire(fps,y,queried,1.0,0.25)
            else:
                selected = np.zeros(len(y),dtype=bool)
                selected[queried] = True
                remaining = np.flatnonzero(~selected)
                model = RandomForestClassifier(
                    n_estimators=256,max_features='sqrt',min_samples_leaf=1,
                    class_weight='balanced_subsample',n_jobs=4,
                    random_state=args.seed+round_no)
                model.fit(X[np.asarray(queried)],y[np.asarray(queried)])
                if model.classes_.tolist()!=[0,1]:
                    raise RuntimeError('RF training lost a class')
                probabilities = model.predict_proba(X[remaining])[:,1]
                choose = remaining[np.argsort(-probabilities,kind='stable')[:64]]
            if len(choose)!=64 or len(set(choose.tolist()))!=64:
                raise RuntimeError('Invalid E79 batch')
            queried.extend(choose.astype(int).tolist())
            info = {'protocol':'E79','source_sha256':pool['screen_sha256'],
                    'start_hash':record['index_sha256'],'method':args.method,
                    'budget':budget,'query_indices':queried,
                    'total_hits':int(y[queried].sum())}
            stage = Path(str(checkpoint)+f'.stage_pid{os.getpid()}')
            stage.write_text(json.dumps(info)+'\n')
            if checkpoint.exists():
                raise RuntimeError('Concurrent E79 checkpoint appeared')
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
    info = {'status':'complete','protocol':'E79','aid':args.aid,'seed':args.seed,
            'method':args.method,'source_sha256':pool['screen_sha256'],
            'start_hash':record['index_sha256'],
            'hits_sha256':sha(stage_out),'query_sha256':sha(stage_query),
            'initial_hits':init_hits,'final_hits':rows[-1]['total_hits'],
            'final_new_hits':rows[-1]['new_hits']}
    stage_marker.write_text(json.dumps(info,indent=2)+'\n')
    stage_out.replace(output);stage_query.replace(query);stage_marker.replace(marker)
    print(json.dumps({'status':'complete','aid':args.aid,'seed':args.seed,
                      'method':args.method,'final_hits':info['final_hits']}),flush=True)


if __name__=='__main__':
    main()
