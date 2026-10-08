"""Round-checkpointed, prospective RF/ECFP baseline for E69 TDC pools."""

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
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'tables'
sys.path.insert(0, str(ROOT / 'scripts'))
from e72_tdc_neighbor_baseline import fingerprints, initial, sha  # noqa: E402


def write_new_json(path: Path, content: dict) -> None:
    if path.exists():
        raise RuntimeError(f'Immutable checkpoint already exists: {path}')
    stage = Path(str(path) + f'.stage_pid{os.getpid()}')
    stage.write_text(json.dumps(content, indent=2) + '\n')
    if path.exists():
        raise RuntimeError('Concurrent checkpoint creation')
    stage.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--aid', choices=('1798','463087','485290'), required=True)
    parser.add_argument('--seed', type=int, choices=range(47001,47006), required=True)
    parser.add_argument('--inject', choices=('none','crash','sleep'), default='none')
    args = parser.parse_args()
    base = TABLES / f'e69_tdc_dev_{args.aid}'
    pool = json.loads(base.with_name(base.name + '_manifest.json').read_text())
    screen_path = base.with_name(base.name + '_screen.csv')
    if pool['status'] != 'frozen' or sha(screen_path) != pool['screen_sha256']:
        raise RuntimeError('E69 pool changed')
    if args.seed not in [x['seed'] for x in pool['seeds']]:
        raise RuntimeError('Unfrozen seed')
    output = TABLES / f'e73_tdc_{args.aid}_seed{args.seed}_rf.csv'
    query = TABLES / f'e73_tdc_{args.aid}_seed{args.seed}_query.csv'
    marker = TABLES / f'e73_tdc_{args.aid}_seed{args.seed}_manifest.json'
    if marker.exists():
        info = json.loads(marker.read_text())
        if (info['status'] != 'complete' or info['source_sha256'] != pool['screen_sha256']
                or sha(output) != info['trajectory_sha256']
                or sha(query) != info['query_sha256']):
            raise RuntimeError('Complete RF result mismatch')
        print(json.dumps({'status': 'skip_verified', 'seed': args.seed,
                          'final_hits': info['final_hits']}))
        return
    if output.exists() or query.exists():
        raise RuntimeError('Partial RF output retained; inspect before retry')
    frame = pd.read_csv(screen_path)
    smiles = frame.smiles.astype(str).tolist()
    y = frame.y.to_numpy(dtype=np.int8)
    packed = fingerprints(args.aid, smiles, pool['screen_sha256'], 'none')
    X = np.unpackbits(packed, axis=1).astype(np.uint8, copy=False)
    if X.shape != (len(y),2048):
        raise RuntimeError('RF fingerprint geometry mismatch')
    init = initial(y, args.seed)
    target_hash = next(x for x in pool['seeds'] if x['seed']==args.seed)['initial_index_sha256']
    if hashlib.sha256(init.tobytes()).hexdigest() != target_hash:
        raise RuntimeError('Common initial indices changed')
    queried = init.astype(int).tolist()
    initial_hits = int(y[init].sum())
    rows = [{'budget':64,'total_hits':initial_hits,'new_hits':0}]
    injected = False
    for round_number, budget in enumerate((128,192,256,320,384), start=1):
        checkpoint = TABLES / f'e73_tdc_{args.aid}_seed{args.seed}_q{budget}.json'
        if checkpoint.exists():
            info = json.loads(checkpoint.read_text())
            if (info['source_sha256'] != pool['screen_sha256']
                    or info['seed'] != args.seed or info['budget'] != budget
                    or info['query_indices'][:len(queried)] != queried
                    or len(info['query_indices']) != budget
                    or len(set(info['query_indices'])) != budget):
                raise RuntimeError('RF checkpoint sequence mismatch')
            queried = info['query_indices']
            if info['total_hits'] != int(y[queried].sum()):
                raise RuntimeError('RF checkpoint label count mismatch')
            print(json.dumps({'status':'resume_verified','budget':budget}), flush=True)
        else:
            mask = np.zeros(len(y), dtype=bool)
            mask[queried] = True
            remaining = np.flatnonzero(~mask)
            model = RandomForestClassifier(
                n_estimators=256, max_features='sqrt', min_samples_leaf=1,
                class_weight='balanced_subsample', n_jobs=4,
                random_state=args.seed + round_number)
            model.fit(X[np.asarray(queried)], y[np.asarray(queried)])
            classes = model.classes_.tolist()
            if classes != [0,1]:
                raise RuntimeError('RF training lost a class')
            probability = model.predict_proba(X[remaining])[:,1]
            choose = remaining[np.argsort(-probability, kind='stable')[:64]]
            if len(choose) != 64 or len(set(choose.tolist())) != 64:
                raise RuntimeError('Invalid RF batch')
            queried.extend(choose.astype(int).tolist())
            info = {'protocol':'E73','source_sha256':pool['screen_sha256'],
                    'seed':args.seed,'budget':budget,'query_indices':queried,
                    'total_hits':int(y[queried].sum())}
            write_new_json(checkpoint, info)
            print(json.dumps({'status':'round_checkpoint','budget':budget,
                              'total_hits':info['total_hits']}), flush=True)
            if not injected:
                injected = True
                if args.inject == 'crash':
                    raise SystemExit(86)
                if args.inject == 'sleep':
                    time.sleep(30)
        total = int(y[queried].sum())
        rows.append({'budget':budget,'total_hits':total,'new_hits':total-initial_hits})
    if len(queried) != 384 or len(set(queried)) != 384:
        raise RuntimeError('Incomplete RF trajectory')
    stage_out = Path(str(output) + f'.stage_pid{os.getpid()}')
    stage_query = Path(str(query) + f'.stage_pid{os.getpid()}')
    with stage_out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('budget','total_hits','new_hits'))
        writer.writeheader(); writer.writerows(rows)
    with stage_query.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('order','index','smiles','y'))
        writer.writeheader()
        for order, index in enumerate(queried, start=1):
            writer.writerow({'order':order,'index':index,'smiles':smiles[index],
                             'y':int(y[index])})
    info = {'status':'complete','protocol':'E73','aid':args.aid,'seed':args.seed,
            'source_sha256':pool['screen_sha256'],
            'trajectory_sha256':sha(stage_out),'query_sha256':sha(stage_query),
            'initial_hits':initial_hits,'final_hits':rows[-1]['total_hits'],
            'final_new_hits':rows[-1]['new_hits']}
    stage_marker = Path(str(marker) + f'.stage_pid{os.getpid()}')
    stage_marker.write_text(json.dumps(info, indent=2) + '\n')
    stage_out.replace(output); stage_query.replace(query); stage_marker.replace(marker)
    print(json.dumps({'status':'complete','aid':args.aid,'seed':args.seed,
                      'final_hits':info['final_hits']}), flush=True)


if __name__ == '__main__':
    main()
