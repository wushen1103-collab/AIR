"""Checkpointed prospective E74 development-only local-risk acquisition grid."""

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
sys.path.insert(0, str(ROOT / 'scripts'))
from e72_tdc_neighbor_baseline import fingerprints, initial, sha  # noqa: E402


VARIANTS = {
    'A': (0.00, 0.00), 'B': (0.15, 0.00), 'C': (0.30, 0.00),
    'D': (0.00, 0.10), 'E': (0.15, 0.10), 'F': (0.30, 0.10),
}


def acquire(fps: list, y: np.ndarray, queried: list[int],
            negative_penalty: float, diversity_penalty: float) -> np.ndarray:
    selected = np.zeros(len(y), dtype=bool)
    selected[queried] = True
    remaining = np.flatnonzero(~selected)
    candidates = [fps[int(i)] for i in remaining]
    known = np.asarray(queried, dtype=np.int64)
    positives = known[y[known] == 1]
    negatives = known[y[known] == 0]
    if not len(positives):
        raise RuntimeError('No acquired positive despite author initialization')
    positive_score = np.zeros(len(remaining), dtype=np.float32)
    for ref in positives:
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(
            fps[int(ref)], candidates), dtype=np.float32)
        np.maximum(positive_score, sims, out=positive_score)
    shortlist_positions = np.argsort(-positive_score, kind='stable')[:4096]
    shortlist = np.sort(remaining[shortlist_positions])
    score_lookup = dict(zip(remaining[shortlist_positions].astype(int),
                            positive_score[shortlist_positions]))
    base = np.asarray([score_lookup[int(i)] for i in shortlist], dtype=np.float32)
    short_fps = [fps[int(i)] for i in shortlist]
    if negative_penalty:
        negative_score = np.zeros(len(shortlist), dtype=np.float32)
        for ref in negatives:
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(
                fps[int(ref)], short_fps), dtype=np.float32)
            np.maximum(negative_score, sims, out=negative_score)
        base -= negative_penalty * negative_score
    if not diversity_penalty:
        choose = shortlist[np.argsort(-base, kind='stable')[:64]]
    else:
        chosen_positions = []
        batch_similarity = np.zeros(len(shortlist), dtype=np.float32)
        occupied = np.zeros(len(shortlist), dtype=bool)
        for _ in range(64):
            adjusted = base - diversity_penalty * batch_similarity
            adjusted[occupied] = -np.inf
            position = int(np.argmax(adjusted))
            if occupied[position]:
                raise RuntimeError('Greedy batch repetition')
            chosen_positions.append(position)
            occupied[position] = True
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(
                short_fps[position], short_fps), dtype=np.float32)
            np.maximum(batch_similarity, sims, out=batch_similarity)
        choose = shortlist[np.asarray(chosen_positions, dtype=np.int64)]
    if len(choose) != 64 or len(set(choose.tolist())) != 64 or selected[choose].any():
        raise RuntimeError('Invalid E74 batch')
    return choose


def completed_or_none(aid: str, seed: int, variant: str, source_hash: str):
    prefix = f'e74_tdc_{aid}_seed{seed}_{variant}'
    result = TABLES / f'{prefix}_hits.csv'
    query = TABLES / f'{prefix}_query.csv'
    marker = TABLES / f'{prefix}_manifest.json'
    if not marker.exists():
        if result.exists() or query.exists():
            raise RuntimeError('Partial E74 result without marker')
        return None
    info = json.loads(marker.read_text())
    if (info['status'] != 'complete' or info['source_sha256'] != source_hash
            or sha(result) != info['hits_sha256']
            or sha(query) != info['query_sha256']):
        raise RuntimeError('Complete E74 result changed')
    return info


def save_final(aid: str, seed: int, variant: str, source_hash: str,
               queried: list[int], rows: list[dict], smiles: list[str],
               y: np.ndarray) -> dict:
    prefix = f'e74_tdc_{aid}_seed{seed}_{variant}'
    result = TABLES / f'{prefix}_hits.csv'
    query = TABLES / f'{prefix}_query.csv'
    marker = TABLES / f'{prefix}_manifest.json'
    result_stage = Path(str(result) + f'.stage_pid{os.getpid()}')
    query_stage = Path(str(query) + f'.stage_pid{os.getpid()}')
    marker_stage = Path(str(marker) + f'.stage_pid{os.getpid()}')
    with result_stage.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('budget','total_hits','new_hits'))
        writer.writeheader(); writer.writerows(rows)
    with query_stage.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('order','index','smiles','y'))
        writer.writeheader()
        for order, index in enumerate(queried, start=1):
            writer.writerow({'order':order,'index':index,
                             'smiles':smiles[index],'y':int(y[index])})
    info = {'status':'complete','protocol':'E74','aid':aid,'seed':seed,
            'variant':variant,'params':VARIANTS[variant],
            'source_sha256':source_hash,
            'hits_sha256':sha(result_stage),'query_sha256':sha(query_stage),
            'initial_hits':rows[0]['total_hits'],
            'final_hits':rows[-1]['total_hits'],
            'final_new_hits':rows[-1]['new_hits']}
    marker_stage.write_text(json.dumps(info, indent=2) + '\n')
    result_stage.replace(result); query_stage.replace(query); marker_stage.replace(marker)
    return info


def verify_A(aid: str, seed: int) -> None:
    reference = TABLES / f'e72_tdc_{aid}_seed{seed}_query.csv'
    candidate = TABLES / f'e74_tdc_{aid}_seed{seed}_A_query.csv'
    if not reference.exists():
        raise RuntimeError('Independent E72 comparator missing')
    with reference.open(newline='') as handle:
        ref = [int(row['index']) for row in csv.DictReader(handle)]
    with candidate.open(newline='') as handle:
        got = [int(row['index']) for row in csv.DictReader(handle)]
    if ref != got:
        raise RuntimeError('Zero-penalty E74 variant does not reproduce E72')
    print(json.dumps({'status':'A_equals_E72','aid':aid,'seed':seed,
                      'query_count':len(got)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--aid', choices=('1798','463087','485290'), required=True)
    parser.add_argument('--seed', type=int, choices=range(47001,47006), required=True)
    parser.add_argument('--inject', choices=('none','crash','sleep'), default='none')
    args = parser.parse_args()
    base = TABLES / f'e69_tdc_dev_{args.aid}'
    pool = json.loads(base.with_name(base.name+'_manifest.json').read_text())
    screen_path = base.with_name(base.name+'_screen.csv')
    if pool['status'] != 'frozen' or sha(screen_path) != pool['screen_sha256']:
        raise RuntimeError('E69 source changed')
    if args.seed not in [s['seed'] for s in pool['seeds']]:
        raise RuntimeError('Unfrozen seed')
    frame = pd.read_csv(screen_path)
    smiles = frame.smiles.astype(str).tolist()
    y = frame.y.to_numpy(dtype=np.int8)
    packed = fingerprints(args.aid, smiles, pool['screen_sha256'], 'none')
    fps = [DataStructs.CreateFromBinaryText(bytes(row)) for row in packed]
    start = initial(y,args.seed)
    expected = next(x for x in pool['seeds'] if x['seed']==args.seed)
    if hashlib.sha256(start.tobytes()).hexdigest() != expected['initial_index_sha256']:
        raise RuntimeError('Initial index hash changed')
    injected = False
    for variant, (negative_penalty, diversity_penalty) in VARIANTS.items():
        completed = completed_or_none(args.aid,args.seed,variant,pool['screen_sha256'])
        if completed:
            print(json.dumps({'status':'skip_verified','variant':variant,
                              'final_hits':completed['final_hits']}), flush=True)
            if variant == 'A':
                verify_A(args.aid,args.seed)
            continue
        queried = start.astype(int).tolist()
        init_hits = int(y[start].sum())
        rows = [{'budget':64,'total_hits':init_hits,'new_hits':0}]
        for budget in (128,192,256,320,384):
            checkpoint = TABLES / f'e74_tdc_{args.aid}_seed{args.seed}_{variant}_q{budget}.json'
            if checkpoint.exists():
                info = json.loads(checkpoint.read_text())
                if (info['source_sha256'] != pool['screen_sha256']
                        or info['variant'] != variant or info['budget'] != budget
                        or info['query_indices'][:len(queried)] != queried
                        or len(info['query_indices']) != budget
                        or len(set(info['query_indices'])) != budget):
                    raise RuntimeError('E74 checkpoint mismatch')
                queried = info['query_indices']
                if info['total_hits'] != int(y[queried].sum()):
                    raise RuntimeError('E74 checkpoint label count mismatch')
                print(json.dumps({'status':'resume_verified','variant':variant,
                                  'budget':budget}), flush=True)
            else:
                choose = acquire(fps,y,queried,negative_penalty,diversity_penalty)
                queried.extend(choose.astype(int).tolist())
                info = {'protocol':'E74','source_sha256':pool['screen_sha256'],
                        'aid':args.aid,'seed':args.seed,'variant':variant,
                        'budget':budget,'query_indices':queried,
                        'total_hits':int(y[queried].sum())}
                stage = Path(str(checkpoint)+f'.stage_pid{os.getpid()}')
                stage.write_text(json.dumps(info) + '\n')
                if checkpoint.exists():
                    raise RuntimeError('Concurrent checkpoint creation')
                stage.replace(checkpoint)
                print(json.dumps({'status':'round_checkpoint','variant':variant,
                                  'budget':budget,'total_hits':info['total_hits']}), flush=True)
                if not injected:
                    injected = True
                    if args.inject == 'crash':
                        raise SystemExit(86)
                    if args.inject == 'sleep':
                        time.sleep(30)
            total = int(y[queried].sum())
            rows.append({'budget':budget,'total_hits':total,'new_hits':total-init_hits})
        if len(queried)!=384 or len(set(queried))!=384:
            raise RuntimeError('Invalid complete E74 sequence')
        result = save_final(args.aid,args.seed,variant,pool['screen_sha256'],
                            queried,rows,smiles,y)
        print(json.dumps({'status':'variant_complete','variant':variant,
                          'final_hits':result['final_hits']}), flush=True)
        if variant == 'A':
            verify_A(args.aid,args.seed)


if __name__ == '__main__':
    main()
