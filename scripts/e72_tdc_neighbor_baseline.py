"""Prospective maximum-positive-Tanimoto baseline on the frozen E69 TDC pool."""

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
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'tables'
CHUNK = 5000
RDLogger.DisableLog('rdApp.warning')


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


def fingerprints(aid: str, smiles: list[str], source_sha: str,
                 inject: str) -> np.ndarray:
    folder = ROOT / 'data/processed/tdc_e72' / aid
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / 'source.json'
    expected = {'protocol': 'E72', 'screen_sha256': source_sha,
                'n': len(smiles), 'radius': 2, 'bits': 2048, 'chirality': True}
    if source.exists():
        if json.loads(source.read_text()) != expected:
            raise RuntimeError('Fingerprint cache source mismatch')
    else:
        source.write_text(json.dumps(expected, indent=2) + '\n')
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True)
    chunks = []
    newly_built = False
    for start in range(0, len(smiles), CHUNK):
        end = min(start + CHUNK, len(smiles))
        number = start // CHUNK
        chunk_file = folder / f'fp_{number:04d}.npy'
        marker = folder / f'fp_{number:04d}.json'
        if marker.exists():
            info = json.loads(marker.read_text())
            if not chunk_file.exists() or info != {
                    'start': start, 'end': end, 'sha256': sha(chunk_file)}:
                raise RuntimeError(f'Fingerprint chunk changed: {number}')
        else:
            if chunk_file.exists():
                raise RuntimeError('Orphan fingerprint chunk without marker')
            packed = np.empty((end-start, 256), dtype=np.uint8)
            for offset, smi in enumerate(smiles[start:end]):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise RuntimeError('E67 SMILES became invalid')
                bits = DataStructs.BitVectToBinaryText(generator.GetFingerprint(mol))
                packed[offset] = np.frombuffer(bits, dtype=np.uint8)
            stage = folder / f'fp_{number:04d}.stage_pid{os.getpid()}.npy'
            np.save(stage, packed)
            stage.replace(chunk_file)
            marker.write_text(json.dumps({'start': start, 'end': end,
                                          'sha256': sha(chunk_file)}) + '\n')
            if not newly_built:
                newly_built = True
                print(json.dumps({'status': 'chunk_checkpoint', 'end': end}), flush=True)
                if inject == 'crash':
                    raise SystemExit(86)
                if inject == 'sleep':
                    time.sleep(30)
        chunks.append(np.load(chunk_file, allow_pickle=False))
    return np.concatenate(chunks, axis=0)


def run_seed(fps: list, y: np.ndarray, seed: int) -> tuple[list[dict], np.ndarray]:
    indices = initial(y, seed)
    selected = np.zeros(len(y), dtype=bool)
    selected[indices] = True
    queried = list(indices.astype(int))
    initial_hits = int(y[indices].sum())
    rows = [{'budget': 64, 'total_hits': initial_hits, 'new_hits': 0}]
    for budget in (128, 192, 256, 320, 384):
        remaining = np.flatnonzero(~selected)
        candidates = [fps[int(i)] for i in remaining]
        positives = np.flatnonzero(selected & (y == 1))
        if len(positives) == 0:
            raise RuntimeError('Author initializer did not supply an active')
        scores = np.zeros(len(remaining), dtype=np.float32)
        for ref in positives:
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(
                fps[int(ref)], candidates), dtype=np.float32)
            np.maximum(scores, sims, out=scores)
        choose = remaining[np.argsort(-scores, kind='stable')[:64]]
        if len(choose) != 64 or selected[choose].any():
            raise RuntimeError('Acquisition duplicate or short batch')
        selected[choose] = True
        queried.extend(choose.astype(int))
        total = int(y[selected].sum())
        rows.append({'budget': budget, 'total_hits': total,
                     'new_hits': total - initial_hits})
    if len(set(queried)) != 384:
        raise RuntimeError('Duplicate query in complete trajectory')
    return rows, np.asarray(queried, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--aid', choices=('1798', '463087', '485290'), required=True)
    parser.add_argument('--seed', type=int, choices=range(47001,47006), required=True)
    parser.add_argument('--inject', choices=('none', 'crash', 'sleep'), default='none')
    args = parser.parse_args()
    base = TABLES / f'e69_tdc_dev_{args.aid}'
    pool = json.loads(base.with_name(base.name + '_manifest.json').read_text())
    screen_path = base.with_name(base.name + '_screen.csv')
    if pool['status'] != 'frozen' or sha(screen_path) != pool['screen_sha256']:
        raise RuntimeError('E69 source mismatch')
    if args.seed not in [x['seed'] for x in pool['seeds']]:
        raise RuntimeError('Seed not frozen')
    output = TABLES / f'e72_tdc_{args.aid}_seed{args.seed}_neighbor.csv'
    sequence = TABLES / f'e72_tdc_{args.aid}_seed{args.seed}_query.csv'
    marker = TABLES / f'e72_tdc_{args.aid}_seed{args.seed}_manifest.json'
    if marker.exists():
        info = json.loads(marker.read_text())
        if (info['status'] != 'complete' or info['source_sha256'] != pool['screen_sha256']
                or sha(output) != info['trajectory_sha256']
                or sha(sequence) != info['query_sha256']):
            raise RuntimeError('Completed E72 result changed')
        print(json.dumps({'status': 'skip_verified', 'seed': args.seed,
                          'final_hits': info['final_hits']}))
        return
    if output.exists() or sequence.exists():
        raise RuntimeError('Partial result retained; inspect before retry')
    frame = pd.read_csv(screen_path)
    y = frame.y.to_numpy(dtype=np.int8)
    smiles = frame.smiles.astype(str).tolist()
    packed = fingerprints(args.aid, smiles, pool['screen_sha256'], args.inject)
    if packed.shape != (len(y), 256):
        raise RuntimeError('Fingerprint cache geometry mismatch')
    fps = [DataStructs.CreateFromBinaryText(bytes(row)) for row in packed]
    rows, queried = run_seed(fps, y, args.seed)
    expected_initial = next(x for x in pool['seeds'] if x['seed'] == args.seed)
    if (hashlib.sha256(queried[:64].tobytes()).hexdigest()
            != expected_initial['initial_index_sha256']):
        raise RuntimeError('Common initializer index hash mismatch')
    stage_out = Path(str(output) + f'.stage_pid{os.getpid()}')
    stage_seq = Path(str(sequence) + f'.stage_pid{os.getpid()}')
    with stage_out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('budget','total_hits','new_hits'))
        writer.writeheader(); writer.writerows(rows)
    with stage_seq.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('order','index','smiles','y'))
        writer.writeheader()
        for order, index in enumerate(queried, start=1):
            writer.writerow({'order': order, 'index': int(index),
                             'smiles': smiles[int(index)], 'y': int(y[int(index)])})
    info = {'status': 'complete', 'protocol': 'E72', 'aid': args.aid,
            'seed': args.seed, 'source_sha256': pool['screen_sha256'],
            'trajectory_sha256': sha(stage_out), 'query_sha256': sha(stage_seq),
            'initial_hits': rows[0]['total_hits'], 'final_hits': rows[-1]['total_hits'],
            'final_new_hits': rows[-1]['new_hits'],
            'method': 'max-positive-Tanimoto, Morgan radius2 2048 chirality-aware'}
    stage_marker = Path(str(marker) + f'.stage_pid{os.getpid()}')
    stage_marker.write_text(json.dumps(info, indent=2) + '\n')
    stage_out.replace(output)
    stage_seq.replace(sequence)
    stage_marker.replace(marker)
    print(json.dumps({'status': 'complete', 'aid': args.aid, 'seed': args.seed,
                      'initial_hits': rows[0]['total_hits'],
                      'final_hits': rows[-1]['total_hits']}), flush=True)


if __name__ == '__main__':
    main()
