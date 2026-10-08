from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def fp_one(smiles: str) -> bytes:
    from rdkit import Chem, DataStructs, RDLogger
    RDLogger.DisableLog('rdApp.*')
    from rdkit.Chem import rdFingerprintGenerator
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return bytes(256)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fp = gen.GetFingerprint(mol)
    return DataStructs.BitVectToBinaryText(fp)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--assays', nargs='*', default=None)
    parser.add_argument('--workers', type=int, default=max(1, min(64, (os.cpu_count() or 32) - 30)))
    parser.add_argument('--chunksize', type=int, default=2000)
    args = parser.parse_args()

    outdir = ROOT / 'data' / 'processed' / 'fingerprints'
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ROOT / 'data' / 'processed' / 'lit_pcba_policy_view.parquet')
    assays = sorted(args.assays or df['assay'].unique().tolist())
    manifest = []
    for assay in assays:
        sub = df[df['assay'] == assay].sort_values('molecule_id', kind='mergesort').reset_index(drop=True)
        ids_path = outdir / f'{assay}_ids.parquet'
        fp_path = outdir / f'{assay}_morgan_r2_2048_chiral.packbits.npy'
        if fp_path.exists() and ids_path.exists():
            manifest.append({'assay': assay, 'n': len(sub), 'fingerprint_path': str(fp_path.relative_to(ROOT)), 'ids_path': str(ids_path.relative_to(ROOT)), 'sha256': sha256_file(fp_path), 'status': 'exists'})
            print(json.dumps(manifest[-1], ensure_ascii=False))
            continue
        smiles = sub['canonical_smiles'].astype(str).tolist()
        print(json.dumps({'assay': assay, 'n': len(smiles), 'workers': args.workers}, ensure_ascii=False), flush=True)
        with mp.Pool(processes=args.workers) as pool:
            fp_bytes = list(tqdm(pool.imap(fp_one, smiles, chunksize=args.chunksize), total=len(smiles)))
        packed = np.frombuffer(b''.join(fp_bytes), dtype=np.uint8).reshape(len(fp_bytes), 256)
        np.save(fp_path, packed)
        sub[['molecule_id', 'assay', 'canonical_smiles']].to_parquet(ids_path, index=False)
        manifest.append({'assay': assay, 'n': len(sub), 'fingerprint_path': str(fp_path.relative_to(ROOT)), 'ids_path': str(ids_path.relative_to(ROOT)), 'sha256': sha256_file(fp_path), 'status': 'built'})
        print(json.dumps(manifest[-1], ensure_ascii=False), flush=True)
    pd.DataFrame(manifest).to_csv(outdir / 'fingerprint_manifest.csv', index=False)


if __name__ == '__main__':
    main()
