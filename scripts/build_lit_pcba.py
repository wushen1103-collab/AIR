from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from airvs.data.registry import LIT_ASSAYS
from airvs.data.standardize import standardize_smiles

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / 'data' / 'raw' / 'lit_pcba' / 'extracted' / 'LIT-PCBA_full'
DIR_ALIASES = {
    'ESR_ago': 'ESR1_ago',
    'ESR_antago': 'ESR1_ant',
}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def molecule_id(assay: str, canonical_smiles: str) -> str:
    h = hashlib.sha256(f'LIT-PCBA::{assay}::{canonical_smiles}'.encode('utf-8')).hexdigest()
    return f'LIT_{assay}_{h[:20]}'


def iter_smi(path: Path, assay: str, y: int) -> Iterable[dict[str, object]]:
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                original_id = f'line{line_no}'
                smiles = parts[0]
            else:
                smiles = parts[0]
                original_id = parts[1]
            yield {
                'assay': assay,
                'source_file': path.name,
                'line_no': line_no,
                'original_id': str(original_id),
                'raw_smiles': smiles,
                'y': int(y),
            }


def standardize_record(row: dict[str, object]) -> dict[str, object]:
    std = standardize_smiles(str(row['raw_smiles']))
    out = dict(row)
    out['canonical_smiles'] = std.canonical_smiles
    out['inchikey'] = std.inchikey
    out['standardize_error'] = std.error
    if std.canonical_smiles:
        out['molecule_id'] = molecule_id(str(row['assay']), std.canonical_smiles)
    else:
        out['molecule_id'] = None
    return out


def collect_raw_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    assay_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    archive = ROOT / 'data' / 'raw' / 'lit_pcba' / 'LIT-PCBA_full.tar.gz'
    manifest_rows.append({
        'dataset': 'LIT-PCBA',
        'kind': 'archive',
        'path': str(archive.relative_to(ROOT)),
        'bytes': archive.stat().st_size,
        'sha256': sha256_file(archive),
    })
    for assay in LIT_ASSAYS:
        dirname = DIR_ALIASES.get(assay.name, assay.name)
        assay_dir = RAW_ROOT / dirname
        active_path = assay_dir / 'actives.smi'
        inactive_path = assay_dir / 'inactives.smi'
        if not active_path.exists() or not inactive_path.exists():
            raise FileNotFoundError(f'missing LIT files for {assay.name}: {assay_dir}')
        active_rows = list(iter_smi(active_path, assay.name, 1))
        inactive_rows = list(iter_smi(inactive_path, assay.name, 0))
        rows.extend(active_rows)
        rows.extend(inactive_rows)
        assay_rows.append({
            'assay': assay.name,
            'expected_active': assay.active,
            'expected_inactive': assay.inactive,
            'raw_active_rows': len(active_rows),
            'raw_inactive_rows': len(inactive_rows),
            'pdb': assay.pdb,
            'use': assay.use,
        })
        for p in [active_path, inactive_path]:
            manifest_rows.append({
                'dataset': 'LIT-PCBA',
                'kind': p.name,
                'assay': assay.name,
                'path': str(p.relative_to(ROOT)),
                'bytes': p.stat().st_size,
                'sha256': sha256_file(p),
            })
    return rows, assay_rows, manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=max(1, min(64, (os.cpu_count() or 32) - 30)))
    parser.add_argument('--chunksize', type=int, default=2000)
    parser.add_argument('--limit-assays', nargs='*', default=None)
    args = parser.parse_args()

    (ROOT / 'data' / 'processed').mkdir(parents=True, exist_ok=True)
    (ROOT / 'data' / 'secure').mkdir(parents=True, exist_ok=True)
    (ROOT / 'data' / 'interim').mkdir(parents=True, exist_ok=True)

    rows, assay_rows, manifest_rows = collect_raw_rows()
    if args.limit_assays:
        keep = set(args.limit_assays)
        rows = [r for r in rows if r['assay'] in keep]
        assay_rows = [r for r in assay_rows if r['assay'] in keep]
        manifest_rows = [r for r in manifest_rows if r.get('assay') in keep or r.get('kind') == 'archive']

    print(json.dumps({'raw_rows': len(rows), 'workers': args.workers, 'chunksize': args.chunksize}, ensure_ascii=False))
    with mp.Pool(processes=args.workers) as pool:
        clean_rows = list(tqdm(pool.imap_unordered(standardize_record, rows, chunksize=args.chunksize), total=len(rows)))

    df = pd.DataFrame(clean_rows)
    failures = df[df['standardize_error'].notna()].copy()
    valid = df[df['standardize_error'].isna()].copy()

    group_cols = ['assay', 'canonical_smiles']
    label_nunique = valid.groupby(group_cols)['y'].nunique().rename('label_nunique').reset_index()
    conflict_keys = label_nunique[label_nunique['label_nunique'] > 1][group_cols]
    if len(conflict_keys):
        conflict_marked = valid.merge(conflict_keys.assign(_conflict=1), on=group_cols, how='left')
        conflicts = conflict_marked[conflict_marked['_conflict'].eq(1)].drop(columns=['_conflict'])
        valid = conflict_marked[conflict_marked['_conflict'].isna()].drop(columns=['_conflict'])
    else:
        conflicts = valid.iloc[[]].copy()

    valid = valid.sort_values(['assay', 'canonical_smiles', 'y', 'original_id', 'line_no'], kind='mergesort')
    dedup = valid.drop_duplicates(group_cols, keep='first').copy()
    dedup['molecule_id'] = [molecule_id(a, s) for a, s in zip(dedup['assay'], dedup['canonical_smiles'])]

    molecules = dedup[['molecule_id', 'assay', 'original_id', 'canonical_smiles', 'inchikey', 'raw_smiles', 'source_file', 'line_no']].copy()
    labels = dedup[['molecule_id', 'assay', 'y']].copy()

    molecules.to_parquet(ROOT / 'data' / 'processed' / 'lit_pcba_molecules.parquet', index=False)
    molecules.drop(columns=['raw_smiles']).to_parquet(ROOT / 'data' / 'processed' / 'lit_pcba_policy_view.parquet', index=False)
    labels.to_parquet(ROOT / 'data' / 'secure' / 'lit_pcba_labels.secure.parquet', index=False)
    failures.to_csv(ROOT / 'data' / 'interim' / 'lit_pcba_standardization_failures.csv', index=False)
    conflicts.to_csv(ROOT / 'data' / 'processed' / 'lit_pcba_conflicts.csv', index=False)

    assays = pd.DataFrame(assay_rows)
    kept_counts = labels.groupby('assay')['y'].agg(kept_total='count', kept_active='sum').reset_index()
    fail_counts = failures.groupby('assay').size().rename('standardize_failures').reset_index() if len(failures) else pd.DataFrame(columns=['assay','standardize_failures'])
    conflict_counts = conflicts.groupby('assay').size().rename('conflict_rows').reset_index() if len(conflicts) else pd.DataFrame(columns=['assay','conflict_rows'])
    assays = assays.merge(kept_counts, on='assay', how='left').merge(fail_counts, on='assay', how='left').merge(conflict_counts, on='assay', how='left')
    assays[['kept_total','kept_active','standardize_failures','conflict_rows']] = assays[['kept_total','kept_active','standardize_failures','conflict_rows']].fillna(0).astype(int)
    assays['kept_inactive'] = assays['kept_total'] - assays['kept_active']
    assays.to_csv(ROOT / 'data' / 'processed' / 'assays.csv', index=False)

    with (ROOT / 'data_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        fieldnames = sorted({k for row in manifest_rows for k in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    lines = ['# LIT-PCBA Cohort Flow', '']
    lines.append(f'- Raw rows processed: {len(df):,}')
    lines.append(f'- Standardization failures: {len(failures):,}')
    lines.append(f'- Label-conflict rows excluded: {len(conflicts):,}')
    lines.append(f'- Deduplicated assay molecules retained: {len(molecules):,}')
    lines.append('')
    lines.append('|assay|raw active|raw inactive|kept active|kept inactive|failures|conflict rows|')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for row in assays.sort_values('assay').to_dict('records'):
        lines.append(f"|{row['assay']}|{row['raw_active_rows']}|{row['raw_inactive_rows']}|{row['kept_active']}|{row['kept_inactive']}|{row['standardize_failures']}|{row['conflict_rows']}|")
    (ROOT / 'data' / 'processed' / 'cohort_flow.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({
        'retained_molecules': int(len(molecules)),
        'standardization_failures': int(len(failures)),
        'conflict_rows': int(len(conflicts)),
        'assay_csv': 'data/processed/assays.csv',
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
