"""Checkpointed, author-featurizer-compatible audit of E64 development HTS rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/tdc_hts_e64'
TABLES = ROOT / 'tables'
SOURCES = {
    '1798': ('m1_muscarinic_receptor_agonists_butkiewicz.tab',
             '6637b5512321fdc6974ea487142c5f6766eb9e935744f2fe4a7c657902a30edf', 61833),
    '463087': ('cav3_t-type_calcium_channels_butkiewicz.tab',
               '201fc8527a0c9db4d26862541f67a89fc38151c1d04f568a9c3ebc809abd9af8', 100875),
    '485290': ('tyrosyl-dna_phosphodiesterase_butkiewicz.tab',
               'da482019d0bb005d684416bca73c8d625cb18d4988e23c0542fa9ffddbc471ba', 341365),
}
COUNTER_NAMES = ('raw_negative', 'raw_positive', 'invalid_smiles',
                 'rejected_scaffold', 'rejected_featurizer', 'accepted_rows')
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')
sys.path.insert(0, str(ROOT / 'external_baselines/official_code/repos/ac_active'))
from utils.utils import check_featurizability  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def initialize(db: sqlite3.Connection, source_hash: str) -> None:
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS mol (smiles TEXT PRIMARY KEY, mask INTEGER NOT NULL, n INTEGER NOT NULL)')
    row = db.execute("SELECT value FROM meta WHERE key='source_sha256'").fetchone()
    if row is None:
        db.execute("INSERT INTO meta VALUES ('source_sha256', ?)", (source_hash,))
        db.execute("INSERT INTO meta VALUES ('cursor', '0')")
        for name in COUNTER_NAMES:
            db.execute('INSERT INTO meta VALUES (?, ?)', (name, '0'))
        db.commit()
    elif row[0] != source_hash:
        raise RuntimeError('Source changed after checkpoint')


def meta(db: sqlite3.Connection, key: str) -> int:
    return int(db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()[0])


def commit_chunk(db: sqlite3.Connection, records: list[tuple[str, int]],
                 delta: dict[str, int], cursor: int) -> None:
    with db:
        db.executemany('INSERT INTO mol (smiles, mask, n) VALUES (?, ?, 1) '
                       'ON CONFLICT(smiles) DO UPDATE SET mask=mask|excluded.mask, n=n+1', records)
        for key, value in delta.items():
            db.execute('UPDATE meta SET value=CAST(value AS INTEGER)+? WHERE key=?', (value, key))
        db.execute("UPDATE meta SET value=? WHERE key='cursor'", (str(cursor),))
    print(json.dumps({'checkpoint': cursor, 'accepted_in_chunk': len(records)}), flush=True)


def classify(row: dict[str, str], delta: dict[str, int]) -> tuple[str, int] | None:
    if row.get('Drug_ID') is None or row.get('Drug') is None or row.get('Y') is None:
        raise RuntimeError('Malformed row or missing required field')
    label_text = row['Y'].strip()
    if label_text not in ('0', '1', '0.0', '1.0'):
        raise RuntimeError('Nonbinary label')
    label = int(float(label_text))
    delta['raw_positive' if label else 'raw_negative'] += 1
    mol = Chem.MolFromSmiles(row['Drug'].strip(), sanitize=True)
    if mol is None or mol.GetNumAtoms() == 0:
        delta['invalid_smiles'] += 1
        return None
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if Chem.MolFromSmiles(scaffold) is None:
            delta['rejected_scaffold'] += 1
            return None
    except Exception:
        delta['rejected_scaffold'] += 1
        return None
    if not check_featurizability(smiles):
        delta['rejected_featurizer'] += 1
        return None
    delta['accepted_rows'] += 1
    return (smiles, 1 << label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--aid', required=True, choices=tuple(SOURCES))
    parser.add_argument('--inject-after', type=int, default=0)
    parser.add_argument('--inject-sleep', type=int, default=0)
    args = parser.parse_args()
    filename, expected_hash, expected_rows = SOURCES[args.aid]
    source = RAW / filename
    actual_hash = sha256(source)
    if actual_hash != expected_hash:
        raise RuntimeError(f'E66 source SHA-256 mismatch for {args.aid}')
    manifest = TABLES / f'e67_tdc_dev_{args.aid}_audit.json'
    if manifest.exists():
        result = json.loads(manifest.read_text())
        if result.get('status') != 'complete' or result.get('source_sha256') != actual_hash:
            raise RuntimeError('Existing manifest not verified')
        print(json.dumps({'status': 'skip_verified', 'aid': args.aid,
                          'retained': result['retained_candidates']}))
        return
    db_path = TABLES / f'e67_tdc_dev_{args.aid}_audit.sqlite'
    TABLES.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    initialize(db, actual_hash)
    cursor = meta(db, 'cursor')
    processed = cursor
    records: list[tuple[str, int]] = []
    delta = {name: 0 for name in COUNTER_NAMES}
    with source.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != ['Drug_ID', 'Drug', 'Y']:
            raise RuntimeError(f'Unexpected source header: {reader.fieldnames}')
        for row_number, row in enumerate(reader, start=1):
            if row_number <= cursor:
                continue
            item = classify(row, delta)
            if item is not None:
                records.append(item)
            processed = row_number
            if row_number % 500 == 0:
                commit_chunk(db, records, delta, row_number)
                records = []
                delta = {name: 0 for name in COUNTER_NAMES}
                if args.inject_after and row_number >= args.inject_after:
                    if args.inject_sleep:
                        time.sleep(args.inject_sleep)
                    raise SystemExit(86)
    if records or processed != cursor:
        commit_chunk(db, records, delta, processed)
    if processed != expected_rows or meta(db, 'cursor') != expected_rows:
        raise RuntimeError(f'Official row count mismatch: {processed}/{expected_rows}')
    totals = {name: meta(db, name) for name in COUNTER_NAMES}
    if totals['raw_negative'] + totals['raw_positive'] != expected_rows:
        raise RuntimeError('Raw label count mismatch')
    if sum(totals[k] for k in ('invalid_smiles', 'rejected_scaffold',
                               'rejected_featurizer', 'accepted_rows')) != expected_rows:
        raise RuntimeError('Filter accounting mismatch')
    unique, conflicting, conflict_rows, repeat_rows, positive = db.execute(
        'SELECT COUNT(*), SUM(mask=3), SUM(CASE WHEN mask=3 THEN n ELSE 0 END), '
        'SUM(CASE WHEN mask!=3 THEN n-1 ELSE 0 END), SUM(mask=2) FROM mol').fetchone()
    retained = unique - conflicting
    result = {
        'status': 'complete', 'protocol': 'E67', 'aid': args.aid,
        'source_filename': filename, 'source_sha256': actual_hash,
        'raw_rows': expected_rows, 'totals': totals,
        'canonical_unique_before_conflict': unique,
        'conflicting_candidates_excluded': conflicting,
        'conflicting_raw_rows_excluded': conflict_rows,
        'same_label_repeat_rows_collapsed': repeat_rows,
        'retained_candidates': retained,
        'retained_positives': positive,
        'retained_negatives': retained - positive,
        'database_sha256': None,
    }
    # Hash before publishing: the DB is closed and no later writes are allowed.
    db.close()
    result['database_sha256'] = sha256(db_path)
    stage = manifest.with_name(manifest.name + f'.stage_pid{os.getpid()}')
    stage.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    if manifest.exists():
        raise RuntimeError('Concurrent manifest appeared; stage retained')
    stage.replace(manifest)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
