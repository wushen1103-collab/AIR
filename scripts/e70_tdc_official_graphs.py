"""Resumable, chunked author-compatible graph preparation for E69 TDC pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[1]
REPOS = ROOT / 'external_baselines/official_code/repos'
TABLES = ROOT / 'tables'
REQUIRED = ('index_smiles', 'smiles_index', 'smiles', 'x', 'y', 'graphs', 'graphs2')
RDLogger.DisableLog('rdApp.warning')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_torch(obj, path: Path) -> None:
    stage = path.with_name(path.name + f'.stage_pid{os.getpid()}')
    torch.save(obj, stage, pickle_protocol=4)
    stage.replace(path)


def simple_graph(mol):
    chirality = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                 Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                 Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                 Chem.rdchem.ChiralType.CHI_OTHER]
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_dirs = [Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
                 Chem.rdchem.BondDir.ENDDOWNRIGHT]
    nodes = [[list(range(1, 119)).index(a.GetAtomicNum()), chirality.index(a.GetChiralTag())]
             for a in mol.GetAtoms()]
    x = torch.tensor(np.asarray(nodes), dtype=torch.long)
    edges, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = [bond_types.index(bond.GetBondType()), bond_dirs.index(bond.GetBondDir())]
        edges.extend([(i, j), (j, i)])
        edge_features.extend([feature, feature])
    if edges:
        edge_index = torch.tensor(np.asarray(edges).T, dtype=torch.long)
        edge_attr = torch.tensor(np.asarray(edge_features), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)
    return x, edge_index, edge_attr


def validate_target(target: Path, expected: dict) -> dict:
    info = json.loads((target / 'e70_manifest.json').read_text())
    if info['status'] != 'complete' or any(info[k] != v for k, v in expected.items()):
        raise RuntimeError('Existing graph target metadata mismatch')
    for relative, digest in info['file_sha256'].items():
        path = target / relative
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f'Graph file changed: {path}')
    return info


def check_chunk(chunk: Path, expected: dict) -> bool:
    if not chunk.exists():
        return False
    marker = chunk / 'chunk_manifest.json'
    if not marker.exists():
        raise RuntimeError(f'Incomplete canonical chunk: {chunk}')
    info = json.loads(marker.read_text())
    if any(info[k] != v for k, v in expected.items()):
        raise RuntimeError(f'Chunk metadata changed: {chunk}')
    for name, digest in info['file_sha256'].items():
        path = chunk / 'part' / name
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f'Chunk hash mismatch: {path}')
    return True


def build_chunk(family_root: Path, stage: Path, usage: str, part: int,
                subset: pd.DataFrame, start: int, end: int, source_sha: str) -> bool:
    key = f'{usage}_{part:03d}'
    chunk = stage / 'chunks' / key
    expected = {'usage': usage, 'part': part, 'start': start, 'end': end,
                'source_sha256': source_sha}
    if check_chunk(chunk, expected):
        print(json.dumps({'chunk': key, 'status': 'resume_verified'}), flush=True)
        return False
    temp_name = f'{key}_stage_pid{os.getpid()}'
    temp = stage / 'chunks' / temp_name
    if temp.exists():
        raise RuntimeError(f'Occupied chunk staging path: {temp}')
    (temp / 'part').mkdir(parents=True)
    sys.path.insert(0, str(family_root / 'utils'))
    from data_prep import MasterDataset  # noqa: E402
    prior = Path.cwd()
    try:
        os.chdir(family_root / 'utils')
        obj = MasterDataset(name='part', df=subset,
                            dataset=f'{stage.name}/chunks/{temp_name}', overwrite=True)
    finally:
        os.chdir(prior)
    if len(obj.graphs) != len(subset) or len(obj.x) != len(subset):
        raise RuntimeError('Official chunk graph count mismatch')
    for offset, graph in enumerate(obj.graphs):
        if str(graph.smiles) != str(subset.smiles.iloc[offset]):
            raise RuntimeError(f'Official chunk order mismatch at {start + offset}')
        graph.fp = torch.tensor([obj.x[offset]], dtype=torch.float32)
        mol = Chem.MolFromSmiles(graph.smiles, sanitize=True)
        if mol is None:
            raise RuntimeError('E67-filtered SMILES became unparseable')
        graph.xp, graph.edgep_index, graph.edgep_attr = simple_graph(mol)
    torch.save(obj.graphs, temp / 'part' / 'graphs2', pickle_protocol=4)
    hashes = {name: sha(temp / 'part' / name) for name in REQUIRED}
    info = dict(expected, file_sha256=hashes)
    (temp / 'chunk_manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    if chunk.exists():
        raise RuntimeError('Concurrent canonical chunk appeared')
    temp.replace(chunk)
    check_chunk(chunk, expected)
    print(json.dumps({'chunk': key, 'status': 'complete', 'rows': len(subset)}), flush=True)
    return True


def assemble(stage: Path, usage: str, df: pd.DataFrame, chunk_size: int) -> None:
    output = stage / usage
    output.mkdir(exist_ok=True)
    all_graphs, all_graphs2, all_x, all_y, all_smiles = [], [], [], [], []
    for start in range(0, len(df), chunk_size):
        part = start // chunk_size
        root = stage / 'chunks' / f'{usage}_{part:03d}' / 'part'
        smiles = torch.load(root / 'smiles')
        x = torch.load(root / 'x')
        y = torch.load(root / 'y')
        graphs = torch.load(root / 'graphs')
        graphs2 = torch.load(root / 'graphs2')
        end = min(start + chunk_size, len(df))
        if (len(smiles) != end-start or len(graphs) != end-start
                or len(graphs2) != end-start or len(x) != end-start):
            raise RuntimeError('Chunk aggregation count mismatch')
        if list(smiles.astype(str)) != df.smiles.iloc[start:end].astype(str).tolist():
            raise RuntimeError('Chunk aggregation SMILES order mismatch')
        all_smiles.extend(smiles)
        all_x.append(x)
        all_y.append(y)
        all_graphs.extend(graphs)
        all_graphs2.extend(graphs2)
    smiles_array = np.asarray(all_smiles)
    x_array = np.concatenate(all_x)
    y_tensor = torch.cat(all_y)
    if len(smiles_array) != len(df) or int(y_tensor.sum()) != int(df.y.sum()):
        raise RuntimeError('Full graph assembly count/label mismatch')
    objects = {
        'index_smiles': OrderedDict(enumerate(smiles_array)),
        'smiles_index': OrderedDict((smi, i) for i, smi in enumerate(smiles_array)),
        'smiles': smiles_array, 'x': x_array, 'y': y_tensor,
        'graphs': all_graphs, 'graphs2': all_graphs2,
    }
    for name, obj in objects.items():
        atomic_torch(obj, output / name)
    print(json.dumps({'usage': usage, 'status': 'assembled', 'rows': len(df)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('ac_active', 'glare'), required=True)
    parser.add_argument('--aid', choices=('1798', '463087', '485290'), required=True)
    parser.add_argument('--mode', choices=('smoke', 'full'), required=True)
    parser.add_argument('--inject-after-first-chunk', choices=('none', 'crash', 'sleep'), default='none')
    args = parser.parse_args()
    family_root = REPOS / args.family
    data_root = (family_root / 'data').resolve()
    alias = f'TDC_E69_AID{args.aid}' + ('_SMOKE' if args.mode == 'smoke' else '')
    target = data_root / alias
    stage = data_root / f'{alias}_E70_STAGE'
    base = TABLES / f'e69_tdc_dev_{args.aid}'
    source_files = {'screen.csv': base.with_name(base.name + '_screen.csv'),
                    'test.csv': base.with_name(base.name + '_test.csv')}
    manifest_path = base.with_name(base.name + '_manifest.json')
    pool = json.loads(manifest_path.read_text())
    if (pool['status'] != 'frozen' or pool['aid'] != args.aid
            or sha(source_files['screen.csv']) != pool['screen_sha256']
            or sha(source_files['test.csv']) != pool['test_sha256']):
        raise RuntimeError('E69 frozen pool hash mismatch')
    source_hashes = {name: sha(path) for name, path in source_files.items()}
    source_hashes['manifest.json'] = sha(manifest_path)
    expected = {'family': args.family, 'aid': args.aid,
                'mode': args.mode, 'source_hashes': source_hashes}
    if target.exists():
        info = validate_target(target, expected)
        print(json.dumps({'status': 'skip_verified', 'alias': alias,
                          'n_screen': info['n_screen']}))
        return
    if stage.exists():
        stamp = json.loads((stage / 'e70_stage.json').read_text())
        if any(stamp[k] != v for k, v in expected.items()):
            raise RuntimeError('Existing stage source mismatch')
    else:
        (stage / 'chunks').mkdir(parents=True)
        (stage / 'original').mkdir()
        (stage / 'e70_stage.json').write_text(json.dumps(expected, indent=2) + '\n')
    screen = pd.read_csv(source_files['screen.csv'])
    test = pd.read_csv(source_files['test.csv'])
    if args.mode == 'smoke':
        screen = screen.iloc[:32].copy()
    elif len(screen) != pool['screen_size'] or int(screen.y.sum()) != pool['screen_positives']:
        raise RuntimeError('E69 full-screen count mismatch')
    if list(screen.columns) != ['smiles', 'y'] or screen.smiles.duplicated().any():
        raise RuntimeError('Screen schema/order invalid')
    for usage, df in (('screen', screen), ('test', test)):
        csv_path = stage / 'original' / f'{usage}.csv'
        content = df.to_csv(index=False, lineterminator='\n').encode()
        if csv_path.exists() and csv_path.read_bytes() != content:
            raise RuntimeError('Staged original CSV differs')
        if not csv_path.exists():
            csv_path.write_bytes(content)
    chunk_size = 10000
    first_completed = False
    for usage, df in (('screen', screen), ('test', test)):
        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            newly = build_chunk(family_root, stage, usage, start // chunk_size,
                                df.iloc[start:end].copy(), start, end,
                                source_hashes[f'{usage}.csv'])
            if newly and not first_completed:
                first_completed = True
                if args.inject_after_first_chunk == 'crash':
                    raise SystemExit(86)
                if args.inject_after_first_chunk == 'sleep':
                    time.sleep(30)
    for usage, df in (('screen', screen), ('test', test)):
        assemble(stage, usage, df, chunk_size)
    file_hashes = {f'original/{name}.csv': sha(stage / 'original' / f'{name}.csv')
                   for name in ('screen', 'test')}
    for usage in ('screen', 'test'):
        for name in REQUIRED:
            file_hashes[f'{usage}/{name}'] = sha(stage / usage / name)
    info = {'status': 'complete', **expected, 'dataset_alias': alias,
            'n_screen': len(screen), 'screen_positive': int(screen.y.sum()),
            'n_test': len(test), 'test_positive': int(test.y.sum()),
            'chunk_size': chunk_size, 'file_sha256': file_hashes}
    (stage / 'e70_manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    if target.exists():
        raise RuntimeError('Concurrent target appeared; stage retained')
    stage.replace(target)
    validate_target(target, expected)
    print(json.dumps({'status': 'complete', 'alias': alias,
                      'n_screen': len(screen), 'positive': int(screen.y.sum())}), flush=True)


if __name__ == '__main__':
    main()
