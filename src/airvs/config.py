from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f'YAML root must be a mapping: {path}')
    return data

def preregistration() -> dict[str, Any]:
    return load_yaml(project_root() / 'configs' / 'preregistration.yaml')
