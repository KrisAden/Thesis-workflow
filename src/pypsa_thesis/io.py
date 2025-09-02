from __future__ import annotations
from pathlib import Path
import yaml
import pypsa

def ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_network(path: str | Path) -> pypsa.Network:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Network file not found: {path}")
    return pypsa.Network(str(path))

def save_network(n: pypsa.Network, path: str | Path) -> None:
    path = Path(path)
    ensure_parent(path)
    n.export_to_netcdf(str(path))
