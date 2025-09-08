# src/pypsa_thesis/apply_renewable_bounds.py
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import pypsa

from . import io as pio
from .renewable_bounds import set_renewable_bounds

def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def main():
    ap = argparse.ArgumentParser(description="Apply renewable capacity bounds.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--network-in")
    ap.add_argument("--network-out")
    args = ap.parse_args()

    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    in_path  = Path(args.network_in or cfg["paths"]["expanded_network"])
    out_path = Path(args.network_out or cfg["paths"]["expanded_network"])

    n = pio.load_network(in_path)

    keep_existing = cfg.get("parameters", {}).get("renewables", {}).get("keep_existing", True)
    set_renewable_bounds(n, keep_existing=keep_existing)

    pio.save_network(n, out_path)
    logging.info(f"Wrote network with renewable bounds: {out_path}")

if __name__ == "__main__":
    main()
