# src/pypsa_thesis/apply_renewable_bounds.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
import pypsa

from . import io as pio
from .renewable_bounds import set_renewable_bounds, disable_hydro_extension


def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _collect_component_carriers(n: pypsa.Network) -> pd.Index:
    """Collect all carrier names referenced by components."""
    series: List[pd.Series] = []
    if len(n.generators):
        series.append(n.generators.carrier)
    if len(n.links):
        series.append(n.links.carrier)
    if len(n.lines) and "carrier" in n.lines:
        series.append(n.lines.carrier)
    if len(n.loads) and "carrier" in n.loads:
        series.append(n.loads.carrier)
    if len(n.storage_units):
        series.append(n.storage_units.carrier)
    if len(series) == 0:
        return pd.Index([])
    return pd.Index(pd.concat(series, axis=0).unique())


def _ensure_all_carriers(n: pypsa.Network) -> pd.Index:
    """
    Ensure every referenced carrier exists in n.carriers.
    Returns the Index of carriers that were added.
    """
    referenced = _collect_component_carriers(n)
    missing = referenced.difference(n.carriers.index)
    if len(missing):
        for c in missing:
            # Add a new Carrier with default (zero) attributes.
            # You can later set costs/emissions in your cost step.
            n.add("Carrier", c)
        logging.warning("Added %d missing carriers to n.carriers: %s",
                        len(missing), list(missing))
    else:
        logging.info("All referenced carriers are present in n.carriers.")
    return missing


def main():
    ap = argparse.ArgumentParser(description="Apply renewable capacity bounds.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--network-in")
    ap.add_argument("--network-out")
    ap.add_argument(
        "--debug-carriers-out",
        help="Optional CSV path to write carriers that were auto-added."
    )
    args = ap.parse_args()

    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    in_path  = Path(args.network_in or cfg["paths"]["expanded_network"])
    out_path = Path(args.network_out or cfg["paths"]["expanded_network"])

    n = pio.load_network(in_path)

    # 1) Apply your generator bounds and hydro restriction
    keep_existing = cfg.get("parameters", {}).get("renewables", {}).get("keep_existing", True)
    set_renewable_bounds(n, keep_existing=keep_existing)
    disable_hydro_extension(n)

    # 2) Ensure carriers are complete (prevents unboundedness from NaN costs)
    missing = _ensure_all_carriers(n)

    # Optional debug dump
    if args.debug_carriers_out:
        Path(args.debug_carriers_out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"added_carrier": list(missing)}).to_csv(args.debug_carriers_out, index=False)

    # 3) Save
    pio.save_network(n, out_path)
    logging.info("Wrote network with renewable bounds: %s", out_path)


if __name__ == "__main__":
    main()
