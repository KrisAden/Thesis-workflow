# src/pypsa_thesis/apply_storage.py
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import pandas as pd
import pypsa

from . import io as pio
from .storage import add_battery_storage, add_hydrogen_chain

def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def main():
    ap = argparse.ArgumentParser(description="Add storage (battery and/or H2) and apply costs.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--network-in")
    ap.add_argument("--network-out")
    ap.add_argument("--table-out")  # writes a simple summary of applied costs
    args = ap.parse_args()

    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    in_path   = Path(args.network_in  or cfg["paths"]["expanded_network"])
    out_path  = Path(args.network_out or cfg["paths"]["network_with_storage"])
    tables    = Path(cfg["paths"].get("tables_dir", "results/tables"))
    tables.mkdir(parents=True, exist_ok=True)
    table_out = Path(args.table_out or (tables / "storage_costs_applied.csv"))

    storage_csv = Path(cfg["paths"]["storage_costs_csv"]).resolve()

    n = pio.load_network(in_path)
    df_costs = pd.read_csv(storage_csv)

    reports = []
    if cfg.get("parameters", {}).get("storage", {}).get("add_battery", False):
        reports.append(add_battery_storage(n, df_costs))
    if cfg.get("parameters", {}).get("storage", {}).get("add_hydrogen", False):
        reports.append(add_hydrogen_chain(n, df_costs))

    if reports:
        pd.concat(reports, ignore_index=True).to_csv(table_out, index=False)
    else:
        logging.info("No storage selected; nothing to add.")
        # still write an empty file so Snakemake has an output
        pd.DataFrame(columns=["component","capital_cost","efficiency"]).to_csv(table_out, index=False)

    pio.save_network(n, out_path)
    logging.info(f"Wrote: {out_path}")
    logging.info(f"Costs table: {table_out}")
    logging.info(f"Read storage costs from: {storage_csv}")

if __name__ == "__main__":
    main()
