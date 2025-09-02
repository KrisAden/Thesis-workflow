from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from . import io as pio

def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def rescale_loads_to_target_twh(n: pypsa.Network, target_twh: pd.Series) -> pd.DataFrame:
    """
    Rescale n.loads_t.p_set country-by-country so that annual energy (TWh)
    matches target_twh (indexed by country code).
    Returns a dataframe with before/after/target/factor per country.
    """
    # Map each load to its country from the bus name suffix "..._<CC>"
    load_country = n.loads["bus"].apply(lambda x: x.split("_")[-1])

    # Annual energy before scaling (TWh): sum MW over hours -> MWh, /1e6 -> TWh
    before_twh_by_load = n.loads_t.p_set.sum() / 1e6
    before_twh_country = before_twh_by_load.groupby(load_country).sum()

    # Keep only countries present in both
    valid_countries = target_twh.index.intersection(before_twh_country.index)
    target = target_twh.loc[valid_countries]
    before = before_twh_country.loc[valid_countries]

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = target / before.replace(0, np.nan)

    # Apply factors per-load according to its country
    for country, factor in factors.dropna().items():
        loads_in_country = load_country[load_country == country].index
        # Scale each load's whole time-series
        n.loads_t.p_set[loads_in_country] = n.loads_t.p_set[loads_in_country] * factor

    # After-scaling check
    after_twh_by_load = n.loads_t.p_set.sum() / 1e6
    after_twh_country = after_twh_by_load.groupby(load_country).sum()

    report = pd.DataFrame({
        "before_TWh": before_twh_country.reindex(valid_countries),
        "target_TWh": target,
        "after_TWh": after_twh_country.reindex(valid_countries),
        "factor": factors.reindex(valid_countries),
    })
    return report

def read_target_twh(csv_path: Path, gtoe_to_twh: float) -> pd.Series:
    # The file you used had index_col "Region" and a column "Value"
    df = pd.read_csv(csv_path, delimiter=",", index_col=0)
    # Try to find the numeric column robustly
    value_col = next((c for c in df.columns if str(df[c].dtype).startswith(("float","int"))), None)
    if value_col is None:
        raise ValueError(f"No numeric value column found in {csv_path}")
    target_twh = df[value_col].astype(float) * gtoe_to_twh
    target_twh.index.name = "country"
    return target_twh

def main():
    parser = argparse.ArgumentParser(description="Rescale network loads to target TWh by country.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--network-in", help="Path to base .nc network (overrides config).")
    parser.add_argument("--gtoe-csv", help="Path to Gtoe CSV (overrides config).")
    parser.add_argument("--network-out", help="Where to save rescaled network .nc (overrides config).")
    parser.add_argument("--report-out", help="Where to save CSV report (optional).")
    args = parser.parse_args()

    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    base_network = Path(args.network_in or cfg["paths"]["base_network"])
    gtoe_csv = Path(args.gtoe_csv or cfg["paths"]["gtoe_csv"])
    out_network = Path(args.network_out or cfg["paths"]["rescaled_network"])
    tables_dir = Path(cfg["paths"].get("tables_dir", "results/tables"))
    report_out = Path(args.report_out) if args.report_out else tables_dir / "load_scaling.csv"
    tables_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading network: {base_network}")
    n = pio.load_network(base_network)

    logging.info(f"Reading target demand (Gtoe→TWh) from: {gtoe_csv}")
    factor = float(cfg["parameters"]["gtoe_to_twh"])
    target_twh = read_target_twh(gtoe_csv, gtoe_to_twh=factor)

    logging.info("Rescaling loads…")
    report = rescale_loads_to_target_twh(n, target_twh)

    logging.info(f"Writing report: {report_out}")
    report.sort_index().to_csv(report_out, float_format="%.6g")

    logging.info(f"Saving rescaled network: {out_network}")
    pio.save_network(n, out_network)

    logging.info("Done.")

if __name__ == "__main__":
    main()

