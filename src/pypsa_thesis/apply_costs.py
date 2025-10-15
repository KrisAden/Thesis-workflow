from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import pypsa

from . import io as pio

def annuity(lifetime_y: float, r: float) -> float:
    if lifetime_y <= 0:
        raise ValueError("lifetime_y must be > 0")
    return r / (1 - (1 + r) ** (-lifetime_y)) if r > 0 else 1.0 / lifetime_y

def capital_cost_from_dea(
    capex_eur_per_mw: float,
    lifetime_y: float,
    discount_rate: float,
    fom_abs_eur_per_mw_a: Optional[float] = None,
    fom_rate_per_a: Optional[float] = None,
) -> float:
    a = annuity(lifetime_y, discount_rate)
    if fom_abs_eur_per_mw_a is not None:
        return capex_eur_per_mw * a + fom_abs_eur_per_mw_a
    if fom_rate_per_a is not None:
        return capex_eur_per_mw * (a + fom_rate_per_a)
    return capex_eur_per_mw * a


def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def apply_generator_capital_costs(n: pypsa.Network, df: pd.DataFrame) -> pd.DataFrame:
    """
    df columns: carrier, capex_eur_per_mw, lifetime_y, discount_rate,
                fom_abs_eur_per_mw_a (opt), fom_rate_per_a (opt)
    """
    rows = []
    gen_carriers = n.generators["carrier"]

    for _, r in df.iterrows():
        carrier = str(r["carrier"])
        capex   = float(r["capex_eur_per_mw"])
        life    = float(r["lifetime_y"])
        disc    = float(r["discount_rate"])
        fom_abs = r.get("fom_abs_eur_per_mw_a")
        fom_rate = r.get("fom_rate_per_a")

        fom_abs = None if pd.isna(fom_abs) else float(fom_abs)
        fom_rate = None if pd.isna(fom_rate) else float(fom_rate)

        cost = capital_cost_from_dea(capex, life, disc, fom_abs_eur_per_mw_a=fom_abs, fom_rate_per_a=fom_rate)

        # exact match first, then case-insensitive fallback
        mask = (gen_carriers == carrier)
        if not mask.any():
            mask = gen_carriers.str.lower() == carrier.lower()

        updated = int(mask.sum())
        if updated > 0:
            n.generators.loc[mask, "capital_cost"] = cost
            rows.append({
                "carrier": carrier,
                "count_generators": updated,
                "capex_eur_per_mw": capex,
                "lifetime_y": life,
                "discount_rate": disc,
                "fom_abs_eur_per_mw_a": fom_abs,
                "fom_rate_per_a": fom_rate,
                "capital_cost_eur_per_mw_a": cost,
            })
        else:
            logging.warning(f"No generators found with carrier '{carrier}'")

    # warn for carriers present in the network but missing in CSV (optional)
    missing = set(gen_carriers.unique()) - {c for c in df["carrier"]}
    if missing:
        logging.info(f"Carriers in network without CSV entries (left unchanged): {sorted(missing)}")

    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Apply generator capital costs from CSV.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--network-in")
    ap.add_argument("--network-out")
    ap.add_argument("--table-out")
    args = ap.parse_args()

    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    in_path   = Path(args.network_in  or cfg["paths"]["expanded_network"])
    out_path  = Path(args.network_out or cfg["paths"]["costed_network"])
    costs_csv = Path(cfg["paths"]["costs_csv"]).resolve()
    tables    = Path(cfg["paths"].get("tables_dir", "results/tables"))
    tables.mkdir(parents=True, exist_ok=True)
    table_out = Path(args.table_out or (tables / "generator_capital_costs.csv"))

    n = pio.load_network(in_path)
    df_costs = pd.read_csv(costs_csv)

    report = apply_generator_capital_costs(n, df_costs)
    report.to_csv(table_out, index=False)
    pio.save_network(n, out_path)

    logging.info(f"Wrote: {out_path}")
    logging.info(f"Costs table: {table_out}")
    logging.info(f"Read costs from: {costs_csv}")

if __name__ == "__main__":
    main()
