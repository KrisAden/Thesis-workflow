# src/pypsa_thesis/apply_renewable_bounds.py
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from . import io as pio

# -------------------------
# Config & logging helpers
# -------------------------

def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

# ---------------------------------
# Core renewable-bounds operations
# ---------------------------------

# Renewables you want to allow to expand (hydro handled separately)
RENEW_CARRIERS = [
    "offwind-ac", "offwind-dc", "onwind", "solar",
    "biomass", "geothermal", "nuclear"
]

# Common names for hydro-like tech across datasets (generators table)
HYDRO_GENERATOR_ALIASES = {"hydro", "PHS"}

def set_renewable_bounds(n: pypsa.Network, keep_existing: bool = True) -> None:
    """
    Make listed renewables extendable; keep existing capacity if requested; fix p_nom_max
    if it was below current p_nom (to avoid min>max).
    """
    if not len(n.generators):
        logging.info("No generators found; skipping renewable bounds.")
        return

    g = n.generators
    g["p_nom"] = g["p_nom"].fillna(0.0)

    mask_renew = g["carrier"].isin(RENEW_CARRIERS)
    if not mask_renew.any():
        logging.info("No renewable generators found for bounds update.")
        return

    # Enable expansion for these carriers
    g.loc[mask_renew, "p_nom_extendable"] = True

    # Lower bound
    if keep_existing:
        g.loc[mask_renew, "p_nom_min"] = g.loc[mask_renew, "p_nom"]
    else:
        g.loc[mask_renew, "p_nom_min"] = 0.0

    # Guard: ensure finite
    g.loc[mask_renew, "p_nom_min"] = g.loc[mask_renew, "p_nom_min"].fillna(0.0)

    # If p_nom_max exists and is below current p_nom, raise it to current p_nom
    if "p_nom_max" in g:
        m = mask_renew & g["p_nom_max"].notnull()
        bad = m & (g["p_nom"] > g["p_nom_max"])
        if bad.any():
            cnt = int(bad.sum())
            logging.warning(
                "Raising p_nom_max to current p_nom for %d renewable generators "
                "to satisfy p_nom_min <= p_nom_max.", cnt
            )
            g.loc[bad, "p_nom_max"] = g.loc[bad, "p_nom"]

    logging.info(
        "Applied renewable bounds (keep_existing=%s) to %d generators. "
        "p_nom_max kept unless raised to current.",
        keep_existing, int(mask_renew.sum())
    )

def disable_hydro_extension(n: pypsa.Network) -> None:
    """
    Ensure hydro storage units and hydro-like generators are fixed at current capacity.
    """
    # Storage units
    if len(n.storage_units):
        su = n.storage_units
        mask_su = su["carrier"].str.lower() == "hydro"
        if mask_su.any():
            su.loc[mask_su, "p_nom"] = su.loc[mask_su, "p_nom"].fillna(0.0)
            su.loc[mask_su, "p_nom_extendable"] = False
            su.loc[mask_su, "p_nom_min"] = su.loc[mask_su, "p_nom"]
            su.loc[mask_su, "p_nom_max"] = su.loc[mask_su, "p_nom"]
            logging.info("Disabled extension for %d hydro storage units.", int(mask_su.sum()))

    # Generators
    if len(n.generators):
        g = n.generators
        g["p_nom"] = g["p_nom"].fillna(0.0)
        mask_gen = g["carrier"].str.lower().isin({c.lower() for c in HYDRO_GENERATOR_ALIASES})
        if mask_gen.any():
            g.loc[mask_gen, "p_nom_extendable"] = False
            g.loc[mask_gen, "p_nom_min"] = g.loc[mask_gen, "p_nom"]
            g.loc[mask_gen, "p_nom_max"] = g.loc[mask_gen, "p_nom"]
            logging.info("Disabled extension for %d hydro generators.", int(mask_gen.sum()))

# ------------------------------
# Diagnostics / small reporting
# ------------------------------

def _bound_violations_gen_su(n: pypsa.Network) -> pd.DataFrame:
    """
    Collect bound violations (min>max) for generators and storage_units.
    """
    rows = []
    if len(n.generators) and {"p_nom_min", "p_nom_max"}.issubset(n.generators.columns):
        bad = n.generators["p_nom_min"].notnull() & n.generators["p_nom_max"].notnull() & (
            n.generators["p_nom_min"] > n.generators["p_nom_max"]
        )
        rows += [("generators", name) for name in n.generators.index[bad]]

    if len(n.storage_units) and {"p_nom_min", "p_nom_max"}.issubset(n.storage_units.columns):
        bad = n.storage_units["p_nom_min"].notnull() & n.storage_units["p_nom_max"].notnull() & (
            n.storage_units["p_nom_min"] > n.storage_units["p_nom_max"]
        )
        rows += [("storage_units", name) for name in n.storage_units.index[bad]]

    if not rows:
        return pd.DataFrame(columns=["component", "name"])
    return pd.DataFrame(rows, columns=["component", "name"])

def _quick_report(n: pypsa.Network) -> pd.DataFrame:
    """
    Small carrier-level summary after applying bounds.
    """
    if not len(n.generators):
        return pd.DataFrame(columns=["carrier", "count", "extendable_count", "total_p_nom", "avg_p_nom_max"])

    g = n.generators.copy()
    if "p_nom_max" not in g:
        g["p_nom_max"] = np.inf
    grp = (
        g.assign(extendable=g["p_nom_extendable"].fillna(False).astype(bool))
         .groupby("carrier", dropna=False)
         .apply(lambda df: pd.Series({
             "count": len(df),
             "extendable_count": int(df["extendable"].sum()),
             "total_p_nom": float(df["p_nom"].fillna(0).sum()),
             "avg_p_nom_max": float(df["p_nom_max"].replace([np.inf, -np.inf], np.nan).mean())
         }))
         .reset_index()
    )
    return grp

# -------------
# CLI entrypoint
# -------------

def main():
    ap = argparse.ArgumentParser(description="Apply renewable bounds and freeze hydro expansion.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--network-in", help="Input .nc (defaults to cfg.paths.costed_network or an upstream product)")
    ap.add_argument("--network-out", help="Output .nc with renewables enabled and hydro frozen")
    ap.add_argument("--report-out", help="CSV report (carrier-level summary)")
    ap.add_argument("--violations-out", help="CSV with any min>max violations on generators/storage_units")
    args = ap.parse_args()

    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    # Default paths – we expect this to run after transmission expansion
    in_path = Path(args.network_in or "data/interim/network_costed_tx.nc")
    out_path = Path(args.network_out or "data/interim/network_costed_tx_ren.nc")

    tables = Path(cfg["paths"].get("tables_dir", "results/tables"))
    tables.mkdir(parents=True, exist_ok=True)

    report_out = Path(args.report_out or (tables / "renewable_bounds_report.csv"))
    violations_out = Path(args.violations_out or (tables / "renewable_bound_violations.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    violations_out.parent.mkdir(parents=True, exist_ok=True)

    n = pio.load_network(in_path)

    keep = bool((cfg.get("parameters", {}) or {}).get("renewables", {}).get("keep_existing", True))
    set_renewable_bounds(n, keep_existing=keep)
    disable_hydro_extension(n)

    # Violations (always write a CSV, even if empty)
    vdf = _bound_violations_gen_su(n)
    vdf.to_csv(violations_out, index=False)
    if len(vdf):
        logging.error("Found %d bound violations (min>max). See: %s", len(vdf), violations_out)
    else:
        logging.info("No generator/storage bound violations.")

    # Small report
    rep = _quick_report(n)
    rep.to_csv(report_out, index=False)

    pio.save_network(n, out_path)
    logging.info("Wrote: %s", out_path)
    logging.info("Report: %s", report_out)
    logging.info("Violations: %s", violations_out)

if __name__ == "__main__":
    main()
