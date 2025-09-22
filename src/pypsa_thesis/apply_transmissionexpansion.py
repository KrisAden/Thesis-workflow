#Importing packages
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from . import io as pio

#Defining logging
def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

#Bound violation logging
def _bound_violations_tx(n: pypsa.Network) -> pd.DataFrame:
    """
    Collect bound violations for transmission assets (lines/links) where min > max.
    """
    rows = []

    if len(n.lines) and {"s_nom_min", "s_nom_max"}.issubset(n.lines.columns):
        bad = (
            n.lines.s_nom_min.notnull()
            & n.lines.s_nom_max.notnull()
            & (n.lines.s_nom_min > n.lines.s_nom_max)
        )
        if bad.any():
            for name in n.lines.index[bad]:
                rows.append(("lines", name))

    if len(n.links) and {"p_nom_min", "p_nom_max"}.issubset(n.links.columns):
        bad = (
            n.links.p_nom_min.notnull()
            & n.links.p_nom_max.notnull()
            & (n.links.p_nom_min > n.links.p_nom_max)
        )
        if bad.any():
            for name in n.links.index[bad]:
                rows.append(("links", name))

    if not rows:
        return pd.DataFrame(columns=["component", "name"])

    return pd.DataFrame(rows, columns=["component", "name"])


#Adding transmission expansion
def enable_transmission_expansion(
        n: pypsa.Network,
        *,
        # Default values if nothing differenti is specified in config
        lines_enable: bool = True,
        links_enable: bool = True,
        line_abs_max: float = 10e5,
        link_abs_max: float = 10e6,
        lines_min_equals_current: bool = True,
        links_min_equals_current: bool = True,
) -> pd.DataFrame:
    rows = []

    #Filling NA values in current capacities with 0 to guard against failure
    if len(n.lines):
        n.lines["s_nom"] = n.lines["s_nom"].fillna(0.0)
    if len(n.links):
        n.links["p_nom"] = n.links["p_nom"].fillna(0.0)

    #Lines (AC)
    if lines_enable and len(n.lines):
        n.lines["s_nom_extendable"] = True

        #Passing values to network
        if lines_min_equals_current:
            n.lines["s_nom_min"] = n.lines["s_nom"].astype(float)
        
        if not np.isinf(line_abs_max):
            n.lines["s_nom_max"] = float(line_abs_max)

        #Guarding against s_nom_max being set below current s_nom
        if "s_nom_max" in n.lines:
            bad = n.lines.s_nom_max.notnull() & (n.lines.s_nom > n.lines.s_nom_max)
            if bad.any():
                logging.warning(
                    "Raising %d lines' s_nom_max up to current s_nom to satisfy minimum <= maximum",
                    bad.sum(),
                )
                n.lines.loc[bad,"s_nom_max"] = n.lines.loc[bad,"s_nom"]

        #Logging handling for lines
        rows.append(
            {
                "component": "lines",
                "count": len(n.lines),
                "min_policy": (
                    "s_nom_min = current" if lines_min_equals_current else "unchanged"
                ),
                "max_policy": (
                    f"s_nom_max = {line_abs_max:g}"
                    if np.isfinite(line_abs_max)
                    else "unchanged"
                ),
            }
        )
    # Links (DC)
    if links_enable and len(n.links):
        n.links["p_nom_extendable"] = True

        #Passing values to network
        if links_min_equals_current:
            n.links["p_nom_min"] = n.links["p_nom"].astype(float)

        if np.isfinite(link_abs_max):
            n.links["p_nom_max"] = float(link_abs_max)

        #Guarding against s_nom_max being set below current s_nom
        if "p_nom_max" in n.links:
            bad = n.links.p_nom_max.notnull() & (n.links["p_nom"] > n.links.p_nom_max)
            if bad.any():
                logging.warning(
                    "Raising %d links' p_nom_max up to current p_nom to satisfy p_nom_min <= p_nom_max.",
                    bad.sum(),
                )
                n.links.loc[bad, "p_nom_max"] = n.links.loc[bad, "p_nom"]

        #Logging handling for lines
        rows.append(
            {
                "component": "links",
                "count": len(n.links),
                "min_policy": (
                    "p_nom_min = current" if links_min_equals_current else "unchanged"
                ),
                "max_policy": (
                    f"p_nom_max = {link_abs_max:g}"
                    if np.isfinite(link_abs_max)
                    else "unchanged"
                ),
            }
        )

    return pd.DataFrame(rows)

#Adding main function to interface with config file and Snakemake
def main():
    parser = argparse.ArgumentParser(
        description="Enable transmission expansion"
    )

    parser.add_argument("--config", default = "config/config.yaml")
    parser.add_argument(
        "--network-in", help = "Override input network (defaults to cfg.paths.network_costed)"
    )

    parser.add_argument(
        "--network-out",
        help = "Override output network (defaults to cfg.paths.network_costed_transmissionexpansion)",
        )
    
    parser.add_argument(
        "--report-out", help="Optional CSV with expansion bounds summary"
    )

    parser.add_argument(
        "--violations-out",
        help="Optional CSV with any line/link bound violations (min>max).",
    )

    args = parser.parse_args()

    #Importing arguments from config
    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level","INFO"))

        #Adding paths from config and defaults

    in_path = Path(args.network_in or cfg["paths"].get("costed_network", "data/interim/network_costed.nc"))
    out_path = Path(args.network_out or "data/interim/network_costed_tx.nc")
    tables = Path(cfg["paths"].get("tables_dir", "results/tables"))

    # NEW: ensure parents exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    report_out = Path(args.report_out or (tables / "tx_expansion_bounds.csv"))
    violations_out = Path(args.violations_out or (tables / "tx_bound_violations.csv"))
    report_out.parent.mkdir(parents=True, exist_ok=True)
    violations_out.parent.mkdir(parents=True, exist_ok=True)

    n = pio.load_network(in_path)

    xp = cfg.get("parameters", {}).get("expansion", {}) or {}
    lines_cfg = xp.get("lines", {}) or {}
    links_cfg = xp.get("links", {}) or {}

    report = enable_transmission_expansion(
        n,
        lines_enable=bool(lines_cfg.get("enable", True)),
        links_enable=bool(links_cfg.get("enable", True)),
        line_abs_max=float(lines_cfg.get("absolute_max", np.inf)),
        link_abs_max=float(links_cfg.get("absolute_max", np.inf)),
        lines_min_equals_current=bool(lines_cfg.get("min_equals_current", True)),
        links_min_equals_current=bool(links_cfg.get("min_equals_current", True)),
    )

    # Write a quick sanity report for contradictions (min>max)
    vdf = _bound_violations_tx(n)
    if len(vdf):
        vdf.to_csv(violations_out, index=False)
        logging.error("Found %d transmission bound violations. Wrote: %s", len(vdf), violations_out)
    else:
        # write an empty file to make the pipeline robust
        vdf.to_csv(violations_out, index=False)
        logging.info("No transmission bound violations.")

    report.to_csv(report_out, index=False)
    pio.save_network(n, out_path)
    logging.info("Wrote: %s", out_path)
    logging.info("Report: %s", report_out)
    logging.info("Violations: %s", violations_out)

if __name__ == "__main__":
	main()
