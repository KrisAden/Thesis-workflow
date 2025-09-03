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

def enable_transmission_expansion(n: pypsa.Network,
                                  lines_enable: bool = True,
                                  links_enable: bool = True,
                                  line_abs_max: float = 1e15,
                                  link_abs_max: float = 1e15,
                                  min_equals_current: bool = True) -> pd.DataFrame:
    """
    Enable expansion on AC lines (s_nom) and DC links (p_nom).
    Sets *_extendable=True, *_min=current capacity (optional), *_max=absolute cap.
    Returns a small report dataframe.
    """
    rows = []

    if lines_enable and len(n.lines):
        n.lines["s_nom_extendable"] = True
        if min_equals_current:
            n.lines["s_nom_min"] = n.lines["s_nom"].fillna(0.0)
        if np.isfinite(line_abs_max):
            n.lines["s_nom_max"] = float(line_abs_max)

        rows.append({
            "component": "lines",
            "count": len(n.lines),
            "min_policy": "s_nom_min = current" if min_equals_current else "unchanged",
            "max_policy": f"s_nom_max = {line_abs_max:g}"
        })

    if links_enable and len(n.links):
        # ensure p_nom is finite
        n.links["p_nom"] = n.links["p_nom"].fillna(0.0)
        n.links["p_nom_extendable"] = True
        if min_equals_current:
            n.links["p_nom_min"] = n.links["p_nom"]
        if np.isfinite(link_abs_max):
            n.links["p_nom_max"] = float(link_abs_max)

        rows.append({
            "component": "links",
            "count": len(n.links),
            "min_policy": "p_nom_min = current" if min_equals_current else "unchanged",
            "max_policy": f"p_nom_max = {link_abs_max:g}"
        })

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Enable transmission expansion on a PyPSA network.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--network-in", help="Override input .nc (defaults to cfg.paths.rescaled_network)")
    parser.add_argument("--network-out", help="Override output .nc (defaults to cfg.paths.expanded_network)")
    parser.add_argument("--report-out", help="Optional CSV with expansion bounds summary")
    args = parser.parse_args()

    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    in_path  = Path(args.network_in  or cfg["paths"]["rescaled_network"])
    out_path = Path(args.network_out or cfg["paths"]["expanded_network"])
    tables   = Path(cfg["paths"].get("tables_dir", "results/tables"))
    report_out = Path(args.report_out) if args.report_out else tables / "tx_expansion_bounds.csv"
    tables.mkdir(parents=True, exist_ok=True)

    n = pio.load_network(in_path)

    xp = cfg.get("parameters", {}).get("expansion", {})
    lines_cfg = xp.get("lines", {})
    links_cfg = xp.get("links", {})

    report = enable_transmission_expansion(
        n,
        lines_enable=bool(lines_cfg.get("enable", True)),
        links_enable=bool(links_cfg.get("enable", True)),
        line_abs_max=float(lines_cfg.get("absolute_max", 1e15)),
        link_abs_max=float(links_cfg.get("absolute_max", 1e15)),
        min_equals_current=bool(lines_cfg.get("min_equals_current", True)),
    )

    report.to_csv(report_out, index=False)
    pio.save_network(n, out_path)
    logging.info(f"Wrote: {out_path}")
    logging.info(f"Report: {report_out}")

if __name__ == "__main__":
    main()

