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


def enable_transmission_expansion(
    n: pypsa.Network,
    *,
    lines_enable: bool = True,
    links_enable: bool = True,
    line_abs_max: float = np.inf,
    link_abs_max: float = np.inf,
    lines_min_equals_current: bool = True,
    links_min_equals_current: bool = True,
) -> pd.DataFrame:
    """
    Enable expansion on AC lines (s_nom) and DC links (p_nom).

    - Sets *_extendable=True.
    - If *_min_equals_current=True, set min to current capacity so existing capacity
      cannot be removed.
    - If absolute max is finite, set *_max accordingly.
    - Guards against NaNs and contradictory bounds (min > max) by raising max to current.

    Returns a small report dataframe.
    """
    rows = []

    # Guard: ensure finite current capacities before using as mins
    if len(n.lines):
        n.lines["s_nom"] = n.lines["s_nom"].fillna(0.0)
    if len(n.links):
        n.links["p_nom"] = n.links["p_nom"].fillna(0.0)

    # Lines (AC)
    if lines_enable and len(n.lines):
        n.lines["s_nom_extendable"] = True

        if lines_min_equals_current:
            n.lines["s_nom_min"] = n.lines["s_nom"].astype(float)

        if np.isfinite(line_abs_max):
            n.lines["s_nom_max"] = float(line_abs_max)

        # If dataset already had s_nom_max below current s_nom, raise it
        if "s_nom_max" in n.lines:
            bad = n.lines.s_nom_max.notnull() & (n.lines.s_nom > n.lines.s_nom_max)
            if bad.any():
                logging.warning(
                    "Raising %d lines' s_nom_max up to current s_nom to satisfy s_nom_min <= s_nom_max.",
                    bad.sum(),
                )
                n.lines.loc[bad, "s_nom_max"] = n.lines.loc[bad, "s_nom"]

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

        if links_min_equals_current:
            n.links["p_nom_min"] = n.links["p_nom"].astype(float)

        if np.isfinite(link_abs_max):
            n.links["p_nom_max"] = float(link_abs_max)

        # If dataset already had p_nom_max below current p_nom, raise it
        if "p_nom_max" in n.links:
            bad = n.links.p_nom_max.notnull() & (n.links["p_nom"] > n.links.p_nom_max)
            if bad.any():
                logging.warning(
                    "Raising %d links' p_nom_max up to current p_nom to satisfy p_nom_min <= p_nom_max.",
                    bad.sum(),
                )
                n.links.loc[bad, "p_nom_max"] = n.links.loc[bad, "p_nom"]

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

def cap_generator_expansion_absolute(
    n: pypsa.Network,
    abs_caps: dict[str, float] | None = None,
    *,
    only_extendable: bool = True,
    min_equals_current: bool = True,
) -> None:
    """
    Apply absolute p_nom_max caps to generators by carrier.

    - abs_caps: mapping like {"biomass": 36000, "nuclear": 64000} (MW).
    - only_extendable: cap only rows with p_nom_extendable==True.
    - min_equals_current: ensure p_nom_min = current so existing capacity isn't removed.
    """
    if not abs_caps or not len(n.generators):
        return

    g = n.generators
    # normalize carrier names to lower for matching
    carr_lower = g["carrier"].astype(str).str.lower()

    if "p_nom_extendable" not in g:
        g["p_nom_extendable"] = False

    ext_mask_all = g["p_nom_extendable"].fillna(False).astype(bool) if only_extendable else pd.Series(True, index=g.index)

    # ensure columns exist
    if "p_nom_max" not in g:
        g["p_nom_max"] = np.inf
    g["p_nom"] = g["p_nom"].fillna(0.0)

    for carr, cap in (abs_caps or {}).items():
        cap = float(cap)
        mask = (carr_lower == str(carr).lower()) & ext_mask_all
        if not mask.any():
            continue

        # new max = min(existing max, absolute cap)
        old = g.loc[mask, "p_nom_max"].replace([np.inf, -np.inf], np.inf)
        g.loc[mask, "p_nom_max"] = np.minimum(old, cap)

        if min_equals_current:
            # Keep existing capacity: p_nom_min = current, but not above max
            g.loc[mask, "p_nom_min"] = np.minimum(g.loc[mask, "p_nom"], g.loc[mask, "p_nom_max"])

        # Guard: if someone had min > max, fix by lowering min
        bad = mask & g["p_nom_min"].notnull() & (g["p_nom_min"] > g["p_nom_max"])
        if bad.any():
            g.loc[bad, "p_nom_min"] = g.loc[bad, "p_nom_max"]



def main():
    parser = argparse.ArgumentParser(
        description="Enable transmission expansion on a PyPSA network."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--network-in", help="Override input .nc (defaults to cfg.paths.rescaled_network)"
    )
    parser.add_argument(
        "--network-out",
        help="Override output .nc (defaults to cfg.paths.expanded_network)",
    )
    parser.add_argument(
        "--report-out", help="Optional CSV with expansion bounds summary"
    )
    parser.add_argument(
        "--violations-out",
        help="Optional CSV with any line/link bound violations (min>max).",
    )
    args = parser.parse_args()

    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    in_path = Path(args.network_in or cfg["paths"]["rescaled_network"])
    out_path = Path(args.network_out or cfg["paths"]["expanded_network"])

    tables = Path(cfg["paths"].get("tables_dir", "results/tables"))
    tables.mkdir(parents=True, exist_ok=True)

    report_out = Path(args.report_out) if args.report_out else tables / "tx_expansion_bounds.csv"
    violations_out = (
        Path(args.violations_out) if args.violations_out else tables / "tx_bound_violations.csv"
    )


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

    xp = cfg.get("parameters", {}).get("expansion", {}) or {}
    gen_cfg = xp.get("generators", {}) or {}

    cap_generator_expansion_absolute(
        n,
        abs_caps=gen_cfg.get("absolute_max_by_carrier", {}),
        only_extendable=True,
        min_equals_current=bool(gen_cfg.get("min_equals_current", True)),
    )


    # Write a quick sanity report for contradictions (min>max)
    vdf = _bound_violations_tx(n)
    if len(vdf):
        vdf.to_csv(violations_out, index=False)
        logging.error("Found %d transmission bound violations. Wrote: %s", len(vdf), violations_out)
    else:
        # write an empty file to make the pipeline deterministic
        vdf.to_csv(violations_out, index=False)
        logging.info("No transmission bound violations.")

    report.to_csv(report_out, index=False)
    pio.save_network(n, out_path)
    logging.info("Wrote: %s", out_path)
    logging.info("Report: %s", report_out)
    logging.info("Violations: %s", violations_out)


if __name__ == "__main__":
    main()
