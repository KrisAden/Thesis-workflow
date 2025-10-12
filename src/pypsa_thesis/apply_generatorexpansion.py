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
def _bound_violations_gen(n: pypsa.Network) -> pd.DataFrame:
    """
    Collect bound violations for generator assets) where min > max.
    """
    rows = []

    if len(n.generators) and {"p_nom_min", "p_nom_max"}.issubset(n.generators.columns):
        bad = (
            n.generators.p_nom_min.notnull()
            & n.generators.p_nom_max.notnull()
            & (n.generators.p_nom_min > n.generators.p_nom_max)
        )
        if bad.any():
            for name in n.generators.index[bad]:
                rows.append(("generators", name))

    if not rows:
        return pd.DataFrame(columns=["component", "name"])

    return pd.DataFrame(rows, columns=["component", "name"])

#Adding Generator expansion
def enable_generator_expansion(
    n: pypsa.Network,
    rel_caps: dict[str, float] | None = None,
    *,
    only_extendable: bool = True,
    min_equals_current: bool = True,
    floor: float = 1.0,
) -> None:
    """
    Apply p_nom_max caps of the form: p_nom_max = p_nom * multiplier,
    with special handling for very large multipliers (effectively unlimited).

    Parameters:
    - rel_caps: mapping like {"geothermal": 10.0, "nuclear": 5.0, "biomass": 2.0} 
                where values are multipliers for p_nom_max = p_nom * multiplier
                If multiplier >= 1e9, sets p_nom_max = 1e9 (effectively unlimited)
    - only_extendable: if True, only apply to generators with p_nom_extendable=True
    - min_equals_current: if True, set p_nom_min = current p_nom to preserve existing capacity
    - floor: minimum value for the calculated p_nom_max
    """
    if not rel_caps or not len(n.generators):
        return

    g = n.generators

    # Ensure required columns exist
    if "p_nom_extendable" not in g:
        g["p_nom_extendable"] = False
    if "p_nom_max" not in g:
        g["p_nom_max"] = np.inf

    # Determine which generators to apply caps to
    ext = g["p_nom_extendable"].fillna(False).astype(bool) if only_extendable else pd.Series(True, index=g.index)
    g["p_nom"] = g["p_nom"].fillna(0.0)

    # Normalize carrier names for matching
    carr = g["carrier"].astype(str).str.lower()

    logging.info("🔧 APPLYING GENERATOR EXPANSION LIMITS:")
    logging.info("-" * 50)

    for carrier_name, mult in (rel_caps or {}).items():
        c_lower = str(carrier_name).lower()
        # Match by carrier only (no bus restriction)
        m = (carr == c_lower) & ext
        if not m.any():
            logging.info(f"⚠️  No extendable {carrier_name} generators found")
            continue
        
        generators_affected = m.sum()
        logging.info(f"🔄 Processing {carrier_name}: {generators_affected} generators")
        
        # Handle very large multipliers as "unlimited"
        if float(mult) >= 1e9:
            # Set to effectively unlimited capacity
            old_max_sample = g.loc[m, "p_nom_max"].iloc[0] if len(g.loc[m]) > 0 else 0
            g.loc[m, "p_nom_max"] = 1e9
            logging.info(f"  ✅ Set UNLIMITED capacity (1e9 MW) for {carrier_name}")
            logging.info(f"     Example: {old_max_sample:.0f} MW → 1,000,000,000 MW")
        else:
            # Calculate new cap: p_nom * multiplier, with floor
            cap = (g.loc[m, "p_nom"].astype(float) * float(mult)).clip(lower=floor)
            
            # Apply minimum of existing max and new cap
            old = g.loc[m, "p_nom_max"].replace([np.inf, -np.inf], np.inf)
            g.loc[m, "p_nom_max"] = np.minimum(old, cap)
            
            avg_new_cap = g.loc[m, "p_nom_max"].mean()
            logging.info(f"  📊 Applied {mult}x multiplier for {carrier_name}")
            logging.info(f"     Average capacity limit: {avg_new_cap:,.0f} MW")

        if min_equals_current:
            # Keep existing capacity but never above the max
            g.loc[m, "p_nom_min"] = np.minimum(g.loc[m, "p_nom"], g.loc[m, "p_nom_max"])

        # Guard: fix any min > max violations
        bad = m & g["p_nom_min"].notnull() & (g["p_nom_min"] > g["p_nom_max"])
        if bad.any():
            g.loc[bad, "p_nom_min"] = g.loc[bad, "p_nom_max"]
            logging.warning(f"🔧 Fixed {bad.sum()} min > max violations for {carrier_name}")

    # VERIFICATION SUMMARY
    logging.info("\n🔍 VERIFICATION SUMMARY:")
    logging.info("-" * 30)
    renewable_carriers = ["solar", "onwind", "offwind-ac", "offwind-dc"]
    for carrier in renewable_carriers:
        carrier_gens = g[g["carrier"] == carrier]
        if len(carrier_gens) > 0:
            unlimited_count = (carrier_gens["p_nom_max"] >= 1e9).sum()
            total_count = len(carrier_gens)
            if unlimited_count == total_count:
                logging.info(f"✅ {carrier}: ALL {total_count} generators unlimited")
            else:
                logging.info(f"❌ {carrier}: {unlimited_count}/{total_count} generators unlimited")
                # Show examples of limited ones
                limited = carrier_gens[carrier_gens["p_nom_max"] < 1e9]
                if len(limited) > 0:
                    sample_limit = limited["p_nom_max"].iloc[0]
                    logging.info(f"   Example limited: {sample_limit:,.0f} MW")

#Adding main function to interface with config file and Snakemake
def main():
    parser = argparse.ArgumentParser(
        description="Enable generator expansion"
    )

    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--network-in", help="Override input network (defaults to cfg.paths.network_costed)"
    )

    parser.add_argument(
        "--network-out",
        help="Override output network (defaults to cfg.paths.network_costed_generatorexpansion)",
    )
    
    parser.add_argument(
        "--report-out", help="Optional CSV with expansion bounds summary"
    )

    parser.add_argument(
        "--violations-out",
        help="Optional CSV with any generator bound violations (min>max).",
    )

    args = parser.parse_args()

    #Importing arguments from config
    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    #Adding paths from config and defaults
    in_path = Path(args.network_in or cfg["paths"].get("costed_network", "data/interim/network_costed.nc"))
    out_path = Path(args.network_out or "data/interim/network_costed_gen.nc")
    tables = Path(cfg["paths"].get("tables_dir", "results/tables"))

    # NEW: ensure parents exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    report_out = Path(args.report_out or (tables / "gen_expansion_bounds.csv"))
    violations_out = Path(args.violations_out or (tables / "gen_bound_violations.csv"))
    report_out.parent.mkdir(parents=True, exist_ok=True)
    violations_out.parent.mkdir(parents=True, exist_ok=True)

    n = pio.load_network(in_path)

    # Get generator expansion configuration
    xp = cfg.get("parameters", {}).get("expansion", {}) or {}
    gen_cfg = xp.get("generators", {}) or {}

    # Apply generator expansion with relative multipliers
    rel_caps = gen_cfg.get("relative_multiplier_by_carrier", {})
    enable_generator_expansion(
        n,
        rel_caps=rel_caps,
        only_extendable=bool(gen_cfg.get("only_extendable", True)),
        min_equals_current=bool(gen_cfg.get("min_equals_current", True)),
        floor=float(gen_cfg.get("floor", 1.0)),
    )

    # Write a quick sanity report for contradictions (min>max)
    vdf = _bound_violations_gen(n)
    if len(vdf):
        vdf.to_csv(violations_out, index=False)
        logging.error("Found %d generator bound violations. Wrote: %s", len(vdf), violations_out)
    else:
        # write an empty file to make the pipeline robust
        vdf.to_csv(violations_out, index=False)
        logging.info("No generator bound violations.")

    # Create a simple report
    gen_summary = []
    if len(n.generators):
        for carrier in n.generators["carrier"].unique():
            carrier_gens = n.generators[n.generators["carrier"] == carrier]
            gen_summary.append({
                "carrier": carrier,
                "count": len(carrier_gens),
                "extendable_count": carrier_gens["p_nom_extendable"].sum() if "p_nom_extendable" in carrier_gens else 0,
                "total_p_nom": carrier_gens["p_nom"].sum(),
                "avg_p_nom_max": carrier_gens["p_nom_max"].replace([np.inf, -np.inf], np.nan).mean()
            })
    
    report_df = pd.DataFrame(gen_summary)
    report_df.to_csv(report_out, index=False)
    
    pio.save_network(n, out_path)
    logging.info("Wrote: %s", out_path)
    logging.info("Report: %s", report_out)
    logging.info("Violations: %s", violations_out)


if __name__ == "__main__":
    main()