# src/pypsa_thesis/solve.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
import pypsa

from . import io as pio


# ---------------------------
# Logging & solver utilities
# ---------------------------

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _solver_from_cfg(cfg: dict) -> Tuple[str, Optional[Dict]]:
    """
    Read solver name and options from config.

    Expected structure:
    parameters:
      solve:
        solver: "gurobi" | "highs" | "cplex" | ...
        solver_options: { ... }   # passed straight to PyPSA/Pyomo
    """
    solve = (cfg.get("parameters", {}) or {}).get("solve", {}) or {}
    name = str(solve.get("solver", "gurobi"))

    opts = solve.get("solver_options", None)
    if opts is not None and not isinstance(opts, dict):
        raise ValueError("parameters.solve.solver_options must be a mapping (YAML dict).")

    # Nothing fancy here: we pass options through as-is. Your Gurobi options
    # (Threads, Method, Crossover, etc.) will be honored.
    return name, opts


# ---------------------------
# Emissions accounting & cap
# ---------------------------

def compute_total_co2(n: pypsa.Network) -> float:
    """
    Compute total CO₂ emissions (tCO2) from the solved dispatch using
    carrier attribute `co2_emissions` in tCO2/MWh of output.

    Counts:
      - Generators: n.generators_t.p
      - Links:      positive n.links_t.p0 (if link carrier has co2_emissions)
    Weighted by snapshot_weightings if present.
    """
    total = 0.0

    # PyPSA snapshot weightings (v0.35+ has separate columns; fall back gracefully)
    w = n.snapshot_weightings
    w_gen = getattr(w, "generators", w)
    w_lnk = getattr(w, "links", w)

    # Generators
    if hasattr(n, "generators_t") and "p" in n.generators_t and len(n.generators):
        # factor per generator index
        fac_g = n.generators["carrier"].map(
            lambda c: float(n.carriers.at[c, "co2_emissions"])
            if ("co2_emissions" in n.carriers and c in n.carriers.index and pd.notna(n.carriers.at[c, "co2_emissions"]))
            else 0.0
        ).fillna(0.0)
        # energy in MWh (sum over time with weights)
        e_g = n.generators_t.p.mul(w_gen, axis=0).sum(axis=0).fillna(0.0)
        total += float(e_g.dot(fac_g))

    # Links (optional)
    if hasattr(n, "links_t") and "p0" in n.links_t and len(n.links):
        fac_l = n.links["carrier"].map(
            lambda c: float(n.carriers.at[c, "co2_emissions"])
            if ("co2_emissions" in n.carriers and c in n.carriers.index and pd.notna(n.carriers.at[c, "co2_emissions"]))
            else 0.0
        ).fillna(0.0)
        # only positive output from bus0 counts as produced energy (convention)
        e_l = n.links_t.p0.clip(lower=0.0).mul(w_lnk, axis=0).sum(axis=0).fillna(0.0)
        total += float(e_l.dot(fac_l))

    return float(total)


def add_global_co2_cap(n: pypsa.Network, cap_tco2: float) -> None:
    """
    Add a PyPSA GlobalConstraint limiting total carrier-attributed CO2 emissions.
    """
    # If a cap already exists (re-runs), update it instead of adding a duplicate
    if "co2_cap" in getattr(n, "global_constraints", pd.DataFrame()).index:
        n.global_constraints.at["co2_cap", "constant"] = float(cap_tco2)
    else:
        n.add(
            "GlobalConstraint",
            "co2_cap",
            sense="<=",
            constant=float(cap_tco2),
            carrier_attribute="co2_emissions",
        )


# ---------------------------
# CLI main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Solve network with optional global CO₂ cap (baseline or constrained).")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--network-in", help="Input network (defaults to cfg.paths.costed_network)")
    ap.add_argument("--network-out", help="Output solved network (.nc)")
    ap.add_argument("--report-out", help="CSV report with objective/status/emissions")
    ap.add_argument("--reduction", type=float, default=0.0,
                    help="Fractional reduction vs baseline (0..1). 0 = baseline (no cap).")
    ap.add_argument("--write-baseline", help="CSV to write baseline_emissions (only used when reduction=0).")
    ap.add_argument("--baseline-file", help="CSV with 'baseline_emissions' column (required when reduction>0).")
    args = ap.parse_args()

    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    in_path = Path(args.network_in or cfg["paths"]["costed_network"])
    out_path = Path(args.network_out or "results/networks/solved.nc")
    rep_path = Path(args.report_out or "results/tables/solve.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.parent.mkdir(parents=True, exist_ok=True)

    # Load network
    n = pio.load_network(in_path)

    # Solver selection + options (passed through as-is)
    solver_name, solver_options = _solver_from_cfg(cfg)
    logging.info(f"Solver: {solver_name} | options: {solver_options or {}} | reduction: {args.reduction:.2%}")

    # ---------------- Baseline solve (no cap) ----------------
    if args.reduction <= 0.0 + 1e-12:
        n.lopf(pyomo=True, solver_name=solver_name, solver_options=solver_options)
        baseline = compute_total_co2(n)
        logging.info(f"Baseline emissions (tCO2): {baseline:,.6f}")

        # Save outputs
        pio.save_network(n, out_path)
        pd.DataFrame([{
            "reduction": 0.0,
            "objective": getattr(n, "objective", float("nan")),
            "status": str(n.solver.get("termination_condition")) if hasattr(n, "solver") else "unknown",
            "allowed_emissions": baseline,
            "actual_emissions": baseline,
        }]).to_csv(rep_path, index=False)

        if args.write_baseline:
            Path(args.write_baseline).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"baseline_emissions": baseline}]).to_csv(args.write_baseline, index=False)

        logging.info(f"Wrote solved baseline: {out_path} | report: {rep_path}")
        return

    # --------------- Constrained solve (cap) -----------------
    if not args.baseline_file:
        raise SystemExit("Constrained run requires --baseline-file pointing to CSV with 'baseline_emissions'.")

    bl = pd.read_csv(args.baseline_file)
    if "baseline_emissions" not in bl.columns or bl.empty:
        raise SystemExit(f"{args.baseline_file} missing 'baseline_emissions'")

    baseline = float(bl["baseline_emissions"].iloc[0])
    cap = baseline * (1.0 - float(args.reduction))
    add_global_co2_cap(n, cap)
    logging.info(f"Applied CO₂ cap (tCO2): {cap:,.6f} from baseline {baseline:,.6f} and reduction {args.reduction:.2%}")

    n.lopf(pyomo=True, solver_name=solver_name, solver_options=solver_options)
    actual = compute_total_co2(n)

    # Save outputs
    pio.save_network(n, out_path)
    pd.DataFrame([{
        "reduction": float(args.reduction),
        "objective": getattr(n, "objective", float("nan")),
        "status": str(n.solver.get("termination_condition")) if hasattr(n, "solver") else "unknown",
        "allowed_emissions": cap,
        "actual_emissions": actual,
    }]).to_csv(rep_path, index=False)

    logging.info(f"Wrote solved constrained: {out_path} | report: {rep_path}")


if __name__ == "__main__":
    main()
