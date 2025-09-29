# src/pypsa_thesis/solve.py
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
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
    solve = (cfg.get("parameters", {}) or {}).get("solve", {}) or {}
    name = str(solve.get("solver", "gurobi"))
    opts = solve.get("solver_options", None)
    if opts is not None and not isinstance(opts, dict):
        raise ValueError("parameters.solve.solver_options must be a mapping (YAML dict).")
    return name, opts


# gurobi options that are allowed to remain strings
_STRING_PARAMS_OK = {
    "LogFile", "ResultFile", "CSManager", "ComputeServer", "Server", "Token",
    "WLSAccessID", "WLSSecret", "LicenseID"
}

def _sanitize_gurobi_opts(opts: Optional[Dict]) -> Optional[Dict]:
    """Convert numeric strings to int/float except for whitelisted string params."""
    if not opts:
        return opts
    out = {}
    for k, v in opts.items():
        if isinstance(v, str) and k not in _STRING_PARAMS_OK:
            s = v.strip()
            try:
                if re.fullmatch(r"[+-]?\d+", s):
                    v = int(s)
                else:
                    v = float(s)
            except Exception:
                # leave as-is if it can't be parsed numerically
                pass
        out[k] = v
    return out


# ---------------------------
# Emissions accounting & cap
# ---------------------------

def compute_total_co2(n: pypsa.Network) -> float:
    """
    Compute total CO2 in *tCO2*, aligned with a primary-energy style cap using
    carriers['co2_emissions'] in tCO2/MWh_fuel.

    Counts:
      - Generators: fuel_MWh = (electric_MWh / efficiency)
      - Links: fuel_MWh from input side (-p0), no extra efficiency factor
    Applies snapshot_weightings (column 'generators' if present, else the Series).
    """

    # snapshot weights (hours represented by each snapshot)
    w = getattr(n, "snapshot_weightings", None)
    if w is None:
        raise ValueError("Network has no snapshot_weightings")
    if hasattr(w, "generators"):
        w = w.generators  # PyPSA≥0.20 DataFrame column
    # else assume it's already a Series of weights

    # emission factors by carrier (tCO2 per MWh_fuel)
    if "co2_emissions" not in n.carriers.columns:
        return 0.0
    ef_by_carrier = pd.to_numeric(n.carriers["co2_emissions"], errors="coerce").fillna(0.0)

    total_t = 0.0

    # ----- Generators -----
    if not n.generators.empty and hasattr(n.generators_t, "p"):
        P = n.generators_t.p  # MW_e
        E_e = P.mul(w, axis=0).sum(axis=0)  # MWh_e per generator
        eff = n.generators.get("efficiency", pd.Series(1.0, index=n.generators.index)).replace(0, np.nan).fillna(1.0)
        E_fuel = (E_e / eff).fillna(E_e)  # MWh_fuel
        ef = n.generators["carrier"].map(ef_by_carrier).fillna(0.0)  # t/MWh_fuel
        total_t += float((E_fuel * ef).sum())

    # ----- Links (fuel input on bus0) -----
    # Assumes thermal links consume fuel on bus0: P_in = -p0 (positive when consuming).
    if not n.links.empty and hasattr(n.links_t, "p0"):
        P_in = (-n.links_t.p0).clip(lower=0.0)  # MW_fuel
        E_fuel = P_in.mul(w, axis=0).sum(axis=0)  # MWh_fuel per link

        # Try link.carrier → carriers.co2_emissions; fallback via bus0's carrier
        ef_link = pd.Series(0.0, index=n.links.index)
        if "carrier" in n.links.columns:
            ef_link = n.links["carrier"].map(ef_by_carrier).fillna(0.0)
        if "bus0" in n.links.columns and "carrier" in n.buses.columns:
            ef_bus0 = n.links["bus0"].map(n.buses["carrier"]).map(ef_by_carrier).fillna(0.0)
            ef_link = ef_link.where(ef_link != 0.0, ef_bus0)

        total_t += float((E_fuel * ef_link).sum())

    return total_t  # tCO2




def add_global_co2_cap(n: pypsa.Network, cap_tco2: float) -> None:
    if "co2_cap" in getattr(n, "global_constraints", pd.DataFrame()).index:
        n.global_constraints.at["co2_cap", "constant"] = float(cap_tco2)
        n.global_constraints.at["co2_cap", "type"] = "primary_energy"
        n.global_constraints.at["co2_cap", "carrier_attribute"] = "co2_emissions"
    else:
        n.add(
            "GlobalConstraint", "co2_cap",
            sense="<=", constant=float(cap_tco2),
            type="primary_energy",
            carrier_attribute="co2_emissions",
        )


def _normalize_reduction(x: float) -> float:
    """
    Accept 0.5 or 50 for '50%'. Returns fraction in [0,1].
    """
    x = float(x)
    if x > 1.0:
        x = x / 100.0
    return max(0.0, min(1.0, x))

# ---------------------------
# Helper to run optimize with pre-save
# ---------------------------

def _run_opt(
    n: pypsa.Network,
    solver_name: str,
    solver_options: Optional[Dict],
    rep_path: Path,
    out_path: Path,
) -> None:
    # Info-log each solver option (helps reproducibility)
    for k, v in (solver_options or {}).items():
        logging.info(f"opt {k}: {v!r} (type={type(v).__name__})")

    # Save pre-optimization snapshot
    pre_path = out_path.with_name(out_path.stem + "_preopt.nc")
    pio.save_network(n, pre_path)
    logging.info(f"Saved pre-optimization network: {pre_path}")

    status = termination = None
    try:
        res = n.optimize(solver_name=solver_name, solver_options=solver_options)
        if isinstance(res, tuple) and len(res) == 2:
            status, termination = res
        else:
            status = getattr(n, "status", None)
            termination = getattr(n, "termination_condition", None)
    except Exception as e:
        status = getattr(n, "status", None)
        termination = getattr(n, "termination_condition", None)
        dbg_path = Path(rep_path).with_suffix(".debug.csv")
        pd.DataFrame([{
            "status": str(status),
            "termination_condition": str(termination),
            "error": repr(e),
        }]).to_csv(dbg_path, index=False)
        logging.exception(
            "Optimization raised exception (status=%s, termination=%s). Debug: %s",
            status, termination, dbg_path
        )
        raise

    ok = str(status).lower() in {"ok", "optimal", "success"} or str(termination).lower() in {"optimal"}
    if not ok:
        dbg_path = Path(rep_path).with_suffix(".debug.csv")
        pd.DataFrame([{
            "status": str(status),
            "termination_condition": str(termination),
            "error": "",
        }]).to_csv(dbg_path, index=False)
        logging.error(
            "Optimization finished non-OK (status=%s, termination=%s). Debug: %s",
            status, termination, dbg_path
        )
        raise RuntimeError(f"Non-OK optimization result: status={status}, termination={termination}")


# ---------------------------
# CLI main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Solve network with optional global CO₂ cap (baseline or constrained)."
    )
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--network-in", help="Input network (defaults to cfg.paths.costed_network)")
    ap.add_argument("--network-out", help="Output solved network (.nc)")
    ap.add_argument("--report-out", help="CSV report with objective/status/emissions")
    ap.add_argument("--reduction", type=float, default=0.0,
                    help="CO2 reduction as fraction (0–1) or percent (0–100).")
    ap.add_argument("--write-baseline", help="CSV to write baseline_emissions (only when reduction=0).")
    ap.add_argument("--baseline-file", help="CSV with 'baseline_emissions' (or legacy 'co2_total') column (required when reduction>0).")
    args = ap.parse_args()

    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    in_path = Path(args.network_in or cfg["paths"]["costed_network"])
    out_path = Path(args.network_out or "results/networks/solved.nc")
    rep_path = Path(args.report_out or "results/tables/solve.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.parent.mkdir(parents=True, exist_ok=True)

    n = pio.load_network(in_path)

    solver_name, solver_options = _solver_from_cfg(cfg)
    solver_options = _sanitize_gurobi_opts(solver_options)

    red_frac = _normalize_reduction(args.reduction)
    logging.info(
        "Solver: %s | options: %s | reduction: %.2f%%",
        solver_name, (solver_options or {}), 100.0 * red_frac
    )

    # Save pre-optimization snapshot
    pre_path = out_path.with_name(out_path.stem + "_preopt.nc")
    pio.save_network(n, pre_path)
    logging.info(f"Saved pre-optimization network: {pre_path}")

    if red_frac <= 0.0 + 1e-12:
        # BASELINE (unconstrained)
        _run_opt(n, solver_name, solver_options, rep_path, out_path)
        pio.save_network(n, out_path)
        logging.info(f"Wrote solved network: {out_path}")

        # Compute and write baseline
        baseline = compute_total_co2(n)
        logging.info("Baseline emissions (tCO2): %,.6f", baseline)

        if args.write_baseline:
            Path(args.write_baseline).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"baseline_emissions": baseline}]).to_csv(args.write_baseline, index=False)
            logging.info("Wrote baseline file: %s", args.write_baseline)

        pd.DataFrame([{
            "reduction": 0.0,
            "objective": getattr(n, "objective", float("nan")),
            "status": str(getattr(n, "status", "")),
            "termination_condition": str(getattr(n, "termination_condition", "")),
            "allowed_emissions": baseline,
            "actual_emissions": baseline,
        }]).to_csv(rep_path, index=False)
        return

    # CONSTRAINED
    if not args.baseline_file:
        raise SystemExit("Constrained run requires --baseline-file with 'baseline_emissions'.")

    bl = pd.read_csv(args.baseline_file)
    if bl.empty or "baseline_emissions" not in bl.columns:
        raise SystemExit(f"{args.baseline_file} missing 'baseline_emissions'")

    baseline = float(bl["baseline_emissions"].iloc[0])
    cap = baseline * (1.0 - red_frac)
    logging.info("Applied CO₂ cap %,.6f (baseline %,.6f, reduction %.2f%%)", cap, baseline, 100.0 * red_frac)

    add_global_co2_cap(n, cap)

    _run_opt(n, solver_name, solver_options, rep_path, out_path)
    actual = compute_total_co2(n)
    pio.save_network(n, out_path)
    pd.DataFrame([{
        "reduction": red_frac,
        "objective": getattr(n, "objective", float("nan")),
        "status": str(getattr(n, "status", "")),
        "termination_condition": str(getattr(n, "termination_condition", "")),
        "allowed_emissions": cap,
        "actual_emissions": actual,
    }]).to_csv(rep_path, index=False)


if __name__ == "__main__":
    main()
