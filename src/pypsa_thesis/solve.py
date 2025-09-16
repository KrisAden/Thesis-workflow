import logging
import pandas as pd

# Attach duals only if the constraint exists
def _assign_dual_if_present(n, cname):
    if hasattr(n, "model") and cname in n.model.constraints:
        try:
            mu = n.model.dual[n.model.constraints[cname]]
            # TODO: map mu into the right n.* table; depends on your structure
        except Exception:
            logging.exception("Failed assigning duals for %s", cname)
    else:
        logging.info("Constraint %s not in model; skipping dual assignment.", cname)
# src/pypsa_thesis/solve.py
from __future__ import annotations

import argparse
import logging
import re
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
    solve = (cfg.get("parameters", {}) or {}).get("solve", {}) or {}
    name = str(solve.get("solver", "gurobi"))
    opts = solve.get("solver_options", None)
    if opts is not None and not isinstance(opts, dict):
        raise ValueError("parameters.solve.solver_options must be a mapping (YAML dict).")
    return name, opts


_STRING_PARAMS_OK = {
    "LogFile", "ResultFile", "CSManager", "ComputeServer", "Server", "Token",
    "WLSAccessID", "WLSSecret", "LicenseID"
}

def _sanitize_gurobi_opts(opts: Optional[Dict]) -> Optional[Dict]:
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
                pass
        out[k] = v
    return out


# ---------------------------
# Emissions accounting & cap
# ---------------------------

def compute_total_co2(n: pypsa.Network) -> float:
    """
    Sum CO₂ = sum_t sum_g p_t,g * co2_factor_g
    Robust to missing columns and misaligned indexes.
    """
    if not hasattr(n, "generators_t") or "p" not in n.generators_t:
        return 0.0

    p = n.generators_t.p                   # DataFrame [snapshots x generators]
    co2 = n.generators.get("co2_factor")   # Series indexed by generator

    if co2 is None:
        return 0.0

    # Align precisely to p's columns, fill missing with 0
    co2 = co2.reindex(p.columns).fillna(0.0).astype(float)

    # Optional: apply snapshot weightings if present
    try:
        w = n.snapshot_weightings["generators"]  # PyPSA >=0.20 style
        p = p.mul(w, axis=0)
    except Exception:
        pass

    # Multiply elementwise and sum — avoids pandas .dot index pitfalls
    return float((p.mul(co2, axis=1)).to_numpy().sum())


def add_global_co2_cap(n: pypsa.Network, cap_tco2: float) -> None:
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
# Helper to run optimize with pre-save
# ---------------------------

def _run_opt(n: pypsa.Network, solver_name: str, solver_options: Optional[Dict], rep_path: Path, out_path: Path) -> None:
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
        logging.exception("Optimization raised exception (status=%s, termination=%s). Debug: %s",
                          status, termination, dbg_path)
        raise

    ok = str(status).lower() in {"ok", "optimal", "success"} or str(termination).lower() in {"optimal"}
    if not ok:
        dbg_path = Path(rep_path).with_suffix(".debug.csv")
        pd.DataFrame([{
            "status": str(status),
            "termination_condition": str(termination),
            "error": "",
        }]).to_csv(dbg_path, index=False)
        logging.error("Optimization finished non-OK (status=%s, termination=%s). Debug: %s",
                      status, termination, dbg_path)
        raise RuntimeError(f"Non-OK optimization result: status={status}, termination={termination}")


# ---------------------------
# CLI main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Solve network with optional global CO₂ cap (baseline or constrained).")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--network-in", help="Input network (defaults to cfg.paths.costed_network)")
    ap.add_argument("--network-out", help="Output solved network (.nc)")
    ap.add_argument("--report-out", help="CSV report with objective/status/emissions")
    ap.add_argument("--reduction", type=float, default=0.0)
    ap.add_argument("--write-baseline", help="CSV to write baseline_emissions (only when reduction=0).")
    ap.add_argument("--baseline-file", help="CSV with 'baseline_emissions' column (required when reduction>0).")
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
    logging.info(f"Solver: {solver_name} | options: {solver_options or {}} | reduction: {args.reduction:.2%}")

    # Always save pre-optimization network before calling optimize
    pre_path = out_path.with_name(out_path.stem + "_preopt.nc")
    pio.save_network(n, pre_path)
    logging.info(f"Saved pre-optimization network: {pre_path}")

    if args.reduction <= 0.0 + 1e-12:
        _run_opt(n, solver_name, solver_options, rep_path, out_path)
        # Always save the solved network, even if CO2 calculation fails
        pio.save_network(n, out_path)
        logging.info(f"Wrote solved network: {out_path}")

        # Attach duals for constraints if needed
        # for cname in ["co2_cap", ...]:
        #     _assign_dual_if_present(n, cname)

        # Baseline CO2 (optional)
        baseline = None
        if args.write_baseline:
            Path(args.write_baseline).parent.mkdir(parents=True, exist_ok=True)
            try:
                baseline = compute_total_co2(n)
                pd.DataFrame([{"co2_total": baseline}]).to_csv(args.write_baseline, index=False)
                logging.info("Wrote baseline emissions: %s", args.write_baseline)
            except Exception:
                logging.exception("Baseline CO2 calculation failed; writing zero.")
                pd.DataFrame([{"co2_total": 0.0}]).to_csv(args.write_baseline, index=False)
                baseline = 0.0
        else:
            try:
                baseline = compute_total_co2(n)
                logging.info(f"Baseline emissions (tCO2): {baseline:,.6f}")
            except Exception:
                logging.exception("Baseline CO2 calculation failed.")
                baseline = 0.0

        pd.DataFrame([{
            "reduction": 0.0,
            "objective": getattr(n, "objective", float("nan")),
            "status": str(getattr(n, "status", "")),
            "termination_condition": str(getattr(n, "termination_condition", "")),
            "allowed_emissions": baseline,
            "actual_emissions": baseline,
        }]).to_csv(rep_path, index=False)
        return

    if not args.baseline_file:
        raise SystemExit("Constrained run requires --baseline-file with 'baseline_emissions'.")

    bl = pd.read_csv(args.baseline_file)
    if "baseline_emissions" not in bl.columns or bl.empty:
        raise SystemExit(f"{args.baseline_file} missing 'baseline_emissions'")

    baseline = float(bl["baseline_emissions"].iloc[0])
    cap = baseline * (1.0 - float(args.reduction))
    add_global_co2_cap(n, cap)
    logging.info(f"Applied CO₂ cap {cap:,.6f} (baseline {baseline:,.6f}, reduction {args.reduction:.2%})")

    _run_opt(n, solver_name, solver_options, rep_path, out_path)
    actual = compute_total_co2(n)
    pio.save_network(n, out_path)
    pd.DataFrame([{
        "reduction": float(args.reduction),
        "objective": getattr(n, "objective", float("nan")),
        "status": str(getattr(n, "status", "")),
        "termination_condition": str(getattr(n, "termination_condition", "")),
        "allowed_emissions": cap,
        "actual_emissions": actual,
    }]).to_csv(rep_path, index=False)


if __name__ == "__main__":
    main()
