# src/pypsa_thesis/try_solve.py
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import pypsa

from . import io as pio

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

def add_load_shedding(n: pypsa.Network, penalty_eur_per_mwh: float = 1e6) -> int:
    if "load_shedding" not in n.carriers.index:
        n.add("Carrier", "load_shedding")
    buses = n.buses.index
    n.madd(
        "Generator",
        buses + " Load Shedding",
        bus=buses,
        carrier="load_shedding",
        p_nom_extendable=True,
        marginal_cost=float(penalty_eur_per_mwh),
        capital_cost=0.0,
        efficiency=1.0,
    )
    return len(buses)

def main() -> None:
    ap = argparse.ArgumentParser(description="Try to optimize a given network (feasibility probe).")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--network-in", required=True, help="Input network (.nc) to test")
    ap.add_argument("--network-out", default="results/networks/try.nc")
    ap.add_argument("--report-out", default="results/tables/try.csv")
    ap.add_argument("--snapshots", type=int, default=0, help="If >0, keep only first N snapshots.")
    ap.add_argument("--add-load-shedding", action="store_true", help="Add high-penalty load shedding to diagnose infeasibility.")
    ap.add_argument("--load-shedding-cost", type=float, default=1e6, help="€/MWh penalty for load shedding.")
    args = ap.parse_args()

    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    in_path  = Path(args.network_in)
    out_path = Path(args.network_out)
    rep_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and optionally slice snapshots
    n = pio.load_network(in_path)
    if args.snapshots and args.snapshots > 0 and len(n.snapshots) > args.snapshots:
        n.set_snapshots(n.snapshots[: args.snapshots])
        logging.info(f"Kept first {args.snapshots} snapshots for quick test.")

    # Optional: add load shedding
    if args.add_load_shedding:
        k = add_load_shedding(n, args.load_shedding_cost)
        logging.info(f"Added {k} load-shedding generators at {args.load_shedding_cost:.0f} €/MWh.")

    # ALWAYS save pre-optimization network
    pre_path = out_path.with_name(out_path.stem + "_preopt.nc")
    pio.save_network(n, pre_path)
    logging.info(f"Saved pre-optimization network: {pre_path}")

    # Solver sanitize + log
    solver_name, solver_options = _solver_from_cfg(cfg)
    solver_options = _sanitize_gurobi_opts(solver_options)
    logging.info(f"Solver: {solver_name} | options: {solver_options or {}}")

    # Try optimize and ALWAYS emit a debug artifact on failure
    status = termination = None
    try:
        res = n.optimize(solver_name=solver_name, solver_options=solver_options)
        if isinstance(res, tuple) and len(res) == 2:
            status, termination = res
        else:
            status = getattr(n, "status", None)
            termination = getattr(n, "termination_condition", None)
    except Exception as e:
        dbg_path = rep_path.with_suffix(".debug.csv")
        pd.DataFrame([{
            "status": str(getattr(n, "status", "")),
            "termination_condition": str(getattr(n, "termination_condition", "")),
            "error": repr(e),
        }]).to_csv(dbg_path, index=False)
        logging.exception("Optimization raised exception. Debug: %s", dbg_path)
        raise

    # Save solved (even if not optimal, for inspection)
    pio.save_network(n, out_path)

    ok = str(status).lower() in {"ok", "optimal", "success"} or str(termination).lower() in {"optimal"}
    pd.DataFrame([{
        "status": str(status),
        "termination_condition": str(termination),
        "snapshots_used": len(n.snapshots),
    }]).to_csv(rep_path, index=False)

    if not ok:
        dbg_path = rep_path.with_suffix(".debug.csv")
        pd.DataFrame([{
            "status": str(status),
            "termination_condition": str(termination),
            "error": "",
        }]).to_csv(dbg_path, index=False)
        logging.error("Non-OK result (status=%s, termination=%s). Debug: %s", status, termination, dbg_path)
        raise SystemExit(f"Non-OK optimization: status={status}, termination={termination}")

if __name__ == "__main__":
    main()
