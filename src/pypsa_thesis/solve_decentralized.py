# src/pypsa_thesis/solve_decentralized.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pypsa

from . import io as pio
from .solve import (
    _setup_logging,
    _solver_from_cfg,
    _sanitize_gurobi_opts,
    add_global_co2_cap,
    _normalize_reduction,
    compute_total_co2,
)


def add_nodal_renewable_constraints(
    n: pypsa.Network,
    k: float,
    renewable_carriers: list[str],
    snapshots,
) -> None:
    """
    Add nodal renewable penetration constraints: 1/k <= gamma <= k
    where gamma = renewable_generation / nodal_load for each bus.
    
    Parameters:
    -----------
    n : pypsa.Network
        The network to add constraints to
    k : float
        The constraint parameter (gamma must be between 1/k and k)
    renewable_carriers : list[str]
        List of carrier names considered renewable
    snapshots : pandas.Index
        Snapshots to consider for the constraints
    """
    logging.info(f"Adding nodal renewable constraints with k={k}")
    
    # Get snapshot weights
    if hasattr(n.snapshot_weightings, 'objective'):
        weights = n.snapshot_weightings.objective
    else:
        weights = n.snapshot_weightings
    
    # For each bus with load, add renewable penetration constraints
    buses_with_load = n.loads.bus.unique()
    
    for i, bus in enumerate(buses_with_load):
        logging.info(f"Processing bus {bus} ({i+1}/{len(buses_with_load)})")
        
        # Get loads at this bus
        loads_at_bus = n.loads[n.loads.bus == bus].index
        
        # Skip if no load
        if not len(loads_at_bus):
            continue
            
        # Get renewable generators at this bus
        gens_at_bus = n.generators[
            (n.generators.bus == bus) & 
            (n.generators.carrier.isin(renewable_carriers))
        ].index
        
        if not len(gens_at_bus):
            # No renewable generators at this bus - skip constraint
            logging.info(f"  Skipping {bus}: no renewable generators")
            continue
            
        # Total load energy at this bus (MWh) - use fixed load values
        load_energy = sum(
            (n.loads_t.p_set[load] * weights).sum()
            for load in loads_at_bus
        )
        
        # Skip buses with zero load
        if load_energy <= 0:
            logging.info(f"  Skipping {bus}: zero load energy")
            continue
            
        # Total renewable generation energy variables at this bus (MWh)
        # Use the proper variable access pattern for PyPSA optimization model
        renewable_energy = 0
        for gen in gens_at_bus:
            for t in snapshots:
                renewable_energy += n.model.variables["Generator-p"][gen, t] * weights[t]
        
        # Add constraints: 1/k <= renewable_energy/load_energy <= k
        # Rearranged: renewable_energy >= load_energy/k AND renewable_energy <= k*load_energy
        
        # Lower bound: renewable_energy >= load_energy/k
        n.model.add_constraints(
            renewable_energy >= load_energy / k,
            name=f"renewable_min_{bus}"
        )
        
        # Upper bound: renewable_energy <= k*load_energy  
        n.model.add_constraints(
            renewable_energy <= k * load_energy,
            name=f"renewable_max_{bus}"
        )
        
        logging.info(f"  Added constraints for {bus}: {len(gens_at_bus)} gens, {load_energy:.0f} MWh load")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Solve network with CO₂ cap and nodal renewable penetration constraints."
    )
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--network-in", help="Input network")
    ap.add_argument("--network-out", help="Output solved network (.nc)")
    ap.add_argument("--report-out", help="CSV report with objective/status/emissions")
    ap.add_argument("--reduction", type=float, required=True,
                    help="CO2 reduction as fraction (0–1) or percent (0–100)")
    ap.add_argument("--k-value", type=float, required=True,
                    help="Decentralization parameter k (gamma constrained to [1/k, k])")
    ap.add_argument("--baseline-file", required=True,
                    help="CSV with 'baseline_emissions' column")
    
    args = ap.parse_args()

    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    in_path = Path(args.network_in)
    out_path = Path(args.network_out)
    rep_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.parent.mkdir(parents=True, exist_ok=True)

    n = pio.load_network(in_path)

    solver_name, solver_options = _solver_from_cfg(cfg)
    solver_options = _sanitize_gurobi_opts(solver_options)

    red_frac = _normalize_reduction(args.reduction)
    k_value = float(args.k_value)
    
    logging.info(
        "Solver: %s | reduction: %.2f%% | k-value: %.2f",
        solver_name, 100.0 * red_frac, k_value
    )

    # Load baseline emissions
    bl = pd.read_csv(args.baseline_file)
    if bl.empty or "baseline_emissions" not in bl.columns:
        raise SystemExit(f"{args.baseline_file} missing 'baseline_emissions'")

    baseline = float(bl["baseline_emissions"].iloc[0])
    cap = baseline * (1.0 - red_frac)
    logging.info("Applied CO₂ cap %.6f (baseline %.6f, reduction %.2f%%)", 
                cap, baseline, 100.0 * red_frac)

    # Add CO2 constraint
    add_global_co2_cap(n, cap)

    # Get renewable carriers from config
    renewable_carriers = cfg.get("parameters", {}).get("decentralization", {}).get(
        "renewable_carriers", ["solar", "onwind", "offwind-ac", "offwind-dc", "ror"]
    )

    # Define extra functionality for nodal constraints
    def extra_functionality(network, snapshots):
        add_nodal_renewable_constraints(
            network, k_value, renewable_carriers, snapshots
        )

    # Save pre-optimization snapshot
    pre_path = out_path.with_name(out_path.stem + "_preopt.nc")
    pio.save_network(n, pre_path)
    logging.info(f"Saved pre-optimization network: {pre_path}")

    # Optimize with constraints
    status = termination = None
    try:
        # Log solver options
        for k, v in (solver_options or {}).items():
            logging.info(f"opt {k}: {v!r} (type={type(v).__name__})")
            
        res = n.optimize(
            solver_name=solver_name,
            solver_options=solver_options,
            extra_functionality=extra_functionality
        )
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
        logging.exception("Optimization failed")
        raise

    # Check optimization result
    ok = str(status).lower() in {"ok", "optimal", "success"} or str(termination).lower() in {"optimal"}
    if not ok:
        dbg_path = Path(rep_path).with_suffix(".debug.csv")
        pd.DataFrame([{
            "status": str(status),
            "termination_condition": str(termination),
            "error": "",
        }]).to_csv(dbg_path, index=False)
        logging.error("Non-optimal optimization result: status=%s, termination=%s", status, termination)
        raise RuntimeError(f"Non-OK optimization result: status={status}, termination={termination}")

    # Save results
    actual = compute_total_co2(n)
    pio.save_network(n, out_path)
    
    # Calculate renewable penetration statistics for reporting
    if hasattr(n.snapshot_weightings, 'objective'):
        weights = n.snapshot_weightings.objective
    else:
        weights = n.snapshot_weightings
        
    bus_stats = []
    buses_with_load = n.loads.bus.unique()
    
    for bus in buses_with_load:
        loads_at_bus = n.loads[n.loads.bus == bus].index
        gens_at_bus = n.generators[
            (n.generators.bus == bus) & 
            (n.generators.carrier.isin(renewable_carriers))
        ].index
        
        if len(loads_at_bus) and len(gens_at_bus):
            # Calculate actual renewable penetration
            load_energy = (n.loads_t.p_set[loads_at_bus].multiply(weights, axis=0)).sum().sum()
            renewable_energy = (n.generators_t.p[gens_at_bus].multiply(weights, axis=0)).sum().sum()
            
            if load_energy > 0:
                gamma = renewable_energy / load_energy
                bus_stats.append({
                    'bus': bus,
                    'renewable_energy_MWh': renewable_energy,
                    'load_energy_MWh': load_energy,
                    'gamma': gamma,
                    'within_bounds': (1/k_value <= gamma <= k_value)
                })

    # Report summary
    avg_gamma = sum(s['gamma'] for s in bus_stats) / len(bus_stats) if bus_stats else 0
    buses_within_bounds = sum(s['within_bounds'] for s in bus_stats) if bus_stats else 0
    
    pd.DataFrame([{
        "reduction": red_frac,
        "k_value": k_value,
        "objective": getattr(n, "objective", float("nan")),
        "status": str(status),
        "termination_condition": str(termination),
        "allowed_emissions": cap,
        "actual_emissions": actual,
        "avg_gamma": avg_gamma,
        "buses_within_bounds": buses_within_bounds,
        "total_buses": len(bus_stats),
    }]).to_csv(rep_path, index=False)
    
    logging.info("Completed decentralized solve with k=%.2f, avg_gamma=%.3f", k_value, avg_gamma)


if __name__ == "__main__":
    main()