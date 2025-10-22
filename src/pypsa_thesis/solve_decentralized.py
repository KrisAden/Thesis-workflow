# src/pypsa_thesis/solve_decentralized.py
"""
Solve network with K-constraints for renewable decentralization analysis.

Key Features:
1. CO2 EQUALITY constraint: Forces exact CO2 emissions to match baseline unconstrained solve
   - Eliminates over-decarbonization that would confound cost comparisons
   - Ensures identical environmental outcomes across different k-values
   
2. SIMPLE K-constraints: 1/k <= gamma <= k where gamma = renewable_generation/load
   - No complex α-scaling that caused numerical instability  
   - Direct, interpretable renewable penetration constraints per bus
   - Optional upper-bound-only mode to test impact of lower bound
   
3. COMPARATIVE ANALYSIS: Clean cost isolation for decentralization impact
   - Baseline: Pure economic optimization (solve.py with CO2 ≤ cap)
   - K-constrained: Same CO2 but with spatial distribution constraints
   - Cost difference = pure decentralization penalty
"""
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
    _normalize_reduction,
    compute_total_co2,
)


def add_global_co2_equality(n: pypsa.Network, target_tco2: float) -> None:
    """Add CO2 equality constraint instead of inequality cap."""
    if "co2_exact" in getattr(n, "global_constraints", pd.DataFrame()).index:
        n.global_constraints.at["co2_exact", "constant"] = float(target_tco2)
        n.global_constraints.at["co2_exact", "sense"] = "=="
        n.global_constraints.at["co2_exact", "type"] = "primary_energy"
        n.global_constraints.at["co2_exact", "carrier_attribute"] = "co2_emissions"
    else:
        n.add(
            "GlobalConstraint", "co2_exact",
            sense="==", constant=float(target_tco2),
            type="primary_energy",
            carrier_attribute="co2_emissions",
        )


def add_simple_nodal_renewable_constraints(
    n: pypsa.Network,
    k: float,
    renewable_carriers: list[str],
    snapshots,
    upper_bound_only: bool = False,
) -> None:
    """
    Add simple nodal renewable penetration constraints: 1/k <= gamma <= k
    where gamma = renewable_generation / nodal_load for each bus.
    
    No α scaling - uses direct load values for clean, interpretable constraints.
    
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
    upper_bound_only : bool
        If True, only apply upper bound (gamma <= k) to avoid over-decarbonization
    """
    constraint_type = "upper-bound-only" if upper_bound_only else "full"
    logging.info(f"Adding simple nodal renewable constraints with k={k} ({constraint_type})")
    
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
            
        # Total load energy at this bus (MWh)
        load_energy = sum(
            (n.loads_t.p_set[load] * weights).sum()
            for load in loads_at_bus
        )
        
        # Skip buses with zero or very small load
        MIN_LOAD_THRESHOLD = 1000.0  # 1 GWh minimum
        if load_energy <= MIN_LOAD_THRESHOLD:
            logging.info(f"  Skipping {bus}: load {load_energy:.0f} MWh below threshold")
            continue
        
        # Build constraint expressions using linopy
        renewable_energy_expr = 0
        
        for gen in gens_at_bus:
            # Access generator variable and multiply by weights, then sum over time
            gen_power = n.model.variables["Generator-p"].sel(Generator=gen)
            renewable_energy_expr += (gen_power * weights).sum()
        
        # Clean bus name for constraint naming
        bus_clean = bus.replace(' ', '_').replace('+', 'plus')
        
        # Add constraints based on mode
        if upper_bound_only:
            # Only upper bound: renewable_energy <= k*load_energy
            upper_bound = k * load_energy
            n.model.add_constraints(
                renewable_energy_expr <= upper_bound,
                name=f"renewable_max_{bus_clean}"
            )
            logging.info(f"  Added upper-bound constraint for {bus}: {len(gens_at_bus)} gens, "
                        f"{load_energy:.0f} MWh load, max {upper_bound:.0f}")
        else:
            # Full constraints: 1/k <= renewable_energy/load_energy <= k
            # Rearranged: load_energy/k <= renewable_energy <= k*load_energy
            lower_bound = load_energy / k
            upper_bound = k * load_energy
            
            n.model.add_constraints(
                renewable_energy_expr >= lower_bound,
                name=f"renewable_min_{bus_clean}"
            )
            
            n.model.add_constraints(
                renewable_energy_expr <= upper_bound,
                name=f"renewable_max_{bus_clean}"
            )
            
            logging.info(f"  Added full constraints for {bus}: {len(gens_at_bus)} gens, "
                        f"{load_energy:.0f} MWh load, bounds [{lower_bound:.0f}, {upper_bound:.0f}]")


# Removed old complex constraint function - replaced with add_simple_nodal_renewable_constraints


# Removed old complex constraint function - using simple constraints instead


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
    ap.add_argument("--upper-bound-only", action="store_true",
                    help="Only apply upper bound constraint (gamma <= k), skip lower bound to avoid over-decarbonization")
    
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

    # Load baseline emissions and actual emissions from the unconstrained solve
    bl = pd.read_csv(args.baseline_file)
    if bl.empty or "baseline_emissions" not in bl.columns:
        raise SystemExit(f"{args.baseline_file} missing 'baseline_emissions'")

    # For CO2 equality constraint, we need the actual emissions from the corresponding
    # unconstrained solve at this reduction level. We'll look for the solve result file.
    baseline_reduction_file = f"results/tables/solve_reduction_{int(args.reduction)}.csv"
    try:
        reduction_result = pd.read_csv(baseline_reduction_file)
        if not reduction_result.empty and "actual_emissions" in reduction_result.columns:
            target_co2 = float(reduction_result["actual_emissions"].iloc[0])
            logging.info("Using actual CO₂ emissions from unconstrained solve: %.6f tCO2", target_co2)
        else:
            # Fallback to calculated cap
            baseline = float(bl["baseline_emissions"].iloc[0])
            target_co2 = baseline * (1.0 - red_frac)
            logging.warning("Could not find unconstrained solve results, using calculated target: %.6f tCO2", target_co2)
    except FileNotFoundError:
        baseline = float(bl["baseline_emissions"].iloc[0])
        target_co2 = baseline * (1.0 - red_frac)
        logging.warning("Unconstrained solve results not found, using calculated target: %.6f tCO2", target_co2)

    # Add CO2 equality constraint (not inequality)
    add_global_co2_equality(n, target_co2)

    # Get renewable carriers from config
    renewable_carriers = cfg.get("parameters", {}).get("decentralization", {}).get(
        "renewable_carriers", ["solar", "onwind", "offwind-ac", "offwind-dc", "ror"]
    )
    
    # Check if we should use CO2 equality constraint (recommended approach)
    use_co2_equality = cfg.get("parameters", {}).get("decentralization", {}).get(
        "use_co2_equality", True
    )
    
    if use_co2_equality:
        logging.info("Using CO2 equality constraint to match baseline decarbonization exactly")
    else:
        logging.info("Using CO2 inequality constraint (may lead to over-decarbonization)")

    # Define extra functionality for nodal constraints
    def extra_functionality(network, snapshots):
        add_simple_nodal_renewable_constraints(
            network, k_value, renewable_carriers, snapshots, args.upper_bound_only
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
        "constraint_type": "upper_only" if args.upper_bound_only else "full",
        "co2_constraint_type": "equality",
        "objective": getattr(n, "objective", float("nan")),
        "status": str(status),
        "termination_condition": str(termination),
        "target_emissions": target_co2,
        "actual_emissions": actual,
        "avg_gamma": avg_gamma,
        "buses_within_bounds": buses_within_bounds,
        "total_buses": len(bus_stats),
    }]).to_csv(rep_path, index=False)
    
    logging.info("Completed decentralized solve with k=%.2f, avg_gamma=%.3f", k_value, avg_gamma)


if __name__ == "__main__":
    main()