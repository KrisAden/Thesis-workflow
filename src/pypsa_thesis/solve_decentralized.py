# src/pypsa_thesis/solve_decentralized.py
"""
Solve network with K-constraints for renewable decentralization analysis.

Key Features:
1. CO2 EQUALITY constraint: Forces exact CO2 emissions to match baseline unconstrained solve
   - Eliminates over-decarbonization that would confound cost comparisons
   - Ensures identical environmental outcomes across different k-values
   
2. LOAD-WEIGHTED CAPACITY constraints: (1/K) * ωₙ * R_total ≤ R_cap_n ≤ K * ωₙ * R_total
   - NEW APPROACH: Constrains capacity distribution, not generation ratios
   - ωₙ = load_n / Σ(load_n) are normalized load weights
   - Eliminates over-generation issues that required compensatory fossil burning
   - Better numerical stability (linear constraints, no ratios)
   - Allows CO₂ constraint to determine optimal total renewable capacity
   - K-constraints only control spatial distribution equity
   
3. COMPARATIVE ANALYSIS: Clean cost isolation for decentralization impact
   - Baseline: Pure economic optimization (solve.py with CO2 ≤ cap)
   - K-constrained: Same CO2 but with load-weighted capacity distribution constraints
   - Cost difference = pure spatial equity penalty (no over-generation waste)
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


def add_global_co2_cap(n: pypsa.Network, target_tco2: float) -> None:
    """Add CO2 inequality constraint (≤ cap)."""
    if "co2_cap" in getattr(n, "global_constraints", pd.DataFrame()).index:
        n.global_constraints.at["co2_cap", "constant"] = float(target_tco2)
        n.global_constraints.at["co2_cap", "sense"] = "<="
        n.global_constraints.at["co2_cap", "type"] = "primary_energy"
        n.global_constraints.at["co2_cap", "carrier_attribute"] = "co2_emissions"
    else:
        n.add(
            "GlobalConstraint", "co2_cap",
            sense="<=", constant=float(target_tco2),
            type="primary_energy",
            carrier_attribute="co2_emissions",
        )


def add_global_co2_equality(n: pypsa.Network, target_tco2: float) -> None:
    """Add CO2 equality constraint (= target) - for legacy comparison only."""
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


def add_load_weighted_capacity_constraints(
    n: pypsa.Network,
    k: float,
    renewable_carriers: list[str],
    snapshots,
    upper_bound_only: bool = False,
) -> None:
    """
    Add load-weighted renewable capacity distribution constraints.
    
    New approach: (1/K) * ωₙ * R_total ≤ R_cap_n ≤ K * ωₙ * R_total
    where ωₙ = load_n / Σ(load_n) are load weights.
    
    This approach:
    - Eliminates over-generation issues
    - Provides clean cost-equity trade-offs  
    - Better numerical stability (linear constraints)
    - Allows CO₂ constraint to determine optimal total renewable capacity
    - K-constraints only control spatial distribution
    
    Parameters:
    -----------
    n : pypsa.Network
        The network to add constraints to
    k : float
        The constraint parameter for capacity distribution bounds
    renewable_carriers : list[str]
        List of carrier names considered renewable
    snapshots : pandas.Index
        Snapshots to consider for the constraints
    upper_bound_only : bool
        If True, only apply upper bound to prevent over-concentration
    """
    constraint_type = "upper-bound-only" if upper_bound_only else "load-weighted-full"
    logging.info(f"Adding load-weighted capacity constraints with k={k} ({constraint_type})")
    
    # Get snapshot weights for load calculation
    if hasattr(n.snapshot_weightings, 'objective'):
        weights = n.snapshot_weightings.objective
    else:
        weights = n.snapshot_weightings
    
    # Calculate total system load and load weights
    buses_with_load = n.loads.bus.unique()
    nodal_loads = {}
    total_system_load = 0.0
    
    for bus in buses_with_load:
        loads_at_bus = n.loads[n.loads.bus == bus].index
        if len(loads_at_bus):
            load_energy = sum(
                (n.loads_t.p_set[load] * weights).sum()
                for load in loads_at_bus
            )
            nodal_loads[bus] = load_energy
            total_system_load += load_energy
    
    # Calculate load weights ωₙ = load_n / total_load
    load_weights = {bus: load / total_system_load for bus, load in nodal_loads.items()}
    
    logging.info(f"Total system load: {total_system_load/1e6:.1f} TWh")
    logging.info(f"Load weight range: [{min(load_weights.values()):.4f}, {max(load_weights.values()):.4f}]")
    
    # Get all renewable generators and calculate total renewable capacity variable
    renewable_gens = n.generators[
        n.generators.carrier.isin(renewable_carriers) & 
        n.generators.p_nom_extendable
    ].index
    
    if len(renewable_gens) == 0:
        logging.warning("No extendable renewable generators found!")
        return
    
    # Total renewable capacity expression: R_total = Σ(p_nom for all renewable generators)
    total_renewable_capacity = sum(n.model["Generator-p_nom"][gen] for gen in renewable_gens)
    
    logging.info(f"Adding constraints for {len(renewable_gens)} renewable generators")
    
    # Add capacity distribution constraints for each bus
    constraint_count = 0
    
    for bus in buses_with_load:
        if bus not in load_weights:
            continue
            
        # Get renewable generators at this bus
        gens_at_bus = [
            gen for gen in renewable_gens 
            if n.generators.loc[gen, 'bus'] == bus
        ]
        
        if not gens_at_bus:
            continue
            
        # Sum of renewable capacity at this bus
        bus_renewable_capacity = sum(n.model["Generator-p_nom"][gen] for gen in gens_at_bus)
        
        # Load weight for this bus
        omega_n = load_weights[bus]
        
        # Clean bus name for constraint naming
        bus_clean = bus.replace(' ', '_').replace('+', 'plus').replace('-', 'minus')
        
        # Add constraints: (1/K) * ωₙ * R_total ≤ R_cap_n ≤ K * ωₙ * R_total
        if upper_bound_only:
            # Only upper bound: R_cap_n ≤ K * ωₙ * R_total
            # Rearranged: bus_renewable_capacity - k * omega_n * total_renewable_capacity <= 0
            n.model.add_constraints(
                bus_renewable_capacity - k * omega_n * total_renewable_capacity <= 0,
                name=f"capacity_upper_{bus_clean}"
            )
            constraint_count += 1
            logging.info(f"  Added upper constraint for {bus}: ω={omega_n:.4f}, {len(gens_at_bus)} gens")
        else:
            # Lower bound: R_cap_n ≥ (1/K) * ωₙ * R_total
            # Rearranged: bus_renewable_capacity - (1/k) * omega_n * total_renewable_capacity >= 0
            n.model.add_constraints(
                bus_renewable_capacity - (1.0/k) * omega_n * total_renewable_capacity >= 0,
                name=f"capacity_lower_{bus_clean}"
            )
            # Upper bound: R_cap_n ≤ K * ωₙ * R_total  
            # Rearranged: bus_renewable_capacity - k * omega_n * total_renewable_capacity <= 0
            n.model.add_constraints(
                bus_renewable_capacity - k * omega_n * total_renewable_capacity <= 0,
                name=f"capacity_upper_{bus_clean}"
            )
            constraint_count += 2
            logging.info(f"  Added bounds for {bus}: ω={omega_n:.4f}, bounds=[{(1.0/k)*omega_n:.4f}R, {k*omega_n:.4f}R], {len(gens_at_bus)} gens")
    
    logging.info(f"Added {constraint_count} load-weighted capacity constraints")
    
    # Add reasonable bounds on total renewable capacity to aid solver
    # Use existing capacity as reference
    existing_renewable_cap = sum(
        n.generators.loc[gen, 'p_nom'] for gen in renewable_gens
        if hasattr(n.generators, 'p_nom_min') and n.generators.loc[gen, 'p_nom_min'] > 0
    ) if hasattr(n.generators, 'p_nom_min') else 0
    
    if existing_renewable_cap == 0:
        existing_renewable_cap = sum(n.generators.loc[gen, 'p_nom'] for gen in renewable_gens) * 0.1
    
    # Set reasonable bounds (allow significant expansion but not unlimited)
    min_total_capacity = max(existing_renewable_cap, total_system_load * 0.01)  # At least 1% of load
    max_total_capacity = total_system_load * 3.0  # Up to 3x annual load (reasonable for renewables)
    
    n.model.add_constraints(
        total_renewable_capacity - min_total_capacity >= 0,
        name="total_renewable_capacity_min"
    )
    n.model.add_constraints(
        total_renewable_capacity - max_total_capacity <= 0, 
        name="total_renewable_capacity_max"
    )
    
    logging.info(f"Total renewable capacity bounds: [{min_total_capacity:.0f}, {max_total_capacity:.0f}] MW")


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

    # Get renewable carriers from config
    renewable_carriers = cfg.get("parameters", {}).get("decentralization", {}).get(
        "renewable_carriers", ["solar", "onwind", "offwind-ac", "offwind-dc", "ror"]
    )
    
    # With new load-weighted approach, use inequality constraint by default
    # (eliminates over-generation issues that previously required equality)
    use_co2_equality = cfg.get("parameters", {}).get("decentralization", {}).get(
        "use_co2_equality", False  # Changed default to False for new approach
    )
    
    if use_co2_equality:
        logging.info("Using CO2 equality constraint (legacy mode for comparison)")
        add_global_co2_equality(n, target_co2)
    else:
        logging.info("Using CO2 inequality constraint (recommended with load-weighted approach)")
        add_global_co2_cap(n, target_co2)

    # Define extra functionality for nodal constraints
    def extra_functionality(network, snapshots):
        add_load_weighted_capacity_constraints(
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
    
    # Calculate both capacity distribution and generation statistics for reporting
    if hasattr(n.snapshot_weightings, 'objective'):
        weights = n.snapshot_weightings.objective
    else:
        weights = n.snapshot_weightings
    
    # Calculate load weights
    buses_with_load = n.loads.bus.unique()
    nodal_loads = {}
    total_system_load = 0.0
    
    for bus in buses_with_load:
        loads_at_bus = n.loads[n.loads.bus == bus].index
        if len(loads_at_bus):
            load_energy = (n.loads_t.p_set[loads_at_bus].multiply(weights, axis=0)).sum().sum()
            nodal_loads[bus] = load_energy
            total_system_load += load_energy
    
    load_weights = {bus: load / total_system_load for bus, load in nodal_loads.items()}
    
    # Calculate capacity statistics
    renewable_gens = n.generators[
        n.generators.carrier.isin(renewable_carriers) & 
        n.generators.p_nom_extendable
    ].index
    
    total_renewable_capacity = sum(n.generators.loc[gen, 'p_nom_opt'] for gen in renewable_gens)
    
    bus_stats = []
    capacity_violations = 0
    
    for bus in buses_with_load:
        if bus not in load_weights:
            continue
            
        loads_at_bus = n.loads[n.loads.bus == bus].index
        gens_at_bus = [gen for gen in renewable_gens if n.generators.loc[gen, 'bus'] == bus]
        
        if len(loads_at_bus) and len(gens_at_bus):
            # Calculate actual values
            load_energy = nodal_loads[bus]
            renewable_energy = (n.generators_t.p[gens_at_bus].multiply(weights, axis=0)).sum().sum()
            bus_renewable_capacity = sum(n.generators.loc[gen, 'p_nom_opt'] for gen in gens_at_bus)
            
            omega_n = load_weights[bus]
            expected_capacity_lower = (1.0/k_value) * omega_n * total_renewable_capacity
            expected_capacity_upper = k_value * omega_n * total_renewable_capacity
            
            # Check capacity constraint compliance
            capacity_within_bounds = (expected_capacity_lower <= bus_renewable_capacity <= expected_capacity_upper)
            if not capacity_within_bounds:
                capacity_violations += 1
            
            if load_energy > 0:
                gamma = renewable_energy / load_energy
                bus_stats.append({
                    'bus': bus,
                    'load_weight': omega_n,
                    'renewable_capacity_MW': bus_renewable_capacity,
                    'expected_capacity_lower_MW': expected_capacity_lower,
                    'expected_capacity_upper_MW': expected_capacity_upper,
                    'capacity_within_bounds': capacity_within_bounds,
                    'renewable_energy_MWh': renewable_energy,
                    'load_energy_MWh': load_energy,
                    'gamma': gamma,
                })

    # Report summary
    avg_gamma = sum(s['gamma'] for s in bus_stats) / len(bus_stats) if bus_stats else 0
    avg_load_weight = sum(s['load_weight'] for s in bus_stats) / len(bus_stats) if bus_stats else 0
    capacity_compliance_rate = (len(bus_stats) - capacity_violations) / len(bus_stats) if bus_stats else 0
    
    pd.DataFrame([{
        "reduction": red_frac,
        "k_value": k_value,
        "constraint_type": "load_weighted_" + ("upper_only" if args.upper_bound_only else "full"),
        "co2_constraint_type": "equality" if use_co2_equality else "inequality",
        "objective": getattr(n, "objective", float("nan")),
        "status": str(status),
        "termination_condition": str(termination),
        "target_emissions": target_co2,
        "actual_emissions": actual,
        "total_renewable_capacity_MW": total_renewable_capacity,
        "avg_gamma": avg_gamma,
        "avg_load_weight": avg_load_weight,
        "capacity_compliance_rate": capacity_compliance_rate,
        "capacity_violations": capacity_violations,
        "total_buses": len(bus_stats),
    }]).to_csv(rep_path, index=False)
    
    logging.info("Completed load-weighted decentralized solve with k=%.2f, capacity compliance=%.1%%, avg_gamma=%.3f", 
                 k_value, capacity_compliance_rate * 100, avg_gamma)


if __name__ == "__main__":
    main()