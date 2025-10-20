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


def calculate_renewable_fraction_needed(
    n: pypsa.Network, co2_cap: float, renewable_carriers: list[str]
) -> float:
    """
    Calculate the fraction of renewable energy needed to meet CO2 constraint.
    
    This computes α = renewable_fraction_needed based on the CO2 cap and
    emission factors, which is used to scale K-constraint denominators.
    
    Parameters:
    -----------
    n : pypsa.Network
        The network to analyze
    co2_cap : float
        CO2 emissions cap (tCO2)
    renewable_carriers : list[str]
        List of carrier names considered renewable
    
    Returns:
    --------
    float
        α: Renewable fraction needed (0-1) to meet CO2 constraint
    """
    # Get emission factors (tCO2/MWh_thermal) - same as solve.py
    if "co2_emissions" not in n.carriers.columns:
        logging.warning("No co2_emissions column found, using α=1.0")
        return 1.0
    emission_factors = pd.to_numeric(n.carriers["co2_emissions"], errors="coerce").fillna(0.0)
    
    # Calculate snapshot weightings
    if hasattr(n.snapshot_weightings, 'objective'):
        weights = n.snapshot_weightings.objective
    else:
        weights = n.snapshot_weightings
    
    # Calculate total load energy
    total_load_energy = sum(
        (n.loads_t.p_set[load] * weights).sum() 
        for load in n.loads.index
    )
    
    if total_load_energy <= 0:
        logging.warning("No load energy found, using α=1.0")
        return 1.0
    
    # Get conventional (non-renewable) carriers and their emission factors
    conventional_carriers = emission_factors[emission_factors > 0].index
    conventional_carriers = conventional_carriers[~conventional_carriers.isin(renewable_carriers)]
    
    if len(conventional_carriers) == 0:
        logging.warning("No conventional carriers found, using α=1.0")
        return 1.0
    
    # Calculate weighted average emission factor for conventional generation
    # This is a simplified approach using mean emission factor
    weighted_emission_factor = emission_factors[conventional_carriers].mean()
    
    if weighted_emission_factor <= 0:
        logging.warning("No positive emission factors found, using α=1.0") 
        return 1.0
    
    # Calculate renewable fraction needed to meet CO2 cap
    # If all energy were conventional: total_emissions = total_load_energy * emission_factor
    # With renewable fraction α: emissions = (1-α) * total_load_energy * emission_factor
    # Constraint: emissions ≤ co2_cap
    # Therefore: (1-α) * total_load_energy * emission_factor ≤ co2_cap
    # Solving for α: α ≥ 1 - (co2_cap / (total_load_energy * emission_factor))
    
    max_conventional_emissions = total_load_energy * weighted_emission_factor
    
    if max_conventional_emissions <= co2_cap:
        # CO2 constraint is not binding - skip K-constraints entirely
        logging.info("CO2 constraint not binding - K-constraints would be meaningless, skipping them")
        return 0.0  # Signal to skip K-constraints
    else:
        alpha = 1.0 - (co2_cap / max_conventional_emissions)
        # Only apply minimal bound for very small values
        if alpha < 0.05:  # Less than 5%
            logging.warning(f"Very small α={alpha:.3f} - using α=0.05 for numerical stability")
            alpha = 0.05
        alpha = max(0.05, min(1.0, alpha))  # Bound between 5% and 100%
    
    logging.info(f"CO2-adjusted α = {alpha:.3f} (total_load={total_load_energy:.1f} MWh, "
               f"cap={co2_cap:.3f} tCO2, avg_emission_factor={weighted_emission_factor:.6f})")
    
    return alpha


def add_nodal_renewable_constraints(
    n: pypsa.Network,
    k: float,
    renewable_carriers: list[str],
    snapshots,
) -> None:
    """
    Add CO2-adjusted nodal renewable penetration constraints: 1/k <= gamma <= k
    where gamma = renewable_generation / (α * nodal_load) for each bus.
    
    The denominator is scaled by α (renewable fraction needed to meet CO2 constraint)
    to prevent over-decarbonization when CO2 constraints are loose.
    
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
    logging.info(f"Adding CO2-adjusted nodal renewable constraints with k={k}")
    
    # Calculate CO2-required renewable fraction
    # Extract CO2 cap - this is a simplified approach
    # In practice, the cap is passed to the global CO2 constraint
    co2_cap = None
    try:
        # Try to estimate from current network state
        current_emissions = compute_total_co2(n)
        # Use a reasonable fraction of current emissions as the cap estimate
        # This is a fallback - ideally the cap would be passed as parameter
        co2_cap = current_emissions * 0.9  # Assume 10% reduction as default
        logging.info(f"Estimated CO2 cap as 90% of current emissions: {co2_cap:.3f}")
    except Exception:
        logging.warning("Could not estimate CO2 cap, using α=1.0 (no scaling)")
        co2_cap = float('inf')
    
    # Calculate the renewable fraction needed
    alpha = calculate_renewable_fraction_needed(n, co2_cap, renewable_carriers)
    
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
        
        # Scale load energy by CO2-required renewable fraction
        effective_load_energy = alpha * load_energy
        
        # Skip constraints if effective load is too small (causes numerical issues)
        MIN_EFFECTIVE_LOAD = 1000.0  # 1 GWh minimum threshold
        if effective_load_energy < MIN_EFFECTIVE_LOAD:
            logging.info(f"  Skipping {bus}: effective load {effective_load_energy:.0f} MWh below threshold")
            continue
        
        # Build constraint expressions using linopy
        # Sum generator power variables for renewable generators at this bus
        renewable_energy_expr = 0
        
        for gen in gens_at_bus:
            # Access generator variable and multiply by weights, then sum over time
            gen_power = n.model.variables["Generator-p"].sel(Generator=gen)
            renewable_energy_expr += (gen_power * weights).sum()
        
        # If no renewable generators, skip
        if len(gens_at_bus) == 0:
            logging.info(f"  Skipping {bus}: no renewable generators")
            continue
        
        # Clean bus name for constraint naming
        bus_clean = bus.replace(' ', '_').replace('+', 'plus')
        
        # Add constraints: 1/k <= renewable_energy/effective_load_energy <= k
        # Rearranged: renewable_energy >= effective_load_energy/k AND renewable_energy <= k*effective_load_energy
        
        # Calculate bounds with numerical safeguards
        lower_bound = effective_load_energy / k
        upper_bound = k * effective_load_energy
        
        # Apply reasonable bounds to avoid extreme values
        MAX_BOUND = 1e8  # 100 TWh maximum
        lower_bound = min(lower_bound, MAX_BOUND)
        upper_bound = min(upper_bound, MAX_BOUND)
        
        # Only add constraints if bounds are reasonable
        if lower_bound > 0 and upper_bound > lower_bound:
            # Lower bound: renewable_energy >= effective_load_energy/k
            n.model.add_constraints(
                renewable_energy_expr >= lower_bound,
                name=f"renewable_min_{bus_clean}"
            )
            
            # Upper bound: renewable_energy <= k*effective_load_energy  
            n.model.add_constraints(
                renewable_energy_expr <= upper_bound,
                name=f"renewable_max_{bus_clean}"
            )
            
            logging.info(f"  Added CO2-adjusted constraints for {bus}: {len(gens_at_bus)} gens, "
                        f"{load_energy:.0f} MWh load, {effective_load_energy:.0f} MWh effective (α={alpha:.3f})")
        else:
            logging.info(f"  Skipping {bus}: invalid bounds (lower={lower_bound:.0f}, upper={upper_bound:.0f})")


def add_nodal_renewable_constraints_with_cap(
    n: pypsa.Network,
    k: float,
    renewable_carriers: list[str],
    snapshots,
    co2_cap: float,
) -> None:
    """
    Wrapper function to add CO2-adjusted nodal renewable constraints with explicit CO2 cap.
    
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
    co2_cap : float
        CO2 emissions cap (tCO2)
    """
    logging.info(f"Adding CO2-adjusted nodal renewable constraints with k={k}, CO2 cap={co2_cap:.3f}")
    
    # Calculate the renewable fraction needed
    alpha = calculate_renewable_fraction_needed(n, co2_cap, renewable_carriers)
    
    # Skip K-constraints if alpha is 0 (CO2 constraint not binding)
    if alpha <= 0:
        logging.info("Skipping K-constraints - CO2 constraint not binding")
        return
    
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
        
        # Scale load energy by CO2-required renewable fraction
        effective_load_energy = alpha * load_energy
        
        # Build constraint expressions using linopy
        # Sum generator power variables for renewable generators at this bus
        renewable_energy_expr = 0
        
        for gen in gens_at_bus:
            # Access generator variable and multiply by weights, then sum over time
            gen_power = n.model.variables["Generator-p"].sel(Generator=gen)
            renewable_energy_expr += (gen_power * weights).sum()
        
        # If no renewable generators, skip
        if len(gens_at_bus) == 0:
            logging.info(f"  Skipping {bus}: no renewable generators")
            continue
        
        # Clean bus name for constraint naming
        bus_clean = bus.replace(' ', '_').replace('+', 'plus')
        
        # Add constraints: 1/k <= renewable_energy/effective_load_energy <= k
        # Rearranged: renewable_energy >= effective_load_energy/k AND renewable_energy <= k*effective_load_energy
        
        # Only add constraints if effective_load_energy > 0
        if effective_load_energy > 0:
            # Lower bound: renewable_energy >= effective_load_energy/k
            n.model.add_constraints(
                renewable_energy_expr >= effective_load_energy / k,
                name=f"renewable_min_{bus_clean}"
            )
            
            # Upper bound: renewable_energy <= k*effective_load_energy  
            n.model.add_constraints(
                renewable_energy_expr <= k * effective_load_energy,
                name=f"renewable_max_{bus_clean}"
            )
            
            logging.info(f"  Added CO2-adjusted constraints for {bus}: {len(gens_at_bus)} gens, "
                        f"{load_energy:.0f} MWh load, {effective_load_energy:.0f} MWh effective (α={alpha:.3f})")
        else:
            logging.info(f"  Skipping {bus}: zero effective load energy (α={alpha:.3f})")


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
        add_nodal_renewable_constraints_with_cap(
            network, k_value, renewable_carriers, snapshots, cap
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