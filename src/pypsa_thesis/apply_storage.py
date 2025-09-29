# src/pypsa_thesis/apply_storage.py
"""
Add optional extendable battery and hydrogen storage to PyPSA networks.

This module adds storage units directly connected to electricity buses using PyPSA's 
StorageUnit component, which combines energy storage and power conversion in a single
component with charge/discharge efficiencies.

Both battery and hydrogen storage are implemented as StorageUnit objects connected
directly to electricity buses, avoiding the need for separate carriers and buses
when hydrogen transport between nodes is not required.
"""
from __future__ import annotations
import argparse
import logging
from typing import Optional, Tuple
from pathlib import Path
import pandas as pd
import pypsa

from . import io as pio

def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def annuity(lifetime_y: float, r: float) -> float:
    if lifetime_y <= 0:
        raise ValueError("lifetime_y must be > 0")
    return r / (1 - (1 + r) ** (-lifetime_y)) if r > 0 else 1.0 / lifetime_y

def _kwh_to_mwh(x: float) -> float:
    return x * 1000.0

def _kw_to_mw(x: float) -> float:
    return x * 1000.0

def _annualize_energy_capex_per_mwh(capex_eur_per_kwh: float, lifetime_y: float, r: float) -> float:
    return annuity(lifetime_y, r) * _kwh_to_mwh(capex_eur_per_kwh)

def _annualize_power_capex_per_mw(capex_eur_per_kw: float, lifetime_y: float, r: float) -> float:
    return annuity(lifetime_y, r) * _kw_to_mw(capex_eur_per_kw)

def _get_cost(df: pd.DataFrame, name: str) -> Tuple[float, Optional[float]]:
    row = df.loc[df["component"].str.lower() == name.lower()]
    if row.empty:
        raise KeyError(f"Missing '{name}' in storage_costs.csv")
    r = row.iloc[0]
    capex_kwh = r.get("capex_eur_per_kwh")
    capex_kw  = r.get("capex_eur_per_kw")
    life      = float(r["lifetime_y"])
    disc      = float(r["discount_rate"])
    eff       = r.get("efficiency")
    eff       = None if pd.isna(eff) else float(eff)
    if not pd.isna(capex_kwh):
        cost = _annualize_energy_capex_per_mwh(float(capex_kwh), life, disc)
    elif not pd.isna(capex_kw):
        cost = _annualize_power_capex_per_mw(float(capex_kw), life, disc)
    else:
        raise ValueError(f"Row '{name}' must specify capex_eur_per_kwh or capex_eur_per_kw")
    return float(cost), eff

def add_battery_storage(n: pypsa.Network, storage_costs: pd.DataFrame) -> pd.DataFrame:
    """Add battery storage units directly connected to electricity buses.
    
    Each storage unit combines energy storage and power conversion (inverter) in a single
    StorageUnit component with symmetric charge/discharge efficiency.
    """
    buses = n.buses.index

    energy_cost, _ = _get_cost(storage_costs, "battery_energy")  # €/MWh/a
    power_cost, eff = _get_cost(storage_costs, "battery_power")  # €/MW/a
    if eff is None:
        eff = 0.96  # Default round-trip efficiency (sqrt(0.96) each direction)

    # StorageUnit separates power and energy costs cleanly
    # capital_cost = power capacity cost, capital_cost_energy = energy capacity cost
    
    storage_names = pd.Index(buses + " Battery")
    need_storage = ~storage_names.isin(n.storage_units.index)

    # Add battery storage units (combines energy storage and power conversion)
    if need_storage.any():
        n.madd(
            "StorageUnit",
            storage_names[need_storage],
            bus=buses[need_storage],
            p_nom_extendable=True,
            e_nom_extendable=True,
            e_cyclic=True,
            efficiency_store=eff,      # Charging efficiency
            efficiency_dispatch=eff,   # Discharging efficiency  
            capital_cost=power_cost,   # €/MW/a for power capacity
            capital_cost_energy=energy_cost,  # €/MWh/a for energy capacity
            marginal_cost=0.0,
        )

    report = pd.DataFrame([
        {"component": "battery", "capital_cost_eur_per_mw_a": power_cost, "capital_cost_eur_per_mwh_a": energy_cost, "efficiency_store": eff, "efficiency_dispatch": eff},
    ])
    return report

def add_hydrogen_storage(n: pypsa.Network, storage_costs: pd.DataFrame) -> pd.DataFrame:
    """Add hydrogen storage units directly connected to electricity buses.
    
    Each storage unit combines H2 tank storage with electrolyser/fuel cell conversion,
    eliminating the need for separate H2 buses and carriers when H2 transport is not needed.
    """
    buses = n.buses.index

    tank_cost, _    = _get_cost(storage_costs, "h2_tank")         # €/MWh/a
    ely_cost, ely_e = _get_cost(storage_costs, "electrolyser")    # €/MW/a
    fc_cost,  fc_e  = _get_cost(storage_costs, "fuel_cell")       # €/MW/a

    # Set default efficiencies if not provided
    if ely_e is None:
        ely_e = 0.66  # Electrolyser efficiency (electricity -> H2)
    if fc_e is None:
        fc_e = 0.50   # Fuel cell efficiency (H2 -> electricity)
    
    # Calculate combined power cost
    # Note: In reality, you need both electrolyser AND fuel cell capacity
    # This is a simplification - you might want to add both costs or use a weighted average
    combined_power_cost = ely_cost + fc_cost  # Conservative approach: add both costs
    
    storage_names = pd.Index(buses + " H2")
    need_storage = ~storage_names.isin(n.storage_units.index)

    # Add hydrogen storage units (directly connected to electricity buses)
    if need_storage.any():
        n.madd(
            "StorageUnit",
            storage_names[need_storage],
            bus=buses[need_storage],
            p_nom_extendable=True,
            e_nom_extendable=True,
            e_cyclic=True,
            efficiency_store=ely_e,        # Electrolyser efficiency (electricity -> H2)
            efficiency_dispatch=fc_e,      # Fuel cell efficiency (H2 -> electricity)
            capital_cost=combined_power_cost,      # €/MW/a for power capacity
            capital_cost_energy=tank_cost,         # €/MWh/a for energy capacity
            marginal_cost=0.0,
            carrier="H2",  # Optional: label for tracking
        )

    report = pd.DataFrame([
        {"component": "hydrogen", "capital_cost_eur_per_mw_a": combined_power_cost, "capital_cost_eur_per_mwh_a": tank_cost, "efficiency_store": ely_e, "efficiency_dispatch": fc_e},
    ])
    return report


def main():
    ap = argparse.ArgumentParser(description="Add storage (battery and/or H2) and apply costs.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--network-in")
    ap.add_argument("--network-out")
    ap.add_argument("--table-out")  # writes a simple summary of applied costs
    args = ap.parse_args()

    cfg = pio.read_yaml(args.config)
    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    in_path   = Path(args.network_in  or cfg["paths"]["expanded_network"])
    out_path  = Path(args.network_out or cfg["paths"]["network_with_storage"])
    tables    = Path(cfg["paths"].get("tables_dir", "results/tables"))
    tables.mkdir(parents=True, exist_ok=True)
    table_out = Path(args.table_out or (tables / "storage_costs_applied.csv"))

    storage_csv = Path(cfg["paths"]["storage_costs_csv"]).resolve()

    n = pio.load_network(in_path)
    df_costs = pd.read_csv(storage_csv)

    reports = []
    if cfg.get("parameters", {}).get("storage", {}).get("add_battery", False):
        reports.append(add_battery_storage(n, df_costs))
    if cfg.get("parameters", {}).get("storage", {}).get("add_hydrogen", False):
        reports.append(add_hydrogen_storage(n, df_costs))

    if reports:
        pd.concat(reports, ignore_index=True).to_csv(table_out, index=False)
    else:
        logging.info("No storage selected; nothing to add.")
        # still write an empty file so Snakemake has an output
        pd.DataFrame(columns=["component","capital_cost","efficiency"]).to_csv(table_out, index=False)

    pio.save_network(n, out_path)
    logging.info(f"Wrote: {out_path}")
    logging.info(f"Costs table: {table_out}")
    logging.info(f"Read storage costs from: {storage_csv}")

if __name__ == "__main__":
    main()
