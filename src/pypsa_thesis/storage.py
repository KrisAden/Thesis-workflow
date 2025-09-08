# src/pypsa_thesis/storage.py
from __future__ import annotations
from typing import Optional, Tuple
import logging
import numpy as np
import pandas as pd
import pypsa

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
    buses = n.buses.index

    energy_cost, _ = _get_cost(storage_costs, "battery_energy")  # €/MWh/a
    power_cost, eff = _get_cost(storage_costs, "battery_power")  # €/MW/a
    if eff is None:
        eff = 0.96

    # idempotency: avoid duplicates if rule re-runs
    store_names = buses + " Battery Storage"
    link_names  = buses + " Battery Inverter"
    missing_store = ~n.stores.index.isin(store_names)
    missing_link  = ~n.links.index.isin(link_names)

    # Add energy store (extendable energy)
    n.madd(
        "Store",
        store_names[missing_store] if hasattr(store_names, "shape") else store_names,
        bus=buses[missing_store] if hasattr(buses, "shape") else buses,
        e_nom_extendable=True,
        e_cyclic=True,
        capital_cost=energy_cost,
        marginal_cost=0.0,
    )

    # Add inverter (extendable power)
    n.madd(
        "Link",
        link_names[missing_link] if hasattr(link_names, "shape") else link_names,
        bus0=buses[missing_link] if hasattr(buses, "shape") else buses,
        bus1=buses[missing_link] if hasattr(buses, "shape") else buses,  # simple one-bus model
        p_nom_extendable=True,
        efficiency=eff,
        capital_cost=power_cost,
        marginal_cost=0.0,
    )

    report = pd.DataFrame([
        {"component": "battery_energy", "capital_cost_eur_per_mwh_a": energy_cost, "efficiency": None},
        {"component": "battery_power",  "capital_cost_eur_per_mw_a":  power_cost,  "efficiency": eff},
    ])
    return report

def add_hydrogen_chain(n: pypsa.Network, storage_costs: pd.DataFrame) -> pd.DataFrame:
    buses = n.buses.index

    tank_cost, _ = _get_cost(storage_costs, "h2_tank")        # €/MWh/a
    ely_cost, ely_eff = _get_cost(storage_costs, "electrolyser")  # €/MW/a
    fc_cost,  fc_eff  = _get_cost(storage_costs, "fuel_cell")     # €/MW/a

    if "H2" not in n.carriers.index:
        n.add("Carrier", "H2")

    h2_bus_names = buses + " H2"
    # idempotent add of H2 buses
    new_mask = ~n.buses.index.isin(h2_bus_names)
    n.madd("Bus", h2_bus_names[new_mask], location=buses[new_mask], carrier="H2")

    # names
    tank_names = buses + " H2 Tank"
    ely_names  = buses + " H2 Electrolysis"
    fc_names   = buses + " H2 Fuel Cell"

    # missing masks
    miss_tank = ~n.stores.index.isin(tank_names)
    miss_ely  = ~n.links.index.isin(ely_names)
    miss_fc   = ~n.links.index.isin(fc_names)

    # H2 storage
    n.madd("Store",
           tank_names[miss_tank],
           bus=h2_bus_names[miss_tank],
           e_nom_extendable=True,
           e_cyclic=True,
           capital_cost=tank_cost,
           marginal_cost=0.0)

    # Electrolyser: electricity -> H2
    n.madd("Link",
           ely_names[miss_ely],
           bus0=buses[miss_ely],
           bus1=h2_bus_names[miss_ely],
           p_nom_extendable=True,
           efficiency=ely_eff if ely_eff is not None else 0.66,
           capital_cost=ely_cost,
           marginal_cost=0.0)

    # Fuel cell: H2 -> electricity
    n.madd("Link",
           fc_names[miss_fc],
           bus0=h2_bus_names[miss_fc],
           bus1=buses[miss_fc],
           p_nom_extendable=True,
           efficiency=fc_eff if fc_eff is not None else 0.50,
           capital_cost=fc_cost,
           marginal_cost=0.0)

    report = pd.DataFrame([
        {"component": "h2_tank",     "capital_cost_eur_per_mwh_a": tank_cost, "efficiency": None},
        {"component": "electrolyser","capital_cost_eur_per_mw_a":  ely_cost,  "efficiency": ely_eff},
        {"component": "fuel_cell",   "capital_cost_eur_per_mw_a":  fc_cost,   "efficiency": fc_eff},
    ])
    return report
