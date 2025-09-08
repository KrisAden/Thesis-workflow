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

    store_names = pd.Index(buses + " Battery Storage")
    link_names  = pd.Index(buses + " Battery Inverter")

    need_store = ~store_names.isin(n.stores.index)
    need_link  = ~link_names.isin(n.links.index)

    # Add energy store (extendable energy)
    if need_store.any():
        n.madd(
            "Store",
            store_names[need_store],
            bus=buses[need_store],
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=energy_cost,
            marginal_cost=0.0,
        )

    # Add inverter (extendable power)
    if need_link.any():
        n.madd(
            "Link",
            link_names[need_link],
            bus0=buses[need_link],
            bus1=buses[need_link],  # simple one-bus model
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

    tank_cost, _    = _get_cost(storage_costs, "h2_tank")         # €/MWh/a
    ely_cost, ely_e = _get_cost(storage_costs, "electrolyser")    # €/MW/a
    fc_cost,  fc_e  = _get_cost(storage_costs, "fuel_cell")       # €/MW/a

    if "H2" not in n.carriers.index:
        n.add("Carrier", "H2")

    h2_bus_names = pd.Index(buses + " H2")
    need_h2_bus  = ~h2_bus_names.isin(n.buses.index)
    if need_h2_bus.any():
        n.madd("Bus", h2_bus_names[need_h2_bus], location=buses[need_h2_bus], carrier="H2")

    tank_names = pd.Index(buses + " H2 Tank")
    ely_names  = pd.Index(buses + " H2 Electrolysis")
    fc_names   = pd.Index(buses + " H2 Fuel Cell")

    need_tank = ~tank_names.isin(n.stores.index)
    need_ely  = ~ely_names.isin(n.links.index)
    need_fc   = ~fc_names.isin(n.links.index)

    if need_tank.any():
        n.madd(
            "Store",
            tank_names[need_tank],
            bus=h2_bus_names[need_tank],
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=tank_cost,
            marginal_cost=0.0,
        )

    if need_ely.any():
        n.madd(
            "Link",
            ely_names[need_ely],
            bus0=buses[need_ely],
            bus1=h2_bus_names[need_ely],
            p_nom_extendable=True,
            efficiency=ely_e if ely_e is not None else 0.66,
            capital_cost=ely_cost,
            marginal_cost=0.0,
        )

    if need_fc.any():
        n.madd(
            "Link",
            fc_names[need_fc],
            bus0=h2_bus_names[need_fc],
            bus1=buses[need_fc],
            p_nom_extendable=True,
            efficiency=fc_e if fc_e is not None else 0.50,
            capital_cost=fc_cost,
            marginal_cost=0.0,
        )

    report = pd.DataFrame([
        {"component": "h2_tank",     "capital_cost_eur_per_mwh_a": tank_cost, "efficiency": None},
        {"component": "electrolyser","capital_cost_eur_per_mw_a":  ely_cost,  "efficiency": ely_e},
        {"component": "fuel_cell",   "capital_cost_eur_per_mw_a":  fc_cost,   "efficiency": fc_e},
    ])
    return report
