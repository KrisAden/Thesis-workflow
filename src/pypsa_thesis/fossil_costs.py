from __future__ import annotations
from typing import Optional

def annuity(lifetime_y: float, r: float) -> float:
    if lifetime_y <= 0:
        raise ValueError("lifetime_y must be > 0")
    return r / (1 - (1 + r) ** (-lifetime_y)) if r > 0 else 1.0 / lifetime_y

def capital_cost_from_dea(
    capex_eur_per_mw: float,
    lifetime_y: float,
    discount_rate: float,
    fom_abs_eur_per_mw_a: Optional[float] = None,
    fom_rate_per_a: Optional[float] = None,
) -> float:
    a = annuity(lifetime_y, discount_rate)
    if fom_abs_eur_per_mw_a is not None:
        return capex_eur_per_mw * a + fom_abs_eur_per_mw_a
    if fom_rate_per_a is not None:
        return capex_eur_per_mw * (a + fom_rate_per_a)
    return capex_eur_per_mw * a
