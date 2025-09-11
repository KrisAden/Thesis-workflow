# src/pypsa_thesis/renewable_bounds.py
from __future__ import annotations
import logging
import pypsa

RENEW_CARRIERS = [
    "offwind-ac", "offwind-dc", "onwind", "solar",
    "biomass", "geothermal", "nuclear"
]

def set_renewable_bounds(n: pypsa.Network, keep_existing: bool = True) -> None:
    """
    Update extendability and capacity bounds for renewable generators.

    - Always set p_nom_extendable = True.
    - If keep_existing=True, enforce current capacities as minimum (p_nom_min = p_nom).
    - If keep_existing=False, allow dropping to zero (p_nom_min = 0).
    - Leave p_nom_max untouched (keeps real-world data).
    """
    mask_renew = n.generators.carrier.isin(RENEW_CARRIERS)
    if not mask_renew.any():
        logging.info("No renewable generators found for bounds update.")
        return

    # enable expansion
    n.generators.loc[mask_renew, "p_nom_extendable"] = True

    # set lower bounds
    if keep_existing:
        n.generators.loc[mask_renew, "p_nom_min"] = n.generators.loc[mask_renew, "p_nom"]
    else:
        n.generators.loc[mask_renew, "p_nom_min"] = 0.0

    # fill NA with 0 for safety
    n.generators.loc[mask_renew, "p_nom_min"] = (
        n.generators.loc[mask_renew, "p_nom_min"].fillna(0.0)
    )

    logging.info(
        f"Applied renewable bounds (keep_existing={keep_existing}) "
        f"to {mask_renew.sum()} generators. p_nom_max left unchanged."
    )


def disable_hydro_extension(n: pypsa.Network) -> None:
    """
    Ensure hydro storage units are non-extendable.
    """
    mask = n.storage_units.carrier.str.lower() == "hydro"
    if mask.any():
        n.storage_units.loc[mask, "p_nom_extendable"] = False
        n.storage_units.loc[mask, "p_nom_min"] = n.storage_units.loc[mask, "p_nom"]
        n.storage_units.loc[mask, "p_nom_max"] = n.storage_units.loc[mask, "p_nom"]
        import logging
        logging.info(f"Disabled extension for {mask.sum()} hydro storage units.")

