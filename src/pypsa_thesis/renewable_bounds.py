from __future__ import annotations
import logging
import numpy as np
import pypsa
import pandas as pd

# Renewables you want to allow to expand (hydro handled separately)
RENEW_CARRIERS = [
    "offwind-ac", "offwind-dc", "onwind", "solar",
    "biomass", "geothermal", "nuclear"
]

# Common names for hydro-like tech across datasets
HYDRO_GENERATOR_ALIASES = {"hydro", "PHS"}


def set_renewable_bounds(n: pypsa.Network, keep_existing: bool = True) -> None:
    """
    Update extendability and capacity bounds for renewable generators.

    - Always set p_nom_extendable = True for RENEW_CARRIERS.
    - If keep_existing=True, enforce current capacities as minimum (p_nom_min = p_nom).
    - If keep_existing=False, allow dropping to zero (p_nom_min = 0).
    - Leave p_nom_max as-is *except* when p_nom > p_nom_max; in that case, raise p_nom_max to p_nom.
    - Guard against NaNs in p_nom.
    """
    if not len(n.generators):
        logging.info("No generators found; skipping renewable bounds.")
        return

    # Ensure finite p_nom for safe min-setting
    n.generators["p_nom"] = n.generators["p_nom"].fillna(0.0)

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

    # ensure p_nom_min finite
    n.generators.loc[mask_renew, "p_nom_min"] = (
        n.generators.loc[mask_renew, "p_nom_min"].fillna(0.0)
    )

    # If p_nom_max exists and is below current p_nom, raise it to avoid min>max contradictions
    if "p_nom_max" in n.generators:
        m = mask_renew & n.generators.p_nom_max.notnull()
        bad = m & (n.generators["p_nom"] > n.generators["p_nom_max"])
        if bad.any():
            cnt = int(bad.sum())
            logging.warning(
                "Raising p_nom_max to current p_nom for %d renewable generators "
                "to satisfy p_nom_min <= p_nom_max.", cnt
            )
            n.generators.loc[bad, "p_nom_max"] = n.generators.loc[bad, "p_nom"]

    logging.info(
        "Applied renewable bounds (keep_existing=%s) to %d generators. p_nom_max kept unless raised to current.",
        keep_existing, int(mask_renew.sum())
    )


def disable_hydro_extension(n: pypsa.Network) -> None:
    """
    Ensure hydro *storage units and generators* are non-extendable and fixed at current capacity.
    """
    # Storage units (reservoir/PHS modeled as storage)
    if len(n.storage_units):
        mask_su = n.storage_units.carrier.str.lower() == "hydro"
        if mask_su.any():
            n.storage_units.loc[mask_su, "p_nom_extendable"] = False
            n.storage_units.loc[mask_su, "p_nom_min"] = n.storage_units.loc[mask_su, "p_nom"].fillna(0.0)
            n.storage_units.loc[mask_su, "p_nom_max"] = n.storage_units.loc[mask_su, "p_nom"].fillna(0.0)
            logging.info("Disabled extension for %d hydro storage units.", int(mask_su.sum()))

    # Generators (run-of-river / reservoir turbines sometimes modeled here)
    if len(n.generators):
        n.generators["p_nom"] = n.generators["p_nom"].fillna(0.0)
        mask_gen = n.generators.carrier.str.lower().isin(HYDRO_GENERATOR_ALIASES)
        if mask_gen.any():
            n.generators.loc[mask_gen, "p_nom_extendable"] = False
            n.generators.loc[mask_gen, "p_nom_min"] = n.generators.loc[mask_gen, "p_nom"]
            n.generators.loc[mask_gen, "p_nom_max"] = n.generators.loc[mask_gen, "p_nom"]
            logging.info("Disabled extension for %d hydro generators.", int(mask_gen.sum()))
