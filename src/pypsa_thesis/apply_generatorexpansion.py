#Importing packages
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from . import io as pio

#Defining logging
def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

#Bound violation logging
def _bound_violations_gen(n: pypsa.Network) -> pd.DataFrame:
    """
    Collect bound violations for generator assets) where min > max.
    """
    rows = []

    if len(n.generators) and {"p_nom_min", "p_nom_max"}.issubset(n.generators.columns):
        bad = (
            n.generators.p_nom_min.notnull()
            & n.generators.p_nom_max.notnull()
            & (n.generators.p_nom_min > n.generators.p_nom_max)
        )
        if bad.any():
            for name in n.generators.index[bad]:
                rows.append(("generators", name))

    if not rows:
        return pd.DataFrame(columns=["component", "name"])

    return pd.DataFrame(rows, columns=["component", "name"])

#Adding Generator expansion
