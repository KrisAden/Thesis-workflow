"""Force unlimited renewable capacity directly."""

import argparse
import logging
from pathlib import Path
import numpy as np
import pypsa
from . import io as pio

def force_unlimited_renewables(n: pypsa.Network) -> None:
    """Directly set unlimited capacity for renewable generators."""
    
    renewable_techs = ['solar', 'onwind', 'offwind-ac', 'offwind-dc']
    
    for tech in renewable_techs:
        mask = n.generators['carrier'] == tech
        if mask.any():
            # Set to effectively unlimited
            n.generators.loc[mask, 'p_nom_max'] = 1e9
            n.generators.loc[mask, 'p_nom_extendable'] = True
            # Keep existing capacity as minimum
            n.generators.loc[mask, 'p_nom_min'] = n.generators.loc[mask, 'p_nom']
            
            logging.info(f"Set {mask.sum()} {tech} generators to unlimited capacity")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-in", required=True)
    parser.add_argument("--network-out", required=True)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    n = pio.load_network(args.network_in)
    force_unlimited_renewables(n)
    pio.save_network(n, args.network_out)
    
    logging.info(f"Wrote unlimited renewable network to {args.network_out}")

if __name__ == "__main__":
    main()
