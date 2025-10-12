"""Diagnostic module to check capacity limits in the network."""

import argparse
import logging
from pathlib import Path
import pandas as pd
import pypsa
from . import io as pio

def main():
    parser = argparse.ArgumentParser(description="Diagnose capacity limits")
    parser.add_argument("--network-in", required=True)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()
    
    n = pio.load_network(args.network_in)
    
    # Check renewable capacity limits
    renewable_techs = ['solar', 'onwind', 'offwind-ac', 'offwind-dc']
    
    diagnostics = []
    for tech in renewable_techs:
        tech_gens = n.generators[n.generators['carrier'] == tech]
        if len(tech_gens) > 0:
            # Check if limits are effectively unlimited (>= 1e6)
            unlimited_count = (tech_gens['p_nom_max'] >= 1e6).sum()
            limited_count = (tech_gens['p_nom_max'] < 1e6).sum()
            
            diagnostics.append({
                'technology': tech,
                'total_generators': len(tech_gens),
                'unlimited_generators': unlimited_count,
                'limited_generators': limited_count,
                'min_p_nom_max': tech_gens['p_nom_max'].min(),
                'max_p_nom_max': tech_gens['p_nom_max'].max(),
                'avg_p_nom_max': tech_gens['p_nom_max'].mean(),
                'all_unlimited': unlimited_count == len(tech_gens)
            })
    
    df = pd.DataFrame(diagnostics)
    df.to_csv(args.report_out, index=False)
    
    # Log summary
    logging.basicConfig(level=logging.INFO)
    for _, row in df.iterrows():
        if row['all_unlimited']:
            logging.info(f"✅ {row['technology']}: All generators unlimited")
        else:
            logging.warning(f"❌ {row['technology']}: {row['limited_generators']}/{row['total_generators']} still limited")

if __name__ == "__main__":
    main()
