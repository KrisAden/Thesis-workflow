#!/usr/bin/env python3
"""
Plotting module for PyPSA-Eur thesis workflow.

This module generates various plots based on the configuration settings in config.yaml.
"""

import argparse
import yaml
import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Try to import pypsa (you may need to install it if not already available)
try:
    import pypsa
except ImportError:
    print("Warning: PyPSA not available. Some network plotting features may not work.")
    pypsa = None

# Try to import geopandas for mapping
try:
    import geopandas as gpd
    from matplotlib.patches import Patch
except ImportError:
    print("Warning: GeoPandas not available. Map plotting features may not work.")
    gpd = None


def gini_coefficient(x):
    """Calculate Gini coefficient for inequality measurement."""
    x = np.array(x, dtype=float)
    if np.amin(x) < 0:
        x -= np.amin(x)
    if np.all(x == 0):
        return 0.0
    x_sorted = np.sort(x)
    n = len(x)
    cumulative = np.cumsum(x_sorted)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return gini


def hhi_index(x):
    """Calculate Herfindahl-Hirschman Index for concentration measurement."""
    x = np.array(x, dtype=float)
    if x.sum() == 0:
        return 0.0
    shares = x / x.sum()
    return np.sum(shares ** 2)


def is_renewable(carrier):
    """Check if a carrier is renewable."""
    renewables = ['wind', 'solar', 'ror', 'biomass', 'geothermal', 'nuclear', 'hydro']
    return isinstance(carrier, str) and any(kw in carrier.lower() for kw in renewables)


def extract_installed_renewable_capacities(network):
    """Extract installed renewable capacities by country from a PyPSA network."""
    gen = network.generators
    renewables = gen[gen['carrier'].apply(is_renewable)]

    if "p_nom_opt" not in renewables.columns:
        raise ValueError("Missing 'p_nom_opt' in generators!")

    bus_to_country = network.buses["country"].to_dict()
    renewables = renewables.copy()
    renewables["country"] = renewables["bus"].map(bus_to_country)

    return renewables.groupby("country")["p_nom_opt"].sum()


def extract_renewable_generation_output(network):
    """Extract annual renewable generation output by country from a PyPSA network."""
    gen = network.generators
    renewables = gen[gen['carrier'].apply(is_renewable)]

    if "p_nom_opt" not in renewables.columns:
        raise ValueError("Missing 'p_nom_opt' in generators!")
    
    if not hasattr(network, 'generators_t') or not hasattr(network.generators_t, 'p'):
        raise ValueError("Missing generation time series data!")

    # Get snapshot weights for proper annual calculation
    weights = getattr(network, 'snapshot_weightings', None)
    if weights is None:
        weights = pd.Series(1.0, index=network.snapshots)
    elif hasattr(weights, 'generators'):
        weights = weights['generators']
    elif hasattr(weights, 'objective'):
        weights = weights['objective']
    else:
        weights = weights

    # Calculate annual generation for renewable generators
    renewable_gens = renewables.index
    gen_timeseries = network.generators_t.p[renewable_gens]  # MW
    
    # Multiply by weights and sum over time to get annual generation (MWh)
    annual_gen_by_generator = (gen_timeseries.T * weights).T.sum()

    # Map generators to countries
    bus_to_country = network.buses["country"].to_dict()
    renewables = renewables.copy()
    renewables["country"] = renewables["bus"].map(bus_to_country)
    renewables["annual_generation_mwh"] = annual_gen_by_generator

    # Group by country and sum annual generation
    return renewables.groupby("country")["annual_generation_mwh"].sum()


def extract_total_nodal_investment(network, baseline_network):
    """Extract total nodal investment into green energy transition by country.
    
    This includes expansion costs for:
    - Storage (compared to baseline)
    - Transmission (allocated by region, compared to baseline) 
    - Generation capacity (compared to baseline)
    
    Args:
        network: PyPSA network for the scenario
        baseline_network: PyPSA baseline network (0% decarbonized)
    
    Returns:
        pd.Series: Total investment by country
    """
    bus_to_country = network.buses["country"].to_dict()
    investment_by_country = {}
    
    # Initialize all countries with zero investment
    all_countries = set(network.buses["country"].unique())
    for country in all_countries:
        investment_by_country[country] = 0.0
    
    # 1. Generation expansion investment
    gen = network.generators.copy()
    baseline_gen = baseline_network.generators.copy()
    
    # Calculate generation expansion (p_nom_opt - baseline p_nom_opt)
    for idx, row in gen.iterrows():
        if idx in baseline_gen.index:
            expansion = max(0, row['p_nom_opt'] - baseline_gen.loc[idx, 'p_nom_opt'])
        else:
            expansion = row['p_nom_opt']  # New generator
        
        if expansion > 0 and 'capital_cost' in row:
            country = bus_to_country.get(row['bus'])
            if country:
                investment_by_country[country] += expansion * row.get('capital_cost', 0)
    
    # 2. Storage expansion investment
    if hasattr(network, 'storage_units') and len(network.storage_units) > 0:
        storage = network.storage_units.copy()
        baseline_storage = baseline_network.storage_units.copy() if hasattr(baseline_network, 'storage_units') else pd.DataFrame()
        
        for idx, row in storage.iterrows():
            if len(baseline_storage) > 0 and idx in baseline_storage.index:
                expansion = max(0, row['p_nom_opt'] - baseline_storage.loc[idx, 'p_nom_opt'])
            else:
                expansion = row['p_nom_opt']  # New storage
            
            if expansion > 0 and 'capital_cost' in row:
                country = bus_to_country.get(row['bus'])
                if country:
                    investment_by_country[country] += expansion * row.get('capital_cost', 0)
    
    # 3. Transmission expansion investment (allocated 50/50 to connected regions)
    # Lines
    if hasattr(network, 'lines') and len(network.lines) > 0:
        lines = network.lines.copy()
        baseline_lines = baseline_network.lines.copy() if hasattr(baseline_network, 'lines') else pd.DataFrame()
        
        for idx, row in lines.iterrows():
            if len(baseline_lines) > 0 and idx in baseline_lines.index:
                expansion = max(0, row['s_nom_opt'] - baseline_lines.loc[idx, 's_nom_opt'])
            else:
                expansion = row['s_nom_opt']  # New line
            
            if expansion > 0 and 'capital_cost' in row:
                bus0_country = bus_to_country.get(row['bus0'])
                bus1_country = bus_to_country.get(row['bus1'])
                investment = expansion * row.get('capital_cost', 0)
                
                # Allocate 50% to each connected region
                if bus0_country:
                    investment_by_country[bus0_country] += investment * 0.5
                if bus1_country:
                    investment_by_country[bus1_country] += investment * 0.5
    
    # Links
    if hasattr(network, 'links') and len(network.links) > 0:
        links = network.links.copy()
        baseline_links = baseline_network.links.copy() if hasattr(baseline_network, 'links') else pd.DataFrame()
        
        for idx, row in links.iterrows():
            if len(baseline_links) > 0 and idx in baseline_links.index:
                expansion = max(0, row['p_nom_opt'] - baseline_links.loc[idx, 'p_nom_opt'])
            else:
                expansion = row['p_nom_opt']  # New link
            
            if expansion > 0 and 'capital_cost' in row:
                bus0_country = bus_to_country.get(row['bus0'])
                bus1_country = bus_to_country.get(row['bus1'])
                investment = expansion * row.get('capital_cost', 0)
                
                # Allocate 50% to each connected region
                if bus0_country:
                    investment_by_country[bus0_country] += investment * 0.5
                if bus1_country:
                    investment_by_country[bus1_country] += investment * 0.5
    
    return pd.Series(investment_by_country)


def extract_total_nodal_investment_scaled_by_load(network, baseline_network):
    """Extract total nodal investment into green energy transition by country, scaled by annual load.
    
    This includes expansion costs for:
    - Storage (compared to baseline)
    - Transmission (allocated by region, compared to baseline) 
    - Generation capacity (compared to baseline)
    
    Investment is then scaled by the annual electricity load of each region.
    
    Args:
        network: PyPSA network for the scenario
        baseline_network: PyPSA baseline network (0% decarbonized)
    
    Returns:
        pd.Series: Total investment per unit load by country (€/MWh)
    """
    # First get total investment by country
    investment_by_country = extract_total_nodal_investment(network, baseline_network)
    
    # Get annual load by country
    bus_to_country = network.buses["country"].to_dict()
    load_by_country = {}
    
    # Initialize all countries with zero load
    all_countries = set(network.buses["country"].unique())
    for country in all_countries:
        load_by_country[country] = 0.0
    
    # Sum up loads for each country
    loads = network.loads_t.p_set
    for load_idx in loads.columns:
        if load_idx in network.loads.index:
            load_bus = network.loads.loc[load_idx, 'bus']
            country = bus_to_country.get(load_bus)
            if country:
                # Sum annual load (assuming hourly data)
                annual_load = loads[load_idx].sum()  # MWh
                load_by_country[country] += annual_load
    
    # Calculate investment per unit load (€/MWh)
    investment_per_load = {}
    for country in all_countries:
        total_investment = investment_by_country.get(country, 0)
        annual_load = load_by_country.get(country, 0)
        
        if annual_load > 0:
            investment_per_load[country] = total_investment / annual_load
        else:
            investment_per_load[country] = 0.0
    
    return pd.Series(investment_per_load)


def calculate_renewable_capacity_concentration(region_caps):
    """Calculate the fraction of total renewable capacity in top N regions.
    
    Args:
        region_caps: pd.Series with renewable capacity by region
    
    Returns:
        dict: Fractions for top 1, 3, and 5 regions
    """
    if len(region_caps) == 0 or region_caps.sum() == 0:
        return {"top_1": 0, "top_3": 0, "top_5": 0}
    
    # Sort regions by capacity (descending)
    sorted_caps = region_caps.sort_values(ascending=False)
    total_capacity = sorted_caps.sum()
    
    # Calculate cumulative fractions
    top_1_fraction = sorted_caps.iloc[0] / total_capacity if len(sorted_caps) >= 1 else 0
    top_3_fraction = sorted_caps.iloc[:3].sum() / total_capacity if len(sorted_caps) >= 3 else sorted_caps.sum() / total_capacity
    top_5_fraction = sorted_caps.iloc[:5].sum() / total_capacity if len(sorted_caps) >= 5 else sorted_caps.sum() / total_capacity
    
    return {
        "top_1": top_1_fraction,
        "top_3": top_3_fraction,
        "top_5": top_5_fraction
    }


def analyze_capacity_by_pentiles(region_caps):
    """Analyze renewable capacity distribution by pentiles.
    
    Args:
        region_caps: pd.Series with renewable capacity by region
    
    Returns:
        dict: Analysis of capacity distribution by pentiles
    """
    if len(region_caps) == 0 or region_caps.sum() == 0:
        return {}
    
    # Sort regions by capacity (descending)
    sorted_caps = region_caps.sort_values(ascending=False)
    total_capacity = sorted_caps.sum()
    n_regions = len(sorted_caps)
    
    # Calculate pentiles (5 groups of ~20% each)
    pentile_size = max(1, n_regions // 5)
    
    pentiles = {}
    for i in range(5):
        start_idx = i * pentile_size
        if i == 4:  # Last pentile gets all remaining regions
            end_idx = n_regions
        else:
            end_idx = min((i + 1) * pentile_size, n_regions)
        
        if start_idx < n_regions:
            pentile_caps = sorted_caps.iloc[start_idx:end_idx]
            pentiles[f"pentile_{i+1}"] = {
                "capacity": pentile_caps.sum(),
                "fraction": pentile_caps.sum() / total_capacity,
                "regions": list(pentile_caps.index),
                "avg_capacity": pentile_caps.mean(),
                "count": len(pentile_caps)
            }
    
    return pentiles


def extract_regional_characteristics(network):
    """Extract various characteristics that might drive renewable investment.
    
    Args:
        network: PyPSA network
    
    Returns:
        dict: Regional characteristics by country
    """
    bus_to_country = network.buses["country"].to_dict()
    characteristics = {}
    
    # Initialize for all countries
    all_countries = set(network.buses["country"].unique())
    for country in all_countries:
        characteristics[country] = {
            "total_load": 0.0,
            "wind_capacity": 0.0,
            "solar_capacity": 0.0,
            "existing_renewable_capacity": 0.0,
            "transmission_connections": 0,
            "storage_capacity": 0.0
        }
    
    # 1. Annual load by country
    if hasattr(network, 'loads_t') and hasattr(network.loads_t, 'p_set'):
        loads = network.loads_t.p_set
        for load_idx in loads.columns:
            if load_idx in network.loads.index:
                load_bus = network.loads.loc[load_idx, 'bus']
                country = bus_to_country.get(load_bus)
                if country:
                    annual_load = loads[load_idx].sum()
                    characteristics[country]["total_load"] += annual_load
    
    # 2. Renewable capacity by technology
    gen = network.generators
    for idx, row in gen.iterrows():
        country = bus_to_country.get(row['bus'])
        if country:
            capacity = row.get('p_nom_opt', 0)
            carrier = str(row.get('carrier', '')).lower()
            
            if 'wind' in carrier:
                characteristics[country]["wind_capacity"] += capacity
            elif 'solar' in carrier:
                characteristics[country]["solar_capacity"] += capacity
            
            if is_renewable(carrier):
                characteristics[country]["existing_renewable_capacity"] += capacity
    
    # 3. Storage capacity
    if hasattr(network, 'storage_units') and len(network.storage_units) > 0:
        storage = network.storage_units
        for idx, row in storage.iterrows():
            country = bus_to_country.get(row['bus'])
            if country:
                capacity = row.get('p_nom_opt', 0)
                characteristics[country]["storage_capacity"] += capacity
    
    # 4. Transmission connections (count of lines/links)
    if hasattr(network, 'lines') and len(network.lines) > 0:
        lines = network.lines
        for idx, row in lines.iterrows():
            country0 = bus_to_country.get(row['bus0'])
            country1 = bus_to_country.get(row['bus1'])
            if country0:
                characteristics[country0]["transmission_connections"] += 1
            if country1:
                characteristics[country1]["transmission_connections"] += 1
    
    if hasattr(network, 'links') and len(network.links) > 0:
        links = network.links
        for idx, row in links.iterrows():
            country0 = bus_to_country.get(row['bus0'])
            country1 = bus_to_country.get(row['bus1'])
            if country0:
                characteristics[country0]["transmission_connections"] += 1
            if country1:
                characteristics[country1]["transmission_connections"] += 1
    
    return characteristics


def analyze_transmission_bottlenecks(network):
    """Analyze transmission capacity constraints and bottlenecks.
    
    Args:
        network: PyPSA network
    
    Returns:
        dict: Analysis of transmission constraints by region and connection
    """
    bus_to_country = network.buses["country"].to_dict()
    bottleneck_analysis = {
        "lines": [],
        "links": [],
        "regional_constraints": {}
    }
    
    # Initialize regional constraints
    all_countries = set(network.buses["country"].unique())
    for country in all_countries:
        bottleneck_analysis["regional_constraints"][country] = {
            "total_capacity_opt": 0.0,
            "total_capacity_max": 0.0,
            "constrained_connections": 0,
            "total_connections": 0,
            "utilization_ratio": 0.0
        }
    
    # Analyze lines (AC transmission)
    if hasattr(network, 'lines') and len(network.lines) > 0:
        lines = network.lines
        for idx, row in lines.iterrows():
            s_nom_opt = row.get('s_nom_opt', 0)
            s_nom_max = row.get('s_nom_max', float('inf'))
            
            # Check if at capacity limit
            at_limit = s_nom_max < float('inf') and s_nom_opt >= 0.95 * s_nom_max
            utilization = s_nom_opt / s_nom_max if s_nom_max > 0 and s_nom_max < float('inf') else 0
            
            country0 = bus_to_country.get(row['bus0'])
            country1 = bus_to_country.get(row['bus1'])
            
            line_data = {
                "name": idx,
                "bus0": row['bus0'],
                "bus1": row['bus1'],
                "country0": country0,
                "country1": country1,
                "s_nom_opt": s_nom_opt,
                "s_nom_max": s_nom_max,
                "utilization": utilization,
                "at_limit": at_limit,
                "type": "line"
            }
            bottleneck_analysis["lines"].append(line_data)
            
            # Update regional statistics
            for country in [country0, country1]:
                if country:
                    bottleneck_analysis["regional_constraints"][country]["total_capacity_opt"] += s_nom_opt / 2
                    if s_nom_max < float('inf'):
                        bottleneck_analysis["regional_constraints"][country]["total_capacity_max"] += s_nom_max / 2
                    bottleneck_analysis["regional_constraints"][country]["total_connections"] += 1
                    if at_limit:
                        bottleneck_analysis["regional_constraints"][country]["constrained_connections"] += 1
    
    # Analyze links (DC transmission, interconnectors)
    if hasattr(network, 'links') and len(network.links) > 0:
        links = network.links
        for idx, row in links.iterrows():
            p_nom_opt = row.get('p_nom_opt', 0)
            p_nom_max = row.get('p_nom_max', float('inf'))
            
            # Check if at capacity limit
            at_limit = p_nom_max < float('inf') and p_nom_opt >= 0.95 * p_nom_max
            utilization = p_nom_opt / p_nom_max if p_nom_max > 0 and p_nom_max < float('inf') else 0
            
            country0 = bus_to_country.get(row['bus0'])
            country1 = bus_to_country.get(row['bus1'])
            
            link_data = {
                "name": idx,
                "bus0": row['bus0'],
                "bus1": row['bus1'],
                "country0": country0,
                "country1": country1,
                "p_nom_opt": p_nom_opt,
                "p_nom_max": p_nom_max,
                "utilization": utilization,
                "at_limit": at_limit,
                "type": "link"
            }
            bottleneck_analysis["links"].append(link_data)
            
            # Update regional statistics
            for country in [country0, country1]:
                if country:
                    bottleneck_analysis["regional_constraints"][country]["total_capacity_opt"] += p_nom_opt / 2
                    if p_nom_max < float('inf'):
                        bottleneck_analysis["regional_constraints"][country]["total_capacity_max"] += p_nom_max / 2
                    bottleneck_analysis["regional_constraints"][country]["total_connections"] += 1
                    if at_limit:
                        bottleneck_analysis["regional_constraints"][country]["constrained_connections"] += 1
    
    # Calculate utilization ratios
    for country, data in bottleneck_analysis["regional_constraints"].items():
        if data["total_capacity_max"] > 0:
            data["utilization_ratio"] = data["total_capacity_opt"] / data["total_capacity_max"]
    
    return bottleneck_analysis


def load_networks_from_results(config, has_baseline=True):
    """Load networks from results/networks directory with proper naming convention."""
    networks_by_percent = {}
    
    # Load baseline (0% reduction)
    if has_baseline:
        try:
            baseline_path = "results/networks/solved_baseline_costed_expansion.nc"
            if os.path.exists(baseline_path) and pypsa:
                print(f"📥 Loading baseline network (0% reduction) from {baseline_path}")
                networks_by_percent[0] = pypsa.Network(baseline_path)
            else:
                print(f"⚠️ Baseline network not found at {baseline_path}")
        except Exception as e:
            print(f"⚠️ Error loading baseline network: {e}")
    
    # Load reduction scenarios
    reductions = config.get("parameters", {}).get("co2_reductions", [])
    for reduction in reductions:
        if float(reduction) > 0:
            try:
                network_path = f"results/networks/solved_reduction_{reduction}.nc"
                if os.path.exists(network_path) and pypsa:
                    print(f"📥 Loading network for {reduction}% reduction from {network_path}")
                    networks_by_percent[float(reduction)] = pypsa.Network(network_path)
                else:
                    print(f"⚠️ Network not found at {network_path}")
            except Exception as e:
                print(f"⚠️ Error loading {reduction}% reduction network: {e}")
    
    return dict(sorted(networks_by_percent.items()))


def load_k_constrained_networks_from_results(config):
    """Load K-constrained networks from results/networks directory."""
    networks_by_reduction_k = {}
    
    # Get parameters from config
    reductions = config.get("parameters", {}).get("co2_reductions", [])
    k_values = config.get("parameters", {}).get("decentralization", {}).get("k_values", [])
    
    for reduction in reductions:
        if float(reduction) > 0:  # Skip baseline
            for k_value in k_values:
                try:
                    network_path = f"results/networks/decentralized_reduction_{reduction}_k_{k_value}.nc"
                    if os.path.exists(network_path) and pypsa:
                        print(f"📥 Loading K-constrained network for {reduction}% reduction, k={k_value} from {network_path}")
                        key = (float(reduction), float(k_value))
                        networks_by_reduction_k[key] = pypsa.Network(network_path)
                    else:
                        print(f"⚠️ K-constrained network not found at {network_path}")
                except Exception as e:
                    print(f"⚠️ Error loading {reduction}% reduction, k={k_value} network: {e}")
    
    return networks_by_reduction_k


def plot_renewable_capacity_inequality(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot Gini coefficient of renewable capacity inequality across scenarios."""
    print("Creating renewable capacity inequality plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping renewable capacity inequality plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Calculate Gini coefficients
    results = []
    for co2_pct, net in networks_by_percent.items():
        try:
            region_capacities = extract_installed_renewable_capacities(net)
            gini = gini_coefficient(region_capacities.values)
            hhi = hhi_index(region_capacities.values)
            results.append({"CO₂ Reduction (%)": int(co2_pct), "Gini": gini, "HHI": hhi})
            print(f"  ✓ Calculated Gini={gini:.3f} for {co2_pct}% reduction")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")

    if not results:
        print("No valid results - cannot create plot")
        return
        
    df = pd.DataFrame(results).sort_values("CO₂ Reduction (%)")

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}

    # Plot Gini coefficient
    ax1.plot(df["CO₂ Reduction (%)"], df["Gini"], marker="o", label="Gini Coefficient", 
             color="tab:blue", linewidth=2, markersize=8)

    ax1.set_ylabel("Gini Coefficient", **font)
    ax1.set_xlabel("CO₂ Reduction (%)", **font)
    ax1.set_title("Inequality of Renewable Capacity Installation", **font)

    # Style the plot
    ax1.grid(True, alpha=0.3)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, loc="upper left")

    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"renewable_generation_inequality.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved plot as {output_file}")

    plt.close(fig)


def plot_green_investment_inequality(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot Gini coefficient of total nodal investment into green energy transition across scenarios."""
    print("Creating green investment inequality plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping green investment inequality plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Need baseline network for comparison
    if 0 not in networks_by_percent:
        print("Baseline network (0% reduction) not found - cannot calculate investment relative to baseline")
        return
    
    baseline_network = networks_by_percent[0]
    
    # Calculate Gini coefficients for investment
    results = []
    for co2_pct, net in networks_by_percent.items():
        if co2_pct == 0:  # Skip baseline for investment calculation
            continue
            
        try:
            investment_by_country = extract_total_nodal_investment(net, baseline_network)
            gini = gini_coefficient(investment_by_country.values)
            hhi = hhi_index(investment_by_country.values)
            total_investment = investment_by_country.sum()
            results.append({
                "CO₂ Reduction (%)": int(co2_pct), 
                "Gini": gini, 
                "HHI": hhi,
                "Total Investment (M€)": total_investment / 1e6  # Convert to millions
            })
            print(f"  ✓ Calculated Gini={gini:.3f}, Total Investment={total_investment/1e6:.1f}M€ for {co2_pct}% reduction")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")

    if not results:
        print("No valid results - cannot create plot")
        return
        
    df = pd.DataFrame(results).sort_values("CO₂ Reduction (%)")

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}

    # Plot Gini coefficient
    ax1.plot(df["CO₂ Reduction (%)"], df["Gini"], marker="o", label="Gini Coefficient", 
             color="tab:green", linewidth=2, markersize=8)

    ax1.set_ylabel("Gini Coefficient", **font)
    ax1.set_xlabel("CO₂ Reduction (%)", **font)
    ax1.set_title("Inequality of Green Energy Transition Investment", **font)

    # Style the plot
    ax1.grid(True, alpha=0.3)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, loc="upper left")

    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"green_investment_inequality.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved plot as {output_file}")

    plt.close(fig)


def plot_renewable_capacity_concentration(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot the fraction of total renewable capacity in top N regions across CO₂ reduction scenarios."""
    print("Creating renewable capacity concentration plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping renewable capacity concentration plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Calculate concentration metrics for each scenario
    results = []
    for co2_pct, net in networks_by_percent.items():
        try:
            region_caps = extract_installed_renewable_capacities(net)
            concentration = calculate_renewable_capacity_concentration(region_caps)
            
            results.append({
                "CO₂ Reduction (%)": int(co2_pct),
                "Top 1 Region": concentration["top_1"],
                "Top 3 Regions": concentration["top_3"],
                "Top 5 Regions": concentration["top_5"]
            })
            print(f"  ✓ Calculated concentration for {co2_pct}% reduction: Top1={concentration['top_1']:.2f}, Top3={concentration['top_3']:.2f}, Top5={concentration['top_5']:.2f}")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")

    if not results:
        print("No valid results - cannot create plot")
        return
        
    df = pd.DataFrame(results).sort_values("CO₂ Reduction (%)")

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}

    # Plot concentration lines
    ax.plot(df["CO₂ Reduction (%)"], df["Top 1 Region"], 
            marker="o", label="Top 1 Region", color="tab:red", linewidth=2, markersize=8)
    ax.plot(df["CO₂ Reduction (%)"], df["Top 3 Regions"], 
            marker="s", label="Top 3 Regions", color="tab:orange", linewidth=2, markersize=8)
    ax.plot(df["CO₂ Reduction (%)"], df["Top 5 Regions"], 
            marker="^", label="Top 5 Regions", color="tab:blue", linewidth=2, markersize=8)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    ax.set_ylabel("Fraction of Total Renewable Capacity", **font)
    ax.set_xlabel("CO₂ Reduction (%)", **font)
    ax.set_title("Concentration of Renewable Capacity in Top Regions", **font)

    # Style the plot
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    ax.set_ylim(0, 1.05)  # Set y-limit from 0 to just above 100%

    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"renewable_capacity_concentration.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved plot as {output_file}")

    plt.close(fig)


def plot_green_investment_inequality_load_scaled(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot Gini coefficient of green investment scaled by annual load across scenarios."""
    print("Creating load-scaled green investment inequality plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping load-scaled green investment inequality plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Need baseline network for comparison
    if 0 not in networks_by_percent:
        print("Baseline network (0% reduction) not found - cannot calculate investment relative to baseline")
        return
    
    baseline_network = networks_by_percent[0]
    
    # Calculate Gini coefficients for load-scaled investment
    results = []
    for co2_pct, net in networks_by_percent.items():
        if co2_pct == 0:  # Skip baseline for investment calculation
            continue
            
        try:
            investment_per_load = extract_total_nodal_investment_scaled_by_load(net, baseline_network)
            gini = gini_coefficient(investment_per_load.values)
            hhi = hhi_index(investment_per_load.values)
            mean_investment_per_load = investment_per_load.mean()
            
            results.append({
                "CO₂ Reduction (%)": int(co2_pct), 
                "Gini": gini, 
                "HHI": hhi,
                "Mean Investment per Load (€/MWh)": mean_investment_per_load
            })
            print(f"  ✓ Calculated Load-scaled Gini={gini:.3f}, Mean Investment/Load={mean_investment_per_load:.2f}€/MWh for {co2_pct}% reduction")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")

    if not results:
        print("No valid results - cannot create plot")
        return
        
    df = pd.DataFrame(results).sort_values("CO₂ Reduction (%)")

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}

    # Plot Gini coefficient
    ax1.plot(df["CO₂ Reduction (%)"], df["Gini"], marker="o", label="Gini Coefficient", 
             color="tab:purple", linewidth=2, markersize=8)

    ax1.set_ylabel("Gini Coefficient", **font)
    ax1.set_xlabel("CO₂ Reduction (%)", **font)
    ax1.set_title("Inequality of Green Investment per Unit Load", **font)

    # Style the plot
    ax1.grid(True, alpha=0.3)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, loc="upper left")

    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"green_investment_inequality_load_scaled.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved plot as {output_file}")

    plt.close(fig)


def extract_renewable_penetration_by_country(network):
    """Extract renewable penetration (%) by country from a PyPSA network."""
    try:
        # Get total generation by country
        bus_to_country = network.buses["country"].to_dict()
        
        # Check if we have the required data
        if not hasattr(network, 'generators_t') or not hasattr(network.generators_t, 'p'):
            raise ValueError("Missing generation time series data (generators_t.p)")
        
        # Get snapshot weights for proper annual calculation
        weights = getattr(network, 'snapshot_weightings', None)
        if weights is None:
            weights = pd.Series(1.0, index=network.snapshots)
        elif hasattr(weights, 'generators'):
            weights = weights['generators']
        elif hasattr(weights, 'objective'):
            weights = weights['objective']
        else:
            weights = weights

        # Calculate total generation by generator
        all_gens = network.generators.index
        gen_timeseries = network.generators_t.p[all_gens]  # MW
        annual_gen_by_generator = (gen_timeseries.T * weights).T.sum()  # MWh

        # Map generators to countries and calculate total generation by country
        gen_with_country = network.generators.copy()
        gen_with_country["country"] = gen_with_country["bus"].map(bus_to_country)
        gen_with_country["annual_generation_mwh"] = annual_gen_by_generator
        
        total_gen_by_country = gen_with_country.groupby("country")["annual_generation_mwh"].sum()
        
        # Calculate renewable generation by country
        renewables = gen_with_country[gen_with_country['carrier'].apply(is_renewable)]
        renewable_gen_by_country = renewables.groupby("country")["annual_generation_mwh"].sum()
        
        # Calculate penetration percentage
        penetration_by_country = (renewable_gen_by_country / total_gen_by_country * 100).fillna(0)
        
        # Ensure we have reasonable values
        penetration_by_country = penetration_by_country.clip(0, 100)
        
        if penetration_by_country.empty:
            raise ValueError("No renewable penetration data calculated")
        
        return penetration_by_country
        
    except Exception as e:
        print(f"Error in extract_renewable_penetration_by_country: {e}")
        raise


def plot_renewable_penetration_gini_by_decarbonization(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot Gini coefficient of renewable penetration with separate subplot for each decarbonization level."""
    print("Creating renewable penetration Gini by decarbonization plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping renewable penetration Gini plot")
        # Create empty files to satisfy Snakemake
        for fmt in output_formats:
            output_file = output_path / f"renewable_penetration_gini_by_decarbonization.{fmt}"
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "PyPSA not available", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Renewable Penetration Gini by Decarbonization")
            fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Created placeholder plot as {output_file}")
        return
    
    try:
        # Load networks
        networks_by_percent = load_networks_from_results(config, has_baseline)
        
        if not networks_by_percent:
            print("No networks found - creating placeholder plot")
            # Create placeholder files
            for fmt in output_formats:
                output_file = output_path / f"renewable_penetration_gini_by_decarbonization.{fmt}"
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No networks found", ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Renewable Penetration Gini by Decarbonization")
                fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ Created placeholder plot as {output_file}")
            return

        # Calculate Gini coefficients for each scenario
        results = []
        penetration_data = {}
        
        for co2_pct, net in networks_by_percent.items():
            try:
                penetration = extract_renewable_penetration_by_country(net)
                gini = gini_coefficient(penetration.values)
                results.append({"CO₂ Reduction (%)": int(co2_pct), "Gini": gini})
                penetration_data[co2_pct] = penetration
                print(f"  ✓ Calculated Gini={gini:.3f} for {co2_pct}% reduction")
            except Exception as e:
                print(f"⚠️ Skipping {co2_pct}% due to error: {e}")

        if not results:
            print("No valid results - creating placeholder plot")
            # Create placeholder files
            for fmt in output_formats:
                output_file = output_path / f"renewable_penetration_gini_by_decarbonization.{fmt}"
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No valid calculation results", ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Renewable Penetration Gini by Decarbonization")
                fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ Created placeholder plot as {output_file}")
            return
            
        df = pd.DataFrame(results).sort_values("CO₂ Reduction (%)")
        
        # Determine subplot layout
        n_scenarios = len(df)
        if n_scenarios <= 2:
            rows, cols = 1, n_scenarios
            figsize = (6 * cols, 5)
        elif n_scenarios <= 4:
            rows, cols = 2, 2
            figsize = (12, 10)
        elif n_scenarios <= 6:
            rows, cols = 2, 3
            figsize = (18, 10)
        else:
            rows, cols = 3, 3
            figsize = (18, 15)

        # Create the plot
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_scenarios == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Set font properties
        font = {'fontsize': 12, 'fontweight': 'bold'}
        title_font = {'fontsize': 14, 'fontweight': 'bold'}
        
        # Color palette for countries
        colors = plt.cm.Set3(np.linspace(0, 1, 20))  # Up to 20 countries
        
        # Plot each scenario
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            co2_pct = f"{int(row['CO₂ Reduction (%)'])}"
            
            # Get penetration data for this scenario
            penetration = penetration_data[co2_pct]
            countries = penetration.index
            
            # Create bar plot
            bars = ax.bar(range(len(countries)), penetration.values, 
                         color=colors[:len(countries)], alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Customize the subplot
            ax.set_title(f'{co2_pct}% CO₂ Reduction\nGini: {row["Gini"]:.3f}', **title_font)
            ax.set_ylabel('Renewable Penetration (%)', **font)
            ax.set_xlabel('Countries', **font)
            
            # Set x-axis labels
            ax.set_xticks(range(len(countries)))
            ax.set_xticklabels(countries, rotation=45, ha='right')
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            
            # Set y-axis limits to 0-100%
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, penetration.values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=9)

        # Hide unused subplots
        for i in range(len(df), len(axes)):
            axes[i].set_visible(False)

        # Add overall title
        fig.suptitle('Renewable Penetration Inequality Across Decarbonization Scenarios', 
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        # Save in all requested formats
        for fmt in output_formats:
            output_file = output_path / f"renewable_penetration_gini_by_decarbonization.{fmt}"
            fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
            print(f"  ✓ Saved plot as {output_file}")

        plt.close(fig)
        print(f"  ✓ Successfully created plot with {len(df)} scenarios")
        
    except Exception as e:
        print(f"⚠️ Error in plot_renewable_penetration_gini_by_decarbonization: {e}")
        import traceback
        traceback.print_exc()
        
        # Create placeholder files to satisfy Snakemake
        for fmt in output_formats:
            output_file = output_path / f"renewable_penetration_gini_by_decarbonization.{fmt}"
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Renewable Penetration Gini by Decarbonization")
            fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Created error placeholder as {output_file}")


def plot_renewable_capacity_pentiles(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot renewable capacity distribution by pentiles across CO₂ reduction scenarios."""
    print("Creating renewable capacity pentiles analysis plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping renewable capacity pentiles plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Analyze pentiles for each scenario
    results = []
    for co2_pct, net in networks_by_percent.items():
        try:
            region_caps = extract_installed_renewable_capacities(net)
            pentiles = analyze_capacity_by_pentiles(region_caps)
            
            result = {"CO₂ Reduction (%)": int(co2_pct)}
            
            for pentile_name, pentile_data in pentiles.items():
                result[f"{pentile_name}_fraction"] = pentile_data["fraction"]
            
            results.append(result)
            print(f"  ✓ Analyzed pentiles for {co2_pct}% reduction")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")

    if not results:
        print("No valid results - cannot create plot")
        return
        
    df = pd.DataFrame(results).sort_values("CO₂ Reduction (%)")

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}

    # Colors for each pentile
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']  # Red, Orange, Green, Blue, Purple
    markers = ['o', 's', '^', 'D', 'v']
    
    # Plot each pentile
    for i, (color, marker) in enumerate(zip(colors, markers)):
        pentile_col = f"pentile_{i+1}_fraction"
        if pentile_col in df.columns:
            ax.plot(df["CO₂ Reduction (%)"], df[pentile_col], 
                   marker=marker, label=f"Pentile {i+1} (Top {20*(i+1)}%)", 
                   color=color, linewidth=2, markersize=8)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    ax.set_ylabel("Fraction of Total Renewable Capacity", **font)
    ax.set_xlabel("CO₂ Reduction (%)", **font)
    ax.set_title("Renewable Capacity Distribution by Region Pentiles", **font)

    # Style the plot
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=10)
    ax.set_ylim(0, max(0.6, df[[col for col in df.columns if col.endswith('_fraction')]].max().max() * 1.1))

    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"renewable_capacity_pentiles.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved plot as {output_file}")

    plt.close(fig)


def plot_middle_pentile_characteristics(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot characteristics of middle pentile regions to understand their investment drivers."""
    print("Creating middle pentile characteristics analysis plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping middle pentile characteristics plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Focus on a specific scenario (e.g., 50% reduction) for detailed analysis
    target_scenario = 50
    if target_scenario not in networks_by_percent:
        # Fall back to highest available scenario
        target_scenario = max(networks_by_percent.keys())
    
    network = networks_by_percent[target_scenario]
    
    try:
        # Get renewable capacities and pentile analysis
        region_caps = extract_installed_renewable_capacities(network)
        pentiles = analyze_capacity_by_pentiles(region_caps)
        characteristics = extract_regional_characteristics(network)
        
        # Focus on middle pentiles (2nd, 3rd, 4th)
        middle_regions = []
        for pentile_num in [2, 3, 4]:
            pentile_key = f"pentile_{pentile_num}"
            if pentile_key in pentiles:
                middle_regions.extend(pentiles[pentile_key]["regions"])
        
        if not middle_regions:
            print("No middle pentile regions found")
            return
        
        # Extract characteristics for middle regions
        middle_chars = {region: characteristics[region] for region in middle_regions if region in characteristics}
        
        if not middle_chars:
            print("No characteristics data for middle regions")
            return
        
        # Prepare data for plotting
        regions = list(middle_chars.keys())
        wind_caps = [middle_chars[r]["wind_capacity"] for r in regions]
        solar_caps = [middle_chars[r]["solar_capacity"] for r in regions]
        loads = [middle_chars[r]["total_load"] for r in regions]
        storage_caps = [middle_chars[r]["storage_capacity"] for r in regions]
        
        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        font = {'fontsize': 11, 'fontweight': 'bold'}
        
        # 1. Wind vs Solar capacity
        ax1.scatter(wind_caps, solar_caps, alpha=0.7, s=100, color='tab:green')
        ax1.set_xlabel("Wind Capacity (MW)", **font)
        ax1.set_ylabel("Solar Capacity (MW)", **font)
        ax1.set_title("Wind vs Solar in Middle Pentile Regions", **font)
        ax1.grid(True, alpha=0.3)
        
        # Add region labels
        for i, region in enumerate(regions):
            ax1.annotate(region, (wind_caps[i], solar_caps[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 2. Load vs Total Renewable Capacity
        total_renewable = [wind_caps[i] + solar_caps[i] for i in range(len(regions))]
        ax2.scatter(loads, total_renewable, alpha=0.7, s=100, color='tab:blue')
        ax2.set_xlabel("Annual Load (MWh)", **font)
        ax2.set_ylabel("Total Renewable Capacity (MW)", **font)
        ax2.set_title("Load vs Renewable Capacity", **font)
        ax2.grid(True, alpha=0.3)
        
        # Add region labels
        for i, region in enumerate(regions):
            ax2.annotate(region, (loads[i], total_renewable[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. Storage capacity distribution
        ax3.bar(range(len(regions)), storage_caps, color='tab:orange', alpha=0.7)
        ax3.set_xlabel("Regions", **font)
        ax3.set_ylabel("Storage Capacity (MW)", **font)
        ax3.set_title("Storage Capacity in Middle Pentile Regions", **font)
        ax3.set_xticks(range(len(regions)))
        ax3.set_xticklabels(regions, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Renewable capacity intensity (capacity per unit load)
        intensity = [total_renewable[i] / max(loads[i], 1) for i in range(len(regions))]
        ax4.bar(range(len(regions)), intensity, color='tab:red', alpha=0.7)
        ax4.set_xlabel("Regions", **font)
        ax4.set_ylabel("Renewable Intensity (MW/MWh)", **font)
        ax4.set_title("Renewable Capacity Intensity", **font)
        ax4.set_xticks(range(len(regions)))
        ax4.set_xticklabels(regions, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f"Middle Pentile Region Characteristics ({target_scenario}% CO₂ Reduction)", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save in all requested formats
        for fmt in output_formats:
            output_file = output_path / f"middle_pentile_characteristics.{fmt}"
            fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
            print(f"  ✓ Saved plot as {output_file}")
        
        plt.close(fig)
        
        print(f"  ✓ Analyzed {len(middle_regions)} middle pentile regions: {', '.join(middle_regions)}")
        
    except Exception as e:
        print(f"⚠️ Error in middle pentile analysis: {e}")


def plot_capacity_expansion_evolution(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot how renewable capacity expansion evolves across scenarios for different region groups."""
    print("Creating capacity expansion evolution plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping capacity expansion evolution plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Need baseline for expansion calculation
    if 0 not in networks_by_percent:
        print("Baseline network not found - cannot calculate expansion")
        return
    
    baseline_network = networks_by_percent[0]
    baseline_caps = extract_installed_renewable_capacities(baseline_network)
    
    # Classify regions based on baseline capacity
    sorted_baseline = baseline_caps.sort_values(ascending=False)
    n_regions = len(sorted_baseline)
    
    # Define region groups
    top_regions = set(sorted_baseline.iloc[:max(1, n_regions//5)].index)  # Top 20%
    middle_regions = set(sorted_baseline.iloc[n_regions//5:4*n_regions//5].index)  # Middle 60%
    bottom_regions = set(sorted_baseline.iloc[4*n_regions//5:].index)  # Bottom 20%
    
    results = []
    for co2_pct, net in networks_by_percent.items():
        if co2_pct == 0:  # Skip baseline
            continue
            
        try:
            current_caps = extract_installed_renewable_capacities(net)
            
            # Calculate expansion for each group
            top_expansion = sum(max(0, current_caps.get(r, 0) - baseline_caps.get(r, 0)) for r in top_regions)
            middle_expansion = sum(max(0, current_caps.get(r, 0) - baseline_caps.get(r, 0)) for r in middle_regions)
            bottom_expansion = sum(max(0, current_caps.get(r, 0) - baseline_caps.get(r, 0)) for r in bottom_regions)
            
            total_expansion = top_expansion + middle_expansion + bottom_expansion
            
            if total_expansion > 0:
                results.append({
                    "CO₂ Reduction (%)": int(co2_pct),
                    "Top Regions (MW)": top_expansion,
                    "Middle Regions (MW)": middle_expansion,
                    "Bottom Regions (MW)": bottom_expansion,
                    "Top Regions (%)": top_expansion / total_expansion * 100,
                    "Middle Regions (%)": middle_expansion / total_expansion * 100,
                    "Bottom Regions (%)": bottom_expansion / total_expansion * 100
                })
            
            print(f"  ✓ Calculated expansion for {co2_pct}% reduction: Top={top_expansion:.0f}MW, Middle={middle_expansion:.0f}MW, Bottom={bottom_expansion:.0f}MW")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")

    if not results:
        print("No valid results - cannot create plot")
        return
        
    df = pd.DataFrame(results).sort_values("CO₂ Reduction (%)")

    # Create subplot figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    font = {'fontsize': 12, 'fontweight': 'bold'}
    
    # Plot 1: Absolute expansion
    ax1.plot(df["CO₂ Reduction (%)"], df["Top Regions (MW)"], 
            marker="o", label="Top 20% Regions", color="tab:red", linewidth=2, markersize=8)
    ax1.plot(df["CO₂ Reduction (%)"], df["Middle Regions (MW)"], 
            marker="s", label="Middle 60% Regions", color="tab:green", linewidth=2, markersize=8)
    ax1.plot(df["CO₂ Reduction (%)"], df["Bottom Regions (MW)"], 
            marker="^", label="Bottom 20% Regions", color="tab:blue", linewidth=2, markersize=8)
    
    ax1.set_ylabel("Renewable Capacity Expansion (MW)", **font)
    ax1.set_xlabel("CO₂ Reduction (%)", **font)
    ax1.set_title("Absolute Renewable Capacity Expansion", **font)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    
    # Plot 2: Percentage share of expansion
    ax2.plot(df["CO₂ Reduction (%)"], df["Top Regions (%)"], 
            marker="o", label="Top 20% Regions", color="tab:red", linewidth=2, markersize=8)
    ax2.plot(df["CO₂ Reduction (%)"], df["Middle Regions (%)"], 
            marker="s", label="Middle 60% Regions", color="tab:green", linewidth=2, markersize=8)
    ax2.plot(df["CO₂ Reduction (%)"], df["Bottom Regions (%)"], 
            marker="^", label="Bottom 20% Regions", color="tab:blue", linewidth=2, markersize=8)
    
    ax2.set_ylabel("Share of Total Expansion (%)", **font)
    ax2.set_xlabel("CO₂ Reduction (%)", **font)
    ax2.set_title("Relative Share of Renewable Expansion", **font)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="center right")
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"capacity_expansion_evolution.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved plot as {output_file}")
    
    plt.close(fig)
    
    print(f"  ✓ Analyzed expansion across {len(top_regions)} top, {len(middle_regions)} middle, {len(bottom_regions)} bottom regions")


def plot_transmission_bottlenecks(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot transmission capacity utilization and bottlenecks across scenarios."""
    print("Creating transmission bottlenecks analysis plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping transmission bottlenecks plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Analyze bottlenecks for each scenario
    scenario_results = []
    regional_utilization = {}
    
    for co2_pct, net in networks_by_percent.items():
        try:
            bottlenecks = analyze_transmission_bottlenecks(net)
            
            # Count constrained connections
            constrained_lines = sum(1 for line in bottlenecks["lines"] if line["at_limit"])
            constrained_links = sum(1 for link in bottlenecks["links"] if link["at_limit"])
            total_constrained = constrained_lines + constrained_links
            total_connections = len(bottlenecks["lines"]) + len(bottlenecks["links"])
            
            # Calculate average utilization
            all_utilizations = [line["utilization"] for line in bottlenecks["lines"] if line["utilization"] > 0]
            all_utilizations.extend([link["utilization"] for link in bottlenecks["links"] if link["utilization"] > 0])
            avg_utilization = np.mean(all_utilizations) if all_utilizations else 0
            
            scenario_results.append({
                "CO₂ Reduction (%)": int(co2_pct),
                "Constrained Connections": total_constrained,
                "Total Connections": total_connections,
                "Constraint Rate (%)": (total_constrained / total_connections * 100) if total_connections > 0 else 0,
                "Average Utilization": avg_utilization
            })
            
            # Store regional data for detailed analysis
            regional_utilization[co2_pct] = bottlenecks["regional_constraints"]
            
            print(f"  ✓ Analyzed {total_constrained}/{total_connections} constrained connections for {co2_pct}% reduction")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")

    if not scenario_results:
        print("No valid results - cannot create plot")
        return
        
    df = pd.DataFrame(scenario_results).sort_values("CO₂ Reduction (%)")

    # Create subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    font = {'fontsize': 11, 'fontweight': 'bold'}
    
    # Plot 1: Number of constrained connections
    ax1.plot(df["CO₂ Reduction (%)"], df["Constrained Connections"], 
            marker="o", color="tab:red", linewidth=2, markersize=8)
    ax1.set_ylabel("Number of Constrained Connections", **font)
    ax1.set_xlabel("CO₂ Reduction (%)", **font)
    ax1.set_title("Transmission Connections at Capacity Limit", **font)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Constraint rate percentage
    ax2.plot(df["CO₂ Reduction (%)"], df["Constraint Rate (%)"], 
            marker="s", color="tab:orange", linewidth=2, markersize=8)
    ax2.set_ylabel("Constraint Rate (%)", **font)
    ax2.set_xlabel("CO₂ Reduction (%)", **font)
    ax2.set_title("Percentage of Connections Constrained", **font)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Plot 3: Average utilization
    ax3.plot(df["CO₂ Reduction (%)"], df["Average Utilization"], 
            marker="^", color="tab:green", linewidth=2, markersize=8)
    ax3.set_ylabel("Average Utilization Ratio", **font)
    ax3.set_xlabel("CO₂ Reduction (%)", **font)
    ax3.set_title("Average Transmission Capacity Utilization", **font)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Regional utilization for a specific scenario (50% reduction)
    target_scenario = 50
    if target_scenario in regional_utilization:
        regional_data = regional_utilization[target_scenario]
        countries = list(regional_data.keys())
        utilizations = [regional_data[country]["utilization_ratio"] for country in countries]
        
        # Filter out countries with zero utilization
        filtered_data = [(country, util) for country, util in zip(countries, utilizations) if util > 0]
        if filtered_data:
            filtered_countries, filtered_utils = zip(*filtered_data)
            
            bars = ax4.bar(range(len(filtered_countries)), filtered_utils, color="tab:blue", alpha=0.7)
            ax4.set_ylabel("Regional Utilization Ratio", **font)
            ax4.set_xlabel("Regions", **font)
            ax4.set_title(f"Regional Transmission Utilization ({target_scenario}% CO₂ Reduction)", **font)
            ax4.set_xticks(range(len(filtered_countries)))
            ax4.set_xticklabels(filtered_countries, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_ylim(0, 1)
            
            # Highlight highly utilized regions (>80%)
            for i, util in enumerate(filtered_utils):
                if util > 0.8:
                    bars[i].set_color('tab:red')
                elif util > 0.6:
                    bars[i].set_color('tab:orange')
    else:
        ax4.text(0.5, 0.5, f"No data for {target_scenario}% scenario", 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("Regional Transmission Utilization", **font)
    
    plt.tight_layout()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"transmission_bottlenecks.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved plot as {output_file}")
    
    plt.close(fig)
    
    # Print summary of most constrained regions
    if target_scenario in regional_utilization:
        print(f"\n📊 Most transmission-constrained regions at {target_scenario}% reduction:")
        regional_data = regional_utilization[target_scenario]
        sorted_regions = sorted(regional_data.items(), 
                              key=lambda x: x[1]["utilization_ratio"], reverse=True)
        for i, (country, data) in enumerate(sorted_regions[:5]):
            if data["utilization_ratio"] > 0:
                print(f"  {i+1}. {country}: {data['utilization_ratio']:.1%} utilization, "
                      f"{data['constrained_connections']}/{data['total_connections']} constrained")


def plot_total_renewable_capacity(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot total installed renewable capacity across CO₂ reduction scenarios."""
    print("Creating total renewable capacity plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping total renewable capacity plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Calculate total installed renewable capacity for each scenario
    total_renewable_capacities = {}
    
    for co2_pct, net in networks_by_percent.items():
        try:
            region_caps = extract_installed_renewable_capacities(net)
            total_renewable_capacities[co2_pct] = region_caps.sum()
            print(f"  ✓ Total renewable capacity for {co2_pct}% reduction: {region_caps.sum():.1f} MW")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")

    if not total_renewable_capacities:
        print("No valid results - cannot create plot")
        return
        
    # Convert the results to a DataFrame for plotting
    df_total_capacity = pd.DataFrame(
        list(total_renewable_capacities.items()),
        columns=["CO₂ Reduction (%)", "Total Renewable Capacity (MW)"]
    ).sort_values("CO₂ Reduction (%)")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}

    # Plot total renewable capacity
    ax.plot(
        df_total_capacity["CO₂ Reduction (%)"],
        df_total_capacity["Total Renewable Capacity (MW)"],
        marker="o",
        label="Total Renewable Capacity",
        color="tab:green",
        linewidth=2,
        markersize=8
    )
    
    ax.set_xlabel("CO₂ Reduction (%)", **font)
    ax.set_ylabel("Total Renewable Capacity (MW)", **font)
    ax.set_title("Total Installed Renewable Capacity vs CO₂ Reduction", **font)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"total_renewable_capacity.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved plot as {output_file}")

    plt.close(fig)


def calculate_true_lcoe_with_sunk_costs(network):
    """
    Calculate full system LCOE including capital cost recovery for ALL capacity,
    even non-extendable (existing) generators.
    
    Returns both transition LCOE (what optimizer minimizes) and true LCOE (full cost).
    """
    # Get snapshot weights
    if hasattr(network, 'snapshot_weightings'):
        weights = network.snapshot_weightings
        if hasattr(weights, 'generators'):
            weights = weights.generators
        elif isinstance(weights, pd.DataFrame) and 'generators' in weights.columns:
            weights = weights['generators']
        else:
            weights = pd.Series(1.0, index=network.snapshots)
    else:
        weights = pd.Series(1.0, index=network.snapshots)
    
    # Calculate actual generation by generator
    weights_arr = weights.values[:, np.newaxis]
    gen_energy = (network.generators_t.p.values * weights_arr).sum(axis=0)
    gen_energy = pd.Series(gen_energy, index=network.generators_t.p.columns)
    
    # Calculate variable costs (fuel + O&M)
    variable_costs = (gen_energy * network.generators['marginal_cost']).sum()
    
    # Calculate capital costs for extendable generators (what optimizer counts)
    extendable_mask = network.generators['p_nom_extendable'].fillna(False)
    capital_costs_counted = (
        network.generators.loc[extendable_mask, 'p_nom_opt'] * 
        network.generators.loc[extendable_mask, 'capital_cost']
    ).sum()
    
    # Calculate sunk capital costs (non-extendable, NOT counted by optimizer)
    non_extendable_mask = ~extendable_mask
    sunk_capital_costs = (
        network.generators.loc[non_extendable_mask, 'p_nom'] * 
        network.generators.loc[non_extendable_mask, 'capital_cost']
    ).sum()
    
    # Total demand
    total_demand = (network.loads_t.p.sum(axis=1) * weights).sum()
    
    # Calculate both LCOEs
    transition_lcoe = (variable_costs + capital_costs_counted) / total_demand
    true_lcoe = (variable_costs + capital_costs_counted + sunk_capital_costs) / total_demand
    
    return {
        'transition_lcoe': transition_lcoe,
        'true_lcoe': true_lcoe,
        'sunk_cost_per_mwh': sunk_capital_costs / total_demand,
        'variable_cost_per_mwh': variable_costs / total_demand,
        'new_capital_per_mwh': capital_costs_counted / total_demand,
    }


def plot_electricity_cost(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot full system electricity cost (€/MWh) across CO₂ reduction scenarios.
    
    This calculates the economically correct electricity price using:
    average_cost = total_system_objective / total_annual_demand
    
    This includes ALL costs: generation capital costs, storage capital costs, 
    transmission capital costs, and variable generation costs (all annualized).
    
    This is the price that electricity would need to be sold at to recover all system costs.
    """
    print("Creating full system electricity cost plot...")
    print("  ✅ Using total system cost approach (includes capital cost recovery)")
    
    if not pypsa:
        print("PyPSA not available - skipping electricity cost plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Calculate electricity cost for each decarbonization level
    electricity_costs = {}
    
    for co2_pct, net in networks_by_percent.items():
        try:
            # Extract total system cost (includes all annualized capital costs)
            total_cost = net.objective  # Total system cost (€/year)
            
            # Calculate total annual demand with proper snapshot weighting
            if hasattr(net, 'snapshot_weightings'):
                weights = net.snapshot_weightings
                if hasattr(weights, 'generators'):
                    weights = weights.generators
                elif isinstance(weights, pd.DataFrame) and 'generators' in weights.columns:
                    weights = weights['generators']
                else:
                    weights = pd.Series(1.0, index=net.snapshots)
            else:
                weights = pd.Series(1.0, index=net.snapshots)
                
            # Total annual demand (MWh/year) - properly weighted
            total_load = (net.loads_t.p.sum(axis=1) * weights).sum()
            
            # Average system cost per MWh (includes capital cost recovery)
            electricity_cost = total_cost / total_load if total_load > 0 else np.nan
            electricity_costs[co2_pct] = electricity_cost
            print(f"  ✓ Full system electricity cost for {co2_pct}% reduction: {electricity_cost:.2f} €/MWh")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")

    if not electricity_costs:
        print("No valid results - cannot create plot")
        return
        
    # Convert the results to a DataFrame for plotting
    df_electricity_costs = pd.DataFrame(
        list(electricity_costs.items()),
        columns=["CO₂ Reduction (%)", "Electricity Cost (€/MWh)"]
    ).sort_values("CO₂ Reduction (%)")

    # Remove any NaN values
    df_electricity_costs = df_electricity_costs.dropna()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}

    # Plot electricity cost
    ax.plot(
        df_electricity_costs["CO₂ Reduction (%)"],
        df_electricity_costs["Electricity Cost (€/MWh)"],
        marker="o",
        label="Electricity Cost",
        color="tab:red",
        linewidth=2,
        markersize=8
    )
    
    ax.set_xlabel("CO₂ Reduction (%)", **font)
    ax.set_ylabel("Electricity Cost (€/MWh)", **font)
    ax.set_title("Electricity Cost vs CO₂ Reduction", **font)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"electricity_cost.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved plot as {output_file}")

    plt.close(fig)


def plot_true_lcoe_with_sunk_costs(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot comparison of transition LCOE vs true LCOE (including sunk costs) across decarbonization levels.
    
    Shows:
    - Transition LCOE: What optimizer minimizes (excludes sunk costs of existing infrastructure)
    - True LCOE: Full cost including capital recovery of pre-existing generators
    - Difference: Sunk cost component that remains constant across scenarios
    """
    print("Creating True LCOE with Sunk Cost Recovery plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping true LCOE plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Calculate both LCOEs for each decarbonization level
    results_by_level = {}
    
    for co2_pct, net in networks_by_percent.items():
        try:
            lcoe_results = calculate_true_lcoe_with_sunk_costs(net)
            results_by_level[co2_pct] = lcoe_results
            print(f"  ✓ {co2_pct}% reduction: Transition LCOE = {lcoe_results['transition_lcoe']:.2f} €/MWh, "
                  f"True LCOE = {lcoe_results['true_lcoe']:.2f} €/MWh")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")
    
    if not results_by_level:
        print("No valid results - cannot create plot")
        return
    
    # Prepare data for plotting
    levels = sorted(results_by_level.keys())
    transition_lcoe = [results_by_level[lvl]['transition_lcoe'] for lvl in levels]
    true_lcoe = [results_by_level[lvl]['true_lcoe'] for lvl in levels]
    sunk_cost = [results_by_level[lvl]['sunk_cost_per_mwh'] for lvl in levels]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}
    
    # Plot both LCOE series
    ax.plot(levels, transition_lcoe, 's-', label='Transition LCOE (Optimizer)', 
            color='tab:blue', linewidth=2.5, markersize=10, alpha=0.8)
    ax.plot(levels, true_lcoe, 'o-', label='True LCOE (Incl. Sunk Costs)', 
            color='tab:red', linewidth=2.5, markersize=10, alpha=0.8)
    
    # Fill area showing sunk cost component
    ax.fill_between(levels, transition_lcoe, true_lcoe, alpha=0.3, 
                     color='orange', label=f'Sunk Cost Component (~{sunk_cost[0]:.1f} €/MWh)')
    
    # Add horizontal line showing constant sunk cost if it's relatively stable
    if max(sunk_cost) - min(sunk_cost) < 2.0:  # If variation < 2 €/MWh
        avg_sunk = np.mean(sunk_cost)
        ax.axhline(y=transition_lcoe[0] + avg_sunk, color='orange', 
                   linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add annotations
    if len(levels) > 0:
        # Annotate baseline
        baseline_idx = 0
        ax.annotate(f'Baseline:\nTransition: {transition_lcoe[baseline_idx]:.1f} €/MWh\nTrue: {true_lcoe[baseline_idx]:.1f} €/MWh',
                    xy=(levels[baseline_idx], true_lcoe[baseline_idx]),
                    xytext=(levels[baseline_idx] + 5, true_lcoe[baseline_idx] + 5),
                    fontsize=9, ha='left',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        # Annotate highest decarbonization
        if len(levels) > 1:
            final_idx = -1
            increase_transition = transition_lcoe[final_idx] - transition_lcoe[0]
            increase_true = true_lcoe[final_idx] - true_lcoe[0]
            ax.annotate(f'{levels[final_idx]}% Reduction:\n'
                        f'Transition: {transition_lcoe[final_idx]:.1f} €/MWh (+{increase_transition:.1f})\n'
                        f'True: {true_lcoe[final_idx]:.1f} €/MWh (+{increase_true:.1f})',
                        xy=(levels[final_idx], true_lcoe[final_idx]),
                        xytext=(levels[final_idx] - 15, true_lcoe[final_idx] + 5),
                        fontsize=9, ha='right',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.set_xlabel("CO₂ Reduction (%)", **font)
    ax.set_ylabel("Levelized Cost of Electricity (€/MWh)", **font)
    ax.set_title("Transition LCOE vs True LCOE (Including Sunk Capital Recovery)", **font)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"true_lcoe_with_sunk_costs.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved true LCOE plot as {output_file}")
    
    plt.close(fig)
    
    # Print summary table
    print("\n=== TRUE LCOE SUMMARY ===")
    print(f"{'Level (%)':<12} {'Transition LCOE':<18} {'True LCOE':<18} {'Sunk Cost':<18}")
    print(f"{'':12} {'(€/MWh)':<18} {'(€/MWh)':<18} {'(€/MWh)':<18}")
    print("-" * 70)
    for level in levels:
        res = results_by_level[level]
        print(f"{level:<12.0f} {res['transition_lcoe']:<18.2f} {res['true_lcoe']:<18.2f} {res['sunk_cost_per_mwh']:<18.2f}")
    
    print("\n💡 Key Insight:")
    print(f"   The sunk cost component (~{np.mean(sunk_cost):.1f} €/MWh) represents capital recovery")
    print(f"   for existing fossil infrastructure, which is constant across all scenarios.")
    print(f"   This accounts for {100*np.mean(sunk_cost)/np.mean(true_lcoe):.0f}% of the true LCOE.")


def plot_marginal_price_vs_lcoe_explanation(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot comparing demand-weighted marginal prices with true LCOE, explaining the merit order effect.
    
    Shows why marginal prices (set by expensive generators) can be higher than average system costs
    (dominated by cheap baseload), and explains the infra-marginal rent concept.
    """
    print("Creating Marginal Price vs LCOE Explanation plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping marginal price vs LCOE plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Calculate demand-weighted marginal prices
    demand_weighted_prices = calculate_demand_weighted_prices_by_level(networks_by_percent)
    
    # Calculate true LCOE for each level
    true_lcoe_by_level = {}
    for co2_pct, net in networks_by_percent.items():
        try:
            lcoe_results = calculate_true_lcoe_with_sunk_costs(net)
            true_lcoe_by_level[co2_pct] = lcoe_results['true_lcoe']
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")
    
    if not demand_weighted_prices or not true_lcoe_by_level:
        print("Insufficient data - cannot create plot")
        return
    
    # Prepare data
    levels = sorted(set(demand_weighted_prices.keys()) & set(true_lcoe_by_level.keys()))
    marginal_prices = [demand_weighted_prices[lvl] for lvl in levels]
    true_lcoe = [true_lcoe_by_level[lvl] for lvl in levels]
    infra_marginal_rent = [marginal_prices[i] - true_lcoe[i] for i in range(len(levels))]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    font = {'fontsize': 12, 'fontweight': 'bold'}
    
    # ========================================================================
    # LEFT PLOT: Price Comparison
    # ========================================================================
    ax1.plot(levels, marginal_prices, 'o-', label='Marginal Price (Demand-weighted)', 
             color='tab:purple', linewidth=2.5, markersize=10, alpha=0.8)
    ax1.plot(levels, true_lcoe, 's-', label='True LCOE (Full Cost)', 
             color='tab:green', linewidth=2.5, markersize=10, alpha=0.8)
    
    # Fill area showing infra-marginal rent
    ax1.fill_between(levels, true_lcoe, marginal_prices, 
                     where=[mp >= lc for mp, lc in zip(marginal_prices, true_lcoe)],
                     alpha=0.3, color='gold', 
                     label='Infra-marginal Rent\n(Profit to baseload)')
    
    # Add annotation explaining the gap
    if len(levels) > 0:
        mid_idx = len(levels) // 2
        gap = marginal_prices[mid_idx] - true_lcoe[mid_idx]
        if gap > 0:
            ax1.annotate(f'Infra-marginal Rent:\n{gap:.1f} €/MWh\n\nProfit to cheap\nbaseload plants',
                        xy=(levels[mid_idx], (marginal_prices[mid_idx] + true_lcoe[mid_idx]) / 2),
                        xytext=(levels[mid_idx] + 10, (marginal_prices[mid_idx] + true_lcoe[mid_idx]) / 2),
                        fontsize=9, ha='left', va='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="gold", alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax1.set_xlabel("CO₂ Reduction (%)", **font)
    ax1.set_ylabel("Electricity Price (€/MWh)", **font)
    ax1.set_title("Marginal Price vs True LCOE", **font)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')
    
    # ========================================================================
    # RIGHT PLOT: Merit Order Explanation
    # ========================================================================
    # Create a stylized merit order curve for baseline
    ax2.text(0.5, 0.95, 'Merit Order Effect Explanation', 
             ha='center', va='top', transform=ax2.transAxes, 
             fontsize=14, fontweight='bold')
    
    # Text explanation
    explanation_text = """
Why Marginal Price > True LCOE?

THE MERIT ORDER EFFECT:

Power plants are dispatched by fuel cost (cheapest first):
  1. Coal/Lignite:  13-24 €/MWh  [BASELOAD - runs most]
  2. Nuclear:       17 €/MWh     [BASELOAD - runs most]
  3. CCGT (Gas):    47 €/MWh     [MID-MERIT]
  4. OCGT (Gas):    58 €/MWh     [PEAKER - sets price]

• Marginal Price = Cost of LAST plant needed (peaker)
  → Set by expensive gas plants (~47-58 €/MWh)
  → This is what markets pay for electricity

• True LCOE = AVERAGE cost of ALL plants
  → Dominated by cheap baseload (13-24 €/MWh)
  → This is what the system actually costs

• Infra-marginal Rent = Marginal Price - LCOE
  → PROFIT earned by cheap baseload plants
  → They produce at 13-24 €/MWh but sell at 47-58 €/MWh
  → This profit helps justify baseload capital investment

WHY THIS MATTERS FOR RENEWABLES:
✓ Renewables have ZERO marginal cost (no fuel)
✓ When they run, they collect the FULL marginal price
✓ This infra-marginal rent helps pay back their
  high capital costs → finances decarbonization!

BOTTOM LINE:
Marginal prices being higher than LCOE is NORMAL
and ECONOMICALLY EFFICIENT in electricity markets.
"""
    
    ax2.text(0.05, 0.88, explanation_text, 
             transform=ax2.transAxes, fontsize=9, 
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"marginal_price_vs_lcoe_explanation.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved explanation plot as {output_file}")
    
    plt.close(fig)
    
    # Print summary
    print("\n=== MARGINAL PRICE VS LCOE SUMMARY ===")
    print(f"{'Level (%)':<12} {'Marginal Price':<18} {'True LCOE':<18} {'Rent/Profit':<18}")
    print(f"{'':12} {'(€/MWh)':<18} {'(€/MWh)':<18} {'(€/MWh)':<18}")
    print("-" * 70)
    for i, level in enumerate(levels):
        print(f"{level:<12.0f} {marginal_prices[i]:<18.2f} {true_lcoe[i]:<18.2f} {infra_marginal_rent[i]:<18.2f}")
    
    avg_rent = np.mean([r for r in infra_marginal_rent if r > 0])
    print(f"\n💡 Average Infra-marginal Rent: {avg_rent:.2f} €/MWh")
    print(f"   This represents profit earned by baseload plants (coal, nuclear, hydro)")
    print(f"   because they produce cheaply but sell at the marginal (gas) price.")


def plot_generation_mix_actual(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot actual electricity generation mix (MWh) as stacked bar chart across CO₂ reduction scenarios."""
    print("Creating actual generation mix plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping generation mix plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Prepare generation mix data as actual electricity generated (MWh)
    generation_mix = []
    
    for co2_pct, net in networks_by_percent.items():
        try:
            # Weighted generation (MWh) across all snapshots
            weighted_gen = net.generators_t.p.multiply(net.snapshot_weightings["objective"], axis=0)

            # Map generator carriers correctly to columns
            carriers = net.generators["carrier"]
            weighted_gen.columns = carriers

            # Sum total generation per carrier over time
            gen_mix = weighted_gen.sum().groupby(level=0).sum()

            # Add CO₂ reduction level
            gen_mix["CO₂ Reduction (%)"] = co2_pct
            generation_mix.append(gen_mix)
            
            print(f"  ✓ Processed generation mix for {co2_pct}% reduction")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")

    if not generation_mix:
        print("No valid results - cannot create plot")
        return
        
    # Convert to DataFrame
    df_generation_mix = pd.DataFrame(generation_mix).fillna(0).set_index("CO₂ Reduction (%)")
    df_generation_mix.sort_index(inplace=True)
    
    # Remove columns with all zeros
    df_generation_mix = df_generation_mix.loc[:, (df_generation_mix != 0).any(axis=0)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}

    # Plot actual generation (MWh) as stacked bar chart
    df_generation_mix.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        colormap="tab20",
        width=0.8
    )
    
    ax.set_xlabel("CO₂ Reduction (%)", **font)
    ax.set_ylabel("Total Generation (MWh)", **font)
    ax.set_title("Electricity Generation Mix (MWh) by CO₂ Reduction Level", **font)
    
    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Move legend outside the plot
    ax.legend(title="Carrier", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"generation_mix_actual.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved plot as {output_file}")

    plt.close(fig)


def calculate_renewable_penetration_by_region(networks_by_percent):
    """Calculate renewable penetration by region for each CO2 reduction level."""
    renewable_penetration_by_region = {}
    
    # Loop through each network and CO2 reduction level
    for co2_pct, net in networks_by_percent.items():
        try:
            weights = net.snapshot_weightings["objective"]

            # Identify renewable generators (including nuclear, biomass, geothermal)
            renewable_carriers = ["solar", "onwind", "offwind-ac", "offwind-dc", "nuclear", "biomass", "geothermal", "ror", "hydro"]
            renewable_gens = net.generators.index[net.generators.carrier.isin(renewable_carriers)]

            # Group buses by region
            region_map = net.buses["country"]
            generation = net.generators_t.p.multiply(weights, axis=0)
            renewable_gen = generation[renewable_gens]

            # Map each generator to its region and sum renewable generation
            gen_regions = net.generators.loc[renewable_gens, "bus"].map(region_map)
            renewable_by_region = renewable_gen.groupby(gen_regions, axis=1).sum().sum()

            # Get load by region
            load = net.loads_t.p_set.multiply(weights, axis=0)
            load_regions = net.loads.bus.map(region_map)
            load_by_region = load.groupby(load_regions, axis=1).sum().sum()

            # Compute penetration (as a fraction of load)
            penetration = (renewable_by_region / load_by_region).fillna(0)
            renewable_penetration_by_region[co2_pct] = penetration
            
            print(f"  ✓ Calculated renewable penetration for {co2_pct}% reduction")

        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")
    
    # Create the DataFrame: Rows = decarbonization levels, Columns = regions
    df_penetration = pd.DataFrame(renewable_penetration_by_region).T.sort_index()
    
    return df_penetration


def plot_renewable_penetration_boxplots(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot boxplots of renewable penetration by region for each CO₂ reduction level."""
    print("Creating renewable penetration boxplots...")
    
    if not pypsa:
        print("PyPSA not available - skipping renewable penetration boxplots")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Calculate renewable penetration by region
    df_penetration = calculate_renewable_penetration_by_region(networks_by_percent)
    
    if df_penetration.empty:
        print("No penetration data - cannot create plot")
        return
    
    # Get all available CO2 reduction levels
    levels_to_plot = sorted(df_penetration.index.tolist())
    print(f"  ✓ Creating boxplots for levels: {levels_to_plot}")
    
    # Prepare data for boxplot
    data = []
    labels = []
    means = []
    medians = []
    region_lists = []
    
    for level in levels_to_plot:
        if level not in df_penetration.index:
            print(f"Level {level} not found in data, skipping.")
            continue
        vals = df_penetration.loc[level].dropna()
        data.append(vals.values)
        labels.append(f"{level}%")
        means.append(vals.mean())
        medians.append(np.median(vals.values))
        region_lists.append(vals.index.tolist())

    # Create the plot
    fig, ax = plt.subplots(figsize=(2 + 2*len(data), 6))
    
    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}
    
    # Create boxplot
    box = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=False,
        meanline=False,
        boxprops=dict(facecolor='lightblue', color='k'),
        medianprops=dict(color='blue', linewidth=2),
        whiskerprops=dict(color='k'),
        capprops=dict(color='k'),
        flierprops=dict(marker='o', color='purple', alpha=0.8)
    )

    # Overlay the mean as a red line and the median as a blue line
    for i, (mean, median) in enumerate(zip(means, medians)):
        ax.plot([i+1-0.2, i+1+0.2], [mean, mean], color='red', linewidth=2, 
                label='Mean' if i == 0 else "")
        ax.plot([i+1-0.2, i+1+0.2], [median, median], color='blue', linewidth=2, 
                label='Median' if i == 0 else "")

    # Annotate outliers with region names
    for i, flier in enumerate(box['fliers']):
        y_outliers = flier.get_ydata()
        x_outliers = flier.get_xdata()
        regions = region_lists[i]
        values = data[i]
        
        if len(values) > 0:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_indices = [j for j, v in enumerate(values) if v < lower_bound or v > upper_bound]
            
            for x, y, idx in zip(x_outliers, y_outliers, outlier_indices):
                if idx < len(regions):
                    ax.annotate(regions[idx], (x, y), textcoords="offset points", 
                               xytext=(5,5), ha='left', fontsize=10, color='purple', 
                               fontweight='bold')

    ax.set_ylabel("Renewable Penetration (Fraction of Load)", **font)
    ax.set_xlabel("CO₂ Reduction Level", **font)
    ax.set_title("Renewable Penetration Distribution by Region", **font)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.5)

    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"renewable_penetration_boxplots.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved plot as {output_file}")

    plt.close(fig)


def plot_renewable_penetration_stacked_bars(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot stacked bar chart showing cumulative renewable penetration by region across CO₂ reduction levels."""
    print("Creating renewable penetration stacked bars plot...")
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # This will store results for each CO2 reduction level
    renewable_penetration_by_region = {}

    # Loop through each network and CO2 reduction level
    for co2_pct, net in networks_by_percent.items():
        try:
            # Get snapshot weights
            weights = getattr(net, 'snapshot_weightings', None)
            if weights is None:
                weights = pd.Series(1.0, index=net.snapshots)
            elif hasattr(weights, 'objective'):
                weights = weights["objective"]
            elif hasattr(weights, 'generators'):
                weights = weights["generators"]
            else:
                weights = weights

            # Identify renewable generators
            renewable_carriers = ["solar", "onwind", "offwind-ac", "offwind-dc", "nuclear", "biomass", "geothermal", "ror", "hydro"]
            renewable_gens = net.generators.index[net.generators.carrier.isin(renewable_carriers)]

            # Group buses by region (country)
            if "country" in net.buses.columns:
                region_map = net.buses["country"]
            else:
                # Extract country from bus names if no country column
                region_map = net.buses.index.to_series().apply(lambda x: x.split("_")[-1])
                net.buses["country"] = region_map

            # Calculate renewable generation with weights
            generation = net.generators_t.p.multiply(weights, axis=0)
            renewable_gen = generation[renewable_gens]

            # Map each generator to its region and sum renewable generation
            gen_regions = net.generators.loc[renewable_gens, "bus"].map(region_map)
            renewable_by_region = renewable_gen.groupby(gen_regions, axis=1).sum().sum()

            # Get load by region
            load = net.loads_t.p_set.multiply(weights, axis=0)
            load_regions = net.loads.bus.map(region_map)
            load_by_region = load.groupby(load_regions, axis=1).sum().sum()

            # Compute penetration (as a fraction of load)
            penetration = (renewable_by_region / load_by_region).fillna(0)
            renewable_penetration_by_region[co2_pct] = penetration

            print(f"  ✓ Calculated penetration for {co2_pct}% reduction ({len(penetration)} regions)")

        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")

    if not renewable_penetration_by_region:
        print("No valid results - cannot create plot")
        return

    # Create the cumulative DataFrame.
    # Rows: decarbonization levels, Columns: regions
    df_penetration = pd.DataFrame(renewable_penetration_by_region).T.sort_index()
    #df_penetration = df_penetration.clip(upper=1)  # Limit values to 1 (100%) if needed

    # Get sorted decarbonization levels and regions.
    levels = df_penetration.index
    regions = df_penetration.columns
    n_regions = len(regions)
    num_levels = len(levels)

    print(f"  ✓ Creating stacked bars for {num_levels} CO₂ levels and {n_regions} regions")

    # Prepare figure with one subplot per decarbonization level.
    fig, axs = plt.subplots(nrows=num_levels, ncols=1, figsize=(16, 4 * num_levels))
    if num_levels == 1:
        axs = [axs]  # Ensure axs is iterable

    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}
    
    # Loop through each decarbonization level to build the stacked bars.
    for i, co2_pct in enumerate(levels):
        ax = axs[i]
        # Retrieve the cumulative penetration for the current level.
        current = df_penetration.loc[co2_pct]
        # For the first level, there is no previous cumulative (use zeros).
        if i == 0:
            prev = pd.Series(0, index=regions)
        else:
            prev = df_penetration.loc[levels[i - 1]]
        # Compute the incremental change.
        increment = current - prev

        # Set x positions for the regions.
        x = np.arange(n_regions)

        # Plot the previous cumulative as the base (in light gray).
        ax.bar(x, prev, color="lightgray", label="Previous cumulative")
        
        # Determine colors for the incremental part: green for positive, red for negative.
        increment_colors = ["green" if val >= 0 else "red" for val in increment]
        
        # Plot the incremental change on top (or below in case of negative values).
        ax.bar(x, increment, bottom=prev, color=increment_colors, label=f"Increment ({co2_pct}%)")
        
        # Set x-axis ticks and labels.
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45)
        ax.set_title(f"Stacked Renewable Penetration at {co2_pct}% CO₂ Reduction", **font)
        ax.set_ylabel("Penetration (Fraction of Load)", **font)
        ax.grid(True, alpha=0.3)
        
        # Optionally, add a legend to the first subplot to reduce clutter.
        if i == 0:
            ax.legend()

    plt.xlabel("Region", **font)
    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"renewable_penetration_stacked_bars.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved plot as {output_file}")

    plt.close(fig)


def compute_mean_interregional_expansion_by_region(network):
    """Compute mean interregional transmission expansion by region."""
    lines = network.lines.copy()
    links = network.links.copy()
    buses = network.buses

    # For lines
    lines["region0"] = buses.loc[lines["bus0"], "country"].values
    lines["region1"] = buses.loc[lines["bus1"], "country"].values
    lines["expansion"] = (lines["s_nom_opt"] - lines["s_nom"]).clip(lower=0)
    inter_lines = lines[lines["region0"] != lines["region1"]]

    # For links
    links["region0"] = buses.loc[links["bus0"], "country"].values
    links["region1"] = buses.loc[links["bus1"], "country"].values
    links["expansion"] = (links["p_nom_opt"] - links["p_nom"]).clip(lower=0)
    inter_links = links[links["region0"] != links["region1"]]

    # Assign half expansion to each region
    expansion_by_region = {}

    for _, row in inter_lines.iterrows():
        for region in [row["region0"], row["region1"]]:
            expansion_by_region.setdefault(region, []).append(row["expansion"] / 2)
    
    for _, row in inter_links.iterrows():
        for region in [row["region0"], row["region1"]]:
            expansion_by_region.setdefault(region, []).append(row["expansion"] / 2)

    # Compute mean for each region
    mean_expansion = {region: np.mean(vals) if vals else 0 for region, vals in expansion_by_region.items()}
    return mean_expansion


def plot_interregional_transmission_expansion(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot boxplots of mean interregional transmission expansion by region for each CO₂ reduction level."""
    print("Creating interregional transmission expansion boxplots...")
    
    if not pypsa:
        print("PyPSA not available - skipping interregional transmission expansion boxplots")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Compute mean interregional expansion for all levels
    mean_expansion_by_level = {}
    
    for co2_pct, net in networks_by_percent.items():
        try:
            mean_expansion_by_level[co2_pct] = compute_mean_interregional_expansion_by_region(net)
            print(f"  ✓ Calculated transmission expansion for {co2_pct}% reduction")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")
    
    if not mean_expansion_by_level:
        print("No expansion data - cannot create plot")
        return
    
    # Create DataFrame
    df_mean_expansion = pd.DataFrame(mean_expansion_by_level).T.sort_index()
    
    # Get all available levels for plotting
    levels_to_plot = sorted(df_mean_expansion.index.tolist())
    print(f"  ✓ Creating boxplots for levels: {levels_to_plot}")
    
    # Prepare boxplot data
    boxplot_data = []
    labels = []
    region_lists = []
    
    for level in levels_to_plot:
        if level not in df_mean_expansion.index:
            print(f"⚠️ Level {level}% not found in data, skipping.")
            continue
        vals = df_mean_expansion.loc[level].dropna()
        boxplot_data.append(vals.values)
        labels.append(f"{level}%")
        region_lists.append(list(vals.index))

    if not boxplot_data:
        print("No valid data for boxplots")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(2 + 2*len(labels), 6))
    
    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}
    
    # Create boxplot
    box = ax.boxplot(
        boxplot_data,
        labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', color='k'),
        medianprops=dict(color='blue', linewidth=2),
        whiskerprops=dict(color='k'),
        capprops=dict(color='k'),
        flierprops=dict(marker='o', color='purple', alpha=0.8)
    )

    # Overlay the mean as a red line and the median as a blue line for each box
    for i, data in enumerate(boxplot_data):
        if len(data) > 0:
            mean_val = np.mean(data)
            median_val = np.median(data)
            ax.plot([i+1-0.2, i+1+0.2], [mean_val, mean_val], color='red', linewidth=2, 
                    label='Mean' if i == 0 else "")
            ax.plot([i+1-0.2, i+1+0.2], [median_val, median_val], color='blue', linewidth=2, 
                    label='Median' if i == 0 else "")

    # Annotate outliers with region names in purple
    for i, flier in enumerate(box['fliers']):
        y_outliers = flier.get_ydata()
        x_outliers = flier.get_xdata()
        regions = region_lists[i]
        values = boxplot_data[i]
        
        # Compute IQR for this box
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outlier indices
        outlier_indices = [j for j, v in enumerate(values) if v < lower_bound or v > upper_bound]
        
        for x, y, idx in zip(x_outliers, y_outliers, outlier_indices):
            if idx < len(regions):  # Safety check
                ax.annotate(regions[idx], (x, y), textcoords="offset points", 
                           xytext=(5, 5), ha='left', fontsize=10, color='purple', 
                           fontweight='bold')

    ax.set_ylabel("Mean Interregional Transmission Expansion (MW)", **font)
    ax.set_xlabel("CO₂ Reduction Level", **font)
    ax.set_title("Interregional Transmission Expansion by Region", **font)
    ax.legend(loc="upper right")
    ax.grid(axis='y', alpha=0.5)

    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"interregional_transmission_expansion.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved plot as {output_file}")

    plt.close(fig)


def compute_storage_expansion_by_region(network):
    """Compute storage expansion by region."""
    storage_units = network.storage_units.copy()
    buses = network.buses
    
    # Map storage units to regions
    storage_units["region"] = buses.loc[storage_units["bus"], "country"].values
    
    # Calculate expansion (p_nom_opt - p_nom), clipped to positive values
    storage_units["expansion"] = (storage_units["p_nom_opt"] - storage_units["p_nom"]).clip(lower=0)
    
    # Group by region and sum expansion
    expansion_by_region = storage_units.groupby("region")["expansion"].sum()
    
    return expansion_by_region.to_dict()


def plot_storage_expansion_boxplots(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot boxplots of storage expansion by region for each CO₂ reduction level."""
    print("Creating storage expansion boxplots...")
    
    if not pypsa:
        print("PyPSA not available - skipping storage expansion boxplots")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Compute storage expansion for all levels
    storage_expansion_by_level = {}
    
    for co2_pct, net in networks_by_percent.items():
        try:
            storage_expansion_by_level[co2_pct] = compute_storage_expansion_by_region(net)
            print(f"  ✓ Calculated storage expansion for {co2_pct}% reduction")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")
    
    if not storage_expansion_by_level:
        print("No storage expansion data - cannot create plot")
        return
    
    # Create DataFrame
    df_storage_expansion = pd.DataFrame(storage_expansion_by_level).T.sort_index()
    df_storage_expansion = df_storage_expansion.fillna(0)  # Fill NaN with 0 for regions without storage
    
    # Get all available levels for plotting
    levels_to_plot = sorted(df_storage_expansion.index.tolist())
    print(f"  ✓ Creating boxplots for levels: {levels_to_plot}")
    
    # Prepare boxplot data
    boxplot_data = []
    labels = []
    region_lists = []
    
    for level in levels_to_plot:
        if level not in df_storage_expansion.index:
            print(f"⚠️ Level {level}% not found in data, skipping.")
            continue
        vals = df_storage_expansion.loc[level]
        # Include all regions, even those with zero expansion
        boxplot_data.append(vals.values)
        labels.append(f"{level}%")
        region_lists.append(list(vals.index))

    if not boxplot_data:
        print("No valid data for boxplots")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(2 + 2*len(labels), 6))
    
    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}
    
    # Create boxplot
    box = ax.boxplot(
        boxplot_data,
        labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor='lightgreen', color='k'),
        medianprops=dict(color='darkgreen', linewidth=2),
        whiskerprops=dict(color='k'),
        capprops=dict(color='k'),
        flierprops=dict(marker='o', color='orange', alpha=0.8)
    )

    # Overlay the mean as a red line and the median as a blue line for each box
    for i, data in enumerate(boxplot_data):
        if len(data) > 0:
            mean_val = np.mean(data)
            median_val = np.median(data)
            ax.plot([i+1-0.2, i+1+0.2], [mean_val, mean_val], color='red', linewidth=2, 
                    label='Mean' if i == 0 else "")
            ax.plot([i+1-0.2, i+1+0.2], [median_val, median_val], color='blue', linewidth=2, 
                    label='Median' if i == 0 else "")

    # Annotate outliers with region names in orange
    for i, flier in enumerate(box['fliers']):
        y_outliers = flier.get_ydata()
        x_outliers = flier.get_xdata()
        regions = region_lists[i]
        values = boxplot_data[i]
        
        if len(values) > 0:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_indices = [j for j, v in enumerate(values) if v < lower_bound or v > upper_bound]
            
            for x, y, idx in zip(x_outliers, y_outliers, outlier_indices):
                if idx < len(regions):
                    ax.annotate(str(regions[idx]), (x, y), textcoords="offset points", 
                               xytext=(5,5), ha='left', fontsize=10, color='orange', 
                               fontweight='bold')

    ax.set_ylabel("Storage Expansion (MW)", **font)
    ax.set_xlabel("CO₂ Reduction Level", **font)
    ax.set_title("Storage Expansion by Region", **font)
    ax.legend(loc="upper left")
    ax.grid(axis='y', alpha=0.5)

    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"storage_expansion_boxplots.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.close(fig)


def plot_total_system_cost(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot total system cost across CO₂ reduction scenarios."""
    print("Creating total system cost plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping total system cost plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Extract system cost for each CO₂ reduction level
    system_costs = {}
    
    for co2_pct, net in networks_by_percent.items():
        try:
            system_costs[co2_pct] = net.objective  # Total system cost
            print(f"  ✓ System cost for {co2_pct}% reduction: {net.objective:.0f} €")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")

    if not system_costs:
        print("No valid results - cannot create plot")
        return
        
    # Convert the results to a DataFrame for plotting
    df_system_costs = pd.DataFrame(
        list(system_costs.items()),
        columns=["CO₂ Reduction (%)", "System Cost (€)"]
    ).sort_values("CO₂ Reduction (%)")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}

    # Plot system cost
    ax.plot(
        df_system_costs["CO₂ Reduction (%)"],
        df_system_costs["System Cost (€)"],
        marker="o",
        label="System Cost",
        color="tab:blue",
        linewidth=2,
        markersize=8
    )
    
    ax.set_xlabel("CO₂ Reduction (%)", **font)
    ax.set_ylabel("System Cost (€)", **font)
    ax.set_title("Total System Cost vs CO₂ Reduction", **font)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Format y-axis to show values in scientific notation or with appropriate scaling
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"total_system_cost.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.close(fig)


def collect_marginal_prices_by_level(networks_by_percent):
    """Collect marginal prices from all networks and organize by CO2 reduction level.
    
    NOTE: These are short-run marginal costs and do NOT include capital cost recovery.
    For full electricity pricing including capital costs, use collect_average_system_costs_by_level().
    """
    dfs = []
    for co2_pct, net in networks_by_percent.items():
        try:
            prices = net.buses_t.marginal_price.copy()
            # Add a column for CO₂ level
            prices["CO2_Level"] = co2_pct
            # Set MultiIndex: (CO2_Level, Snapshot)
            prices = prices.set_index("CO2_Level", append=True)
            prices = prices.reorder_levels(["CO2_Level", prices.index.names[0]])
            dfs.append(prices)
            print(f"  ✓ Collected marginal prices for {co2_pct}% reduction")
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")
    
    # Concatenate all
    if not dfs:
        print("No marginal price data collected.")
        return None
    df_all = pd.concat(dfs).sort_index()
    return df_all


def collect_average_system_costs_by_level(networks_by_percent):
    """Calculate average system cost (including capital cost recovery) for each CO2 reduction level.
    
    This provides realistic electricity pricing that includes:
    - Generator capital costs (annualized)
    - Storage capital costs (annualized) 
    - Transmission capital costs (annualized)
    - Variable generation costs (fuel + O&M)
    
    This is the economically correct electricity price that covers all system costs.
    """
    system_costs = {}
    
    for co2_pct, net in networks_by_percent.items():
        try:
            # Calculate total annual system cost from objective function
            total_system_cost = net.objective  # €/year (includes all annualized costs)
            
            # Calculate total annual demand
            if hasattr(net, 'snapshot_weightings'):
                weights = net.snapshot_weightings
                if hasattr(weights, 'generators'):
                    weights = weights.generators
                elif isinstance(weights, pd.DataFrame) and 'generators' in weights.columns:
                    weights = weights['generators']
                else:
                    weights = pd.Series(1.0, index=net.snapshots)
            else:
                weights = pd.Series(1.0, index=net.snapshots)
                
            # Total annual demand (MWh/year)
            total_demand = (net.loads_t.p.sum(axis=1) * weights).sum()
            
            # Average system cost per MWh
            avg_system_cost = total_system_cost / total_demand if total_demand > 0 else np.nan
            
            system_costs[co2_pct] = {
                'avg_cost_per_mwh': avg_system_cost,
                'total_system_cost': total_system_cost,
                'total_demand': total_demand
            }
            
            print(f"  ✓ Calculated average system cost for {co2_pct}% reduction: {avg_system_cost:.2f} €/MWh")
            
        except Exception as e:
            print(f"⚠️ Skipping {co2_pct}% due to error: {e}")
    
    return system_costs


def calculate_mean_prices_by_level(df_marginal_prices):
    """Calculate mean marginal price per region for each decarbonization level.
    
    NOTE: These are marginal prices only - they do NOT include capital cost recovery.
    """
    # Remove the 'CO2_Level' column if it's present in the columns (should only be in the index)
    if "CO2_Level" in df_marginal_prices.columns:
        df_marginal_prices = df_marginal_prices.drop(columns=["CO2_Level"])
    
    # Group by CO₂ level (first index), then take mean across all snapshots for each region
    mean_prices_by_level = df_marginal_prices.groupby(level="CO2_Level").mean()
    
    return mean_prices_by_level


def calculate_demand_weighted_prices_by_level(networks_by_percent):
    """Calculate demand-weighted average marginal prices for each decarbonization level.
    
    This function computes the average marginal price where each bus-time observation
    is weighted by its load share of the total annual load across the entire network.
    
    Formula: mean_price = Σ_t Σ_bus (price_bus_t * load_bus_t * weight_t) / total_annual_load
    
    NOTE: These are marginal prices only - they do NOT include capital cost recovery.
    """
    demand_weighted_prices = {}
    
    for co2_pct, net in networks_by_percent.items():
        try:
            # Get marginal prices
            prices = net.buses_t.marginal_price
            
            # Get loads (demand) for each bus
            loads = net.loads_t.p
            
            # Get snapshot weights
            if hasattr(net, 'snapshot_weightings'):
                weights = net.snapshot_weightings
                if hasattr(weights, 'generators'):
                    weights = weights.generators
                elif isinstance(weights, pd.DataFrame) and 'generators' in weights.columns:
                    weights = weights['generators']
                else:
                    weights = pd.Series(1.0, index=net.snapshots)
            else:
                weights = pd.Series(1.0, index=net.snapshots)
            
            # Create DataFrame with loads aggregated by bus
            load_by_bus = pd.DataFrame(0.0, index=net.snapshots, columns=net.buses.index)
            
            for load_id in loads.columns:
                if load_id in net.loads.index:
                    bus = net.loads.loc[load_id, 'bus']
                    load_by_bus[bus] += loads[load_id]
            
            # Only keep buses that have loads and marginal prices
            common_buses = load_by_bus.columns.intersection(prices.columns)
            load_by_bus = load_by_bus[common_buses]
            prices_by_bus = prices[common_buses]
            
            # Calculate weighted loads (load * time weight) for each bus-time pair
            weighted_loads = load_by_bus.mul(weights, axis=0)
            
            # Calculate total annual load across all buses and snapshots
            total_annual_load = weighted_loads.sum().sum()
            
            if total_annual_load == 0:
                print(f"⚠️ Warning: Zero total load for {co2_pct}%")
                demand_weighted_prices[co2_pct] = 0.0
                continue
            
            # Calculate demand-weighted price:
            # Each price is weighted by (load_bus_t * weight_t) / total_annual_load
            weighted_price_sum = (prices_by_bus * weighted_loads).sum().sum()
            demand_weighted_price = weighted_price_sum / total_annual_load
            
            demand_weighted_prices[co2_pct] = demand_weighted_price
            
            # Print diagnostic information
            # Calculate per-snapshot average for diagnostics
            total_load_per_snapshot = load_by_bus.sum(axis=1)
            valid_snapshots = total_load_per_snapshot > 0
            avg_price_per_snapshot = (prices_by_bus * load_by_bus).sum(axis=1)[valid_snapshots] / total_load_per_snapshot[valid_snapshots]
            
            price_min = avg_price_per_snapshot.min()
            price_max = avg_price_per_snapshot.max()
            price_median = avg_price_per_snapshot.median()
            negative_pct = (avg_price_per_snapshot < 0).sum() / len(avg_price_per_snapshot) * 100
            
            print(f"  ✓ {co2_pct}% reduction: {demand_weighted_price:.2f} €/MWh")
            print(f"     Range: [{price_min:.2f}, {price_max:.2f}] | Median: {price_median:.2f} | Negative: {negative_pct:.1f}%")
                
        except Exception as e:
            print(f"⚠️ Error calculating demand-weighted price for {co2_pct}%: {e}")
    
    return demand_weighted_prices


def plot_electricity_cost_comparison(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot comparison between marginal prices and full system costs (including capital recovery)."""
    print("Creating electricity cost comparison plot (marginal vs. full system cost)...")
    
    if not pypsa:
        print("PyPSA not available - skipping electricity cost comparison plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Collect both marginal prices and average system costs
    print("  → Collecting marginal prices...")
    df_marginal_prices = collect_marginal_prices_by_level(networks_by_percent)
    
    print("  → Calculating average system costs...")
    system_costs = collect_average_system_costs_by_level(networks_by_percent)
    
    print("  → Calculating demand-weighted marginal prices...")
    demand_weighted_marginal_prices = calculate_demand_weighted_prices_by_level(networks_by_percent)
    
    if df_marginal_prices is None or not system_costs:
        print("Insufficient data - cannot create comparison plot")
        return
    
    # Calculate mean marginal prices by level
    mean_marginal_prices = calculate_mean_prices_by_level(df_marginal_prices)
    overall_marginal_by_level = mean_marginal_prices.mean(axis=1)  # Average across all regions
    
    # Prepare data for plotting
    levels = []
    marginal_prices = []
    demand_weighted_prices = []
    avg_system_costs = []
    
    for level in sorted(system_costs.keys()):
        if level in overall_marginal_by_level.index:
            levels.append(level)
            marginal_prices.append(overall_marginal_by_level[level])
            avg_system_costs.append(system_costs[level]['avg_cost_per_mwh'])
            if level in demand_weighted_marginal_prices:
                demand_weighted_prices.append(demand_weighted_marginal_prices[level])
            else:
                demand_weighted_prices.append(np.nan)
    
    if not levels:
        print("No matching data for comparison - cannot create plot")
        return
    
    # Create the comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}
    
    # Plot all price series
    ax.plot(levels, marginal_prices, 'o-', label='Marginal Prices (Spatial avg.)', 
            color='tab:blue', linewidth=2, markersize=8)
    ax.plot(levels, demand_weighted_prices, '^-', label='Marginal Prices (Demand-weighted)', 
            color='tab:cyan', linewidth=2, markersize=8)
    ax.plot(levels, avg_system_costs, 's-', label='Average System Cost (Full cost incl. capital)', 
            color='tab:red', linewidth=2, markersize=8)
    
    # Fill area between demand-weighted marginal and full system cost
    ax.fill_between(levels, demand_weighted_prices, avg_system_costs, alpha=0.3, 
                   color='orange', label='Missing Capital Cost Recovery')
    
    ax.set_xlabel("CO₂ Reduction (%)", **font)
    ax.set_ylabel("Electricity Price (€/MWh)", **font)
    ax.set_title("Electricity Pricing: Marginal Cost vs. Full System Cost\n(Demonstrates Missing Capital Cost Recovery)", **font)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    # Add text annotation explaining the difference
    if len(levels) > 0:
        mid_idx = len(levels) // 2
        mid_level = levels[mid_idx]
        mid_marginal = marginal_prices[mid_idx]
        mid_avg = avg_system_costs[mid_idx]
        gap = mid_avg - mid_marginal
        
        ax.annotate(f'Capital cost gap:\n~{gap:.1f} €/MWh', 
                   xy=(mid_level, (mid_marginal + mid_avg) / 2),
                   xytext=(mid_level + 10, (mid_marginal + mid_avg) / 2),
                   fontsize=10, ha='left', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1))
    
    plt.tight_layout()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"electricity_cost_comparison.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved comparison plot as {output_file}")
    
    plt.close(fig)
    
    # Print summary statistics
    print("\n=== ELECTRICITY PRICING ANALYSIS SUMMARY ===")
    print(f"{'Level':<8} {'Marginal (Spatial)':<20} {'Marginal (Demand-wtd)':<23} {'Full Cost':<15} {'Gap':<20}")
    print(f"{'(%)':<8} {'(€/MWh)':<20} {'(€/MWh)':<23} {'(€/MWh)':<15} {'(€/MWh, %)':<20}")
    print("-" * 100)
    
    for i, level in enumerate(levels):
        gap_spatial = avg_system_costs[i] - marginal_prices[i]
        gap_pct_spatial = (gap_spatial / avg_system_costs[i]) * 100
        
        if not np.isnan(demand_weighted_prices[i]):
            gap_demand = avg_system_costs[i] - demand_weighted_prices[i]
            gap_pct_demand = (gap_demand / avg_system_costs[i]) * 100
            print(f"{level:<8.0f} {marginal_prices[i]:<20.2f} {demand_weighted_prices[i]:<23.2f} "
                  f"{avg_system_costs[i]:<15.2f} {gap_demand:<8.2f} ({gap_pct_demand:<.1f}%)")
        else:
            print(f"{level:<8.0f} {marginal_prices[i]:<20.2f} {'N/A':<23} "
                  f"{avg_system_costs[i]:<15.2f} {gap_spatial:<8.2f} ({gap_pct_spatial:<.1f}%)")
    
    if levels:
        # Calculate averages for demand-weighted prices where available
        valid_demand_weighted = [p for p in demand_weighted_prices if not np.isnan(p)]
        if valid_demand_weighted:
            avg_demand_weighted = np.mean(valid_demand_weighted)
            avg_gap_demand = np.mean([avg_system_costs[i] - demand_weighted_prices[i] 
                                     for i in range(len(levels)) if not np.isnan(demand_weighted_prices[i])])
            avg_gap_pct_demand = np.mean([(avg_system_costs[i] - demand_weighted_prices[i]) / avg_system_costs[i] * 100 
                                          for i in range(len(levels)) if not np.isnan(demand_weighted_prices[i])])
            print(f"\nAverage demand-weighted marginal price: {avg_demand_weighted:.2f} €/MWh")
            print(f"Average capital cost gap (demand-weighted): {avg_gap_demand:.1f} €/MWh ({avg_gap_pct_demand:.1f}% of total cost)")
        
        avg_gap = np.mean([avg_system_costs[i] - marginal_prices[i] for i in range(len(levels))])
        avg_gap_pct = np.mean([(avg_system_costs[i] - marginal_prices[i]) / avg_system_costs[i] * 100 for i in range(len(levels))])
        print(f"Average capital cost gap (spatial avg.): {avg_gap:.1f} €/MWh ({avg_gap_pct:.1f}% of total cost)")
        print("\nNote: Gap represents annualized capital cost recovery not captured in marginal pricing.")


def plot_demand_weighted_marginal_prices(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot demand-weighted average marginal prices across decarbonization levels.
    
    This shows how the economically relevant electricity price (weighted by actual consumption)
    evolves with decarbonization, capturing the merit order effect and scarcity pricing.
    """
    print("Creating demand-weighted marginal price plot...")
    
    if not pypsa:
        print("PyPSA not available - skipping demand-weighted price plot")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Calculate demand-weighted marginal prices
    print("  → Calculating demand-weighted marginal prices...")
    demand_weighted_prices = calculate_demand_weighted_prices_by_level(networks_by_percent)
    
    if not demand_weighted_prices:
        print("No demand-weighted price data - cannot create plot")
        return
    
    # Prepare data for plotting
    levels = sorted(demand_weighted_prices.keys())
    prices = [demand_weighted_prices[level] for level in levels]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}
    
    # Plot the demand-weighted price curve
    ax.plot(levels, prices, 'o-', label='Demand-weighted Marginal Price', 
            color='tab:cyan', linewidth=2.5, markersize=10)
    
    # Add horizontal line at baseline if available
    if 0 in levels:
        baseline_price = demand_weighted_prices[0]
        ax.axhline(baseline_price, color='gray', linewidth=1.5, linestyle='--', 
                  label=f'Baseline: {baseline_price:.1f} €/MWh', alpha=0.7)
    
    # Find and annotate the minimum price point
    min_idx = np.argmin(prices)
    min_level = levels[min_idx]
    min_price = prices[min_idx]
    ax.plot(min_level, min_price, 'r*', markersize=20, label=f'Minimum: {min_price:.1f} €/MWh at {min_level}%')
    
    # Add annotation for the minimum
    ax.annotate(f'Lowest price\n{min_price:.1f} €/MWh', 
               xy=(min_level, min_price),
               xytext=(min_level - 15, min_price - 5),
               fontsize=10, ha='center', va='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Calculate and show price change from baseline to 100%
    if 0 in levels and 100 in levels:
        baseline = demand_weighted_prices[0]
        final = demand_weighted_prices[100]
        change = final - baseline
        change_pct = (change / baseline) * 100
        
        ax.annotate(f'Total change:\n+{change:.1f} €/MWh\n(+{change_pct:.1f}%)', 
                   xy=(100, final),
                   xytext=(85, (baseline + final) / 2),
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
    
    ax.set_xlabel("CO₂ Reduction (%)", **font)
    ax.set_ylabel("Demand-Weighted Marginal Price (€/MWh)", **font)
    ax.set_title("Demand-Weighted Marginal Electricity Prices\nAcross Decarbonization Pathway", **font)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"demand_weighted_marginal_prices.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved demand-weighted price plot as {output_file}")
    
    plt.close(fig)
    
    # Print summary statistics
    print("\n=== DEMAND-WEIGHTED MARGINAL PRICE SUMMARY ===")
    print(f"{'Level (%)':<12} {'Price (€/MWh)':<18} {'Change from Baseline':<25}")
    print("-" * 60)
    
    baseline_val = demand_weighted_prices.get(0, None)
    for level in levels:
        price = demand_weighted_prices[level]
        if baseline_val is not None and level > 0:
            change = price - baseline_val
            change_pct = (change / baseline_val) * 100
            change_str = f"{change:+.2f} €/MWh ({change_pct:+.1f}%)"
        else:
            change_str = "Baseline" if level == 0 else "N/A"
        print(f"{level:<12} {price:>16.2f}  {change_str:<25}")
    
    # Identify key transition points
    print("\n=== KEY OBSERVATIONS ===")
    if min_level > 0:
        savings = baseline_val - min_price if baseline_val else 0
        print(f"Merit order effect: Prices decrease by {savings:.2f} €/MWh from baseline to {min_level}%")
        print(f"This represents early displacement of expensive fossil generation by cheap renewables")
    
    if 100 in levels and baseline_val:
        high_decarb_increase = demand_weighted_prices[100] - min_price
        print(f"\nHigh decarbonization challenge: Prices increase by {high_decarb_increase:.2f} €/MWh")
        print(f"from minimum ({min_level}%) to 100% due to integration costs and scarcity events")


def plot_mean_price_bellcurve(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot mean marginal prices by region in bell curve arrangement for each CO₂ reduction level.
    
    NOTE: This plots MARGINAL PRICES (short-run costs) which do NOT include capital cost recovery.
    For full electricity costs including capital recovery, see plot_electricity_cost().
    """
    print("Creating mean marginal price bell curve plots...")
    print("  ⚠️  NOTE: These are marginal prices (short-run costs) - do NOT include capital cost recovery")
    
    if not pypsa:
        print("PyPSA not available - skipping mean price bell curve plots")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Collect marginal prices from all networks
    print("  → Collecting marginal prices...")
    df_marginal_prices = collect_marginal_prices_by_level(networks_by_percent)
    
    if df_marginal_prices is None or df_marginal_prices.empty:
        print("No marginal price data - cannot create plot")
        return
    
    # Calculate mean prices by level
    print("  → Calculating mean prices by level...")
    mean_prices_by_level = calculate_mean_prices_by_level(df_marginal_prices)
    
    # Get all available levels
    levels_to_plot = sorted(mean_prices_by_level.index.tolist())
    print(f"  ✓ Creating bell curve plots for levels: {levels_to_plot}")
    
    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}
    
    # Create plots for each level (we'll save all levels in a single multi-page PDF or separate files)
    for level in levels_to_plot:
        if level not in mean_prices_by_level.index:
            print(f"Level {level} not found in data, skipping.")
            continue
            
        prices = mean_prices_by_level.loc[level].dropna()
        
        if prices.empty:
            print(f"No price data for level {level}%, skipping.")
            continue
            
        # Sort regions by mean price (descending)
        sorted_prices = prices.sort_values(ascending=False)
        n = len(sorted_prices)
        
        # Arrange so largest is in the middle, next largest to the right, next to the left, etc.
        bellcurve_prices = [None] * n
        bellcurve_regions = [None] * n
        center = n // 2
        
        for i, (region, value) in enumerate(sorted_prices.items()):
            pos = center + ((i+1)//2) * (-1 if i%2 else 1)
            bellcurve_prices[pos] = value
            bellcurve_regions[pos] = region

        mean_val = prices.mean()
        std_val = prices.std()

        # Assign colors: darker blue for bars below (mean - std), else skyblue
        bar_colors = [
            'royalblue' if (v is not None and v < mean_val - std_val) else 'skyblue'
            for v in bellcurve_prices
        ]

        # Create the plot
        fig, ax = plt.subplots(figsize=(max(10, n//2), 6))
        
        bars = ax.bar(range(n), bellcurve_prices, color=bar_colors, edgecolor='k', alpha=0.8)
        ax.axhline(mean_val, color='red', linestyle='-', linewidth=3, 
                   label=f"Mean: {mean_val:.2f}")
        ax.axhline(mean_val + std_val, color='orange', linestyle='--', linewidth=3, 
                   label=f"+1 Std: {mean_val + std_val:.2f}")
        ax.axhline(mean_val - std_val, color='orange', linestyle='--', linewidth=3, 
                   label=f"-1 Std: {mean_val - std_val:.2f}")
        
        ax.set_xticks(range(n))
        ax.set_xticklabels(bellcurve_regions, rotation=90)
        ax.set_ylabel("Mean Marginal Price (€/MWh)", **font)
        ax.set_xlabel("Region", **font)
        ax.set_title(f"Mean Regional Prices\nCO₂ Reduction Level: {level}%", **font)
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.5)
        
        plt.tight_layout()

        # Save in all requested formats
        for fmt in output_formats:
            output_file = output_path / f"mean_price_bellcurve_{level}pct.{fmt}"
            fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
            print(f"  ✓ Saved plot as {output_file}")

        plt.close(fig)
    
    # Create summary files that Snakemake expects
    # These will be empty placeholder files since the real plots are the individual level files
    for fmt in output_formats:
        summary_file = output_path / f"mean_price_bellcurve.{fmt}"
        # Create a simple text-based summary plot or copy the first level's plot
        if levels_to_plot:
            first_level_file = output_path / f"mean_price_bellcurve_{levels_to_plot[0]}pct.{fmt}"
            if first_level_file.exists():
                shutil.copy2(first_level_file, summary_file)
                print(f"  ✓ Created summary file {summary_file}")
    
    print(f"  ✓ Created bell curve plots for {len(levels_to_plot)} CO₂ reduction levels")


def plot_mean_price_boxplots(config, output_path, output_formats, dpi=300, has_baseline=True):
    """
    Plot boxplots of mean marginal prices by region for each CO₂ reduction level.
    Shows whiskers, outliers in purple, and overlays mean (red) and median (blue) lines.
    Outliers are annotated with their region names.
    
    NOTE: This plots MARGINAL PRICES (short-run costs) which do NOT include capital cost recovery.
    For full electricity costs including capital recovery, see plot_electricity_cost().
    """
    print("Creating mean marginal price boxplots...")
    print("  ⚠️  NOTE: These are marginal prices (short-run costs) - do NOT include capital cost recovery")
    
    if not pypsa:
        print("PyPSA not available - skipping mean price boxplots")
        return
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create plot")
        return
    
    # Collect marginal prices from all networks
    print("  → Collecting marginal prices...")
    df_marginal_prices = collect_marginal_prices_by_level(networks_by_percent)
    
    if df_marginal_prices is None or df_marginal_prices.empty:
        print("No marginal price data - cannot create plot")
        return
    
    # Calculate mean prices by level
    print("  → Calculating mean prices by level...")
    mean_prices_by_level = calculate_mean_prices_by_level(df_marginal_prices)
    
    # Get all available levels
    levels_to_plot = sorted(mean_prices_by_level.index.tolist())
    print(f"  ✓ Creating boxplots for levels: {levels_to_plot}")
    
    # Prepare data for boxplots
    data = []
    labels = []
    means = []
    medians = []
    region_lists = []
    
    for level in levels_to_plot:
        if level not in mean_prices_by_level.index:
            print(f"Level {level} not found in data, skipping.")
            continue
        prices = mean_prices_by_level.loc[level].dropna()
        if prices.empty:
            print(f"No price data for level {level}%, skipping.")
            continue
            
        data.append(prices.values)
        labels.append(f"{level}%")
        means.append(prices.mean())
        medians.append(np.median(prices.values))
        region_lists.append(prices.index.tolist())
    
    if not data:
        print("No valid data for boxplots")
        return
    
    # Set font properties
    font = {'fontsize': 12, 'fontweight': 'bold'}
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(8, 2 + 2*len(data)), 6))
    
    box = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showmeans=False,
        meanline=False,
        boxprops=dict(facecolor='skyblue', color='k'),
        medianprops=dict(color='blue', linewidth=2),
        whiskerprops=dict(color='k'),
        capprops=dict(color='k'),
        flierprops=dict(marker='o', color='purple', alpha=0.8)  # Outliers in purple
    )

    # Overlay the mean as a red line and the median as a blue line for each box
    for i, (mean, median) in enumerate(zip(means, medians)):
        ax.plot([i+1-0.2, i+1+0.2], [mean, mean], color='red', linewidth=2, 
                label='Mean' if i == 0 else "")
        ax.plot([i+1-0.2, i+1+0.2], [median, median], color='blue', linewidth=2, 
                label='Median' if i == 0 else "")

    # Annotate outliers with region names in purple
    for i, flier in enumerate(box['fliers']):
        y_outliers = flier.get_ydata()
        x_outliers = flier.get_xdata()
        regions = region_lists[i]
        values = data[i]
        
        # Compute IQR for this box
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outlier indices
        outlier_indices = [j for j, v in enumerate(values) if v < lower_bound or v > upper_bound]
        
        for x, y, idx in zip(x_outliers, y_outliers, outlier_indices):
            if idx < len(regions):  # Safety check
                ax.annotate(regions[idx], (x, y), textcoords="offset points", 
                           xytext=(5, 5), ha='left', fontsize=10, color='purple', 
                           fontweight='bold')

    ax.set_ylabel("Mean Marginal Price (€/MWh)", **font)
    ax.set_xlabel("CO₂ Reduction Level", **font)
    ax.set_title("Regional Mean Marginal Price Distribution by CO₂ Reduction Level", **font)
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"mean_price_boxplots.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved plot as {output_file}")

    plt.close(fig)
    print(f"  ✓ Created boxplot with {len(levels_to_plot)} CO₂ reduction levels")


def get_europe_map():
    """Get Europe map data, handling GeoPandas version differences."""
    # Method 1: Try the old naturalearth_lowres dataset
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        europe = world[world['continent'] == 'Europe']
        if len(europe) > 0:
            return europe
    except (AttributeError, Exception):
        pass
    
    # Method 2: Try downloading from Natural Earth directly
    try:
        print("  → Downloading map data from Natural Earth...")
        url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)
        europe = world[world['CONTINENT'] == 'Europe']
        if len(europe) > 0:
            print("  ✓ Successfully downloaded Europe map")
            return europe
    except Exception as e:
        print(f"  ⚠️ Could not download from Natural Earth: {e}")
    
    # Method 3: Create a simple Europe bounding box as fallback
    try:
        print("  → Using simplified Europe bounding box for maps")
        from shapely.geometry import Polygon
        
        # Europe bounding box coordinates
        europe_bounds = Polygon([(-15, 35), (35, 35), (35, 72), (-15, 72), (-15, 35)])
        europe_gdf = gpd.GeoDataFrame([{'name': 'Europe Box'}], geometry=[europe_bounds], crs="EPSG:4326")
        return europe_gdf
    except Exception as e:
        print(f"  ⚠️ Could not create bounding box: {e}")
        return None


def get_carrier_color_map():
    """Define consistent color mapping for energy carriers/technologies."""
    return {
        'hydro': '#377eb8',      # Blue
        'nuclear': '#e41a1c',    # Red
        'coal': '#8B4513',       # Brown  
        'lignite': '#A0522D',    # Darker brown
        'CCGT': '#4daf4a',       # Green
        'OCGT': '#90EE90',       # Light green
        'oil': '#000000',        # Black
        'biomass': '#32CD32',    # Lime green
        'geothermal': '#FF6347', # Tomato
        'ror': '#377eb8',        # Blue
        'solar': '#ffff33',      # Yellow
        'onwind': '#984ea3',     # Purple
        'offwind-ac': '#ff7f00', # Orange
        'offwind-dc': '#a65628', # Dark orange
    }


def get_carrier_color_map():
    """Define consistent color mapping for energy carriers/technologies."""
    return {
        'hydro': '#377eb8',      # Blue
        'nuclear': '#e41a1c',    # Red
        'coal': '#8B4513',       # Brown  
        'lignite': '#A0522D',    # Darker brown
        'CCGT': '#4daf4a',       # Green
        'OCGT': '#90EE90',       # Light green
        'oil': '#000000',        # Black
        'biomass': '#32CD32',    # Lime green
        'geothermal': '#FF6347', # Tomato
        'ror': '#377eb8',        # Blue
        'solar': '#ffff33',      # Yellow
        'onwind': '#984ea3',     # Purple
        'offwind-ac': '#ff7f00', # Orange
        'offwind-dc': '#a65628', # Dark orange
    }


def get_storage_color_map():
    """Define consistent color mapping for storage technologies."""
    return {
        'PHS': '#1f77b4',      # Blue for Pumped Hydro Storage
        'hydro': '#17becf',    # Cyan for other hydro storage
        'battery': '#2ca02c',  # Green for batteries
        'Battery': '#2ca02c',  # Green for batteries (uppercase)
        'H2': '#ff7f0e',       # Orange for hydrogen
        'hydrogen': '#ff7f0e'  # Orange for hydrogen (lowercase)
    }


def prepare_network_data_for_maps(network):
    """Prepare network data for mapping visualization."""
    if not network:
        return None, None, None, None, None
    
    # Prepare bus data
    bus_df = network.buses.copy()
    
    # Prepare generation data
    generators = network.generators.copy()
    generators["capacity"] = generators["p_nom_opt"]  # Use optimized capacity
    cap_by_node_carrier = generators.groupby(["bus", "carrier"])["capacity"].sum().unstack(fill_value=0)
    cap_by_node_carrier = cap_by_node_carrier.reindex(bus_df.index, fill_value=0) / 1000  # Convert to GW
    
    # Prepare transmission lines data
    lines = network.lines.copy()
    lines["bus0_x"] = network.buses.loc[lines["bus0"], "x"].values
    lines["bus0_y"] = network.buses.loc[lines["bus0"], "y"].values  
    lines["bus1_x"] = network.buses.loc[lines["bus1"], "x"].values
    lines["bus1_y"] = network.buses.loc[lines["bus1"], "y"].values
    lines["capacity"] = lines["s_nom_opt"] / 1000  # Convert to GW
    
    # Prepare links data  
    links = network.links.copy()
    links["bus0_x"] = network.buses.loc[links["bus0"], "x"].values
    links["bus0_y"] = network.buses.loc[links["bus0"], "y"].values
    links["bus1_x"] = network.buses.loc[links["bus1"], "x"].values  
    links["bus1_y"] = network.buses.loc[links["bus1"], "y"].values
    links["capacity"] = links["p_nom_opt"] / 1000  # Convert to GW
    
    # Prepare storage data
    storage_by_node_carrier = None
    if hasattr(network, "storage_units") and len(network.storage_units) > 0:
        storage = network.storage_units.copy()
        storage["node"] = storage["bus"]
        storage_by_node_carrier = storage.groupby(["node", "carrier"])["p_nom_opt"].sum().unstack(fill_value=0)
        storage_by_node_carrier = storage_by_node_carrier.reindex(bus_df.index, fill_value=0) / 1000  # Convert to GW
    
    return bus_df, cap_by_node_carrier, lines, links, storage_by_node_carrier


def plot_generation_map(network, level, output_path, output_formats, dpi=300):
    """Create generation capacity map for a specific CO₂ reduction level."""
    if not gpd or not pypsa:
        print(f"Skipping generation map for {level}% - missing dependencies")
        return
    
    # Prepare data
    bus_df, cap_by_node_carrier, _, _, _ = prepare_network_data_for_maps(network)
    if bus_df is None:
        return
    
    carrier_color_map = get_carrier_color_map()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Load world map and focus on Europe
    europe = get_europe_map()
    if europe is not None:
        europe.plot(ax=ax, color='lightgray', edgecolor='k', alpha=0.7, zorder=0)
    
    # Calculate maximum total capacity for scaling pie sizes
    max_total_cap = cap_by_node_carrier.sum(axis=1).max()
    
    # Create pie charts at each bus location
    buses_with_capacity = 0
    for node, row in bus_df.iterrows():
        caps = cap_by_node_carrier.loc[node] 
        total_cap = caps.sum()
        
        if total_cap < 0.1:  # Skip nodes with very small capacity (< 100 MW)
            continue
            
        buses_with_capacity += 1
        
        # Calculate pie size based on total capacity
        size = 18000 * total_cap / max_total_cap
        
        # Prepare data for pie chart
        fracs = []
        colors = []
        labels = []
        
        for carrier in cap_by_node_carrier.columns:
            val = caps.get(carrier, 0)
            if val > 0.05:  # Only show technologies with >50 MW
                fracs.append(val)
                colors.append(carrier_color_map.get(carrier, 'gray'))
                labels.append(carrier)
        
        if fracs:  # Only create pie if there's data
            x, y = row["x"], row["y"]
            ax.pie(fracs, colors=colors, radius=np.sqrt(size)/100, center=(x, y), frame=True)
            # Add small black dot at center
            ax.plot(x, y, "o", color="k", markersize=2, zorder=3)
    
    # Create legend for technologies that are present
    present_carriers = [c for c in cap_by_node_carrier.columns if cap_by_node_carrier[c].sum() > 0.1]
    legend_patches = [Patch(color=carrier_color_map.get(c, 'gray'), label=c) for c in present_carriers]
    ax.legend(handles=legend_patches, title="Generation Technologies", 
              loc="upper right", fontsize=11, title_fontsize=13, 
              bbox_to_anchor=(1.25, 1))
    
    # Set map extent to cover the network area with some padding
    ax.set_xlim(bus_df['x'].min() - 2, bus_df['x'].max() + 2)
    ax.set_ylim(bus_df['y'].min() - 2, bus_df['y'].max() + 2)
    ax.set_title(f"Installed Generation Capacity by Technology\nCO₂ Reduction Level: {level}%\n(Pie size proportional to total capacity)", 
                 fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")
    
    plt.tight_layout()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / "maps" / f"generation_map_{level}pct.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.close(fig)


def plot_transmission_map(network, level, output_path, output_formats, dpi=300):
    """Create transmission network map for a specific CO₂ reduction level."""
    if not gpd or not pypsa:
        print(f"Skipping transmission map for {level}% - missing dependencies")
        return
    
    # Prepare data
    bus_df, _, lines, links, _ = prepare_network_data_for_maps(network)
    if bus_df is None:
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Load world map and focus on Europe
    europe = get_europe_map()
    if europe is not None:
        europe.plot(ax=ax, color='lightgray', edgecolor='k', alpha=0.7, zorder=0)
    
    # Find maximum capacity for line width scaling
    max_line_cap = max(lines["capacity"].max(), links["capacity"].max())
    
    # Plot AC lines (green)
    line_count = 0
    for _, row in lines.iterrows():
        if row["capacity"] > 0.0001:  # Only show lines with >100 kW capacity
            lw = 2 + 15 * row["capacity"] / max_line_cap  # Line width scaling
            ax.plot([row["bus0_x"], row["bus1_x"]], [row["bus0_y"], row["bus1_y"]], 
                    color="green", linewidth=lw, alpha=0.8, zorder=1)
            line_count += 1
    
    # Plot DC links (purple, dashed)
    link_count = 0
    for _, row in links.iterrows():
        if row["capacity"] > 0.0001:  # Only show links with >100 kW capacity
            lw = 2 + 15 * row["capacity"] / max_line_cap  # Line width scaling
            ax.plot([row["bus0_x"], row["bus1_x"]], [row["bus0_y"], row["bus1_y"]], 
                    color="purple", linewidth=lw, alpha=0.8, zorder=1, linestyle="--")
            link_count += 1
    
    # Add bus locations as small dots
    for node, row in bus_df.iterrows():
        ax.plot(row["x"], row["y"], "o", color="red", markersize=4, zorder=3, alpha=0.7)
    
    # Create legend
    line_patch = Patch(color="green", label="AC Lines")
    link_patch = Patch(color="purple", label="DC Links")
    bus_patch = Patch(color="red", label="Buses/Nodes")
    ax.legend(handles=[line_patch, link_patch, bus_patch], 
              loc="upper right", fontsize=12, title="Transmission Infrastructure", 
              title_fontsize=14, bbox_to_anchor=(1.25, 1))
    
    # Set map extent
    ax.set_xlim(bus_df['x'].min() - 2, bus_df['x'].max() + 2)
    ax.set_ylim(bus_df['y'].min() - 2, bus_df['y'].max() + 2)
    ax.set_title(f"Transmission Infrastructure\nCO₂ Reduction Level: {level}%\n(Line width proportional to capacity)", 
                 fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")
    
    plt.tight_layout()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / "maps" / f"transmission_map_{level}pct.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.close(fig)


def plot_storage_map(network, level, output_path, output_formats, dpi=300):
    """Create storage capacity map for a specific CO₂ reduction level."""
    if not gpd or not pypsa:
        print(f"Skipping storage map for {level}% - missing dependencies")
        return
    
    # Prepare data
    bus_df, _, _, _, storage_by_node_carrier = prepare_network_data_for_maps(network)
    if bus_df is None:
        return
    
    storage_color_map = get_storage_color_map()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Load world map and focus on Europe
    europe = get_europe_map()
    if europe is not None:
        europe.plot(ax=ax, color='lightgray', edgecolor='k', alpha=0.7, zorder=0)
    
    storage_legend_handles = []
    
    if storage_by_node_carrier is not None and storage_by_node_carrier.sum().sum() > 0:
        max_storage = storage_by_node_carrier.sum(axis=1).max()
        
        # Storage types present
        storage_types = storage_by_node_carrier.columns.tolist()
        
        # Create pie charts for nodes with storage
        storage_nodes = 0
        for node, row in bus_df.iterrows():
            caps = storage_by_node_carrier.loc[node]
            total_storage = caps.sum()
            
            if total_storage < 0.01:  # Skip nodes with very small storage (< 10 MW)
                continue
                
            storage_nodes += 1
            
            # Calculate pie size based on total storage capacity
            size = 18000 * total_storage / max_storage
            
            # Prepare data for pie chart
            fracs = []
            colors = []
            labels = []
            
            for storage_type in storage_types:
                val = caps.get(storage_type, 0)
                if val > 0.005:  # Only show storage types with >5 MW
                    fracs.append(val)
                    colors.append(storage_color_map.get(storage_type, 'gray'))
                    labels.append(storage_type)
            
            if fracs:  # Only create pie if there's data
                x, y = row["x"], row["y"]
                ax.pie(fracs, colors=colors, radius=np.sqrt(size)/100, center=(x, y), frame=True)
                # Add small black dot at center
                ax.plot(x, y, "o", color="k", markersize=2, zorder=3)
        
        # Create legend for storage types that are present
        present_storage_types = [c for c in storage_types if storage_by_node_carrier[c].sum() > 0]
        storage_legend_handles = [Patch(color=storage_color_map.get(c, 'gray'), label=c) 
                                 for c in present_storage_types]
        
        ax.set_title(f"Storage Capacity by Node and Type\nCO₂ Reduction Level: {level}%\n(Pie size proportional to total storage capacity)", 
                     fontsize=16, fontweight="bold", pad=20)
    else:
        ax.set_title(f"Storage Capacity\nCO₂ Reduction Level: {level}%\nNo Storage Units Present", 
                     fontsize=16, fontweight="bold", pad=20)
    
    # Add legend if storage exists
    if storage_legend_handles:
        ax.legend(handles=storage_legend_handles, 
                  loc="upper right", fontsize=12, title="Storage Technologies", 
                  title_fontsize=14, bbox_to_anchor=(1.25, 1))
    
    # Set map extent
    ax.set_xlim(bus_df['x'].min() - 2, bus_df['x'].max() + 2)
    ax.set_ylim(bus_df['y'].min() - 2, bus_df['y'].max() + 2)
    ax.axis("off")
    
    plt.tight_layout()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / "maps" / f"storage_map_{level}pct.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.close(fig)


def plot_network_maps(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Create generation, transmission, and storage maps for all CO₂ reduction levels."""
    print("Creating network maps for all CO₂ reduction levels...")
    
    if not gpd or not pypsa:
        print("GeoPandas or PyPSA not available - skipping network maps")
        return
    
    # Create maps subdirectory
    maps_path = output_path / "maps"
    maps_path.mkdir(exist_ok=True)
    print(f"  → Created maps directory: {maps_path}")
    
    # Load networks
    networks_by_percent = load_networks_from_results(config, has_baseline)
    
    if not networks_by_percent:
        print("No networks found - cannot create maps")
        return
    
    # Get all available levels
    levels_to_plot = sorted(networks_by_percent.keys())
    print(f"  ✓ Creating maps for levels: {levels_to_plot}")
    
    map_counts = {"generation": 0, "transmission": 0, "storage": 0}
    
    for level, network in networks_by_percent.items():
        print(f"  → Processing {level}% CO₂ reduction level...")
        
        try:
            # Generation map
            plot_generation_map(network, level, output_path, output_formats, dpi)
            map_counts["generation"] += 1
            print(f"    ✓ Created generation map")
            
            # Transmission map
            plot_transmission_map(network, level, output_path, output_formats, dpi)
            map_counts["transmission"] += 1
            print(f"    ✓ Created transmission map")
            
            # Storage map
            plot_storage_map(network, level, output_path, output_formats, dpi)
            map_counts["storage"] += 1
            print(f"    ✓ Created storage map")
            
        except Exception as e:
            print(f"    ⚠️ Error creating maps for {level}%: {e}")
    
    print(f"  ✓ Created {map_counts['generation']} generation maps, {map_counts['transmission']} transmission maps, {map_counts['storage']} storage maps")
    print(f"  ✓ All maps saved to {maps_path}")
    
    # Create summary files that Snakemake expects
    for fmt in output_formats:
        summary_file = output_path / f"network_maps.{fmt}"
        # Create a simple text-based summary plot since we created individual maps
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.7, f"Network Maps Created", ha='center', va='center', 
                fontsize=20, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.5, f"Generated maps for {len(levels_to_plot)} CO₂ reduction levels:", 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.text(0.5, 0.4, f"• {map_counts['generation']} generation capacity maps", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.35, f"• {map_counts['transmission']} transmission network maps", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.3, f"• {map_counts['storage']} storage capacity maps", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.2, f"All detailed maps saved to: {maps_path}", 
                ha='center', va='center', fontsize=10, style='italic', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        fig.savefig(summary_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Created summary file {summary_file}")
        
    # Plot K-constrained networks if enabled
    k_maps_enabled = config.get("parameters", {}).get("plotting", {}).get("k_constrained_maps", False)
    if k_maps_enabled:
        plot_k_constrained_network_maps(config, output_path, output_formats, dpi)


def plot_k_constrained_network_maps(config, output_path, output_formats, dpi=300):
    """Create generation, transmission, and storage maps for K-constrained networks."""
    print("Creating K-constrained network maps...")
    
    if not gpd or not pypsa:
        print("GeoPandas or PyPSA not available - skipping K-constrained network maps")
        return
    
    # Create K-constrained maps subdirectory
    k_maps_path = output_path / "maps" / "k_constrained"
    k_maps_path.mkdir(exist_ok=True, parents=True)
    print(f"  → Created K-constrained maps directory: {k_maps_path}")
    
    # Load K-constrained networks
    networks_by_reduction_k = load_k_constrained_networks_from_results(config)
    
    if not networks_by_reduction_k:
        print("No K-constrained networks found - cannot create K-constrained maps")
        return
    
    # Get available reduction levels and k values
    available_combos = sorted(networks_by_reduction_k.keys())
    print(f"  ✓ Creating K-constrained maps for {len(available_combos)} reduction-k combinations")
    
    k_map_counts = {"generation": 0, "transmission": 0, "storage": 0}
    
    for (reduction, k_value), network in networks_by_reduction_k.items():
        print(f"  → Processing {reduction}% CO₂ reduction, k={k_value}...")
        
        try:
            # Create subdirectory for this k-value
            k_subdir = k_maps_path / f"k_{k_value}"
            k_subdir.mkdir(exist_ok=True)
            
            # Generation map
            plot_generation_map_k_constrained(network, reduction, k_value, k_subdir, output_formats, dpi)
            k_map_counts["generation"] += 1
            print(f"    ✓ Created generation map")
            
            # Transmission map
            plot_transmission_map_k_constrained(network, reduction, k_value, k_subdir, output_formats, dpi)
            k_map_counts["transmission"] += 1
            print(f"    ✓ Created transmission map")
            
            # Storage map
            plot_storage_map_k_constrained(network, reduction, k_value, k_subdir, output_formats, dpi)
            k_map_counts["storage"] += 1
            print(f"    ✓ Created storage map")
            
        except Exception as e:
            print(f"    ⚠️ Error creating K-constrained maps for {reduction}%, k={k_value}: {e}")
    
    print(f"  ✓ Created {k_map_counts['generation']} K-constrained generation maps, {k_map_counts['transmission']} transmission maps, {k_map_counts['storage']} storage maps")
    print(f"  ✓ All K-constrained maps saved to {k_maps_path}")
    
    # Create summary file for K-constrained maps
    for fmt in output_formats:
        summary_file = k_maps_path / f"k_constrained_maps_summary.{fmt}"
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.7, f"K-Constrained Network Maps Created", ha='center', va='center', 
                fontsize=20, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.5, f"Generated maps for {len(available_combos)} reduction-k combinations:", 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.text(0.5, 0.4, f"• {k_map_counts['generation']} generation capacity maps", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.35, f"• {k_map_counts['transmission']} transmission network maps", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.3, f"• {k_map_counts['storage']} storage capacity maps", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.2, f"All detailed maps saved to: {k_maps_path}", 
                ha='center', va='center', fontsize=10, style='italic', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        fig.savefig(summary_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Created K-constrained summary file {summary_file}")


def plot_generation_map_k_constrained(network, reduction, k_value, output_path, output_formats, dpi=300):
    """Create generation capacity map for a K-constrained network."""
    if not gpd or not pypsa:
        print(f"Skipping K-constrained generation map for {reduction}%, k={k_value} - missing dependencies")
        return
    
    # Prepare data
    bus_df, cap_by_node_carrier, _, _, _ = prepare_network_data_for_maps(network)
    if bus_df is None:
        return
    
    carrier_color_map = get_carrier_color_map()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Load world map and focus on Europe
    europe = get_europe_map()
    if europe is not None:
        europe.plot(ax=ax, color='lightgray', edgecolor='k', alpha=0.7, zorder=0)
    
    # Calculate maximum total capacity for scaling pie sizes
    max_total_cap = cap_by_node_carrier.sum(axis=1).max()
    
    # Create pie charts at each bus location
    buses_with_capacity = 0
    for node, row in bus_df.iterrows():
        caps = cap_by_node_carrier.loc[node] 
        total_cap = caps.sum()
        
        if total_cap < 0.1:  # Skip nodes with very small capacity (< 100 MW)
            continue
            
        buses_with_capacity += 1
        
        # Calculate pie size based on total capacity
        size = 18000 * total_cap / max_total_cap
        
        # Prepare data for pie chart
        fracs = []
        colors = []
        labels = []
        
        for carrier in cap_by_node_carrier.columns:
            val = caps.get(carrier, 0)
            if val > 0.05:  # Only show technologies with >50 MW
                fracs.append(val)
                colors.append(carrier_color_map.get(carrier, 'gray'))
                labels.append(carrier)
        
        if fracs:  # Only create pie if there's data
            x, y = row["x"], row["y"]
            ax.pie(fracs, colors=colors, radius=np.sqrt(size)/100, center=(x, y), frame=True)
            # Add small black dot at center
            ax.plot(x, y, "o", color="k", markersize=2, zorder=3)
    
    # Create legend for technologies that are present
    present_carriers = [c for c in cap_by_node_carrier.columns if cap_by_node_carrier[c].sum() > 0.1]
    legend_patches = [Patch(color=carrier_color_map.get(c, 'gray'), label=c) for c in present_carriers]
    ax.legend(handles=legend_patches, title="Generation Technologies", 
              loc="upper right", fontsize=11, title_fontsize=13, 
              bbox_to_anchor=(1.25, 1))
    
    # Set map extent to cover the network area with some padding
    ax.set_xlim(bus_df['x'].min() - 2, bus_df['x'].max() + 2)
    ax.set_ylim(bus_df['y'].min() - 2, bus_df['y'].max() + 2)
    ax.set_title(f"Installed Generation Capacity by Technology (K-constrained)\nCO₂ Reduction: {reduction}%, k={k_value}\n(Pie size proportional to total capacity)", 
                 fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")
    
    plt.tight_layout()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"generation_map_{reduction}pct_k_{k_value}.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.close(fig)


def plot_transmission_map_k_constrained(network, reduction, k_value, output_path, output_formats, dpi=300):
    """Create transmission network map for a K-constrained network."""
    if not gpd or not pypsa:
        print(f"Skipping K-constrained transmission map for {reduction}%, k={k_value} - missing dependencies")
        return
    
    # Prepare data
    bus_df, _, lines, links, _ = prepare_network_data_for_maps(network)
    if bus_df is None:
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Load world map and focus on Europe
    europe = get_europe_map()
    if europe is not None:
        europe.plot(ax=ax, color='lightgray', edgecolor='k', alpha=0.7, zorder=0)
    
    # Find maximum capacity for line width scaling
    max_line_cap = max(lines["capacity"].max(), links["capacity"].max())
    
    # Plot AC lines (green)
    line_count = 0
    for _, row in lines.iterrows():
        if row["capacity"] > 0.0001:  # Only show lines with >100 kW capacity
            lw = 2 + 15 * row["capacity"] / max_line_cap  # Line width scaling
            ax.plot([row["bus0_x"], row["bus1_x"]], [row["bus0_y"], row["bus1_y"]], 
                    color="green", linewidth=lw, alpha=0.8, zorder=1)
            line_count += 1
    
    # Plot DC links (purple, dashed)
    link_count = 0
    for _, row in links.iterrows():
        if row["capacity"] > 0.0001:  # Only show links with >100 kW capacity
            lw = 2 + 15 * row["capacity"] / max_line_cap  # Line width scaling
            ax.plot([row["bus0_x"], row["bus1_x"]], [row["bus0_y"], row["bus1_y"]], 
                    color="purple", linewidth=lw, alpha=0.8, zorder=1, linestyle="--")
            link_count += 1
    
    # Add bus locations as small dots
    for node, row in bus_df.iterrows():
        ax.plot(row["x"], row["y"], "o", color="black", markersize=3, zorder=2, alpha=0.7)
    
    # Create legend for line types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=3, label=f'AC Lines ({line_count})'),
        Line2D([0], [0], color='purple', lw=3, linestyle='--', label=f'DC Links ({link_count})')
    ]
    ax.legend(handles=legend_elements, title="Transmission Infrastructure", 
              loc="upper right", fontsize=11, title_fontsize=13,
              bbox_to_anchor=(1.25, 1))
    
    # Set map extent to cover the network area with some padding  
    ax.set_xlim(bus_df['x'].min() - 2, bus_df['x'].max() + 2)
    ax.set_ylim(bus_df['y'].min() - 2, bus_df['y'].max() + 2)
    ax.set_title(f"Transmission Network Infrastructure (K-constrained)\nCO₂ Reduction: {reduction}%, k={k_value}\n(Line width proportional to capacity)", 
                 fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")
    
    plt.tight_layout()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"transmission_map_{reduction}pct_k_{k_value}.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.close(fig)


def plot_storage_map_k_constrained(network, reduction, k_value, output_path, output_formats, dpi=300):
    """Create storage capacity map for a K-constrained network."""
    if not gpd or not pypsa:
        print(f"Skipping K-constrained storage map for {reduction}%, k={k_value} - missing dependencies")
        return
    
    # Prepare data
    bus_df, _, _, _, storage_by_node_carrier = prepare_network_data_for_maps(network)
    if bus_df is None or storage_by_node_carrier is None:
        return
    
    storage_color_map = get_storage_color_map()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Load world map and focus on Europe
    europe = get_europe_map()
    if europe is not None:
        europe.plot(ax=ax, color='lightgray', edgecolor='k', alpha=0.7, zorder=0)
    
    # Calculate maximum total capacity for scaling pie sizes
    max_total_cap = storage_by_node_carrier.sum(axis=1).max()
    
    # Create pie charts at each bus location
    buses_with_storage = 0
    for node, row in bus_df.iterrows():
        caps = storage_by_node_carrier.loc[node]
        total_cap = caps.sum()
        
        if total_cap < 0.1:  # Skip nodes with very small capacity (< 100 MW)
            continue
            
        buses_with_storage += 1
        
        # Calculate pie size based on total capacity
        size = 15000 * total_cap / max_total_cap
        
        # Prepare data for pie chart
        fracs = []
        colors = []
        labels = []
        
        for carrier in storage_by_node_carrier.columns:
            val = caps.get(carrier, 0)
            if val > 0.05:  # Only show technologies with >50 MW
                fracs.append(val)
                colors.append(storage_color_map.get(carrier, 'gray'))
                labels.append(carrier)
        
        if fracs:  # Only create pie if there's data
            x, y = row["x"], row["y"]
            ax.pie(fracs, colors=colors, radius=np.sqrt(size)/100, center=(x, y), frame=True)
            # Add small black dot at center
            ax.plot(x, y, "o", color="k", markersize=2, zorder=3)
    
    # Create legend for storage technologies that are present
    present_carriers = [c for c in storage_by_node_carrier.columns if storage_by_node_carrier[c].sum() > 0.1]
    legend_patches = [Patch(color=storage_color_map.get(c, 'gray'), label=c) for c in present_carriers]
    ax.legend(handles=legend_patches, title="Storage Technologies", 
              loc="upper right", fontsize=11, title_fontsize=13, 
              bbox_to_anchor=(1.25, 1))
    
    # Set map extent to cover the network area with some padding
    ax.set_xlim(bus_df['x'].min() - 2, bus_df['x'].max() + 2)
    ax.set_ylim(bus_df['y'].min() - 2, bus_df['y'].max() + 2)
    ax.set_title(f"Installed Storage Capacity by Technology (K-constrained)\nCO₂ Reduction: {reduction}%, k={k_value}\n(Pie size proportional to total capacity)", 
                 fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")
    
    plt.tight_layout()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"storage_map_{reduction}pct_k_{k_value}.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.close(fig)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_matplotlib():
    """Setup matplotlib with sensible defaults for thesis plots."""
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })


def plot_network_topology(networks, output_path, output_formats, dpi=300):
    """Plot network topology showing buses and transmission lines."""
    print("Creating network topology plot...")
    
    # Create plot using the baseline or first reduction network
    if networks:
        network = networks[0]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Basic network plot - you may need to adjust this based on your network structure
        try:
            network.plot(ax=ax, bus_sizes=0.02, line_widths=0.5)
            ax.set_title("Network Topology")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        except Exception as e:
            print(f"Error plotting network: {e}")
            # Create a placeholder plot
            ax.text(0.5, 0.5, "Network topology plot\n(requires geographic data)", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Network Topology")
        
        # Save in all requested formats
        for fmt in output_formats:
            output_file = output_path / f"network_topology.{fmt}"
            fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        
        plt.close(fig)


def plot_generation_mix(networks, tables, output_path, output_formats, dpi=300):
    """Plot generation mix across different scenarios."""
    print("Creating generation mix plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Example plot - you'll need to adapt this to your data structure
    scenarios = [f"Reduction {i}%" for i in range(0, 50, 10)]
    carriers = ['Solar', 'Wind', 'Hydro', 'Nuclear', 'Gas', 'Coal']
    
    # Create sample data - replace with actual data extraction
    data = np.random.rand(len(scenarios), len(carriers))
    data = data / data.sum(axis=1, keepdims=True) * 100  # Normalize to percentages
    
    # Create stacked bar chart
    bottom = np.zeros(len(scenarios))
    colors = plt.cm.tab10(np.linspace(0, 1, len(carriers)))
    
    for i, carrier in enumerate(carriers):
        ax.bar(scenarios, data[:, i], bottom=bottom, label=carrier, color=colors[i])
        bottom += data[:, i]
    
    ax.set_title("Generation Mix by Scenario")
    ax.set_xlabel("CO₂ Reduction Scenario")
    ax.set_ylabel("Generation Share (%)")
    ax.legend()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"generation_mix.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.close(fig)


def plot_transmission_flows(networks, output_path, output_formats, dpi=300):
    """Plot transmission flows and congestion."""
    print("Creating transmission flows plot...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Placeholder plot - adapt to your data
    lines = [f"Line {i+1}" for i in range(10)]
    flows = np.random.rand(10) * 1000  # Example flow data
    capacities = flows + np.random.rand(10) * 200  # Example capacity data
    
    x = np.arange(len(lines))
    width = 0.35
    
    ax.bar(x - width/2, flows, width, label='Flow', alpha=0.8)
    ax.bar(x + width/2, capacities, width, label='Capacity', alpha=0.8)
    
    ax.set_title("Transmission Line Utilization")
    ax.set_xlabel("Transmission Lines")
    ax.set_ylabel("Power (MW)")
    ax.set_xticks(x)
    ax.set_xticklabels(lines, rotation=45)
    ax.legend()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"transmission_flows.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.close(fig)


def plot_storage_utilization(networks, output_path, output_formats, dpi=300):
    """Plot storage capacity and utilization."""
    print("Creating storage utilization plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Battery storage
    scenarios = [f"R{i}%" for i in range(0, 50, 10)]
    battery_cap = np.random.rand(len(scenarios)) * 1000
    h2_cap = np.random.rand(len(scenarios)) * 500
    
    ax1.plot(scenarios, battery_cap, 'o-', label='Battery', linewidth=2, markersize=8)
    ax1.plot(scenarios, h2_cap, 's-', label='Hydrogen', linewidth=2, markersize=8)
    ax1.set_title("Storage Capacity by Scenario")
    ax1.set_xlabel("CO₂ Reduction Scenario")
    ax1.set_ylabel("Storage Capacity (MWh)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Storage utilization
    storage_types = ['Battery', 'H2 Storage', 'Pumped Hydro']
    utilization = np.random.rand(3) * 100
    
    bars = ax2.bar(storage_types, utilization, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax2.set_title("Average Storage Utilization")
    ax2.set_ylabel("Utilization (%)")
    ax2.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, val in zip(bars, utilization):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"storage_utilization.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.close(fig)


def plot_emissions_summary(tables, baseline_data, output_path, output_formats, dpi=300):
    """Plot CO2 emissions reduction summary."""
    print("Creating emissions summary plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Example data - replace with actual emissions data
    scenarios = [f"{i}%" for i in range(0, 50, 10)]
    emissions = [100, 90, 80, 70, 60]  # Example: decreasing emissions
    
    ax.plot(scenarios, emissions, 'ro-', linewidth=3, markersize=10)
    ax.fill_between(scenarios, emissions, alpha=0.3)
    
    ax.set_title("CO₂ Emissions by Reduction Scenario")
    ax.set_xlabel("CO₂ Reduction Target")
    ax.set_ylabel("CO₂ Emissions (Mt CO₂/year)")
    ax.grid(True, alpha=0.3)
    
    # Add target line
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Target')
    ax.legend()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"emissions_summary.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.close(fig)


def plot_cost_breakdown(tables, output_path, output_formats, dpi=300):
    """Plot system cost breakdown by technology and scenario."""
    print("Creating cost breakdown plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Example data - replace with actual cost data
    scenarios = [f"R{i}%" for i in range(0, 50, 10)]
    cost_categories = ['Generation', 'Transmission', 'Storage', 'Operation']
    
    # Sample cost data
    costs = np.random.rand(len(scenarios), len(cost_categories)) * 1000
    
    # Create stacked bar chart
    bottom = np.zeros(len(scenarios))
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    
    for i, category in enumerate(cost_categories):
        ax.bar(scenarios, costs[:, i], bottom=bottom, label=category, color=colors[i])
        bottom += costs[:, i]
    
    ax.set_title("System Cost Breakdown by Scenario")
    ax.set_xlabel("CO₂ Reduction Scenario")
    ax.set_ylabel("Cost (M€/year)")
    ax.legend()
    
    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"cost_breakdown.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.close(fig)


def load_networks_and_data(config, has_baseline=True):
    """Load network files and data tables."""
    networks = []
    tables = []
    baseline_data = None
    
    # Load baseline if it exists
    if has_baseline:
        try:
            baseline_path = "results/networks/solved_baseline_costed_expansion.nc"
            if os.path.exists(baseline_path) and pypsa:
                baseline_net = pypsa.Network(baseline_path)
                networks.append(baseline_net)
            
            baseline_table_path = "results/tables/solve_baseline_costed_expansion.csv"
            if os.path.exists(baseline_table_path):
                baseline_data = pd.read_csv(baseline_table_path)
        except Exception as e:
            print(f"Warning: Could not load baseline data: {e}")
    
    # Load reduction scenario networks
    reductions = config.get("parameters", {}).get("co2_reductions", [])
    for reduction in reductions:
        if float(reduction) > 0:
            try:
                net_path = f"results/networks/solved_reduction_{reduction}.nc"
                if os.path.exists(net_path) and pypsa:
                    net = pypsa.Network(net_path)
                    networks.append(net)
                
                table_path = f"results/tables/solve_reduction_{reduction}.csv"
                if os.path.exists(table_path):
                    table = pd.read_csv(table_path)
                    tables.append(table)
            except Exception as e:
                print(f"Warning: Could not load reduction {reduction}% data: {e}")
    
    return networks, tables, baseline_data


def load_networks_and_data(config, has_baseline=True):
    """Load network files and data tables (backward compatibility function)."""
    networks = []
    tables = []
    baseline_data = None
    
    # Load baseline if it exists
    if has_baseline:
        try:
            baseline_path = "results/networks/solved_baseline_costed_expansion.nc"
            if os.path.exists(baseline_path) and pypsa:
                baseline_net = pypsa.Network(baseline_path)
                networks.append(baseline_net)
            
            baseline_table_path = "results/tables/solve_baseline_costed_expansion.csv"
            if os.path.exists(baseline_table_path):
                baseline_data = pd.read_csv(baseline_table_path)
        except Exception as e:
            print(f"Warning: Could not load baseline data: {e}")
    
    # Load reduction scenario networks
    reductions = config.get("parameters", {}).get("co2_reductions", [])
    for reduction in reductions:
        if float(reduction) > 0:
            try:
                net_path = f"results/networks/solved_reduction_{reduction}.nc"
                if os.path.exists(net_path) and pypsa:
                    net = pypsa.Network(net_path)
                    networks.append(net)
                
                table_path = f"results/tables/solve_reduction_{reduction}.csv"
                if os.path.exists(table_path):
                    table = pd.read_csv(table_path)
                    tables.append(table)
            except Exception as e:
                print(f"Warning: Could not load reduction {reduction}% data: {e}")
    
    return networks, tables, baseline_data


def main():
    """Main plotting function."""
    parser = argparse.ArgumentParser(description="Generate plots for PyPSA-Eur thesis workflow")
    parser.add_argument("--config", required=True, help="Path to config.yaml file")
    parser.add_argument("--plot-types", required=True, help="Comma-separated list of plot types")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    parser.add_argument("--output-formats", required=True, help="Comma-separated list of output formats")
    parser.add_argument("--has-baseline", type=bool, default=True, help="Whether baseline scenario exists")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    plot_config = config.get("parameters", {}).get("plotting", {})
    
    # Setup output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse arguments
    plot_types = [pt.strip() for pt in args.plot_types.split(",")]
    output_formats = [fmt.strip() for fmt in args.output_formats.split(",")]
    dpi = plot_config.get("dpi", 300)
    
    # Setup matplotlib
    setup_matplotlib()
    
    # Load data (for backward compatibility with other plots)
    print("Loading networks and data...")
    networks, tables, baseline_data = load_networks_and_data(config, args.has_baseline)
    
    print(f"Loaded {len(networks)} networks and {len(tables)} data tables")
    
    # Generate requested plots
    plot_functions = {
        "renewable_generation_inequality": lambda: plot_renewable_capacity_inequality(config, output_path, output_formats, dpi, args.has_baseline),
        "green_investment_inequality": lambda: plot_green_investment_inequality(config, output_path, output_formats, dpi, args.has_baseline),
        "green_investment_inequality_load_scaled": lambda: plot_green_investment_inequality_load_scaled(config, output_path, output_formats, dpi, args.has_baseline),
        "renewable_capacity_concentration": lambda: plot_renewable_capacity_concentration(config, output_path, output_formats, dpi, args.has_baseline),
        "renewable_capacity_pentiles": lambda: plot_renewable_capacity_pentiles(config, output_path, output_formats, dpi, args.has_baseline),
        "middle_pentile_characteristics": lambda: plot_middle_pentile_characteristics(config, output_path, output_formats, dpi, args.has_baseline),
        "capacity_expansion_evolution": lambda: plot_capacity_expansion_evolution(config, output_path, output_formats, dpi, args.has_baseline),
        "transmission_bottlenecks": lambda: plot_transmission_bottlenecks(config, output_path, output_formats, dpi, args.has_baseline),
        "total_renewable_capacity": lambda: plot_total_renewable_capacity(config, output_path, output_formats, dpi, args.has_baseline),
        "electricity_cost": lambda: plot_electricity_cost(config, output_path, output_formats, dpi, args.has_baseline),
        "true_lcoe_with_sunk_costs": lambda: plot_true_lcoe_with_sunk_costs(config, output_path, output_formats, dpi, args.has_baseline),
        "marginal_price_vs_lcoe_explanation": lambda: plot_marginal_price_vs_lcoe_explanation(config, output_path, output_formats, dpi, args.has_baseline),
        "electricity_cost_comparison": lambda: plot_electricity_cost_comparison(config, output_path, output_formats, dpi, args.has_baseline),
        "demand_weighted_marginal_prices": lambda: plot_demand_weighted_marginal_prices(config, output_path, output_formats, dpi, args.has_baseline),
        "generation_mix_actual": lambda: plot_generation_mix_actual(config, output_path, output_formats, dpi, args.has_baseline),
        "renewable_penetration_boxplots": lambda: plot_renewable_penetration_boxplots(config, output_path, output_formats, dpi, args.has_baseline),
        "renewable_penetration_stacked_bars": lambda: plot_renewable_penetration_stacked_bars(config, output_path, output_formats, dpi, args.has_baseline),
        "renewable_penetration_gini_by_decarbonization": lambda: plot_renewable_penetration_gini_by_decarbonization(config, output_path, output_formats, dpi, args.has_baseline),
        "interregional_transmission_expansion": lambda: plot_interregional_transmission_expansion(config, output_path, output_formats, dpi, args.has_baseline),
        "storage_expansion_boxplots": lambda: plot_storage_expansion_boxplots(config, output_path, output_formats, dpi, args.has_baseline),
        "total_system_cost": lambda: plot_total_system_cost(config, output_path, output_formats, dpi, args.has_baseline),
        "mean_price_bellcurve": lambda: plot_mean_price_bellcurve(config, output_path, output_formats, dpi, args.has_baseline),
        "mean_price_boxplots": lambda: plot_mean_price_boxplots(config, output_path, output_formats, dpi, args.has_baseline),
        "network_maps": lambda: plot_network_maps(config, output_path, output_formats, dpi, args.has_baseline),
        "network_topology": lambda: plot_network_topology(networks, output_path, output_formats, dpi),
        "generation_mix": lambda: plot_generation_mix(networks, tables, output_path, output_formats, dpi),
        "transmission_flows": lambda: plot_transmission_flows(networks, output_path, output_formats, dpi),
        "storage_utilization": lambda: plot_storage_utilization(networks, output_path, output_formats, dpi),
        "emissions_summary": lambda: plot_emissions_summary(tables, baseline_data, output_path, output_formats, dpi),
        "cost_breakdown": lambda: plot_cost_breakdown(tables, output_path, output_formats, dpi)
    }
    
    for plot_type in plot_types:
        if plot_type in plot_functions:
            try:
                plot_functions[plot_type]()
                print(f"✓ Generated {plot_type} plot")
            except Exception as e:
                print(f"✗ Error generating {plot_type} plot: {e}")
        else:
            print(f"Warning: Unknown plot type '{plot_type}'")
    
    print(f"Plotting complete! Outputs saved to {output_path}")


if __name__ == "__main__":
    main()