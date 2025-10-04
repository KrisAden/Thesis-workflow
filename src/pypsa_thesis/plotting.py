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


def plot_renewable_capacity_inequality(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot Gini coefficient of installed renewable capacity inequality across scenarios."""
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
            region_caps = extract_installed_renewable_capacities(net)
            gini = gini_coefficient(region_caps.values)
            hhi = hhi_index(region_caps.values)
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
    ax1.set_title("Inequality of Installed Renewable Capacity", **font)

    # Style the plot
    ax1.grid(True, alpha=0.3)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, loc="upper left")

    plt.tight_layout()

    # Save in all requested formats
    for fmt in output_formats:
        output_file = output_path / f"renewable_capacity_inequality.{fmt}"
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


def plot_electricity_cost(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot electricity cost (€/MWh) across CO₂ reduction scenarios."""
    print("Creating electricity cost plot...")
    
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
            # Extract total system cost and divide by total load to get cost per MWh
            total_cost = net.objective  # Total system cost
            total_load = net.loads_t.p.sum().sum()  # Total load in MWh
            electricity_cost = total_cost / total_load if total_load > 0 else np.nan
            electricity_costs[co2_pct] = electricity_cost
            print(f"  ✓ Electricity cost for {co2_pct}% reduction: {electricity_cost:.2f} €/MWh")
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
                               xytext=(5,5), ha='left', fontsize=10, color='purple', 
                               fontweight='bold')

    ax.set_ylabel("Mean Interregional Transmission Expansion (MW)", **font)
    ax.set_xlabel("CO₂ Reduction Level", **font)
    ax.set_title("Interregional Transmission Expansion by Region", **font)
    ax.legend(loc="lower right")
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

    # Overlay the mean as a red line and the median as a green line for each box
    for i, data in enumerate(boxplot_data):
        if len(data) > 0:
            mean_val = np.mean(data)
            median_val = np.median(data)
            ax.plot([i+1-0.2, i+1+0.2], [mean_val, mean_val], color='red', linewidth=2, 
                    label='Mean' if i == 0 else "")
            ax.plot([i+1-0.2, i+1+0.2], [median_val, median_val], color='darkgreen', linewidth=2, 
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
        print(f"  ✓ Saved plot as {output_file}")

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
        print(f"  ✓ Saved plot as {output_file}")

    plt.close(fig)


def collect_marginal_prices_by_level(networks_by_percent):
    """Collect marginal prices from all networks and organize by CO2 reduction level."""
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


def calculate_mean_prices_by_level(df_marginal_prices):
    """Calculate mean marginal price per region for each decarbonization level."""
    # Remove the 'CO2_Level' column if it's present in the columns (should only be in the index)
    if "CO2_Level" in df_marginal_prices.columns:
        df_marginal_prices = df_marginal_prices.drop(columns=["CO2_Level"])
    
    # Group by CO₂ level (first index), then take mean across all snapshots for each region
    mean_prices_by_level = df_marginal_prices.groupby(level="CO2_Level").mean()
    
    return mean_prices_by_level


def plot_mean_price_bellcurve(config, output_path, output_formats, dpi=300, has_baseline=True):
    """Plot mean marginal prices by region in bell curve arrangement for each CO₂ reduction level."""
    print("Creating mean price bell curve plots...")
    
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
        # Create a simple summary plot or copy the first level's plot
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
    """
    print("Creating mean price boxplots...")
    
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
    try:
        # Try the old method first for backward compatibility
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        europe = world[world['continent'] == 'Europe']
        return europe
    except (AttributeError, Exception):
        try:
            # Create a simple Europe bounding box as fallback
            print("  → Using simplified Europe bounding box for maps")
            # Create a simple rectangular background for Europe
            from shapely.geometry import Polygon
            
            # Europe bounding box coordinates
            europe_bounds = Polygon([(-15, 35), (35, 35), (35, 72), (-15, 72), (-15, 35)])
            europe_gdf = gpd.GeoDataFrame([1], geometry=[europe_bounds], crs="EPSG:4326")
            return europe_gdf
        except Exception as e:
            print(f"  ⚠️ Could not load map data: {e}")
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
              bbox_to_anchor=(1.22, 1))
    
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
    
    if not pypsa:
        print("PyPSA not available - skipping network topology plot")
        return
    
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
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
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
        "renewable_capacity_inequality": lambda: plot_renewable_capacity_inequality(config, output_path, output_formats, dpi, args.has_baseline),
        "green_investment_inequality": lambda: plot_green_investment_inequality(config, output_path, output_formats, dpi, args.has_baseline),
        "total_renewable_capacity": lambda: plot_total_renewable_capacity(config, output_path, output_formats, dpi, args.has_baseline),
        "electricity_cost": lambda: plot_electricity_cost(config, output_path, output_formats, dpi, args.has_baseline),
        "generation_mix_actual": lambda: plot_generation_mix_actual(config, output_path, output_formats, dpi, args.has_baseline),
        "renewable_penetration_boxplots": lambda: plot_renewable_penetration_boxplots(config, output_path, output_formats, dpi, args.has_baseline),
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