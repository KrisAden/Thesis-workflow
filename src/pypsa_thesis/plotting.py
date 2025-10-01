#!/usr/bin/env python3
"""
Plotting module for PyPSA-Eur thesis workflow.

This module generates various plots based on the configuration settings in config.yaml.
"""

import argparse
import yaml
import os
import sys
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
        "total_renewable_capacity": lambda: plot_total_renewable_capacity(config, output_path, output_formats, dpi, args.has_baseline),
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