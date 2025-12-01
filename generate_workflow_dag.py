"""
Generate a visual DAG of the Snakemake workflow
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import yaml
from pathlib import Path
import os

# Change to the script's directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Read config to get parameters
config_path = script_dir / "config" / "config.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

REDUCTIONS = cfg.get("parameters", {}).get("co2_reductions", [0])
K_VALUES = cfg.get("parameters", {}).get("decentralization", {}).get("k_values", [1.0])
HAS_BASELINE = any(float(r) == 0 for r in REDUCTIONS)
REDUC_GT0 = [r for r in REDUCTIONS if float(r) > 0]

# Define data sources (inputs)
data_sources = {
    "base_network": {
        "path": cfg.get("paths", {}).get("base_network", "data/raw/base_network.nc"),
        "type": "PyPSA Network",
        "color": "#FFE6E6",
        "shape": "data"
    },
    "generator_costs": {
        "path": "data/raw/generator_costs.csv",
        "type": "Cost Data",
        "color": "#FFE6E6",
        "shape": "data"
    },
    "storage_costs": {
        "path": "data/raw/storage_costs.csv",
        "type": "Cost Data",
        "color": "#FFE6E6",
        "shape": "data"
    }
}

# Define the workflow structure
workflow = {
    "rescale_loads": {
        "outputs": ["network_rescaled.nc", "load_scaling.csv"],
        "dependencies": ["base_network"],
        "color": "#E8F4F8"
    },
    "apply_costs": {
        "outputs": ["network_costed.nc", "generator_capital_costs.csv"],
        "dependencies": ["rescale_loads", "generator_costs"],
        "color": "#D4E9F7"
    },
    "enable_transmission_expansion": {
        "outputs": ["network_costed_tx.nc", "tx_expansion_bounds.csv"],
        "dependencies": ["apply_costs"],
        "color": "#B8D8EB"
    },
    "enable_generator_expansion": {
        "outputs": ["network_costed_tx_gen.nc", "gen_expansion_bounds.csv"],
        "dependencies": ["enable_transmission_expansion"],
        "color": "#9CC7E0"
    },
    "add_storage": {
        "outputs": ["network_costed_tx_gen_sto.nc", "storage_costs_applied.csv"],
        "dependencies": ["enable_generator_expansion", "storage_costs"],
        "color": "#7FB3D5"
    },
}

# Add conditional baseline solving
if HAS_BASELINE:
    workflow["solve_baseline"] = {
        "outputs": ["solved_baseline_costed_expansion.nc", 
                   "baseline_emissions.csv"],
        "dependencies": ["add_storage"],
        "color": "#FFF4E6",
        "label": "solve_baseline\n(C=0%)"
    }
    
# Add reduction solving
workflow["solve_with_cap"] = {
    "outputs": [f"solved_reduction_{r}.nc" for r in REDUC_GT0[:2]],  # Show first 2
    "dependencies": ["add_storage", "solve_baseline"] if HAS_BASELINE else ["add_storage"],
    "color": "#FFE5CC",
    "label": f"solve_with_cap\n(C>0%, n={len(REDUC_GT0)})"
}

# Add decentralization
workflow["solve_decentralized"] = {
    "outputs": [f"decentralized_reduction_{{r}}_k_{{k}}.nc"],
    "dependencies": ["add_storage", "solve_baseline"] if HAS_BASELINE else ["add_storage"],
    "color": "#FFD6B3",
    "label": f"solve_decentralized\n(C>0%, K∈{K_VALUES})"
}

# Add plotting
workflow["make_plots"] = {
    "outputs": ["plots/*.png"],
    "dependencies": ["solve_with_cap"],
    "color": "#C8E6C9"
}

# Create figure
fig, ax = plt.subplots(figsize=(14, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13.5)
ax.axis('off')

# Position nodes (data sources at top)
positions = {
    # Data sources
    "base_network": (3, 12.5),
    "generator_costs": (5, 12.5),
    "storage_costs": (7, 12.5),
    # Processing pipeline
    "rescale_loads": (3, 11),
    "apply_costs": (5, 9.5),
    "enable_transmission_expansion": (5, 8),
    "enable_generator_expansion": (5, 6.5),
    "add_storage": (5, 5),
}

if HAS_BASELINE:
    positions["solve_baseline"] = (2.5, 3.5)
    positions["solve_with_cap"] = (5, 2)
    positions["solve_decentralized"] = (7.5, 2)
    positions["make_plots"] = (5, 0.5)
else:
    positions["solve_with_cap"] = (4.5, 2.5)
    positions["solve_decentralized"] = (7, 2.5)
    positions["make_plots"] = (5.75, 1)

# Draw data source nodes (cylinders/parallelograms for data)
data_width = 1.8
data_height = 0.6

for source_name, source_data in data_sources.items():
    x, y = positions[source_name]
    
    # Draw parallelogram for data
    from matplotlib.patches import Polygon
    offset = 0.15
    vertices = [
        (x - data_width/2 + offset, y - data_height/2),
        (x + data_width/2 + offset, y - data_height/2),
        (x + data_width/2 - offset, y + data_height/2),
        (x - data_width/2 - offset, y + data_height/2)
    ]
    
    poly = Polygon(vertices, 
                   facecolor=source_data["color"],
                   edgecolor='#CC0000',
                   linewidth=2,
                   linestyle='--')
    ax.add_patch(poly)
    
    # Add text
    label = source_name.replace('_', '\n')
    ax.text(x, y, label, 
            ha='center', va='center',
            fontsize=8, fontweight='bold',
            style='italic')

# Draw processing nodes
node_width = 2.2
node_height = 0.8

for rule_name, data in workflow.items():
    x, y = positions[rule_name]
    label = data.get("label", rule_name)
    
    # Draw box
    box = FancyBboxPatch(
        (x - node_width/2, y - node_height/2),
        node_width, node_height,
        boxstyle="round,pad=0.1",
        facecolor=data["color"],
        edgecolor='#333333',
        linewidth=2
    )
    ax.add_patch(box)
    
    # Add text
    ax.text(x, y, label, 
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            wrap=True)

# Draw edges
def draw_arrow(from_rule, to_rule, offset=0):
    x1, y1 = positions[from_rule]
    x2, y2 = positions[to_rule]
    
    # Calculate start and end points
    start_y = y1 - node_height/2
    end_y = y2 + node_height/2
    
    arrow = FancyArrowPatch(
        (x1 + offset, start_y),
        (x2 + offset, end_y),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color='#555555',
        linewidth=1.5,
        connectionstyle="arc3,rad=0.0" if offset == 0 else f"arc3,rad={offset*0.1}"
    )
    ax.add_patch(arrow)

# Draw all dependencies
for rule_name, data in workflow.items():
    for dep in data["dependencies"]:
        if dep in positions:
            offset = 0
            # Add offset for multiple connections
            if rule_name in ["solve_with_cap", "solve_decentralized"] and dep == "add_storage":
                offset = -0.3 if rule_name == "solve_with_cap" else 0.3
            
            # Use dashed line for data source connections
            if dep in data_sources:
                x1, y1 = positions[dep]
                x2, y2 = positions[rule_name]
                start_y = y1 - data_height/2
                end_y = y2 + node_height/2
                
                arrow = FancyArrowPatch(
                    (x1 + offset, start_y),
                    (x2 + offset, end_y),
                    arrowstyle='->,head_width=0.4,head_length=0.4',
                    color='#CC0000',
                    linewidth=1.5,
                    linestyle='--',
                    connectionstyle="arc3,rad=0.0" if offset == 0 else f"arc3,rad={offset*0.1}"
                )
                ax.add_patch(arrow)
            else:
                draw_arrow(dep, rule_name, offset)

# Add title
ax.text(5, 13.2, 'PyPSA Thesis Workflow DAG', 
        ha='center', va='bottom',
        fontsize=16, fontweight='bold')

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    mpatches.Patch(facecolor='#FFE6E6', edgecolor='#CC0000', linestyle='--', label='Data Sources'),
    mpatches.Patch(facecolor='#E8F4F8', edgecolor='#333', label='Preprocessing'),
    mpatches.Patch(facecolor='#FFE5CC', edgecolor='#333', label='Optimization'),
    mpatches.Patch(facecolor='#C8E6C9', edgecolor='#333', label='Visualization'),
    Line2D([0], [0], color='#CC0000', linestyle='--', linewidth=1.5, label='Data Flow'),
    Line2D([0], [0], color='#555555', linewidth=1.5, label='Process Flow')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

# Add parameter info
param_text = f"Parameters:\n• CO₂ reductions: {len(REDUCTIONS)} levels\n"
param_text += f"• Baseline (C=0%): {'Yes' if HAS_BASELINE else 'No'}\n"
param_text += f"• Decentralization K values: {len(K_VALUES)}"
ax.text(0.5, 0.3, param_text, 
        fontsize=8, va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('workflow_dag.png', dpi=300, bbox_inches='tight')
plt.savefig('workflow_dag.pdf', bbox_inches='tight')
print("✓ Generated workflow_dag.png")
print("✓ Generated workflow_dag.pdf")

# Also create a detailed rule graph
fig2, ax2 = plt.subplots(figsize=(16, 12))
ax2.axis('off')

# Create a more detailed view
y_pos = 11
x_left = 2
x_right = 8

ax2.text(5, y_pos + 0.5, 'Detailed Workflow Structure', 
         ha='center', fontsize=14, fontweight='bold')

# First add data sources
ax2.text(x_left + 0.25, y_pos - 0.3, "DATA SOURCES", 
         fontsize=10, fontweight='bold', va='center', color='#CC0000')

for i, (source_name, source_data) in enumerate(data_sources.items()):
    y = y_pos - 0.8 - i*0.7
    
    # Source name
    ax2.add_patch(FancyBboxPatch(
        (x_left - 1, y - 0.25),
        2.5, 0.5,
        boxstyle="round,pad=0.05",
        facecolor=source_data["color"],
        edgecolor='#CC0000',
        linewidth=1.5,
        linestyle='--'
    ))
    ax2.text(x_left + 0.25, y, source_name,
             fontsize=8, fontweight='bold', va='center', style='italic')
    
    # Path
    ax2.text(x_right, y, source_data["path"],
             fontsize=7, va='center', family='monospace', color='#666')

# Add processing rules
rule_start_y = y_pos - 0.8 - len(data_sources)*0.7 - 0.8
ax2.text(x_left + 0.25, rule_start_y, "PROCESSING RULES", 
         fontsize=10, fontweight='bold', va='center', color='#333')

for i, (rule_name, data) in enumerate(workflow.items()):
    y = rule_start_y - 0.6 - i*1.2
    color = data["color"]
    
    # Rule name
    ax2.add_patch(FancyBboxPatch(
        (x_left - 1, y - 0.3),
        2.5, 0.6,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor='black',
        linewidth=1.5
    ))
    ax2.text(x_left + 0.25, y, rule_name,
             fontsize=9, fontweight='bold', va='center')
    
    # Outputs
    outputs_text = "\n".join([f"  • {out}" for out in data["outputs"][:3]])
    if len(data["outputs"]) > 3:
        outputs_text += f"\n  • ... ({len(data['outputs'])} total)"
    
    ax2.text(x_right, y, outputs_text,
             fontsize=7, va='center', family='monospace')
    
    # Dependencies
    if data["dependencies"]:
        deps_text = f"← {', '.join(data['dependencies'])}"
        ax2.text(x_left + 0.25, y - 0.5, deps_text,
                fontsize=7, va='top', style='italic', color='#666')

ax2.set_xlim(0, 12)
ax2.set_ylim(-3, 13)

# Add column headers
ax2.text(x_left + 0.25, y_pos + 0.2, "Rule", fontsize=10, fontweight='bold')
ax2.text(x_right, y_pos + 0.2, "Outputs", fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('workflow_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Generated workflow_detailed.png")

print(f"\n📊 Workflow Statistics:")
print(f"   Total rules: {len(workflow)}")
print(f"   Preprocessing steps: 5")
print(f"   Optimization runs: {len(REDUC_GT0)} baseline + {len(REDUC_GT0) * len(K_VALUES)} decentralized")
print(f"   Total network files: {len(REDUC_GT0) * (1 + len(K_VALUES)) + (1 if HAS_BASELINE else 0)}")

plt.show()
