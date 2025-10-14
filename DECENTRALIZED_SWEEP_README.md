# Decentralized Sweep Implementation

## Overview

This implementation adds a new rule `solve_decentralized` to your PyPSA-EUR thesis workflow that studies the interaction between decarbonization level, total system cost, and forced decentralization.

## What's Been Added

### 1. Configuration Parameters (`config/config.yaml`)

Added new section under `parameters`:

```yaml
decentralization:
  k_values: [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]  # k values for 1/k < gamma < k constraint
  renewable_carriers: ["solar", "onwind", "offwind-ac", "offwind-dc", "ror"]  # carriers considered renewable
```

- `k_values`: List of k parameters for the decentralization constraint
- `renewable_carriers`: List of generator carriers considered as renewable for the gamma calculation

### 2. New Python Module (`src/pypsa_thesis/solve_decentralized.py`)

This module implements:

- **Nodal renewable penetration constraints**: For each bus with load, constrains renewable penetration γ to be within [1/k, k]
- **Gamma calculation**: γ = renewable_generation / nodal_load for each node
- **Individual node constraints**: Each bus gets its own constraint based on its specific load
- **Integration with existing CO₂ constraints**: Works alongside the global CO₂ cap

### 3. New Snakemake Rule (`workflow/Snakefile`)

The `solve_decentralized` rule:

- **Input**: Same as `solve_with_cap` (processed network + baseline emissions)
- **Output**: Networks and reports for each (reduction, k) combination
- **Parameters**: Both CO₂ reduction level and k-value
- **Execution**: For every decarbonization level and every k-value

## How It Works

### The Constraint

For each bus `i` with load, the constraint ensures:

```
1/k ≤ γᵢ ≤ k
```

Where:
- γᵢ = (renewable generation at bus i) / (load at bus i)
- k is the decentralization parameter

### Implementation Details

1. **Bus-level analysis**: Each bus with electrical load gets individual constraints
2. **Renewable identification**: Uses the configured `renewable_carriers` list
3. **Energy calculation**: Uses snapshot weightings to convert power to energy
4. **Constraint formulation**: Adds linear constraints to the optimization model

### Mathematical Formulation

For each bus `i`:

```
renewable_energy_i ≥ load_energy_i / k     (lower bound)
renewable_energy_i ≤ k × load_energy_i     (upper bound)
```

Where:
- `renewable_energy_i = Σₜ Σ₍gens at bus i₎ P_g,t × weight_t`
- `load_energy_i = Σₜ Σ₍loads at bus i₎ P_l,t × weight_t`

## Output Files

For each combination of reduction `r` and k-value `k`:

- **Network**: `results/networks/decentralized_reduction_{r}_k_{k}.nc`
- **Report**: `results/tables/decentralized_reduction_{r}_k_{k}.csv`

The report includes:
- Optimization status and objective value
- CO₂ emissions (allowed vs actual)  
- Average gamma across all buses
- Number of buses satisfying the gamma constraints

## Usage

### Running the Workflow

```bash
# Dry run to check everything
snakemake -n

# Execute with multiple cores
snakemake --cores 4

# Run only decentralized sweep
snakemake --cores 4 results/networks/decentralized_reduction_50_k_2.0.nc

# Test with less restrictive constraint first
snakemake --cores 4 results/networks/decentralized_reduction_50_k_5.0.nc
```

## Troubleshooting

### Common Issues

1. **KeyError 'Load-p'**: Fixed in the implementation - the constraint formulation now uses the correct variable access pattern
2. **Logging format errors**: Fixed - removed comma formatting from log messages  
3. **k=1.0 may be infeasible**: For 100% decarbonization, k=1.0 (perfect self-sufficiency) might be impossible. Try larger k values first.

### Testing Strategy

Start with less restrictive constraints:

```bash
# Test with high decentralization flexibility first
snakemake --cores 4 results/networks/decentralized_reduction_50_k_5.0.nc

# Then try more restrictive
snakemake --cores 4 results/networks/decentralized_reduction_50_k_2.0.nc

# Finally test very restrictive (may be infeasible)
snakemake --cores 4 results/networks/decentralized_reduction_100_k_1.0.nc
```

### Expected Behavior

- **k=1.0**: Forces perfect renewable self-sufficiency (γ=1) at every bus
- **k>1.0**: Allows increasing imbalance between local generation and consumption
- **Higher k**: More centralized solutions (some regions import, others export)
- **Lower k**: More decentralized solutions (each region closer to self-sufficient)

## Analysis Possibilities

With this implementation, you can study:

1. **Cost of decentralization**: How does total system cost change with k?
2. **Feasibility boundaries**: What's the minimum k for each decarbonization level?
3. **Spatial patterns**: Which regions become exporters/importers under different k values?
4. **Technology deployment**: How does renewable capacity distribution change with k?
5. **Transmission needs**: How does grid expansion change with decentralization level?

## Files Modified/Created

- ✅ `config/config.yaml` - Added decentralization parameters
- ✅ `src/pypsa_thesis/solve_decentralized.py` - New module for decentralized solving
- ✅ `workflow/Snakefile` - Added new rule and updated rule all
- ✅ `test_decentralized_implementation.py` - Validation script

## Next Steps

1. **Test the implementation**: Run the test script to validate setup
2. **Run a small test**: Try one decarbonization level with one k-value first
3. **Analyze results**: Create plots showing cost vs k for different reduction levels
4. **Extend analysis**: Add post-processing rules for visualization and analysis