# Thesis-workflow

This repository contains all scripts, dependencies, environments, and networks necessary to generate the results for the paper. It includes a full PyPSA-based energy system optimization workflow managed by Snakemake, solving 112 optimization jobs across centralized and decentralized CO₂ reduction scenarios using the Gurobi solver.

The workflow covers:
- **1 baseline solve** (unconstrained, r=0)
- **13 centralized CO₂ reduction scenarios** (r = 10, 20, 30, 40, 50, 60, 70, 80, 90, 93, 96, 99, 100%)
- **91 decentralized scenarios** (13 reduction levels × 7 load-weighted k-values)
- **25 result figures** generated automatically from solved networks

 **Runtime warning:** This workflow was executed on an HPC cluster using approximately **2,000 core-hours**. Running the full workflow locally is not feasible. This guide is provided for transparency and reproducibility verification. Reviewers wishing to inspect the methodology without running it can browse the `scripts/`, `src/`, and `workflow/` directories directly, and verify the workflow structure using the dry-run in Step 5.

---

## Prerequisites

Before starting, ensure you have the following installed:

- **Git** — [git-scm.com](https://git-scm.com)
- **Miniconda or Anaconda** — [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html)
- **Gurobi license** — Required for solving. Free academic licenses are available at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/). After obtaining a license, place your `gurobi.lic` file in your home directory (`~/` on Mac/Linux, `C:\Users\yourname\` on Windows).
- **Windows only — Microsoft C++ Build Tools** — Required to compile certain Python dependencies. Download from [visualstudio.microsoft.com/visual-cpp-build-tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), select **"Desktop development with C++"** during installation (~5 GB), and restart your computer afterwards.

---

## Local Setup

### Step 1 — Clone the repository

Open a terminal and navigate to a directory of your choice, then clone the repository:

```bash
cd /path/to/your/projects        # e.g. ~/Documents on Mac/Linux
                                 # or C:\Users\yourname\Documents on Windows
git clone https://github.com/KrisAden/Thesis-workflow
cd Thesis-workflow
```

---

### Step 2 — Set up the conda environment

**Option A — Create a fresh environment from this repository (recommended):**

```bash
conda env create -f environment.yml -n pypsa-thesis
```

**Option B — Update an existing environment:**

```bash
conda activate YOUR_ENV_NAME
conda env update -n YOUR_ENV_NAME -f environment.yml --prune
```

Installation may take 10–20 minutes depending on your internet connection.

**Windows users:** If installation fails on a package called `datrie` with an error about "Microsoft Visual C++ 14.0 or greater is required", install the C++ Build Tools listed in the prerequisites above, then retry.

---

### Step 3 — Activate the environment

```bash
conda activate pypsa-thesis
```

 **Troubleshooting — "Run 'conda init' before 'conda activate'":**
 This is common on a fresh conda installation. Run:
 ```bash
 conda init
 ```
 Then **close and reopen your terminal completely** and try activating again.

 **Troubleshooting — Windows PowerShell execution policy error:**
 If you see a message about scripts being disabled, run this once:
 ```powershell
 Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
 ```
 Then open a **new** PowerShell window and activate again.

---

### Step 4 — Fix a known dependency conflict

Due to a version incompatibility between `snakemake=7` and newer `pulp` releases, run this after environment creation:

```bash
pip install "pulp=2.8,<3"
```

Verify snakemake is working:

```bash
python -m snakemake --version
```

 **Note:** Snakemake is invoked as `python -m snakemake` throughout this guide. This is equivalent to the `snakemake` command but works reliably regardless of how your shell PATH is configured.

---

### Step 5 — Verify the workflow with a dry-run

From the repository root, run:

```bash
python -m snakemake -n -p
```

The `-n` flag performs a dry-run (no jobs are executed). A successful dry-run ends with this job summary:

```
Job stats:
job                              count
-----------------------------  -------
add_storage                          1
all                                  1
apply_costs                          1
enable_generator_expansion           1
enable_transmission_expansion        1
make_plots                           1
rescale_loads                        1
solve_baseline                       1
solve_decentralized                 91
solve_with_cap                      13
total                              112

This was a dry-run (flag -n). The order of jobs does not reflect the order of execution.
```

If you see exactly 112 jobs, the workflow is correctly configured on your machine.

---

### Step 6 — Run the workflow (HPC recommended)

To run the full workflow locally (not recommended — see runtime warning above):

```bash
python -m snakemake --cores 4
```

Replace `4` with the number of CPU cores available on your machine.

 **Memory note:** The solver is configured to use up to 120 GB of RAM (`MemLimit: 120000` in `config/config.yaml`). On a local machine, Gurobi will use whatever is available, but solve times will increase significantly. You can reduce this limit by editing that value in the config file.

To run a single scenario rather than the full workflow, specify the target output file explicitly:

```bash
python -m snakemake --cores 4 results/networks/solved_reduction_10.nc results/tables/solve_reduction_10.csv
```

---

## HPC Setup (CLAAUDIA / AAU)

The workflow was originally executed on the AAU HPC cluster. To reproduce the HPC run:

```bash
git clone https://github.com/KrisAden/Thesis-workflow
cd "/work/KristofferHedegaardAden#2272/Thesis-workflow"
source scripts/init_session.sh
```

To sync changes with the remote repository:

```bash
git push origin main   # push local changes
git pull origin main   # pull remote changes
```

When prompted for authentication, use your personal access token.

---

## Configuration

The workflow is controlled by `config/config.yaml`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `co2_reductions` | `[0,10,...,100]` | CO₂ reduction targets (%) to solve |
| `decentralization.k_values` | `[1,2,3,4,5,6,7]` | Load-weighted equity constraint values |
| `solve.solver` | `gurobi` | Solver (Gurobi required) |
| `solve.solver_options.Threads` | `4` | CPU threads per solve job |
| `solve.solver_options.MemLimit` | `120000` | Max RAM for Gurobi in MB (reduce for local runs) |
| `plotting.enable` | `true` | Whether to generate figures after solving |
| `preprocess.bypass` | `false` | Skip load rescaling (use base network as-is) |

---

## Output

After a successful run, results are written to:

| Directory | Contents |
|---|---|
| `results/networks/` | Solved PyPSA network files (`.nc`) for each scenario |
| `results/tables/` | Summary statistics and cost tables (`.csv`) |
| `results/plots/` | All 25 figures used in the thesis (`.png`) |

---

## Repository structure

```
Thesis-workflow/
├── config/          # config.yaml — workflow parameters and solver settings
├── data/
│   ├── raw/         # Input data (base network, cost CSVs, geographic data)
│   └── interim/     # Intermediate processed networks (generated during run)
├── results/
│   ├── networks/    # Solved network outputs
│   ├── tables/      # Result tables
│   └── plots/       # Figures
├── scripts/         # HPC session init and helper scripts
├── src/             # Python package (pypsa_thesis) — all model logic
├── workflow/        # Snakefile
├── environment.yml  # Full conda environment specification
├── baseline.nc      # Pre-computed baseline network
└── CITATION.cff     # Citation metadata
```

---

## Citation

If you use this workflow, please cite it using the information in `CITATION.cff`.
