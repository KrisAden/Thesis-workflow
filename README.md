# Thesis-workflow
This repository contains all scripts, dependencies, enviroments and networks necessary to generate the results for my Master's thesis

 ## Current status
 Loads precompiled 37-node network, reads country level electricity deman, and rescales to projected loads.
 Writes:
 data/interim/network_rescaled.nc
results/tables/load_scaling.csv

How to excecute:
 -Prerequisites, Git, conda (Anaconda or Miniconda)
Start by cloning the repo and then from repo root:

# Option A: create a fresh env from this repo's file
conda env create -f environment.yml -n pypsa-thesis
conda activate pypsa-thesis

# Option B: use your existing env  and update it
conda activate YOUR_env
conda env update -n YOUR_env -f environment.yml --prune

    Sidenote: If snakemake isn’t found afterwards, install it in your env:  pip install "snakemake>=7,<8"

Basis network and rescale values can be edited through config/config.yaml

To run the re-scaling workflow in repo root:
snakemake -s workflow/Snakefile --cores 1 -p

Outputs will appear in : data/interim/network_rescaled.nc, results/tables/load_scaling.csv


HPC Setup should hopefully be:
git clone https://github.com/KrisAden/Thesis-workflow
cd Thesis-workflow
bash scripts/install_env.sh           # auto-installs conda-lock if needed
conda activate pypsa-thesis
snakemake -s workflow/Snakefile --cores 1 -p

