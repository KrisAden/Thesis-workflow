# Thesis-workflow
This repository contains all scripts, dependencies, enviroments and networks necessary to generate the results for my Master's thesis

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

To run the workflow for constrained runs: Update Config to include the correct levels. Then
snakemake -s workflow/Snakefile -p --cores 4 results/networks/solved_reduction_XX.nc results/tables/solve_reduction_XX.csv


Outputs will appear in : data/interim/network_rescaled.nc, results/tables/load_scaling.csv


HPC Setup is:
git clone https://github.com/KrisAden/Thesis-workflow


cd "/work/KristofferHedegaardAden#2272/Thesis-workflow"
source scripts/init_session.sh

to push and pull git push/pull orginin main and when prompted paste token from sticky note

