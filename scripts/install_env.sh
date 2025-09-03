#!/usr/bin/env bash
set -euo pipefail

# Ensure 'conda activate' works in this shell without 'conda init'
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "conda not found. Load your module or install (mini)conda." >&2
  exit 1
fi

# Make sure conda-lock is available (install to user site if missing)
if ! command -v conda-lock >/dev/null 2>&1; then
  python -m pip install --user conda-lock
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create or update the environment from the lock file
ENV_NAME="${1:-pypsa-thesis}"
conda-lock install --name "$ENV_NAME" conda-lock.yml

# Show result
conda activate "$ENV_NAME"
python -c "import sys; print('Environment ready ->', sys.executable)"
