#!/usr/bin/env bash
# One-command setup: creates a virtual environment and installs everything
# needed to run the simulation. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

echo "Installing dependencies..."
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ""
echo "Setup complete. To run the live visual demo:"
echo ""
echo "    source venv/bin/activate"
echo "    python3 run.py --n_particles 20 --p_driv 0.004"
echo ""
echo "(close the window, or Ctrl+C in the terminal, to stop)"
