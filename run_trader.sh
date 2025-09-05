#!/bin/bash

# This script automates the execution of the trading bot.
# It should be scheduled to run via a cron job.
set -e
# --- Configuration ---
# The absolute path to your project directory
PROJECT_DIR="/home/eli/alpaca-predict"
# The name of your Python virtual environment directory
VENV_DIR=".venv"
# -------------------

echo "--- Starting Trader cron job at $(date) ---"

# Navigate to the project directory. The '|| exit' will stop the script if the directory doesn't exist.
cd "$PROJECT_DIR" || exit

# Activate the Python virtual environment
source "$VENV_DIR/bin/activate"

# Ensure the log directory exists
mkdir -p "$PROJECT_DIR/logs"

# Run the Python script and redirect all output (stdout and stderr) to a dated log file
python trader.py >> "$PROJECT_DIR/logs/trader_log_$(date +\%Y-\%m-\%d).log" 2>&1

echo "--- Finished Trader cron job at $(date) ---"
