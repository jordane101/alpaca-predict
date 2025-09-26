#!/bin/bash

# This script automates the execution of the monthly_pnl_reporter.py script.
# It's intended to be run by a cron job on the first day of each month.

# Set the absolute path to the project directory
PROJECT_DIR="/home/eli/alpaca-predict"

# Navigate to the project directory. This is crucial for the Python script
# to find the .env file and any other relative paths.
cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR" >&2; exit 1; }

# --- Optional: Activate Python Virtual Environment ---
# If your project uses a virtual environment (e.g., named 'venv'),
# uncomment the following line to activate it before running the script.
#
# source "$PROJECT_DIR/venv/bin/activate"

# Define the log file path. A new log is created for each run, named by date.
LOG_FILE="$PROJECT_DIR/reports/pnl_report_$(date +\%Y-\%m-\%d).log"

# Create the logs directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Execute the Python script and redirect all output (stdout and stderr) to the log file.
# The 'tee -a' command both prints to the log and to standard out if you run it manually.
echo "--- Starting Monthly P/L Report: $(date) ---" | tee -a "$LOG_FILE"
python3 "$PROJECT_DIR/monthly_pnl_reporter.py" | tee -a "$LOG_FILE"
echo "--- Finished: $(date) ---" | tee -a "$LOG_FILE"