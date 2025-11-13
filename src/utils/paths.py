"""
Path constants for the alpaca-predict project.
"""
from pathlib import Path

# Project root directory (go up from src/utils/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
CAUSALITY_CACHE_DIR = DATA_DIR / "causality_cache"
HMM_MODELS_DIR = DATA_DIR / "hmm_models"
CSV_DIR = DATA_DIR / "csv"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = DATA_DIR / "models"
OUTPUTS_DIR = DATA_DIR / "outputs"
REPORTS_DIR = DATA_DIR / "reports"

# Config directory
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure directories exist
for directory in [
    DATA_DIR, CAUSALITY_CACHE_DIR, HMM_MODELS_DIR, CSV_DIR,
    LOGS_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, CONFIG_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# Default file paths
DEFAULT_DAG_FILE = CAUSALITY_CACHE_DIR / "large_network_graph_dag.pkl"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
