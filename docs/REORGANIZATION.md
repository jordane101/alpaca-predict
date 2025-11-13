# Project Reorganization Complete! 🎉

## Summary

The alpaca-predict project has been successfully reorganized into a clean, modular structure. All imports have been updated, and the code is now properly organized into logical modules.

## New Directory Structure

```
alpaca-predict/
├── src/                          # Source code
│   ├── causality/               # Causality analysis
│   │   ├── market_causality_dag.py
│   │   ├── causal_feature_engine.py
│   │   └── __init__.py
│   ├── hmm/                     # Hidden Markov Models
│   │   ├── hmm_analysis.py
│   │   └── __init__.py
│   ├── trading/                 # Trading strategies & agents
│   │   ├── strategies.py
│   │   ├── trading_agent.py
│   │   ├── orchestrator.py
│   │   ├── position.py
│   │   └── __init__.py
│   ├── backtest/                # Backtesting framework
│   │   ├── backtester.py
│   │   ├── optimizer.py
│   │   └── __init__.py
│   ├── api/                     # Flask API
│   │   ├── api.py
│   │   └── __init__.py
│   ├── utils/                   # Utilities & constants
│   │   ├── paths.py            # Centralized path management
│   │   └── __init__.py
│   └── __init__.py
├── scripts/                      # Executable scripts
│   ├── trader.py                # Main trading bot
│   ├── build_large_dag.py       # Build causality DAG
│   ├── test_quantile_causality.py
│   ├── backtest_causality_comparison.py
│   ├── run_trader.sh            # Cron job script
│   └── run_monthly_report.sh    # Monthly report script
├── data/                        # Data storage
│   ├── causality_cache/         # Cached causality matrices
│   ├── hmm_models/              # Trained HMM models
│   ├── csv/                     # CSV data files
│   ├── logs/                    # Application logs
│   ├── models/                  # Other model files
│   ├── outputs/                 # Generated outputs
│   └── reports/                 # Generated reports
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── config/                      # Configuration files
├── pyproject.toml               # Package configuration
├── requirements.txt             # Dependencies
├── .env                         # Environment variables
└── README.md                    # This file
```

## Key Changes

### 1. Module Organization
- **src/causality/**: All causality-related code
  - `MarketCausalityDAG`: Build and analyze market causality networks
  - `CausalFeatureEngine`: Extract features from DAG for ML

- **src/hmm/**: HMM analysis
  - `AnalyzeHMM`: Train and use HMM models
  - `setup_logging()`: Configure logging

- **src/trading/**: Trading logic
  - `BaseStrategy`, `HMMStrategy`, `DonchianBreakoutStrategy`
  - `TradingAgent`: Manages trading for individual strategies
  - `Orchestrator`: Coordinates multiple agents
  - `Position`: Position management classes

- **src/backtest/**: Backtesting
  - `Backtester`: Vectorized backtesting with vectorbt
  - `Optimizer`: Parameter optimization

- **src/api/**: Web API
  - Flask API for portfolio data

- **src/utils/**: Shared utilities
  - `paths.py`: Centralized path constants

### 2. Path Management
All paths are now managed centrally in `src/utils/paths.py`:

```python
from src.utils.paths import (
    PROJECT_ROOT,           # /home/eli/alpaca-predict
    DATA_DIR,               # /home/eli/alpaca-predict/data
    CAUSALITY_CACHE_DIR,    # /home/eli/alpaca-predict/data/causality_cache
    HMM_MODELS_DIR,         # /home/eli/alpaca-predict/data/hmm_models
    OUTPUTS_DIR,            # /home/eli/alpaca-predict/data/outputs
    DEFAULT_DAG_FILE,       # Default DAG file path
)
```

### 3. Import Updates
All imports have been updated to use the new structure:

**Old:**
```python
from hmm_analysis import AnalyzeHMM
from market_causality_dag import MarketCausalityDAG
from strategies import HMMStrategy
```

**New:**
```python
from src.hmm.hmm_analysis import AnalyzeHMM
from src.causality.market_causality_dag import MarketCausalityDAG
from src.trading.strategies import HMMStrategy
```

Or use package imports:
```python
from src.hmm import AnalyzeHMM
from src.causality import MarketCausalityDAG, CausalFeatureEngine
from src.trading import HMMStrategy, TradingAgent, Orchestrator
from src.backtest import Backtester, Optimizer
```

### 4. Script Updates
All scripts in `scripts/` have been updated with proper imports:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hmm.hmm_analysis import AnalyzeHMM
```

### 5. Data Organization
All data is now under `data/`:
- `data/causality_cache/` - Causality matrices and DAGs
- `data/hmm_models/` - Trained HMM models
- `data/logs/` - Application logs
- `data/outputs/` - Generated visualizations
- `data/reports/` - Monthly P/L reports

## Usage

### Running Scripts
Scripts automatically add the project root to Python path:

```bash
# From project root
python3 scripts/trader.py
python3 scripts/build_large_dag.py
python3 scripts/test_quantile_causality.py
```

### Using Modules
Import modules from anywhere:

```python
# In any script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Then import normally
from src.causality import CausalFeatureEngine
from src.hmm import AnalyzeHMM
from src.trading import HMMStrategy
```

### Shell Scripts
Updated to use new paths:

```bash
./scripts/run_trader.sh          # Runs scripts/trader.py
./scripts/run_monthly_report.sh  # Runs scripts/monthly_pnl_reporter.py
```

## Testing

Run the test script to verify all imports:

```bash
python3 test_reorganization.py
```

This will test:
- ✓ All module imports
- ✓ Cross-module dependencies
- ✓ Path configuration
- ✓ Package structure

## Notes

1. **NumPy Compatibility**: If you see "numpy.dtype size changed" errors, rebuild numpy:
   ```bash
   pip install --no-binary :all: --force-reinstall numpy
   ```

2. **Virtual Environment**: The project was designed to work with or without a virtual environment. Scripts use `sys.path.insert(0, ...)` to ensure imports work.

3. **Backwards Compatibility**: Old cache files in `causality_cache/` and `hmm_models/` at the root have been moved to `data/causality_cache/` and `data/hmm_models/`. The code automatically uses the new paths.

4. **Package Installation**: For development, you can install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Benefits

✓ **Clean separation of concerns**: Each module has a clear purpose  
✓ **Easier navigation**: Find code quickly based on functionality  
✓ **Better imports**: Clear, hierarchical import structure  
✓ **Centralized paths**: No more hardcoded paths scattered throughout  
✓ **Test-friendly**: Easy to test individual modules  
✓ **Scalable**: Easy to add new modules without cluttering root  
✓ **Professional structure**: Follows Python best practices  

## Next Steps

1. Continue with HMM-DAG integration (add causal features to HMM training)
2. Test backtesting with new structure
3. Add unit tests in `tests/` directory
4. Consider adding a `setup.py` for pip installation
5. Update documentation in `docs/` as features evolve
