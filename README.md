# alpaca-predict

> **🎉 Project Recently Reorganized!** The codebase has been restructured into a clean, modular package. See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the new organization and [docs/REORGANIZATION.md](docs/REORGANIZATION.md) for migration details.

This is an exploration into Agentic AI in the stock market featuring:
- **Hidden Markov Models (HMM)** for market regime detection
- **Market Causality DAG** using quantile Granger causality on 95 stocks
- **Causal Feature Engine** to extract DAG-based features for ML
- **Multi-agent orchestration** with conflict resolution
- **Backtesting framework** with vectorbt integration

The system uses HMM to predict the next day's market state and implements automated trading with configurable capital limits, strategies, and risk management. 


## Quick Start

### Setup

1. **Get Alpaca API Keys**: Create account at [alpaca.markets](https://alpaca.markets/)
2. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/alpaca-predict.git
   cd alpaca-predict
   ```
3. **Configure API Keys**: Copy keys to `.env` file
   ```bash
   echo "PAPER_KEY=your_api_key" >> .env
   echo "PAPER_SEC=your_secret_key" >> .env
   ```
4. **Create Virtual Environment**
   ```bash
   python -m virtualenv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```
5. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Bot

#### Option 1: Config-Driven (Recommended) ✨ NEW
```bash
# Edit configuration
nano config/trader_config.yaml

# Test without executing trades
python scripts/trader_with_config.py --dry-run

# Run live trading
python scripts/trader_with_config.py
```

#### Option 2: Original Script
```bash
python scripts/trader.py
```

**📖 Configuration Guides:**
- [Trader Config Guide](docs/TRADER_CONFIG_GUIDE.md) - Live trading configuration
- [Backtest Config Guide](docs/BACKTEST_CONFIG_GUIDE.md) - Backtesting configuration  
- [Quick Reference](docs/CONFIG_QUICK_REFERENCE.md) - All configs at a glance

### Backtesting

Test strategies before deploying:

```bash
# Edit backtest config
nano config/backtest_config.yaml

# Run backtest
python scripts/backtest_with_config.py

# Quick backtest (top 20 stocks)
python scripts/quick_backtest_tech.py
```

## Automation & Production Deployment

### Systemctl Service (Linux - Recommended)

Run the trading bot as a system service that starts automatically and restarts on failure:

```bash
# Test configuration
./setup_service.sh test

# Install service
sudo ./setup_service.sh install

# Start trading
sudo systemctl start alpaca-trader

# Check status
./setup_service.sh status
```

Full documentation: [docs/SERVICE_SETUP.md](docs/SERVICE_SETUP.md)

### Cron Jobs

Alternative scheduling options:

```bash
# Run trader daily at market open
crontab -e
# Add: 30 9 * * 1-5 /home/eli/alpaca-predict/scripts/run_trader.sh

# Monthly P&L report on 1st of month
# Add: 0 9 1 * * /home/eli/alpaca-predict/scripts/run_monthly_report.sh
```

### Note
This program uses websockets to run constantly, monitoring asset changes during market hours. The systemctl service provides automatic restarts, logging, and proper process management. 