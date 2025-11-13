# Configuration System Summary

**Date:** November 13, 2025  
**Version:** v2.1 - Config-Driven Trading & Backtesting

## Overview

The alpaca-predict system now supports **fully configurable trading and backtesting** through YAML configuration files. No code changes needed to adjust parameters!

## What's New

### ✨ Configuration Files Created

1. **`config/backtest_config.yaml`** (144 lines)
   - Portfolio parameters (capital, positions, rebalancing)
   - Strategy configurations (2-state, 4-state, causal, etc.)
   - HMM settings (confidence, position sizing, shorts)
   - Universe selection (top 20 or full DAG)
   - Output options

2. **`config/trader_config.yaml`** (314 lines)
   - Multi-agent configuration
   - Trading schedule (cron-style)
   - Risk management (stops, position limits, daily caps)
   - Strategy settings per agent
   - API configuration
   - Monitoring & alerts

### 📜 New Scripts

1. **`scripts/backtest_with_config.py`** (393 lines)
   - Reads from `backtest_config.yaml`
   - Runs portfolio backtests
   - Compares multiple strategies
   - Saves results automatically
   - Usage: `python scripts/backtest_with_config.py [--config file.yaml]`

2. **`scripts/trader_with_config.py`** (283 lines)
   - Reads from `trader_config.yaml`
   - Runs live trading bot
   - Supports multiple agents
   - Dry-run mode for testing
   - Usage: `python scripts/trader_with_config.py [--dry-run]`

### 📚 Documentation Created

1. **`docs/BACKTEST_CONFIG_GUIDE.md`** (360 lines)
   - Complete backtest configuration guide
   - Parameter explanations
   - Example configurations (aggressive, conservative, balanced)
   - Performance impact analysis
   - Troubleshooting tips

2. **`docs/TRADER_CONFIG_GUIDE.md`** (495 lines)
   - Complete live trading configuration guide
   - Pre-made configurations for different goals
   - Parameter deep dive with tables
   - Safety warnings and checklist
   - Advanced features (causal, confidence sizing, shorts)

3. **`docs/CONFIG_QUICK_REFERENCE.md`** (264 lines)
   - Quick reference for all configs
   - Common use cases
   - Parameter impact matrix
   - Safety checklist
   - Pro tips

4. **Updated `README.md`**
   - Added "Quick Start" section
   - Links to new config guides
   - Simplified setup instructions

## Key Features

### 🎯 Easy Parameter Adjustment

**Before (Code Changes Required):**
```python
# Had to edit scripts/trader.py
strategy = HMMStrategy(
    n_components=2,
    use_causal_features=False,
    # ... etc
)
```

**After (Config File):**
```yaml
# Just edit config/trader_config.yaml
strategy:
  n_components: 4
  use_causal_features: true
  min_confidence: 0.6
```

### 🔄 Multiple Configurations

Create different config files for different scenarios:
- `config/aggressive.yaml` - High risk/high reward
- `config/conservative.yaml` - Capital preservation
- `config/testing.yaml` - Experimental strategies

Switch between them:
```bash
python scripts/trader_with_config.py --config aggressive.yaml
```

### 🧪 Safe Testing

Built-in dry-run mode:
```bash
# Test without executing trades
python scripts/trader_with_config.py --dry-run
```

### 📊 Multi-Agent Support

Run multiple strategies simultaneously:
```yaml
agents:
  - name: "Causal_Agent"
    total_allocation_pct: 0.60
    strategy:
      use_causal_features: true
      
  - name: "Technical_Agent"
    total_allocation_pct: 0.30
    strategy:
      use_causal_features: false
```

## Configuration Highlights

### Backtest Config (Most Important Settings)

```yaml
# Time period
backtest:
  years: 2

# Portfolio
portfolio:
  initial_capital: 100000
  max_positions: 10
  rebalance_days: 63        # Quarterly

# Strategy
strategies:
  - name: "4-State Causal + Confidence"
    enabled: true
    n_components: 4
    use_causal_features: true
    optimize_n_components: true

# HMM
hmm:
  min_confidence: 0.5
  allow_shorts: true
  short_confidence_threshold: 0.7
```

### Trading Config (Most Important Settings)

```yaml
# Schedule
schedule:
  hour: "8,16"              # 8 AM and 4 PM
  minute: "45"
  timezone: "America/New_York"

# Agents
agents:
  - name: "HMM_Causal_Agent"
    enabled: true
    max_positions: 20
    total_allocation_pct: 0.60
    stop_loss_pct: 0.05
    take_profit_pct: 0.10
    
    strategy:
      n_components: 4
      use_causal_features: true
      min_confidence: 0.6
      allow_shorts: false

# Risk
risk:
  max_total_allocation: 0.90
  max_trades_per_day: 50
  max_loss_per_day: 0.05
```

## File Structure

```
alpaca-predict/
├── config/
│   ├── backtest_config.yaml          # NEW - Backtest parameters
│   └── trader_config.yaml            # NEW - Live trading parameters
│
├── scripts/
│   ├── backtest_with_config.py       # NEW - Config-driven backtest
│   ├── trader_with_config.py         # NEW - Config-driven trader
│   ├── quick_backtest_tech.py        # NEW - Fast 20-stock backtest
│   ├── backtest_tech_portfolio.py    # NEW - Full portfolio backtest
│   ├── trader.py                     # Original (still works)
│   └── ...
│
├── docs/
│   ├── BACKTEST_CONFIG_GUIDE.md      # NEW - Backtest guide
│   ├── TRADER_CONFIG_GUIDE.md        # NEW - Trading guide
│   ├── CONFIG_QUICK_REFERENCE.md     # NEW - Quick reference
│   ├── CONFIDENCE_POSITION_SIZING.md # v2.0 feature docs
│   ├── CAUSAL_HMM_INTEGRATION_SUMMARY.md
│   └── ...
│
└── README.md                          # Updated with config info
```

## Usage Examples

### 1. Backtest a New Strategy

```bash
# Edit config
nano config/backtest_config.yaml

# Enable strategy
strategies:
  - name: "4-State Causal"
    enabled: true

# Run
python scripts/backtest_with_config.py

# Review results in data/backtest_results/
```

### 2. Deploy to Live Trading

```bash
# Copy successful backtest settings
cp config/backtest_config.yaml config/my_live_config.yaml

# Adjust for live trading
nano config/my_live_config.yaml

# Test first
python scripts/trader_with_config.py --config my_live_config.yaml --dry-run

# If looks good, go live
python scripts/trader_with_config.py --config my_live_config.yaml
```

### 3. Compare Multiple Strategies

```yaml
# In backtest_config.yaml
strategies:
  - name: "2-State Technical"
    enabled: true
  - name: "4-State Technical"
    enabled: true
  - name: "4-State Causal"
    enabled: true
```

Run once, get comparison table:
```
Strategy                        Return    Annual   Sharpe    MaxDD
------------------------------------------------------------------
QQQ (Buy & Hold)                61.18%    26.96%    1.27   -22.88%
2-State Technical               33.26%    15.44%    6.12   -14.91%
4-State Causal + Confidence     29.30%    13.71%    8.08    -9.07%
```

### 4. Adjust Risk On-The-Fly

```bash
# Edit config while bot is stopped
nano config/trader_config.yaml

# Change from aggressive to conservative
agents:
  - min_confidence: 0.7        # Was 0.5
    max_positions: 10          # Was 20
    total_allocation_pct: 0.40 # Was 0.80

# Restart bot
python scripts/trader_with_config.py
```

## Migration from Old Scripts

### Old Way (Hardcoded)
```python
# scripts/trader.py - line 20
strategy = HMMStrategy(
    n_components=2,
    model_order=1,
    ranking_metric='sharpe'
)
```

### New Way (Config)
```yaml
# config/trader_config.yaml
strategy:
  type: "HMM"
  n_components: 2
  model_order: 1
  ranking_metric: "sharpe"
```

**Both still work!** Old scripts unchanged for backwards compatibility.

## Benefits

✅ **No Code Changes** - Adjust parameters via YAML  
✅ **Version Control** - Track configuration history  
✅ **Multiple Configs** - Different files for different scenarios  
✅ **Safe Testing** - Dry-run mode built-in  
✅ **Multi-Agent** - Run multiple strategies simultaneously  
✅ **Comprehensive Docs** - 3 detailed guides + quick reference  
✅ **Backwards Compatible** - Old scripts still work  

## Next Steps

1. **Review Guides:**
   - Read `docs/CONFIG_QUICK_REFERENCE.md` first
   - Deep dive into `docs/TRADER_CONFIG_GUIDE.md` for live trading
   - Study `docs/BACKTEST_CONFIG_GUIDE.md` for testing

2. **Test Backtests:**
   ```bash
   python scripts/backtest_with_config.py
   ```

3. **Dry Run Trading:**
   ```bash
   python scripts/trader_with_config.py --dry-run
   ```

4. **Customize:**
   - Create your own config variations
   - Test different parameter combinations
   - Document what works

5. **Deploy:**
   - Start with paper trading
   - Monitor closely for first week
   - Gradually increase allocation

## Support

- **Configuration Issues:** See `docs/CONFIG_QUICK_REFERENCE.md`
- **Backtest Help:** See `docs/BACKTEST_CONFIG_GUIDE.md`
- **Trading Help:** See `docs/TRADER_CONFIG_GUIDE.md`
- **HMM v2.0 Features:** See `docs/CONFIDENCE_POSITION_SIZING.md`
- **Service Setup:** See `docs/SERVICE_SETUP.md`

## Version History

- **v2.1** (Nov 13, 2025) - Configuration system
- **v2.0** (Nov 13, 2025) - Causal features + confidence sizing
- **v1.0** - Initial release

---

**Enjoy your configurable trading bot! 🚀**
