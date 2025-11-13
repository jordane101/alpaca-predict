# Configuration Files Quick Reference

This guide provides a quick overview of all configuration files in the alpaca-predict system.

## 📁 Configuration Files

### 1. **Backtest Configuration**
**File:** `config/backtest_config.yaml`  
**Script:** `scripts/backtest_with_config.py`  
**Purpose:** Configure portfolio backtesting parameters

**Quick adjustments:**
```yaml
# Make more aggressive
portfolio:
  max_positions: 15-20
hmm:
  min_confidence: 0.3

# Make more conservative  
portfolio:
  max_positions: 5-8
hmm:
  min_confidence: 0.7
```

**Run:**
```bash
.venv/bin/python scripts/backtest_with_config.py
```

**Guide:** `docs/BACKTEST_CONFIG_GUIDE.md`

---

### 2. **Live Trading Configuration**
**File:** `config/trader_config.yaml`  
**Script:** `scripts/trader_with_config.py`  
**Purpose:** Configure live trading bot behavior

**Quick adjustments:**
```yaml
# Enable/disable agents
agents:
  - name: "HMM_Causal_Agent"
    enabled: true          # Set false to disable

# Adjust risk
agents:
  - stop_loss_pct: 0.05    # Tighter = 0.03, Looser = 0.07
    min_confidence: 0.6    # Higher = fewer trades

# Change schedule
schedule:
  hour: "9,15"             # Twice daily
```

**Run:**
```bash
# Test without executing
.venv/bin/python scripts/trader_with_config.py --dry-run

# Live trading
.venv/bin/python scripts/trader_with_config.py
```

**Guide:** `docs/TRADER_CONFIG_GUIDE.md`

---

## 🎯 Common Use Cases

### Change Time Period (Backtest)
```yaml
# config/backtest_config.yaml
backtest:
  start_date: "2020-01-01"
  end_date: "2023-12-31"
```

### Increase Portfolio Size (Live Trading)
```yaml
# config/trader_config.yaml
agents:
  - max_positions: 30              # From 20
    total_allocation_pct: 0.80     # From 0.60
```

### Enable Short Positions (Both)
```yaml
hmm:
  allow_shorts: true
  short_confidence_threshold: 0.75
```

### Change Rebalancing Frequency (Backtest)
```yaml
portfolio:
  rebalance_days: 21    # Monthly (was 63 = quarterly)
```

### Change Analysis Schedule (Live Trading)
```yaml
schedule:
  hour: "9,12,15"    # 3x per day (was 2x)
  minute: "45"
```

### Use Different Stock Universe (Backtest)
```yaml
universe:
  type: "full_dag"    # All ~80 stocks (was "top_20")
```

### Enable Multiple Strategies (Live Trading)
```yaml
agents:
  - name: "Agent_1"
    enabled: true
    total_allocation_pct: 0.60
    
  - name: "Agent_2"
    enabled: true       # Change from false
    total_allocation_pct: 0.30
```

---

## 📊 Parameter Impact Matrix

| Parameter | ↑ Increase | ↓ Decrease |
|-----------|------------|------------|
| `max_positions` | More diversification | More concentration |
| `min_confidence` | Fewer, safer trades | More trades, higher risk |
| `stop_loss_pct` | Wider stops, more volatility | Tighter risk control |
| `total_allocation_pct` | Higher exposure | More cash buffer |
| `rebalance_days` | Less turnover, lower costs | More responsive |
| `n_components` | More nuanced regimes | Simpler classification |

---

## 🚀 Quick Start Workflows

### Test a New Strategy
1. Edit `config/backtest_config.yaml`
2. Enable desired strategy
3. Run: `.venv/bin/python scripts/backtest_with_config.py`
4. Review results
5. If good, update `config/trader_config.yaml`
6. Test: `.venv/bin/python scripts/trader_with_config.py --dry-run`

### Deploy to Production
1. Backtest configuration first
2. Copy successful settings to `trader_config.yaml`
3. Set `paper_trading: true`
4. Run with `--dry-run` for 1 day
5. Run paper trading for 1 week
6. Switch to live (if confident)

### Adjust Risk After Losses
1. Increase `min_confidence` (0.6 → 0.7)
2. Decrease `total_allocation_pct` (0.8 → 0.6)
3. Reduce `max_positions` (20 → 15)
4. Tighten `stop_loss_pct` (0.05 → 0.03)

### Adjust Risk After Success
1. Decrease `min_confidence` (0.7 → 0.6)
2. Increase `total_allocation_pct` (0.6 → 0.8)
3. Add more positions (15 → 20)
4. Enable second agent

---

## 📝 Configuration Principles

### 1. Start Conservative
- High `min_confidence` (0.7+)
- Low `total_allocation` (0.5 or less)
- Tight `stop_loss` (0.03-0.05)
- Few `max_positions` (5-10)

### 2. Test Before Deploying
- Always backtest first
- Use `--dry-run` mode
- Paper trade for at least 1 week
- Monitor closely during first month

### 3. Make Small Changes
- Adjust one parameter at a time
- Change by 10-20% increments
- Observe for 1-2 weeks
- Document what works

### 4. Keep Safety Limits
- Total allocation ≤ 90%
- Min cash reserve ≥ 10%
- Daily loss limits enabled
- Stop losses always active

---

## 🔧 Files Overview

```
config/
├── backtest_config.yaml        # Backtesting parameters
└── trader_config.yaml          # Live trading parameters

scripts/
├── backtest_with_config.py     # Run backtests
├── trader_with_config.py       # Run live trading
├── quick_backtest_tech.py      # Quick backtest (hardcoded)
└── trader.py                   # Original trader (hardcoded)

docs/
├── BACKTEST_CONFIG_GUIDE.md    # Detailed backtest guide
├── TRADER_CONFIG_GUIDE.md      # Detailed trading guide
└── QUICK_REFERENCE.md          # This file
```

---

## 💡 Pro Tips

1. **Version Control**: Commit config files to track what works
2. **Separate Configs**: Create `aggressive.yaml`, `conservative.yaml` variants
3. **Document Changes**: Add comments to config files explaining changes
4. **Regular Review**: Reassess parameters monthly based on performance
5. **Market Adaptation**: Adjust for market regime changes (bull → bear)

---

## ⚠️ Safety Checklist

Before going live:
- [ ] Backtested configuration shows positive results
- [ ] Total agent allocations ≤ 100%
- [ ] Stop loss percentages are reasonable
- [ ] `paper_trading: true` initially
- [ ] Tested with `--dry-run` first
- [ ] API credentials in `.env` are correct
- [ ] Position limits make sense for account size
- [ ] Schedule matches market hours
- [ ] Monitoring/alerts configured
- [ ] Emergency stop plan in place

---

## 📞 Getting Help

- **Backtest issues**: See `docs/BACKTEST_CONFIG_GUIDE.md`
- **Trading issues**: See `docs/TRADER_CONFIG_GUIDE.md`
- **HMM details**: See `docs/CONFIDENCE_POSITION_SIZING.md`
- **Causal features**: See `docs/CAUSAL_HMM_INTEGRATION_SUMMARY.md`
- **Service setup**: See `docs/SERVICE_SETUP.md`

---

**Last Updated:** November 13, 2025
