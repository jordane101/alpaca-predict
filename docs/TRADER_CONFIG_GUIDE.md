# Live Trading Configuration Guide

## 📁 Configuration File Location

```
config/trader_config.yaml
```

## 🚀 Quick Start

### Run with default config:
```bash
.venv/bin/python scripts/trader_with_config.py
```

### Test without executing trades (DRY RUN):
```bash
.venv/bin/python scripts/trader_with_config.py --dry-run
```

### Use custom config:
```bash
.venv/bin/python scripts/trader_with_config.py --config my_config.yaml
```

## 🎯 Key Configuration Areas

### 1. **Trading Schedule**

Controls when the bot analyzes stocks and places trades:

```yaml
schedule:
  hour: "8,16"      # 8 AM and 4 PM Eastern
  minute: "45"      # At :45 minutes
  timezone: "America/New_York"
```

**Common patterns:**
- **Conservative**: `hour: "9", minute: "45"` (once per day at market open)
- **Moderate**: `hour: "9,15", minute: "45"` (morning and afternoon)
- **Aggressive**: `hour: "9-16", minute: "30"` (every hour during market)

### 2. **Agent Configuration**

You can run multiple agents with different strategies:

```yaml
agents:
  - name: "HMM_Causal_Agent"
    enabled: true                   # Enable/disable this agent
    max_positions: 20               # Max positions to hold
    total_allocation_pct: 0.60     # Use 60% of account equity
    
    strategy:
      type: "HMM"
      n_components: 4
      use_causal_features: true
      min_confidence: 0.6
```

**Important:** Total allocation across all agents should be ≤ 100%

### 3. **Risk Management**

Configure position sizing and risk limits:

```yaml
agents:
  - name: "My_Agent"
    stop_loss_pct: 0.05           # 5% stop loss
    take_profit_pct: 0.10         # 10% take profit (2:1 ratio)
    max_positions: 20             # Portfolio diversification

risk:
  max_position_size: 0.15         # No position > 15% of portfolio
  max_trades_per_day: 50          # Daily trade limit
  max_loss_per_day: 0.05          # Circuit breaker at 5% daily loss
```

### 4. **Strategy Settings**

Fine-tune the HMM strategy:

```yaml
strategy:
  # Model configuration
  n_components: 4                      # Number of market regimes (2-4)
  optimize_n_components: true          # Auto-select optimal states
  
  # Causal features (v2.0)
  use_causal_features: true            # Use DAG relationships
  
  # Confidence-based position sizing
  min_confidence: 0.6                  # 60% minimum confidence
  allow_shorts: false                  # Enable/disable shorts
  short_confidence_threshold: 0.75     # 75% confidence for shorts
  
  # Retraining
  retrain_max_age_days: 1              # Retrain daily
```

## 📊 Pre-Made Configurations

### Configuration 1: Aggressive Growth (High Risk/High Reward)

```yaml
agents:
  - name: "Aggressive_Agent"
    enabled: true
    max_positions: 30
    total_allocation_pct: 0.90       # Use 90% of capital
    stop_loss_pct: 0.07              # Wider stops
    take_profit_pct: 0.14
    
    strategy:
      n_components: 4
      use_causal_features: true
      min_confidence: 0.5            # Lower threshold = more trades
      allow_shorts: true
      short_confidence_threshold: 0.7
      ranking_metric: "return"       # Maximize returns

schedule:
  hour: "9,12,15"                    # Trade 3x per day
  minute: "45"
```

### Configuration 2: Conservative Income (Low Risk)

```yaml
agents:
  - name: "Conservative_Agent"
    enabled: true
    max_positions: 10
    total_allocation_pct: 0.50       # Only 50% invested
    stop_loss_pct: 0.03              # Tight stops
    take_profit_pct: 0.09
    
    strategy:
      n_components: 2                # Simple bull/bear
      use_causal_features: false     # Technical only
      min_confidence: 0.8            # High confidence required
      allow_shorts: false            # Longs-only
      ranking_metric: "sharpe"       # Risk-adjusted

schedule:
  hour: "9"                          # Once per day
  minute: "45"
```

### Configuration 3: Multi-Agent Diversified

```yaml
agents:
  # Agent 1: Causal HMM (60% allocation)
  - name: "HMM_Causal_Agent"
    enabled: true
    max_positions: 15
    total_allocation_pct: 0.60
    
    strategy:
      n_components: 4
      use_causal_features: true
      min_confidence: 0.6
      allow_shorts: false
  
  # Agent 2: Technical HMM (30% allocation)
  - name: "HMM_Technical_Agent"
    enabled: true
    max_positions: 10
    total_allocation_pct: 0.30
    
    strategy:
      n_components: 2
      use_causal_features: false
      min_confidence: 0.7

schedule:
  hour: "9,15"
  minute: "45"
```

## 🔧 Parameter Deep Dive

### Agent Parameters

| Parameter | Default | Description | Adjustment Impact |
|-----------|---------|-------------|-------------------|
| `max_positions` | 10 | Max stocks to hold | ↑ = More diversification |
| `total_allocation_pct` | 0.5 | % of equity to use | ↑ = Higher exposure |
| `stop_loss_pct` | 0.05 | Stop loss % | ↓ = Tighter risk control |
| `take_profit_pct` | 0.10 | Take profit % | ↑ = Let winners run |
| `max_analysis_workers` | 4 | Parallel workers | ↑ = Faster (up to CPU limit) |

### Strategy Parameters

| Parameter | Default | Description | Adjustment Impact |
|-----------|---------|-------------|-------------------|
| `n_components` | 4 | HMM states | 2 = simple, 4 = nuanced |
| `use_causal_features` | true | Use DAG | true = better predictions |
| `min_confidence` | 0.6 | Entry threshold | ↓ = More trades, ↑ risk |
| `allow_shorts` | false | Enable shorts | true = profit in bear markets |
| `short_confidence_threshold` | 0.75 | Short entry bar | ↑ = Safer shorts |
| `retrain_max_age_days` | 1 | Retrain frequency | 0 = daily, adaptive |

### Risk Parameters

| Parameter | Default | Description | Purpose |
|-----------|---------|-------------|---------|
| `max_total_allocation` | 0.90 | Max invested | Capital preservation |
| `min_cash_reserve` | 0.10 | Cash buffer | Liquidity for opportunities |
| `max_position_size` | 0.15 | Single stock limit | Concentration risk |
| `max_trades_per_day` | 50 | Daily trade cap | Prevent overtrading |
| `max_loss_per_day` | 0.05 | Circuit breaker | Stop losses at 5% down |

## ⚙️ Advanced Features

### 1. **Causal Features** (v2.0)

Uses market relationships from the DAG to improve predictions:

```yaml
strategy:
  use_causal_features: true
  causal_dag_file: null              # Uses default DAG
```

**Requirements:**
- DAG must be built: `python scripts/build_large_dag.py`
- Stocks must be in DAG universe

### 2. **Confidence-Based Position Sizing** (v2.0)

Scales position sizes based on model confidence:

```yaml
strategy:
  min_confidence: 0.6                # 60% min to trade
  max_position: 1.0                  # 100% max size
```

**How it works:**
- Confidence 50% → Skip trade
- Confidence 60% → 60% of normal size
- Confidence 100% → Full position size

### 3. **Short Selling**

Enable short positions for bearish signals:

```yaml
strategy:
  allow_shorts: true
  short_confidence_threshold: 0.75   # Higher bar for shorts
```

**Caution:**
- Requires margin account
- Higher risk than longs-only
- Check account permissions

### 4. **Multi-Agent Setup**

Run multiple strategies simultaneously:

```yaml
agents:
  - name: "Aggressive_Agent"
    total_allocation_pct: 0.60
  - name: "Conservative_Agent"
    total_allocation_pct: 0.30
```

**Benefits:**
- Diversification across strategies
- Different risk profiles
- Complementary signals

## 🚨 Important Warnings

### ⚠️ Before Going Live:

1. **Backtest First**: Run backtests with your config
   ```bash
   python scripts/backtest_with_config.py
   ```

2. **Use Dry Run**: Test without executing
   ```bash
   python scripts/trader_with_config.py --dry-run
   ```

3. **Start Small**: Use low `total_allocation_pct` initially

4. **Paper Trade**: Set `api.paper_trading: true`

5. **Monitor Closely**: Watch first few days of live trading

### ⚠️ Common Mistakes:

❌ **Total allocation > 100%**
```yaml
# BAD: 60% + 50% = 110%
agents:
  - total_allocation_pct: 0.60
  - total_allocation_pct: 0.50
```

✅ **Correct:**
```yaml
# GOOD: 60% + 30% = 90%
agents:
  - total_allocation_pct: 0.60
  - total_allocation_pct: 0.30
```

❌ **Too low confidence threshold**
```yaml
# Risky: Will trade on weak signals
min_confidence: 0.3
```

✅ **Better:**
```yaml
# Safer: Only high-conviction trades
min_confidence: 0.6
```

❌ **Insufficient cash reserve**
```yaml
# Risky: Fully invested
max_total_allocation: 1.0
min_cash_reserve: 0.0
```

✅ **Better:**
```yaml
# Safer: Keep buffer
max_total_allocation: 0.90
min_cash_reserve: 0.10
```

## 📝 Configuration Checklist

Before going live, verify:

- [ ] Total agent allocations ≤ 100%
- [ ] `paper_trading: true` for testing
- [ ] Stop loss % makes sense for your risk tolerance
- [ ] Confidence thresholds are appropriate (0.6+ recommended)
- [ ] Schedule matches your availability to monitor
- [ ] DAG is built if using causal features
- [ ] Account has sufficient equity for position sizes
- [ ] Margin enabled if using shorts
- [ ] Monitoring/alerts configured

## 🔍 Debugging

Enable debug mode for troubleshooting:

```yaml
advanced:
  debug: true                        # Verbose logging
  dry_run: true                      # Don't execute trades
```

Check logs:
```bash
tail -f data/logs/trader.log
```

## 📞 Support

If you encounter issues:

1. Check `data/logs/trader.log` for errors
2. Verify `.env` has API credentials
3. Test with `--dry-run` first
4. Review `docs/SERVICE_SETUP.md` for systemd setup
5. Check `position_ownership.json` for position state

## 🎓 Learning Path

**Beginner → Advanced:**

1. **Week 1**: Run with default config in paper trading
2. **Week 2**: Adjust `min_confidence` and `max_positions`
3. **Week 3**: Enable causal features, compare performance
4. **Week 4**: Add second agent with different strategy
5. **Month 2**: Enable shorts (if comfortable)
6. **Month 3**: Fine-tune based on real performance

**Remember**: Start conservative, increase risk gradually!
