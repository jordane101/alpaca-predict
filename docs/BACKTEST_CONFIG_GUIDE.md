# Backtest Configuration Guide

## 📁 Configuration File Location

```
config/backtest_config.yaml
```

## 🎯 Quick Parameter Guide

### Most Common Adjustments:

#### 1. **Make Strategy More Aggressive** (Higher Returns, Higher Risk)
```yaml
portfolio:
  max_positions: 15-20        # Increase from 10
  cash_usage: 0.98            # Use more cash (from 0.95)

hmm:
  min_confidence: 0.3         # Lower threshold (from 0.5)
  min_position_size: 0.05     # Allow smaller positions (from 0.1)
```

#### 2. **Make Strategy More Conservative** (Lower Risk, Lower Returns)
```yaml
portfolio:
  max_positions: 5-8          # Fewer positions (from 10)
  cash_usage: 0.80            # Keep more cash (from 0.95)

hmm:
  min_confidence: 0.7         # Higher threshold (from 0.5)
  min_position_size: 0.2      # Only high-conviction positions (from 0.1)
```

#### 3. **Change Rebalancing Frequency**
```yaml
portfolio:
  rebalance_days: 21    # Monthly
  # or
  rebalance_days: 63    # Quarterly (current)
  # or
  rebalance_days: 5     # Weekly (high turnover!)
```

#### 4. **Test Different Time Periods**
```yaml
backtest:
  years: 3              # 3 years instead of 2
  
  # Or use specific dates:
  # start_date: "2020-01-01"
  # end_date: "2023-12-31"
```

#### 5. **Enable/Disable Strategies**
```yaml
strategies:
  - name: "2-State Technical"
    enabled: true         # Set to false to skip
    
  - name: "4-State Causal + Confidence"
    enabled: true         # Test multiple at once
```

#### 6. **Change Universe Size**
```yaml
universe:
  type: "top_20"         # Use top 20 stocks (fast)
  # or
  type: "full_dag"       # Use all ~80 stocks from DAG (slower)
```

#### 7. **Adjust Short Position Settings**
```yaml
hmm:
  allow_shorts: false              # Disable shorts (longs-only)
  short_confidence_threshold: 0.8  # Require 80% confidence for shorts
```

## 🚀 How to Run

### Using the config file:
```bash
.venv/bin/python scripts/backtest_with_config.py
```

### Using a custom config:
```bash
.venv/bin/python scripts/backtest_with_config.py --config my_config.yaml
```

## 📊 Example Configurations

### Configuration 1: Aggressive Growth
**Goal:** Maximum returns, willing to accept higher risk
```yaml
portfolio:
  max_positions: 20
  cash_usage: 0.98
  rebalance_days: 21  # Monthly rebalancing

hmm:
  min_confidence: 0.3
  min_position_size: 0.05
  allow_shorts: true
  short_confidence_threshold: 0.6
```

### Configuration 2: Conservative Income
**Goal:** Lower volatility, capital preservation
```yaml
portfolio:
  max_positions: 5
  cash_usage: 0.80
  rebalance_days: 126  # Semi-annual

hmm:
  min_confidence: 0.8
  min_position_size: 0.3
  allow_shorts: false  # Longs-only
```

### Configuration 3: Balanced Approach
**Goal:** Moderate risk/reward balance
```yaml
portfolio:
  max_positions: 10
  cash_usage: 0.90
  rebalance_days: 63  # Quarterly

hmm:
  min_confidence: 0.6
  min_position_size: 0.15
  allow_shorts: true
  short_confidence_threshold: 0.75
```

## 🔧 Strategy Parameters Explained

### `n_components`
- **2**: Simple bull/bear classification
- **4**: Nuanced (Strong Bear, Mild Bear, Mild Bull, Strong Bull)

### `use_causal_features`
- **true**: Use causal DAG relationships (advanced)
- **false**: Use technical indicators only (simpler)

### `optimize_n_components`
- **true**: Automatically select best 2-4 states using AIC/BIC
- **false**: Use fixed `n_components`

## 📈 Performance Impact

| Parameter | Effect on Returns | Effect on Risk |
|-----------|------------------|----------------|
| ↑ max_positions | ↑ Diversification | ↓ Volatility |
| ↑ cash_usage | ↑ Exposure | ↑ Risk |
| ↓ rebalance_days | ↑ Turnover | ↑ Transaction costs |
| ↓ min_confidence | ↑ Positions taken | ↑ Risk |
| enable shorts | ↑ Potential alpha | ↑ Complexity |

## 📝 Notes

1. **Lower confidence thresholds** = More trades, more risk
2. **Higher max_positions** = More diversification, lower concentration risk
3. **Shorter rebalance periods** = Higher turnover, more responsive to market changes
4. **Causal features** = Better market structure understanding but slower training
5. **Optimization** = Better model selection but longer computation time

## 💡 Recommended Starting Points

**For Testing/Development:**
- Universe: top_20
- Rebalance: 63 days (quarterly)
- Max positions: 10
- Years: 2

**For Production:**
- Universe: full_dag
- Rebalance: 21 days (monthly)
- Max positions: 15-20
- Years: 3-5 (longer backtest)

## ⚠️ Important

- Always test parameter changes on historical data before live trading
- Lower confidence thresholds increase risk
- Shorter rebalance periods increase transaction costs
- Full DAG universe takes ~10-15 minutes per backtest
