# Quantile Granger Causality for HMM Feature Selection

## Overview

This document describes the implementation of **Quantile Granger Causality** testing for automatic feature selection in Hidden Markov Model (HMM) training. This advanced feature helps identify which technical indicators and market features truly have predictive power for future returns at different quantiles of the return distribution.

## What is Quantile Granger Causality?

### Traditional Granger Causality
Traditional Granger causality tests whether past values of variable X provide statistically significant information about future values of variable Y, beyond what Y's own past values provide.

### Quantile Extension
**Quantile Granger causality** extends this concept to different quantiles of the conditional distribution:
- Tests predictive power at **tail events** (10th, 90th percentiles) 
- Tests predictive power at **median** behavior (50th percentile)
- Tests predictive power at **intermediate** regions (25th, 75th percentiles)

### Why This Matters
Stock returns are **non-Gaussian** - they have:
- Fat tails (extreme events more common than normal distribution predicts)
- Asymmetry (different behavior in bull vs bear markets)
- Regime-dependent relationships

A feature might be causal only during extreme market conditions (quantile tails) but not during normal times (median), or vice versa.

## Implementation Details

### Algorithm

For each feature and each quantile:

1. **Create lagged data**: Use lags 1-5 of both target (Return) and feature
2. **Fit restricted model**: Quantile regression of Return on its own lags only
3. **Fit unrestricted model**: Quantile regression of Return on its lags + feature lags
4. **Compare fit**: Calculate improvement in quantile loss function
5. **Significance test**: Approximate p-value using chi-square distribution
6. **Decision**: Feature is "causal" if p-value < 0.05 at ANY quantile

### Code Structure

```python
# Enable causality filtering when creating analyzer
ah = AnalyzeHMM(
    ticker="AAPL",
    n_components=2,
    model_order=1,
    use_causality_filter=True,      # Enable causality testing
    causality_significance=0.05,     # P-value threshold
    force_retrain=True
)
```

### Quantiles Tested

Default quantiles (configurable):
- **0.10**: Lower tail (bear market conditions)
- **0.25**: Lower quartile
- **0.50**: Median (typical conditions)
- **0.75**: Upper quartile  
- **0.90**: Upper tail (bull market conditions)

### Output Format

#### Console Output
```
======================================================================
QUANTILE GRANGER CAUSALITY ANALYSIS
======================================================================
Target: Return
Quantiles: [0.1, 0.25, 0.5, 0.75, 0.9]
Max lag: 5
Significance level: 0.05

--- Testing Volatility -> Return ---
  Q=0.10: p-value=0.0234 (lag=2) ***
  Q=0.25: p-value=0.0891 (lag=1)
  Q=0.50: p-value=0.1234 (lag=3)
  Q=0.75: p-value=0.0456 (lag=2) ***
  Q=0.90: p-value=0.0123 (lag=1) ***
  ✓ Volatility IS CAUSAL (best: q=0.9, p=0.0123)

--- Testing SMA_50 -> Return ---
  Q=0.10: p-value=0.3456 (lag=5)
  Q=0.25: p-value=0.2891 (lag=3)
  Q=0.50: p-value=0.4123 (lag=2)
  Q=0.75: p-value=0.3789 (lag=4)
  Q=0.90: p-value=0.2345 (lag=1)
  ✗ SMA_50 NOT CAUSAL (best p=0.2345)

======================================================================
SUMMARY: 2/3 features show Granger causality
Causal features: Volatility, SP500_Return
======================================================================
```

#### Model JSON Output

The causality results are saved to the model's JSON file:

```json
{
  "ticker": "AAPL",
  "n_components": 2,
  "features_used": ["Return", "Volatility", "SP500_Return"],
  "causality_analysis": {
    "enabled": true,
    "significance_level": 0.05,
    "results": {
      "Volatility": {
        "is_causal": true,
        "min_p_value": 0.0123,
        "best_quantile": 0.9,
        "best_lag": 1,
        "quantile_results": {
          "0.1": {"lag": 2, "p_value": 0.0234},
          "0.25": {"lag": 1, "p_value": 0.0891},
          "0.5": {"lag": 3, "p_value": 0.1234},
          "0.75": {"lag": 2, "p_value": 0.0456},
          "0.9": {"lag": 1, "p_value": 0.0123}
        }
      },
      "SMA_50": {
        "is_causal": false,
        "min_p_value": 0.2345,
        "best_quantile": 0.9,
        "best_lag": 1,
        "quantile_results": { ... }
      },
      "SP500_Return": {
        "is_causal": true,
        "min_p_value": 0.0089,
        "best_quantile": 0.5,
        "best_lag": 1,
        "quantile_results": { ... }
      }
    }
  }
}
```

## Usage Examples

### Example 1: Basic Usage with Causality Filtering

```python
from hmm_analysis import AnalyzeHMM

# Train model with causality-filtered features
analyzer = AnalyzeHMM(
    ticker="TSLA",
    n_components=2,
    model_order=1,
    use_causality_filter=True,
    force_retrain=True
)

# Check which features were selected
print(f"Features used: {analyzer.features}")
print(f"Causality results: {analyzer.causality_results}")
```

### Example 2: Compare Models With/Without Causality

```python
# Model 1: All features
model_all = AnalyzeHMM(
    ticker="NVDA",
    n_components=2,
    model_order=1,
    use_causality_filter=False,
    force_retrain=True
)

# Model 2: Only causal features
model_causal = AnalyzeHMM(
    ticker="NVDA",
    n_components=2,
    model_order=1,
    use_causality_filter=True,
    force_retrain=True
)

print(f"All features: {model_all.features}")
print(f"Causal features: {model_causal.features}")
```

### Example 3: Custom Causality Settings

```python
# More stringent causality requirement
analyzer = AnalyzeHMM(
    ticker="SPY",
    n_components=2,
    model_order=2,
    use_causality_filter=True,
    causality_significance=0.01,  # Require p < 0.01 instead of 0.05
    force_retrain=True
)
```

### Example 4: Standalone Causality Testing

```python
# Just run causality analysis without training
analyzer = AnalyzeHMM(
    ticker="AAPL",
    n_components=2,
    model_order=1,
    use_causality_filter=False
)

# Run causality test manually
causality_results = analyzer.quantile_granger_causality_test(
    target='Return',
    quantiles=[0.1, 0.5, 0.9],  # Test fewer quantiles
    maxlag=3,
    significance_level=0.05
)

# Analyze results
for feature, result in causality_results.items():
    if result['is_causal']:
        print(f"{feature}: Causal at quantile {result['best_quantile']} (p={result['min_p_value']:.4f})")
```

## Integration with Trading Strategy

Update `strategies.py` to use causality filtering:

```python
class HMMStrategy(BaseStrategy):
    def __init__(self, n_components=2, model_order=1, 
                 use_causality=False, **kwargs):
        self.use_causality = use_causality
        # ... other init code ...
    
    def analyze(self, ticker, bars_data, sp500_data=None):
        analyzer = AnalyzeHMM(
            ticker=ticker,
            n_components=self.n_components,
            model_order=self.model_order,
            bars_data=bars_data,
            sp500_data=sp500_data,
            use_causality_filter=self.use_causality,  # Pass through
            max_age_days=self.retrain_max_age_days
        )
        # ... rest of analysis ...
```

Then use in trader configuration:

```python
# In trader.py or orchestrator.py
strategy = HMMStrategy(
    n_components=2,
    model_order=1,
    use_causality=True,  # Enable causality filtering
    optimize_order=False
)
```

## Benefits

### 1. **Improved Generalization**
- Removes spurious correlations that don't hold out-of-sample
- Focuses model on features with genuine predictive power
- Reduces overfitting to noise in training data

### 2. **Model Interpretability**
- Clear documentation of WHY features were included
- Quantifiable evidence of feature importance
- Transparency in feature selection process

### 3. **Computational Efficiency**
- Fewer features = faster training and prediction
- Reduced model complexity
- Lower memory requirements

### 4. **Robustness**
- Features tested across multiple market conditions (quantiles)
- More likely to work in different market regimes
- Adapts to non-linear, regime-dependent relationships

## Performance Considerations

### Computational Cost
- Causality testing adds ~30-60 seconds per ticker to training time
- Tests all feature combinations across all quantiles and lags
- **Recommendation**: Cache models and only retrain when necessary

### Sample Size Requirements
- Requires sufficient historical data for statistical power
- Minimum ~200 observations (days) recommended
- More data = more reliable causality tests

### False Positives/Negatives
- Type I error: Rejecting a causal feature (false negative) - protected by testing multiple quantiles
- Type II error: Including a non-causal feature (false positive) - controlled by significance level

## Theoretical Foundation

### Quantile Regression
Uses the check function (quantile loss):
```
ρ_τ(u) = u(τ - 𝟙(u < 0))
```

Where:
- τ is the quantile (0.1, 0.5, 0.9, etc.)
- u is the residual
- 𝟙 is the indicator function

### Causality Test Statistic
Improvement in fit:
```
Δ = (Loss_restricted - Loss_unrestricted) / Loss_restricted
```

Approximate p-value:
```
p = P(χ²_df > n × Δ)
```

Where n is sample size and df is degrees of freedom (number of added parameters).

## Limitations and Future Work

### Current Limitations
1. **Linear quantile regression**: Assumes linear relationships at each quantile
2. **Independence assumption**: Doesn't account for feature interactions
3. **Computational cost**: Can be slow for many features
4. **Approximate p-values**: Uses asymptotic approximation rather than bootstrap

### Potential Enhancements
1. **Bootstrap p-values**: More accurate significance testing
2. **Feature interactions**: Test joint causality of feature pairs
3. **Non-linear quantile regression**: Use neural networks or splines
4. **Cross-validation**: Validate causality on hold-out data
5. **Adaptive quantiles**: Select quantiles based on data distribution

## References

1. Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
2. Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models and Cross-spectral Methods". *Econometrica*, 37(3), 424–438.
3. Troster, V. (2018). "Testing for Granger-causality in quantiles". *Econometric Reviews*, 37(8), 850-866.
4. White, H., Kim, T. H., & Manganelli, S. (2015). "VAR for VaR: Measuring tail dependence using multivariate regression quantiles". *Journal of Econometrics*, 187(1), 169-188.

## Support

For questions or issues:
- Check logs in `logs/hmm_training_*.log`
- Review causality results in model JSON files
- Examine `causality_results` attribute on AnalyzeHMM instances

---

**Last Updated**: November 12, 2025  
**Author**: Eli Jordan
