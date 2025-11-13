# HMM Refactoring - October 2025

## Changes Made

This refactoring updates the Hidden Markov Model (HMM) implementation to use 2 states instead of 3 and incorporates S&P 500 return data as a feature.

### Key Changes

1. **2-State HMM (Binary Regime Classification)**
   - Changed from 3 states (negative/neutral/positive) to 2 states (negative/positive)
   - Updated `n_components` default from 3 to 2
   - Simplified regime classification logic
   - Updated files:
     - `hmm_analysis.py`: Changed default `n_components` to 2
     - `strategies.py`: Changed default `n_components` to 2

2. **S&P 500 Returns as Feature**
   - Added S&P 500 (SPY) return data as a feature for all stock HMMs
   - Fetches SPY data automatically for the same time period as the target stock
   - Aligns SPY returns with stock data by date
   - Forward-fills missing values to handle market closure differences
   - New features added to base_features:
     - `SP500_Return` (daily timeframe)
     - `SP500_Return` (weekly timeframe)
   
3. **Quantizer Improvement**
   - Changed QuantileTransformer to use `min(len(X), 100)` quantiles instead of `n_components`
   - Provides better distribution mapping regardless of number of states

4. **API Changes**
   - `AnalyzeHMM.__init__()` now accepts optional `sp500_data` parameter
   - If not provided and ticker is not SPY, it fetches SPY data automatically
   - Added `_get_sp500_data()` method to fetch S&P 500 returns

### Migration Notes

1. **Old Model Files**
   - Models trained with 3 components are incompatible with the new 2-component system
   - Run `python cleanup_old_models.py` to delete old 3-component models
   - New models will be automatically created when needed

2. **Regime Classification**
   - With 2 states, there is no "neutral" regime anymore
   - State 0 = Negative regime (lower returns)
   - State 1 = Positive regime (higher returns)
   - The `predict_next_day_outlook()` method now returns only "negative" or "positive"

3. **Feature Set**
   - All models now include S&P 500 returns as a feature
   - This provides market context for individual stock predictions
   - For SPY itself, SP500_Return will be 0 (or the feature can be excluded)

### Files Modified

- `hmm_analysis.py` - Core HMM implementation
- `strategies.py` - Strategy default parameters
- `cleanup_old_models.py` - New utility script to clean old models

### Testing Recommendations

1. Test with a single stock first: `python hmm_analysis.py`
2. Verify S&P 500 data is being fetched correctly
3. Check that 2-state regime classification works as expected
4. Run backtests to compare performance with old 3-state models
5. Monitor for any issues with the new feature set

### Performance Considerations

- S&P 500 data is fetched once per ticker analysis
- Data is cached in `self.sp500_data` to avoid redundant API calls
- For backtesting with "backtest" ticker, S&P 500 fetch is skipped
- Models will retrain automatically when old cached models are detected

## Usage Examples

### Basic Usage (will use 2 components by default)
```python
from hmm_analysis import AnalyzeHMM

# Analyze a stock with default 2-component HMM
analyzer = AnalyzeHMM("AAPL")
prediction = analyzer.predict_next_day_outlook()
print(f"Outlook: {prediction['outlook']}")  # Will be 'positive' or 'negative'
```

### With Pre-fetched S&P 500 Data (for batch processing)
```python
# Fetch SPY data once
spy_analyzer = AnalyzeHMM("SPY")
spy_data = spy_analyzer.data[['SP500_Return']]

# Reuse for multiple stocks
for ticker in ['AAPL', 'MSFT', 'GOOGL']:
    analyzer = AnalyzeHMM(ticker, sp500_data=spy_data)
    prediction = analyzer.predict_next_day_outlook()
```

### Forcing Retrain
```python
# Force retrain to get new 2-component model
analyzer = AnalyzeHMM("AAPL", force_retrain=True)
```
