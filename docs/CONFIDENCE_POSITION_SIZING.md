# Confidence-Based Position Sizing with Short Capability

## Overview

The HMM trading system now includes sophisticated confidence-based position sizing that:
- Uses the HMM forward algorithm to calculate state probabilities
- Scales position sizes based on prediction confidence
- Supports short positions for high-confidence bearish regimes
- Integrates seamlessly with the multi-state regime classification

## Features

### 1. **State Probability Calculation**
Uses the forward algorithm to compute probability distribution over all states:
```python
prob_info = analyzer.get_state_probabilities()
# Returns:
# {
#   'probabilities': [0.15, 0.70, 0.10, 0.05],  # Probability for each state
#   'most_likely_state': 1,                      # Index of most likely state
#   'confidence': 0.70,                          # Probability of most likely state
#   'state_probs_dict': {0: 0.15, 1: 0.70, ...} # Dict mapping states to probs
# }
```

### 2. **Confidence-Based Position Sizing**
Automatically calculates position size based on:
- **Confidence level**: How certain the model is about the prediction
- **Regime classification**: Bullish, neutral, or bearish
- **Sentiment score**: Weighted average of state probabilities (-1 to +1)
- **Expected return**: Probability-weighted average of state returns

```python
position_info = analyzer.calculate_position_size(
    min_confidence=0.5,              # Minimum confidence to take any position
    max_position=1.0,                # Maximum position size (100%)
    allow_shorts=True,               # Enable short positions
    short_confidence_threshold=0.7   # Minimum confidence for shorts
)
# Returns:
# {
#   'position_size': 0.75,           # +0.75 = 75% long, -0.75 = 75% short
#   'confidence': 0.85,              # Confidence in the prediction
#   'regime': 'bullish',             # bearish/neutral/bullish
#   'action': 'buy',                 # buy/sell/short/hold
#   'reasoning': 'Bullish regime...',
#   'sentiment_score': 0.65,         # -1 (bearish) to +1 (bullish)
#   'expected_return': 0.0123        # Probability-weighted expected return
# }
```

### 3. **Short Position Support**
The system can now take short positions when:
- Regime is classified as bearish (sentiment_score < -0.3)
- Confidence exceeds the short threshold (default 70%)
- Short positions are enabled

**Example scenarios:**
- **High confidence bullish**: 85% confidence, bullish → 85% long position
- **High confidence bearish**: 75% confidence, bearish → 75% short position
- **Low confidence bearish**: 55% confidence, bearish → 0% position (below short threshold)
- **Neutral regime**: 80% confidence, neutral → 40% position (reduced)

## Configuration Parameters

### Position Sizing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_confidence` | 0.5 | Minimum confidence (50%) to take any position |
| `max_position` | 1.0 | Maximum position size as fraction of portfolio (100%) |
| `allow_shorts` | True | Whether to allow negative (short) positions |
| `short_confidence_threshold` | 0.7 | Minimum confidence (70%) required for shorts |

### Regime Classification Thresholds

| Sentiment Score | Regime | Description |
|----------------|--------|-------------|
| < -0.3 | Bearish | Weighted probability favors low-return states |
| -0.3 to 0.3 | Neutral | Mixed probability across states |
| > 0.3 | Bullish | Weighted probability favors high-return states |

## Integration with Trading Strategy

### HMMStrategy Changes

The `HMMStrategy` class now:
1. Passes confidence and position sizing to predictions
2. Weights ranking strength by confidence
3. Handles both long and short signals
4. Scales waterfall allocation by recommended position size

```python
# Example: Strategy initialization
strategy = HMMStrategy(
    n_components=3,
    use_causal_features=True,
    optimize_n_components=True
)

# The strategy now returns enhanced predictions:
outlook, prediction = strategy.analyze(ticker, bars_data)
# prediction now includes:
# - confidence: 0.85
# - position_size: +0.75 (long) or -0.75 (short)
# - position_action: 'buy', 'short', or 'hold'
# - regime: 'bullish', 'bearish', or 'neutral'
```

### TradingAgent Changes

The `TradingAgent` now:
1. Recognizes short signals (`position_action == 'short'`)
2. Scales waterfall allocation by `abs(position_size)`
3. Makes notional value negative for short positions
4. Displays position action and confidence in logs

```python
# Example output:
#   - AAPL: Action=BUY, Strength=2.1234, Confidence=85%, PosSize=+75%, Notional=$7,500
#   - TSLA: Action=SHORT, Strength=1.8765, Confidence=72%, PosSize=-60%, Notional=$-6,000
```

## Usage Examples

### Example 1: Conservative Long-Only Strategy
```python
from src.hmm.hmm_analysis import AnalyzeHMM

analyzer = AnalyzeHMM(
    ticker="AAPL",
    n_components=3,
    use_causal_features=True,
    optimize_n_components=True
)

# Get position sizing (longs only, high threshold)
position = analyzer.calculate_position_size(
    min_confidence=0.6,        # Need 60% confidence minimum
    max_position=0.5,          # Max 50% position size
    allow_shorts=False,        # No shorts
    short_confidence_threshold=0.8
)

print(f"Recommended: {position['action']} {abs(position['position_size']):.1%} of portfolio")
# Output: "Recommended: buy 45.0% of portfolio"
```

### Example 2: Aggressive Long-Short Strategy
```python
# Get position sizing (aggressive with shorts)
position = analyzer.calculate_position_size(
    min_confidence=0.4,        # Lower threshold (40%)
    max_position=1.0,          # Full position allowed
    allow_shorts=True,         # Enable shorts
    short_confidence_threshold=0.6  # Lower short threshold (60%)
)

if position['position_size'] > 0:
    print(f"GO LONG: {position['position_size']:.1%} position")
elif position['position_size'] < 0:
    print(f"GO SHORT: {abs(position['position_size']):.1%} position")
else:
    print("HOLD: Confidence too low or neutral regime")
```

### Example 3: Full Prediction with Confidence
```python
# Get complete prediction including confidence and position sizing
prediction = analyzer.predict_next_day_outlook()

print(f"Outlook: {prediction['outlook']}")
print(f"Confidence: {prediction['confidence']:.1%}")
print(f"Action: {prediction['position_action']}")
print(f"Position Size: {prediction['position_size']:+.1%}")
print(f"Regime: {prediction['regime']}")

# State probabilities
for state, prob in prediction['state_probabilities'].items():
    print(f"  State {state}: {prob:.1%}")
```

## Mathematical Foundation

### Confidence Calculation
Confidence is the probability of the most likely state from the forward algorithm:
```
confidence = max(P(s₁|x), P(s₂|x), ..., P(sₙ|x))
```

### Sentiment Score
Weighted average of state positions (-1 to +1):
```
sentiment = Σᵢ P(sᵢ|x) × position_scoreᵢ
where position_scoreᵢ = (2i / (n-1)) - 1
```

### Position Size Calculation
```
confidence_scaled = (confidence - min_confidence) / (1 - min_confidence)
base_position = confidence_scaled × max_position

if regime == bullish:
    position_size = base_position
elif regime == bearish and confidence ≥ short_threshold:
    position_size = -base_position
else:
    position_size = 0 or base_position × 0.5 (neutral)
```

## Testing

Run the comprehensive test suite:
```bash
.venv/bin/python scripts/test_causal_hmm_integration.py
```

The test suite includes:
- **Test 1**: Causal Feature Engine initialization
- **Test 2**: Model selection with AIC/BIC
- **Test 3**: Multi-state regime classification
- **Test 4**: Causal vs technical indicators
- **Test 5**: End-to-end integration
- **Test 6**: Confidence-based position sizing ← NEW

## Performance Considerations

### Benefits
✅ **Risk-adjusted sizing**: Positions scale with confidence, reducing exposure on uncertain signals  
✅ **Short capability**: Can profit from bearish regimes, not just bullish  
✅ **Confidence threshold**: Prevents weak signals from generating positions  
✅ **Regime awareness**: Different handling for bull/bear/neutral markets  

### Trade-offs
⚠️ **Higher requirements for shorts**: Need higher confidence to short than to go long  
⚠️ **Reduced position sizes**: Low confidence → smaller positions → potentially lower returns  
⚠️ **Complexity**: More parameters to tune and monitor  

## Recommended Settings

### For Conservative Traders
```python
min_confidence=0.6              # Higher threshold
max_position=0.5                # Limit to 50% positions
allow_shorts=False              # Long-only
short_confidence_threshold=0.8  # N/A
```

### For Moderate Traders
```python
min_confidence=0.5              # Default threshold
max_position=0.75               # Up to 75% positions
allow_shorts=True               # Enable shorts
short_confidence_threshold=0.7  # Require high confidence for shorts
```

### For Aggressive Traders
```python
min_confidence=0.4              # Lower threshold
max_position=1.0                # Full positions allowed
allow_shorts=True               # Enable shorts
short_confidence_threshold=0.6  # More permissive short threshold
```

## Future Enhancements

Potential improvements:
1. **Dynamic threshold adjustment**: Adjust confidence thresholds based on market volatility
2. **Sector-specific thresholds**: Different confidence requirements by sector
3. **Kelly criterion**: Optimal position sizing based on expected return and variance
4. **Risk parity**: Balance risk across positions rather than dollar amounts
5. **Drawdown-based scaling**: Reduce position sizes after drawdowns

## See Also

- [HMM Refactoring Summary](HMM_REFACTORING_SUMMARY.md)
- [Market Causality DAG](MARKET_CAUSALITY_DAG.md)
- [Multi-State Implementation](IMPLEMENTATION_COMPLETE.md)
