# Market Causality DAG Implementation

## Overview
Building a Directed Acyclic Graph (DAG) of market causality relationships to use causal structure for feature engineering instead of raw price data.

## Concept

Instead of using just technical indicators (SMA, volatility, etc.), we:
1. Test Granger causality between all pairs of stocks/sectors
2. Build a DAG showing which assets influence others
3. For each stock, use returns from its "causal parents" as features
4. Train HMMs using this causality-informed feature set

## Benefits

1. **Captures market structure**: Identifies which stocks/sectors lead vs lag
2. **Reduces dimensionality**: Only include causally relevant stocks
3. **Better generalization**: Features based on actual market relationships
4. **Sector analysis**: Can identify sector-level causality
5. **Portfolio construction**: Understand contagion and spillover effects

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MARKET CAUSALITY DAG                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SPY (Market) ──→ Tech Sector ──→ Individual Tech Stocks        │
│      │                │                                          │
│      ├──→ Energy ────→ XLE ──→ Individual Energy Stocks         │
│      │                                                           │
│      └──→ Financials ─→ XLF ──→ Individual Financial Stocks     │
│                                                                  │
│  For each stock, features = returns of causal ancestors         │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Pairwise Causality Testing
- Test Granger causality between all stock pairs
- Create adjacency matrix of causal relationships
- Store p-values and lag structures

### Phase 2: DAG Construction
- Build directed graph from causality matrix
- Check for cycles (ensure it's acyclic)
- Identify strongly connected components
- Topological sorting for training order

### Phase 3: Feature Engineering
- For each stock, identify causal parents in DAG
- Use parent returns as features instead of technical indicators
- Handle lagged relationships (e.g., if A causes B with lag 2)

### Phase 4: HMM Training
- Train HMMs using causality-based features
- Compare performance vs technical indicator features
- Analyze prediction improvements

## Example

```python
# Universe of stocks
universe = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

# Build causality DAG
dag = MarketCausalityDAG(universe, start_date='2023-01-01')
dag.test_all_pairwise_causality()
dag.build_graph()

# For AAPL, find causal parents
parents = dag.get_causal_parents('AAPL')
# → ['SPY', 'QQQ', 'MSFT']  # These Granger-cause AAPL

# Train HMM with causal features
model = AnalyzeHMM(
    ticker='AAPL',
    use_causal_features=True,
    causal_parents=parents,
    causal_dag=dag
)
```

## Key Classes to Implement

1. **MarketCausalityDAG**
   - test_pairwise_causality(stock_a, stock_b)
   - build_adjacency_matrix()
   - detect_cycles()
   - get_causal_parents(stock)
   - get_causal_children(stock)
   - visualize_graph()

2. **CausalFeatureEngine**
   - extract_causal_features(stock, dag)
   - create_lagged_features(parent_returns, optimal_lag)
   - handle_missing_data()

3. **AnalyzeHMM (Extended)**
   - use_causal_dag parameter
   - load features from causal parents
   - train with network-aware features

## Data Requirements

- Historical returns for all stocks in universe
- Sufficient history (2+ years) for reliable causality tests
- Aligned timestamps (handle different trading hours)
- Handle corporate actions (splits, dividends)

## Validation

- Compare prediction accuracy: causal features vs technical indicators
- Test on out-of-sample period
- Analyze feature importance
- Check DAG stability over time (does structure change?)

## Next Steps

1. Implement MarketCausalityDAG class
2. Create pairwise causality testing infrastructure
3. Build graph visualization tools
4. Integrate with existing HMM training
5. Run comprehensive backtests
6. Analyze sector-level causality patterns
