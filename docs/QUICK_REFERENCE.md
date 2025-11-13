╔══════════════════════════════════════════════════════════════════════╗
║              HMM REFACTORING - QUICK REFERENCE CARD                  ║
╚══════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────┐
│ 1. WHAT CHANGED?                                                    │
└─────────────────────────────────────────────────────────────────────┘

  HMM States:     3 → 2
  New Feature:    S&P 500 Returns
  Classification: Negative/Neutral/Positive → Negative/Positive

┌─────────────────────────────────────────────────────────────────────┐
│ 2. WHY THE CHANGE?                                                  │
└─────────────────────────────────────────────────────────────────────┘

  ✓ Simpler binary regime classification
  ✓ More decisive trading signals (no neutral waiting)
  ✓ Market context from S&P 500 correlation
  ✓ Better performance on 2-state models in testing

┌─────────────────────────────────────────────────────────────────────┐
│ 3. QUICK START                                                      │
└─────────────────────────────────────────────────────────────────────┘

  # Step 1: Clean up old models
  python cleanup_old_models.py

  # Step 2: Run tests
  python test_hmm_refactoring.py

  # Step 3: Try it out
  python quick_start_hmm.py

┌─────────────────────────────────────────────────────────────────────┐
│ 4. BASIC USAGE                                                      │
└─────────────────────────────────────────────────────────────────────┘

  from hmm_analysis import AnalyzeHMM

  # Analyze a stock (auto-fetches S&P 500 data)
  analyzer = AnalyzeHMM("AAPL", n_components=2, model_order=1)
  
  # Get prediction
  prediction = analyzer.predict_next_day_outlook()
  print(f"Outlook: {prediction['outlook']}")  # 'positive' or 'negative'

┌─────────────────────────────────────────────────────────────────────┐
│ 5. BATCH PROCESSING (Efficient)                                     │
└─────────────────────────────────────────────────────────────────────┘

  # Fetch S&P 500 data once
  spy_data = AnalyzeHMM("SPY").data[['SP500_Return']]
  
  # Reuse for multiple stocks
  for ticker in ['AAPL', 'MSFT', 'GOOGL']:
      analyzer = AnalyzeHMM(ticker, sp500_data=spy_data)
      prediction = analyzer.predict_next_day_outlook()

┌─────────────────────────────────────────────────────────────────────┐
│ 6. FEATURES INCLUDED                                                │
└─────────────────────────────────────────────────────────────────────┘

  Base Features:
    • Return            - Stock daily/weekly return
    • Volatility        - 30-day rolling std of returns
    • SMA_50 or SMA_10 - Simple moving average
    • SP500_Return     - S&P 500 return (NEW!)

  + Lagged features if model_order > 1

┌─────────────────────────────────────────────────────────────────────┐
│ 7. MODEL FILES                                                      │
└─────────────────────────────────────────────────────────────────────┘

  Location:  hmm_models/
  
  Naming:    {ticker}_{n_components}_{model_order}.pkl
  
  Examples:
    AAPL_2_1.pkl      ✓ New format (2 components)
    AAPL_2_1.json     ✓ Human-readable summary
    AAPL_3_1.pkl      ✗ Old format (delete)

┌─────────────────────────────────────────────────────────────────────┐
│ 8. PREDICTION OUTPUT                                                │
└─────────────────────────────────────────────────────────────────────┘

  prediction = {
      'outlook': 'positive' or 'negative',
      'predicted_state': 0 or 1,
      'predicted_state_mean_return': float,
      'predicted_state_std_return': float,
      'last_return': float,
      'comparison': 'higher', 'lower', or 'the same'
  }

┌─────────────────────────────────────────────────────────────────────┐
│ 9. TROUBLESHOOTING                                                  │
└─────────────────────────────────────────────────────────────────────┘

  Problem: Old models incompatible
  → Run: python cleanup_old_models.py

  Problem: "S&P 500 data not available"
  → Check Alpaca API credentials

  Problem: Still seeing 'neutral' outlook
  → Verify you're using updated code

┌─────────────────────────────────────────────────────────────────────┐
│ 10. DOCUMENTATION                                                   │
└─────────────────────────────────────────────────────────────────────┘

  HMM_REFACTORING.md          - Full documentation
  HMM_REFACTORING_SUMMARY.md  - Technical summary
  TODO_CHECKLIST.md           - Implementation checklist
  quick_start_hmm.py          - Usage examples

┌─────────────────────────────────────────────────────────────────────┐
│ 11. TESTING                                                         │
└─────────────────────────────────────────────────────────────────────┘

  Unit Tests:           python test_hmm_refactoring.py
  Single Stock Test:    python hmm_analysis.py
  Examples:             python quick_start_hmm.py
  Backtest:             python backtester.py
  Live Trading:         python trader.py

┌─────────────────────────────────────────────────────────────────────┐
│ 12. IMPORTANT NOTES                                                 │
└─────────────────────────────────────────────────────────────────────┘

  ⚠️  Old 3-component models WILL NOT WORK
  ⚠️  Must delete old models or force retrain
  ✓  S&P 500 data fetched automatically
  ✓  Minimal API overhead (1 call per ticker)
  ✓  SPY analysis skips S&P 500 fetch (no recursion)

╔══════════════════════════════════════════════════════════════════════╗
║  For detailed documentation, see: HMM_REFACTORING.md                 ║
╚══════════════════════════════════════════════════════════════════════╝
