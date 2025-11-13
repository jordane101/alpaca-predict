#!/usr/bin/env python3
"""
Quick Start Guide for Refactored 2-State HMM with S&P 500 Features

This script demonstrates how to use the new 2-state HMM implementation
with S&P 500 return features.

Author - Eli Jordan
Date - 10/17/2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from src.hmm.hmm_analysis import AnalyzeHMM

def example_1_basic_usage():
    """Example 1: Basic usage with automatic S&P 500 feature"""
    print("\n" + "="*60)
    print("Example 1: Basic 2-State HMM Analysis")
    print("="*60)
    
    ticker = "AAPL"
    print(f"\nAnalyzing {ticker}...")
    
    # Create analyzer - will automatically fetch S&P 500 data
    analyzer = AnalyzeHMM(
        ticker=ticker,
        n_components=2,  # 2-state model (negative/positive)
        model_order=1,   # First-order Markov model
        force_retrain=False  # Use cached model if available
    )
    
    # Get prediction
    prediction = analyzer.predict_next_day_outlook()
    
    # Display results
    print(f"\n{ticker} Analysis Results:")
    print(f"  Current State: {analyzer.data['Hidden_State'].iloc[-1]}")
    print(f"  Today's Return: {prediction['last_return']:.4f}")
    print(f"  Predicted Next State: {prediction['predicted_state']}")
    print(f"  Outlook: {prediction['outlook'].upper()}")
    print(f"  Expected Return: {prediction['predicted_state_mean_return']:.4f}")
    print(f"  Return Volatility: {prediction['predicted_state_std_return']:.4f}")
    
    # Show features used
    print(f"\nFeatures used: {analyzer.features}")
    
    # Show that S&P 500 data is included
    if 'SP500_Return' in analyzer.data.columns:
        sp500_stats = analyzer.data['SP500_Return'].describe()
        print(f"\nS&P 500 Return Statistics:")
        print(f"  Mean: {sp500_stats['mean']:.6f}")
        print(f"  Std: {sp500_stats['std']:.6f}")

def example_2_batch_with_shared_sp500():
    """Example 2: Batch analysis with shared S&P 500 data"""
    print("\n" + "="*60)
    print("Example 2: Batch Analysis with Shared S&P 500 Data")
    print("="*60)
    
    # First, get S&P 500 data once
    print("\nFetching S&P 500 data...")
    spy_analyzer = AnalyzeHMM("SPY", n_components=2, model_order=1)
    sp500_data = spy_analyzer.data[['SP500_Return']].copy()
    print(f"S&P 500 data loaded: {len(sp500_data)} rows")
    
    # Analyze multiple tickers using the same S&P 500 data
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    
    print(f"\nAnalyzing {len(tickers)} stocks...")
    results = []
    
    for ticker in tickers:
        # Pass pre-fetched S&P 500 data to avoid redundant API calls
        analyzer = AnalyzeHMM(
            ticker=ticker,
            n_components=2,
            model_order=1,
            sp500_data=sp500_data,
            verbose=False
        )
        
        prediction = analyzer.predict_next_day_outlook()
        results.append({
            'ticker': ticker,
            'outlook': prediction['outlook'],
            'expected_return': prediction['predicted_state_mean_return'],
            'current_state': int(analyzer.data['Hidden_State'].iloc[-1])
        })
    
    # Display summary
    print(f"\n{'Ticker':<8} {'State':<6} {'Outlook':<10} {'Expected Return':<15}")
    print("-" * 45)
    for r in results:
        print(f"{r['ticker']:<8} {r['current_state']:<6} {r['outlook']:<10} {r['expected_return']:>14.4f}")

def example_3_force_retrain():
    """Example 3: Force retrain to get fresh model"""
    print("\n" + "="*60)
    print("Example 3: Force Retrain for Fresh Model")
    print("="*60)
    
    ticker = "NVDA"
    print(f"\nForce retraining {ticker} model...")
    
    analyzer = AnalyzeHMM(
        ticker=ticker,
        n_components=2,
        model_order=1,
        force_retrain=True,  # Force retrain regardless of cache
        verbose=True
    )
    
    print(f"\nModel path: {analyzer.model_path}")
    print(f"Model exists: {analyzer.model_path.exists()}")
    
    prediction = analyzer.predict_next_day_outlook()
    print(f"\nOutlook: {prediction['outlook'].upper()}")

def example_4_regime_characteristics():
    """Example 4: Examine regime characteristics"""
    print("\n" + "="*60)
    print("Example 4: Regime Characteristics Analysis")
    print("="*60)
    
    ticker = "TSLA"
    print(f"\nAnalyzing regime characteristics for {ticker}...")
    
    analyzer = AnalyzeHMM(ticker=ticker, n_components=2, model_order=1)
    
    print(f"\nRegime Characteristics:")
    print(f"\nState Means:")
    print(analyzer.state_means)
    
    print(f"\nState Standard Deviations:")
    print(analyzer.state_stds)
    
    print(f"\nState Regimes (sorted by return):")
    for i, state in enumerate(analyzer.state_regimes):
        regime_type = 'negative' if i == 0 else 'positive'
        mean_return = analyzer.state_means.loc[state, 'Return']
        print(f"  State {state}: {regime_type.upper()} (Mean Return: {mean_return:.4f})")
    
    print(f"\nTransition Matrix:")
    print(analyzer.model.transmat_)

def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("HMM Refactoring - Quick Start Examples")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        example_1_basic_usage()
        example_2_batch_with_shared_sp500()
        example_3_force_retrain()
        example_4_regime_characteristics()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
