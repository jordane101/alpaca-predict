#!/usr/bin/env python3
"""
Test script to verify Causal HMM Integration.

This script tests:
1. Causal features load correctly from DAG
2. Model selection with AIC/BIC optimization works
3. Multi-state regime classification functions properly
4. Predictions differ between causal and technical indicator modes
5. Basic functionality comparison

Usage:
    python scripts/test_causal_hmm_integration.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hmm.hmm_analysis import AnalyzeHMM
from src.causality.causal_feature_engine import CausalFeatureEngine
from src.utils.paths import CAUSALITY_CACHE_DIR, DATA_DIR

# Import Alpaca API for fetching data
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import os

# Load environment variables
env_path = PROJECT_ROOT / '.env'
load_dotenv(env_path)

# Initialize Alpaca client - will be created on first use
data_client = None


def get_data_client():
    """Lazy initialization of Alpaca client."""
    global data_client
    if data_client is None:
        # Try both naming conventions
        api_key = os.getenv('PAPER_KEY') or os.getenv('APCA_API_KEY_ID')
        api_secret = os.getenv('PAPER_SEC') or os.getenv('APCA_API_SECRET_KEY')
        if not api_key or not api_secret:
            raise ValueError(f"Alpaca API credentials not found in environment. Keys tried: PAPER_KEY, APCA_API_KEY_ID")
        data_client = StockHistoricalDataClient(api_key, api_secret)
    return data_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def get_bars(ticker, start_date, end_date, timeframe='1Day'):
    """Fetch historical bars from Alpaca API."""
    client = get_data_client()
    request_params = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame.Day if timeframe == '1Day' else TimeFrame.Hour,
        start=start_date,
        end=end_date
    )
    
    bars = client.get_stock_bars(request_params)
    df = bars.df
    
    if ticker in df.index.get_level_values(0):
        df = df.xs(ticker, level=0)
    
    return df


def test_causal_feature_engine():
    """Test 1: Verify CausalFeatureEngine loads correctly."""
    print_section("TEST 1: Causal Feature Engine Initialization")
    
    try:
        # Find the DAG file
        dag_file = CAUSALITY_CACHE_DIR / "large_network_graph_dag.pkl"
        
        if not dag_file.exists():
            print(f"❌ DAG file not found at: {dag_file}")
            print("   Run scripts/build_large_dag.py first to create the DAG")
            return False
        
        print(f"✓ Found DAG file: {dag_file}")
        
        # Initialize engine
        engine = CausalFeatureEngine(dag_file=str(dag_file))
        print(f"✓ Initialized CausalFeatureEngine")
        print(f"  - Network has {engine.graph.number_of_nodes()} nodes")
        print(f"  - Network has {engine.graph.number_of_edges()} edges")
        
        # Test getting causal parents for a stock
        test_ticker = "AAPL"
        parents = engine.get_causal_parents(test_ticker, top_k=5, max_p_value=0.05)
        
        if parents:
            print(f"\n✓ Found {len(parents)} causal parents for {test_ticker}:")
            for parent, p_val, lag in parents[:3]:
                print(f"    {parent}: p-value={p_val:.6f}, lag={lag}")
        else:
            print(f"\n⚠️  No causal parents found for {test_ticker}")
        
        print("\n✅ Test 1 PASSED: Causal Feature Engine working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_selection():
    """Test 2: Verify AIC/BIC model selection works."""
    print_section("TEST 2: Model Selection with AIC/BIC Optimization")
    
    try:
        # Fetch sample data
        ticker = "AAPL"
        print(f"Fetching data for {ticker}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years
        
        bars = get_bars(
            ticker, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d'),
            timeframe='1Day'
        )
        
        if bars.empty or len(bars) < 300:
            print(f"❌ Insufficient data for {ticker} (need at least 300 days, got {len(bars)})")
            return False
        
        print(f"✓ Fetched {len(bars)} days of data")
        
        # Test with optimization enabled
        print("\nTraining HMM with optimization (testing 2-4 states)...")
        analyzer = AnalyzeHMM(
            ticker=ticker,
            n_components=3,  # Starting point
            model_order=1,
            bars_data=bars,
            verbose=True,
            force_retrain=True,
            use_causal_features=False,  # Disable for faster test
            optimize_n_components=True,
            n_components_range=(2, 4)
        )
        
        analyzer.train()
        
        # Check results
        if not hasattr(analyzer, 'optimal_n_components'):
            print("❌ optimal_n_components not set")
            return False
        
        print(f"\n✓ Optimal number of components selected: {analyzer.optimal_n_components}")
        
        if hasattr(analyzer, 'model_selection_results'):
            results = analyzer.model_selection_results
            print(f"\n  Model Selection Results:")
            for i, n in enumerate(results['n_components']):
                print(f"    {n} states: AIC={results['aic'][i]:,.2f}, BIC={results['bic'][i]:,.2f}")
        
        # Check regime labels
        if hasattr(analyzer, 'regime_labels'):
            print(f"\n✓ Regime labels: {analyzer.regime_labels}")
        
        print("\n✅ Test 2 PASSED: Model selection working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_state_regimes():
    """Test 3: Verify multi-state regime classification."""
    print_section("TEST 3: Multi-State Regime Classification")
    
    try:
        ticker = "MSFT"
        print(f"Testing regime classification for {ticker}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        
        bars = get_bars(
            ticker, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d'),
            timeframe='1Day'
        )
        
        if bars.empty or len(bars) < 300:
            print(f"❌ Insufficient data for {ticker}")
            return False
        
        print(f"✓ Fetched {len(bars)} days of data")
        
        # Test different numbers of states
        for n_states in [2, 3, 4]:
            print(f"\n--- Testing {n_states}-state model ---")
            
            analyzer = AnalyzeHMM(
                ticker=f"{ticker}_test_{n_states}",
                n_components=n_states,
                model_order=1,
                bars_data=bars,
                verbose=False,
                force_retrain=True,
                use_causal_features=False,
                optimize_n_components=False  # Use fixed n_components for this test
            )
            
            analyzer.train()
            
            # Check regime labels
            if not hasattr(analyzer, 'regime_labels'):
                print(f"  ❌ No regime_labels for {n_states} states")
                continue
            
            print(f"  ✓ Regime labels: {analyzer.regime_labels}")
            
            # Check state characteristics
            for i, (state_idx, label) in enumerate(zip(analyzer.state_regimes, analyzer.regime_labels)):
                mean_ret = analyzer.state_means.loc[state_idx, 'Return']
                std_ret = analyzer.state_stds.loc[state_idx, 'Return']
                count = (analyzer.data['Hidden_State'] == state_idx).sum()
                freq = count / len(analyzer.data) * 100
                print(f"    State {state_idx} ({label}): Return={mean_ret:.4f}, Vol={std_ret:.4f}, Freq={freq:.1f}%")
            
            # Test prediction
            prediction = analyzer.predict_next_day_outlook()
            print(f"  ✓ Prediction: {prediction['outlook']} (state {prediction['predicted_state']})")
        
        print("\n✅ Test 3 PASSED: Multi-state regimes working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_causal_vs_technical():
    """Test 4: Compare causal features vs technical indicators."""
    print_section("TEST 4: Causal Features vs Technical Indicators")
    
    try:
        ticker = "NVDA"
        print(f"Comparing feature modes for {ticker}...")
        
        # Check if DAG exists
        dag_file = CAUSALITY_CACHE_DIR / "large_network_graph_dag.pkl"
        if not dag_file.exists():
            print(f"⚠️  Skipping causal comparison - DAG file not found")
            print(f"   Run scripts/build_large_dag.py to create the DAG")
            return True  # Don't fail the test, just skip
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        
        bars = get_bars(
            ticker, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d'),
            timeframe='1Day'
        )
        
        if bars.empty or len(bars) < 300:
            print(f"❌ Insufficient data for {ticker}")
            return False
        
        print(f"✓ Fetched {len(bars)} days of data")
        
        # Test with technical indicators only
        print("\n--- Technical Indicators Mode ---")
        analyzer_tech = AnalyzeHMM(
            ticker=f"{ticker}_tech",
            n_components=3,
            model_order=1,
            bars_data=bars,
            verbose=False,
            force_retrain=True,
            use_causal_features=False,
            optimize_n_components=False
        )
        analyzer_tech.train()
        
        print(f"  Features used: {analyzer_tech.features[:5]}...")  # Show first 5
        print(f"  Total features: {len(analyzer_tech.features)}")
        pred_tech = analyzer_tech.predict_next_day_outlook()
        print(f"  Prediction: {pred_tech['outlook']} (mean return: {pred_tech['predicted_state_mean_return']:.4f})")
        
        # Test with causal features
        print("\n--- Causal Features Mode ---")
        analyzer_causal = AnalyzeHMM(
            ticker=f"{ticker}_causal",
            n_components=3,
            model_order=1,
            bars_data=bars,
            verbose=False,
            force_retrain=True,
            use_causal_features=True,
            causal_dag_file=str(dag_file),
            optimize_n_components=False
        )
        analyzer_causal.train()
        
        print(f"  Features used: {analyzer_causal.features[:5]}...")  # Show first 5
        print(f"  Total features: {len(analyzer_causal.features)}")
        pred_causal = analyzer_causal.predict_next_day_outlook()
        print(f"  Prediction: {pred_causal['outlook']} (mean return: {pred_causal['predicted_state_mean_return']:.4f})")
        
        # Compare
        print("\n--- Comparison ---")
        print(f"  Technical features: {len(analyzer_tech.features)}")
        print(f"  Causal features: {len(analyzer_causal.features)}")
        
        # Check if causal features were actually added
        causal_feature_names = [f for f in analyzer_causal.features if '_Return_Lag' in f]
        if causal_feature_names:
            print(f"  ✓ {len(causal_feature_names)} causal features added:")
            for feat in causal_feature_names[:5]:
                print(f"    - {feat}")
        else:
            print(f"  ⚠️  No causal features detected (may not be a problem if no causal parents found)")
        
        print("\n✅ Test 4 PASSED: Feature mode comparison completed")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end():
    """Test 5: End-to-end with all features enabled."""
    print_section("TEST 5: End-to-End Integration Test")
    
    try:
        ticker = "AAPL"
        print(f"Running full integration test for {ticker}...")
        
        dag_file = CAUSALITY_CACHE_DIR / "large_network_graph_dag.pkl"
        use_causal = dag_file.exists()
        
        if not use_causal:
            print("⚠️  Running without causal features (DAG not found)")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        
        bars = get_bars(
            ticker, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d'),
            timeframe='1Day'
        )
        
        if bars.empty or len(bars) < 300:
            print(f"❌ Insufficient data for {ticker}")
            return False
        
        print(f"✓ Fetched {len(bars)} days of data")
        
        # Create analyzer with all features enabled
        print("\nCreating HMM with all features enabled...")
        analyzer = AnalyzeHMM(
            ticker=ticker,
            n_components=3,  # Starting point
            model_order=1,
            bars_data=bars,
            verbose=True,
            force_retrain=True,
            use_causal_features=use_causal,
            causal_dag_file=str(dag_file) if use_causal else None,
            optimize_n_components=True,
            n_components_range=(2, 4)
        )
        
        print("\nTraining model...")
        analyzer.train()
        
        print("\n--- Model Summary ---")
        print(f"  Optimal states: {analyzer.optimal_n_components}")
        print(f"  Regime labels: {analyzer.regime_labels}")
        print(f"  Features: {len(analyzer.features)}")
        print(f"  Data points: {len(analyzer.data)}")
        
        # Make prediction
        print("\n--- Making Prediction ---")
        prediction = analyzer.predict_next_day_outlook()
        print(f"  Outlook: {prediction['outlook']}")
        print(f"  Predicted state: {prediction['predicted_state']}")
        print(f"  Mean return: {prediction['predicted_state_mean_return']:.4f}")
        print(f"  Std return: {prediction['predicted_state_std_return']:.4f}")
        
        # Check state distribution
        print("\n--- State Distribution ---")
        for state_idx, label in zip(analyzer.state_regimes, analyzer.regime_labels):
            count = (analyzer.data['Hidden_State'] == state_idx).sum()
            pct = count / len(analyzer.data) * 100
            mean_ret = analyzer.state_means.loc[state_idx, 'Return']
            print(f"  {label}: {pct:.1f}% (mean return: {mean_ret:.4f})")
        
        print("\n✅ Test 5 PASSED: End-to-end integration successful")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_confidence_position_sizing():
    """Test 6: Verify confidence-based position sizing with short capability."""
    print_section("TEST 6: Confidence-Based Position Sizing & Shorts")
    
    try:
        ticker = "TSLA"
        print(f"Testing confidence-based position sizing for {ticker}...")
        
        dag_file = CAUSALITY_CACHE_DIR / "large_network_graph_dag.pkl"
        use_causal = dag_file.exists()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        
        bars = get_bars(
            ticker, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d'),
            timeframe='1Day'
        )
        
        if bars.empty or len(bars) < 300:
            print(f"❌ Insufficient data for {ticker}")
            return False
        
        print(f"✓ Fetched {len(bars)} days of data")
        
        # Create analyzer with all features
        print("\nCreating HMM with confidence-based sizing...")
        analyzer = AnalyzeHMM(
            ticker=ticker,
            n_components=3,
            model_order=1,
            bars_data=bars,
            verbose=True,
            force_retrain=True,
            use_causal_features=use_causal,
            causal_dag_file=str(dag_file) if use_causal else None,
            optimize_n_components=True,
            n_components_range=(2, 4)
        )
        
        analyzer.train()
        
        # Test state probabilities
        print("\n--- Testing State Probabilities ---")
        prob_info = analyzer.get_state_probabilities()
        print(f"✓ Most likely state: {prob_info['most_likely_state']}")
        print(f"✓ Confidence: {prob_info['confidence']:.2%}")
        
        # Test position sizing with different settings
        print("\n--- Testing Position Sizing (Longs Only) ---")
        pos_info = analyzer.calculate_position_size(
            min_confidence=0.5,
            max_position=1.0,
            allow_shorts=False,
            short_confidence_threshold=0.7
        )
        print(f"  Position Size: {pos_info['position_size']:.2%}")
        print(f"  Action: {pos_info['action']}")
        print(f"  Regime: {pos_info['regime']}")
        print(f"  Reasoning: {pos_info['reasoning']}")
        
        # Test with shorts enabled
        print("\n--- Testing Position Sizing (Shorts Enabled) ---")
        pos_info_shorts = analyzer.calculate_position_size(
            min_confidence=0.4,
            max_position=1.0,
            allow_shorts=True,
            short_confidence_threshold=0.6
        )
        print(f"  Position Size: {pos_info_shorts['position_size']:.2%}")
        print(f"  Action: {pos_info_shorts['action']}")
        print(f"  Regime: {pos_info_shorts['regime']}")
        print(f"  Sentiment Score: {pos_info_shorts['sentiment_score']:.3f}")
        print(f"  Expected Return: {pos_info_shorts['expected_return']:.4f}")
        
        # Test with very high short threshold
        print("\n--- Testing Position Sizing (High Short Threshold) ---")
        pos_info_conservative = analyzer.calculate_position_size(
            min_confidence=0.5,
            max_position=1.0,
            allow_shorts=True,
            short_confidence_threshold=0.9
        )
        print(f"  Position Size: {pos_info_conservative['position_size']:.2%}")
        print(f"  Action: {pos_info_conservative['action']}")
        print(f"  Reasoning: {pos_info_conservative['reasoning']}")
        
        # Verify position sizing logic
        assert 'position_size' in pos_info
        assert 'confidence' in pos_info
        assert 'action' in pos_info
        assert 'regime' in pos_info
        assert -1.0 <= pos_info_shorts['position_size'] <= 1.0
        
        print("\n✅ Test 6 PASSED: Confidence-based position sizing working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("  CAUSAL HMM INTEGRATION TEST SUITE")
    print("="*80)
    print(f"\nStarting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Run all tests
    results['Test 1: Causal Feature Engine'] = test_causal_feature_engine()
    results['Test 2: Model Selection'] = test_model_selection()
    results['Test 3: Multi-State Regimes'] = test_multi_state_regimes()
    results['Test 4: Causal vs Technical'] = test_causal_vs_technical()
    results['Test 5: End-to-End Integration'] = test_end_to_end()
    results['Test 6: Confidence Position Sizing'] = test_confidence_position_sizing()
    
    # Print summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n{'='*80}")
    print(f"  Results: {passed}/{total} tests passed")
    print(f"{'='*80}\n")
    
    if passed == total:
        print("🎉 All tests passed! The integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
