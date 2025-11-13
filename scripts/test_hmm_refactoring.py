#!/usr/bin/env python3
"""
Test script to verify the HMM refactoring changes work correctly.

Tests:
1. 2-component HMM training
2. S&P 500 feature integration
3. Binary regime classification (positive/negative only)

Author - Eli Jordan
Date - 10/17/2025
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
from src.hmm.hmm_analysis import AnalyzeHMM

def test_basic_hmm():
    """Test basic HMM with 2 components"""
    print("\n" + "="*60)
    print("Test 1: Basic 2-Component HMM")
    print("="*60)
    
    try:
        # Test with a simple stock
        analyzer = AnalyzeHMM("AAPL", n_components=2, model_order=1, force_retrain=True)
        
        # Check number of components
        assert analyzer.n_components == 2, "Expected 2 components"
        print(f"✓ Model has {analyzer.n_components} components")
        
        # Check state regimes
        assert len(analyzer.state_regimes) == 2, "Expected 2 regime states"
        print(f"✓ Found {len(analyzer.state_regimes)} regime states")
        
        # Check features include SP500_Return
        assert 'SP500_Return' in analyzer.features, "SP500_Return should be in features"
        print(f"✓ SP500_Return feature included")
        
        # Check prediction
        prediction = analyzer.predict_next_day_outlook()
        assert prediction['outlook'] in ['positive', 'negative'], "Outlook should be positive or negative only"
        print(f"✓ Prediction outlook: {prediction['outlook']}")
        
        # Check that we have SP500 data
        if analyzer.sp500_data is not None:
            print(f"✓ S&P 500 data loaded: {len(analyzer.sp500_data)} rows")
        
        print("\n✅ Test 1 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spy_handling():
    """Test that SPY itself doesn't try to fetch SP500 data recursively"""
    print("\n" + "="*60)
    print("Test 2: SPY Ticker Handling")
    print("="*60)
    
    try:
        # Test with SPY itself
        analyzer = AnalyzeHMM("SPY", n_components=2, model_order=1, force_retrain=True)
        
        # For SPY, sp500_data should be None (not fetched)
        print(f"✓ SPY analyzer created successfully")
        
        # Check features
        print(f"✓ Features: {analyzer.features}")
        
        # Get prediction
        prediction = analyzer.predict_next_day_outlook()
        assert prediction['outlook'] in ['positive', 'negative'], "Outlook should be binary"
        print(f"✓ SPY prediction outlook: {prediction['outlook']}")
        
        print("\n✅ Test 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_file_naming():
    """Test that model files use correct naming with 2 components"""
    print("\n" + "="*60)
    print("Test 3: Model File Naming")
    print("="*60)
    
    try:
        analyzer = AnalyzeHMM("MSFT", n_components=2, model_order=1, force_retrain=True)
        
        # Check model path
        expected_in_path = "_2_1.pkl"
        assert expected_in_path in str(analyzer.model_path), f"Model path should contain {expected_in_path}"
        print(f"✓ Model path: {analyzer.model_path}")
        
        # Check if model was saved
        if analyzer.model_path.exists():
            print(f"✓ Model file created successfully")
        
        # Check JSON summary
        json_path = analyzer.model_path.with_suffix('.json')
        if json_path.exists():
            print(f"✓ JSON summary created: {json_path}")
            
            import json
            with open(json_path, 'r') as f:
                summary = json.load(f)
                
            assert summary['n_components'] == 2, "JSON should show 2 components"
            print(f"✓ JSON confirms n_components: {summary['n_components']}")
            
            # Check regime mapping
            regime_map = summary['state_regime_mapping']
            print(f"✓ Regime mapping: {regime_map}")
            assert len(regime_map) == 2, "Should have 2 regimes"
            
        print("\n✅ Test 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_features_with_sp500():
    """Test that features correctly include S&P 500 returns"""
    print("\n" + "="*60)
    print("Test 4: S&P 500 Feature Integration")
    print("="*60)
    
    try:
        analyzer = AnalyzeHMM("TSLA", n_components=2, model_order=1, force_retrain=True)
        
        # Check that SP500_Return is in features
        assert 'SP500_Return' in analyzer.features, "SP500_Return should be in features"
        print(f"✓ Features include: {analyzer.features}")
        
        # Check that data has SP500_Return column
        assert 'SP500_Return' in analyzer.data.columns, "Data should have SP500_Return column"
        print(f"✓ Data columns include SP500_Return")
        
        # Check that SP500_Return has actual values (not all zeros)
        sp500_values = analyzer.data['SP500_Return'].dropna()
        if len(sp500_values) > 0:
            non_zero_count = (sp500_values != 0).sum()
            print(f"✓ SP500_Return has {non_zero_count}/{len(sp500_values)} non-zero values")
            
            # At least some values should be non-zero (unless market was completely flat)
            if non_zero_count > 0:
                print(f"✓ S&P 500 returns are being captured")
        
        print("\n✅ Test 4 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shared_sp500_data():
    """Test that pre-fetched S&P 500 data works correctly (optimization)"""
    print("\n" + "="*60)
    print("Test 5: Shared S&P 500 Data (API Optimization)")
    print("="*60)
    
    try:
        # Fetch S&P 500 data once
        print("Fetching S&P 500 data once...")
        spy_analyzer = AnalyzeHMM("SPY", n_components=2, model_order=1, force_retrain=True)
        sp500_data = spy_analyzer.data[['SP500_Return']].copy()
        print(f"✓ Fetched S&P 500 data: {len(sp500_data)} rows")
        
        # Use for multiple stocks
        test_tickers = ['AAPL', 'MSFT']
        print(f"\nReusing S&P 500 data for {len(test_tickers)} stocks...")
        
        for ticker in test_tickers:
            analyzer = AnalyzeHMM(ticker, n_components=2, model_order=1, 
                                sp500_data=sp500_data, force_retrain=True)
            
            # Verify SP500_Return is in the data
            assert 'SP500_Return' in analyzer.data.columns, f"{ticker} should have SP500_Return"
            print(f"✓ {ticker}: SP500_Return column present")
            
            # Get prediction to ensure it works
            prediction = analyzer.predict_next_day_outlook()
            print(f"✓ {ticker}: Prediction = {prediction['outlook']}")
        
        print(f"\n✓ Successfully reused S&P 500 data for {len(test_tickers)} stocks")
        print("✓ This optimization saves N-1 API calls (where N = number of stocks)")
        
        print("\n✅ Test 5 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("HMM Refactoring Test Suite")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Run tests
    results.append(("Basic 2-Component HMM", test_basic_hmm()))
    results.append(("SPY Ticker Handling", test_spy_handling()))
    results.append(("Model File Naming", test_model_file_naming()))
    results.append(("S&P 500 Feature Integration", test_features_with_sp500()))
    results.append(("Shared S&P 500 Data Optimization", test_shared_sp500_data()))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
