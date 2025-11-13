#!/usr/bin/env python3
"""
Test script to verify all imports work correctly after reorganization.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all critical imports from the reorganized structure."""
    
    print("="*80)
    print("TESTING IMPORTS AFTER REORGANIZATION")
    print("="*80)
    
    errors = []
    successes = []
    
    # Test 1: Causality module
    print("\n1. Testing causality module...")
    try:
        from src.causality.market_causality_dag import MarketCausalityDAG
        from src.causality.causal_feature_engine import CausalFeatureEngine
        from src.causality import MarketCausalityDAG as MarketCausalityDAG2
        from src.causality import CausalFeatureEngine as CausalFeatureEngine2
        successes.append("✓ Causality module imports successful")
    except Exception as e:
        errors.append(f"✗ Causality module import failed: {e}")
    
    # Test 2: HMM module
    print("2. Testing HMM module...")
    try:
        from src.hmm.hmm_analysis import AnalyzeHMM, setup_logging
        from src.hmm import AnalyzeHMM as AnalyzeHMM2
        successes.append("✓ HMM module imports successful")
    except Exception as e:
        errors.append(f"✗ HMM module import failed: {e}")
    
    # Test 3: Trading module
    print("3. Testing trading module...")
    try:
        from src.trading.strategies import BaseStrategy, HMMStrategy, DonchianBreakoutStrategy
        from src.trading.trading_agent import TradingAgent
        from src.trading.orchestrator import Orchestrator
        from src.trading.position import ManagedPosition, PositionState, CooldownReason
        from src.trading import HMMStrategy as HMMStrategy2
        successes.append("✓ Trading module imports successful")
    except Exception as e:
        errors.append(f"✗ Trading module import failed: {e}")
    
    # Test 4: Backtest module
    print("4. Testing backtest module...")
    try:
        from src.backtest.backtester import Backtester
        from src.backtest.optimizer import Optimizer
        from src.backtest import Backtester as Backtester2
        successes.append("✓ Backtest module imports successful")
    except Exception as e:
        errors.append(f"✗ Backtest module import failed: {e}")
    
    # Test 5: API module
    print("5. Testing API module...")
    try:
        from src.api.api import app
        from src.api import app as app2
        successes.append("✓ API module imports successful")
    except Exception as e:
        errors.append(f"✗ API module import failed: {e}")
    
    # Test 6: Utils module (paths)
    print("6. Testing utils module...")
    try:
        from src.utils.paths import (
            PROJECT_ROOT, DATA_DIR, CAUSALITY_CACHE_DIR, HMM_MODELS_DIR,
            CSV_DIR, LOGS_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR,
            CONFIG_DIR, DEFAULT_DAG_FILE, DEFAULT_ENV_FILE
        )
        from src.utils import CAUSALITY_CACHE_DIR as CACHE_DIR2
        
        # Verify paths exist
        assert DATA_DIR.exists(), "DATA_DIR doesn't exist"
        assert CAUSALITY_CACHE_DIR.exists(), "CAUSALITY_CACHE_DIR doesn't exist"
        assert HMM_MODELS_DIR.exists(), "HMM_MODELS_DIR doesn't exist"
        
        successes.append("✓ Utils module imports successful")
        successes.append(f"  PROJECT_ROOT: {PROJECT_ROOT}")
        successes.append(f"  DATA_DIR: {DATA_DIR}")
        successes.append(f"  CAUSALITY_CACHE_DIR: {CAUSALITY_CACHE_DIR}")
    except Exception as e:
        errors.append(f"✗ Utils module import failed: {e}")
    
    # Test 7: Cross-module dependencies
    print("7. Testing cross-module dependencies...")
    try:
        # Test that HMM can use paths
        from src.hmm.hmm_analysis import AnalyzeHMM
        assert hasattr(AnalyzeHMM, 'MODEL_DIR'), "AnalyzeHMM.MODEL_DIR not found"
        
        # Test that strategies can import HMM
        from src.trading.strategies import HMMStrategy
        
        # Test that backtester can import strategies
        from src.backtest.backtester import Backtester
        
        successes.append("✓ Cross-module dependencies work correctly")
    except Exception as e:
        errors.append(f"✗ Cross-module dependency failed: {e}")
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    if successes:
        print("\n✓ SUCCESSES:")
        for success in successes:
            print(f"  {success}")
    
    if errors:
        print("\n✗ ERRORS:")
        for error in errors:
            print(f"  {error}")
    
    print("\n" + "="*80)
    if not errors:
        print("✓ ALL TESTS PASSED! The reorganization was successful.")
        print("="*80)
        return True
    else:
        print(f"✗ {len(errors)} TEST(S) FAILED. Please review the errors above.")
        print("="*80)
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
