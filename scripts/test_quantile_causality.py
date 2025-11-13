#!/usr/bin/env python3
"""
Test script for the refactored HMM analysis module.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime
from src.hmm.hmm_analysis import AnalyzeHMM, setup_logging

def test_causality_comparison(ticker="NVDA"):
    """
    Train two models for the same ticker:
    1. Without causality filtering (all features)
    2. With causality filtering (only causal features)
    
    Compare the results.
    """
    setup_logging()
    
    logging.info(f"\n{'='*80}")
    logging.info(f"QUANTILE GRANGER CAUSALITY COMPARISON TEST")
    logging.info(f"Ticker: {ticker}")
    logging.info(f"{'='*80}\n")
    
    # Model 1: All features (no causality filtering)
    logging.info(f"\n{'='*80}")
    logging.info(f"MODEL 1: ALL FEATURES (NO CAUSALITY FILTERING)")
    logging.info(f"{'='*80}\n")
    
    start_time = datetime.now()
    model_all = AnalyzeHMM(
        ticker=ticker,
        n_components=2,
        model_order=1,
        use_causality_filter=False,
        force_retrain=True
    )
    time_all = datetime.now() - start_time
    
    logging.info(f"\nModel 1 Training Time: {time_all}")
    logging.info(f"Features Used: {model_all.features}")
    logging.info(f"Number of Features: {len(model_all.features)}")
    
    prediction_all = model_all.predict_next_day_outlook()
    logging.info(f"Prediction: {prediction_all['outlook']} (state {prediction_all['predicted_state']})")
    
    # Model 2: Only causal features
    logging.info(f"\n{'='*80}")
    logging.info(f"MODEL 2: CAUSAL FEATURES ONLY (WITH CAUSALITY FILTERING)")
    logging.info(f"{'='*80}\n")
    
    # Delete the model file so it retrains
    model_path = model_all.MODEL_DIR / f"{ticker}_2_1.pkl"
    if model_path.exists():
        os.remove(model_path)
    
    start_time = datetime.now()
    model_causal = AnalyzeHMM(
        ticker=ticker,
        n_components=2,
        model_order=1,
        use_causality_filter=True,
        causality_significance=0.05,
        force_retrain=True
    )
    time_causal = datetime.now() - start_time
    
    logging.info(f"\nModel 2 Training Time: {time_causal}")
    logging.info(f"Features Used: {model_causal.features}")
    logging.info(f"Number of Features: {len(model_causal.features)}")
    
    prediction_causal = model_causal.predict_next_day_outlook()
    logging.info(f"Prediction: {prediction_causal['outlook']} (state {prediction_causal['predicted_state']})")
    
    # Comparison
    logging.info(f"\n{'='*80}")
    logging.info(f"COMPARISON")
    logging.info(f"{'='*80}\n")
    
    logging.info(f"Training Time:")
    logging.info(f"  Model 1 (All features):     {time_all}")
    logging.info(f"  Model 2 (Causal features):  {time_causal}")
    logging.info(f"  Difference:                 {time_causal - time_all}")
    
    logging.info(f"\nFeature Count:")
    logging.info(f"  Model 1: {len(model_all.features)} features")
    logging.info(f"  Model 2: {len(model_causal.features)} features")
    logging.info(f"  Reduction: {len(model_all.features) - len(model_causal.features)} features removed")
    
    if model_causal.causality_results:
        logging.info(f"\nCausality Test Results:")
        for feature, result in model_causal.causality_results.items():
            status = "✓ CAUSAL" if result['is_causal'] else "✗ NOT CAUSAL"
            logging.info(f"  {feature:20s}: {status:15s} (p={result['min_p_value']:.4f}, q={result['best_quantile']})")
        
        removed_features = [f for f, r in model_causal.causality_results.items() if not r['is_causal']]
        if removed_features:
            logging.info(f"\nRemoved Features (non-causal):")
            for f in removed_features:
                logging.info(f"  - {f}")
        else:
            logging.info(f"\nNo features were removed (all were causal)")
    
    logging.info(f"\nPrediction Agreement:")
    if prediction_all['outlook'] == prediction_causal['outlook']:
        logging.info(f"  ✓ Both models predict: {prediction_all['outlook'].upper()}")
    else:
        logging.info(f"  ✗ Models disagree!")
        logging.info(f"    Model 1 (all features): {prediction_all['outlook'].upper()}")
        logging.info(f"    Model 2 (causal):       {prediction_causal['outlook'].upper()}")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"TEST COMPLETE")
    logging.info(f"{'='*80}\n")
    
    return {
        'model_all': model_all,
        'model_causal': model_causal,
        'time_all': time_all,
        'time_causal': time_causal,
        'prediction_all': prediction_all,
        'prediction_causal': prediction_causal
    }

if __name__ == "__main__":
    # Test with a ticker (default NVDA, or pass as command line argument)
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    
    try:
        results = test_causality_comparison(ticker)
        sys.exit(0)
    except Exception as e:
        logging.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
