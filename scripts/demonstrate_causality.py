#!/usr/bin/env python3
"""
Demonstrate causality filtering by testing with a ticker where some features may not be causal,
or by artificially adjusting the significance threshold.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime
from src.hmm.hmm_analysis import AnalyzeHMM, setup_logging

def demonstrate_causality_filtering(ticker="NVDA", significance=0.05):
    """
    Show the difference in features used with different causality thresholds.
    """
    setup_logging()
    
    logging.info(f"\n{'='*80}")
    logging.info(f"CAUSALITY FILTERING DEMONSTRATION")
    logging.info(f"Ticker: {ticker}")
    logging.info(f"Significance Level: {significance}")
    logging.info(f"{'='*80}\n")
    
    # Test 1: Without causality filtering (all features)
    logging.info(f"{'='*80}")
    logging.info(f"TEST 1: WITHOUT CAUSALITY FILTERING")
    logging.info(f"{'='*80}\n")
    
    model_no_filter = AnalyzeHMM(
        ticker=ticker,
        n_components=2,
        model_order=1,
        use_causality_filter=False,
        force_retrain=True
    )
    
    logging.info(f"\nFeatures used (NO filtering): {model_no_filter.features}")
    logging.info(f"Total features: {len(model_no_filter.features)}")
    
    # Test 2: With causality filtering
    logging.info(f"\n{'='*80}")
    logging.info(f"TEST 2: WITH CAUSALITY FILTERING (p < {significance})")
    logging.info(f"{'='*80}\n")
    
    model_with_filter = AnalyzeHMM(
        ticker=ticker,
        n_components=2,
        model_order=1,
        use_causality_filter=True,
        causality_significance=significance,
        force_retrain=True
    )
    
    logging.info(f"\nFeatures used (WITH filtering): {model_with_filter.features}")
    logging.info(f"Total features: {len(model_with_filter.features)}")
    
    # Comparison
    logging.info(f"\n{'='*80}")
    logging.info(f"COMPARISON")
    logging.info(f"{'='*80}\n")
    
    removed_features = [f for f in model_no_filter.base_features if f not in model_with_filter.base_features]
    kept_features = [f for f in model_with_filter.base_features if f in model_no_filter.base_features]
    
    logging.info(f"Original features: {model_no_filter.base_features}")
    logging.info(f"Kept features:     {kept_features}")
    logging.info(f"Removed features:  {removed_features if removed_features else 'None - all features passed causality test'}")
    logging.info(f"\nFeature reduction: {len(model_no_filter.features)} → {len(model_with_filter.features)} features")
    
    if model_with_filter.causality_results:
        logging.info(f"\n{'='*80}")
        logging.info(f"DETAILED CAUSALITY RESULTS")
        logging.info(f"{'='*80}\n")
        
        for feature, result in model_with_filter.causality_results.items():
            status = "✓ KEPT" if result['is_causal'] else "✗ REMOVED"
            logging.info(f"{feature:20s}: {status:15s} p={result['min_p_value']:.4f} at q={result['best_quantile']}")
            
            # Show quantile-specific results
            logging.info(f"{'':20s}   Quantile results:")
            for q, qresult in sorted(result['quantile_results'].items()):
                sig = "***" if qresult['p_value'] < significance else ""
                logging.info(f"{'':20s}     q={q}: p={qresult['p_value']:.4f} (lag={qresult['lag']}) {sig}")
    
    # Show model predictions
    logging.info(f"\n{'='*80}")
    logging.info(f"MODEL PREDICTIONS")
    logging.info(f"{'='*80}\n")
    
    pred_no_filter = model_no_filter.predict_next_day_outlook()
    pred_with_filter = model_with_filter.predict_next_day_outlook()
    
    logging.info(f"Without filtering: {pred_no_filter['outlook'].upper()} (state {pred_no_filter['predicted_state']})")
    logging.info(f"With filtering:    {pred_with_filter['outlook'].upper()} (state {pred_with_filter['predicted_state']})")
    
    if pred_no_filter['outlook'] != pred_with_filter['outlook']:
        logging.info(f"\n⚠️  PREDICTIONS DIFFER!")
    else:
        logging.info(f"\n✓ Predictions agree")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"KEY INSIGHT")
    logging.info(f"{'='*80}\n")
    
    if not removed_features:
        logging.info("All features passed the causality test at p < {}".format(significance))
        logging.info("This means ALL your features have statistically significant")
        logging.info("predictive power for future returns - your feature engineering is sound!")
        logging.info("\nTo see filtering in action, try:")
        logging.info("  1. A stricter significance level (e.g., p < 0.01)")
        logging.info("  2. A different ticker with weaker feature relationships")
        logging.info("  3. Adding a random/noise feature to the base features")
    else:
        logging.info(f"Causality filtering removed {len(removed_features)} non-predictive features")
        logging.info("The HMM is now trained on a more focused feature set")
        logging.info("This should improve out-of-sample performance and reduce overfitting")
    
    logging.info(f"\n{'='*80}\n")
    
    return {
        'no_filter': model_no_filter,
        'with_filter': model_with_filter,
        'removed': removed_features
    }

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    significance = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    
    try:
        results = demonstrate_causality_filtering(ticker, significance)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"\nOriginal features: {results['no_filter'].base_features}")
        print(f"Filtered features: {results['with_filter'].base_features}")
        print(f"Removed: {results['removed'] if results['removed'] else 'None'}")
        
        sys.exit(0)
    except Exception as e:
        logging.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
