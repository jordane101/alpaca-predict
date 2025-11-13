#!/usr/bin/env python3
"""
Backtest comparison: HMM with vs without quantile Granger causality filtering.

This script runs backtests for the same ticker with:
1. HMM using all features (baseline)
2. HMM using only causal features (causality-filtered)

Compares performance metrics to evaluate if causality filtering improves
out-of-sample trading performance.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from datetime import datetime
from src.backtest.backtester import Backtester
from src.trading.strategies import HMMStrategy
from src.hmm.hmm_analysis import setup_logging

def run_backtest_comparison(ticker="AAPL", start_date="2023-01-01", end_date="2024-12-31", initial_cash=100000):
    """
    Run backtest comparison between HMM with and without causality filtering.
    
    Args:
        ticker: Stock ticker to backtest
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        initial_cash: Starting capital
    
    Returns:
        Dictionary with comparison results
    """
    setup_logging()
    
    logging.info(f"\n{'='*80}")
    logging.info(f"BACKTEST COMPARISON: CAUSALITY FILTERING")
    logging.info(f"{'='*80}")
    logging.info(f"Ticker: {ticker}")
    logging.info(f"Period: {start_date} to {end_date}")
    logging.info(f"Initial Cash: ${initial_cash:,}")
    logging.info(f"{'='*80}\n")
    
    # Clean up any existing models to force fresh training
    from pathlib import Path
    model_dir = Path("hmm_models")
    model_files = [
        model_dir / f"{ticker}_2_1.pkl",
        model_dir / f"{ticker}_2_1.json"
    ]
    for f in model_files:
        if f.exists():
            os.remove(f)
            logging.info(f"Removed existing model: {f}")
    
    results = {}
    
    # ============================================================================
    # BACKTEST 1: ALL FEATURES (NO CAUSALITY FILTERING)
    # ============================================================================
    
    logging.info(f"\n{'='*80}")
    logging.info(f"BACKTEST 1: HMM WITH ALL FEATURES (BASELINE)")
    logging.info(f"{'='*80}\n")
    
    strategy_all = HMMStrategy(
        n_components=2,
        model_order=1,
        optimize_order=False,
        walk_forward_window=252,  # 1 year training window
        retrain_period=63         # Retrain quarterly
    )
    
    # Temporarily add causality parameters (set to False)
    strategy_all.use_causality_filter = False
    
    start_time = datetime.now()
    backtester_all = Backtester(
        strategy=strategy_all,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )
    stats_all = backtester_all.run(initial_cash=initial_cash, verbose=True)
    time_all = datetime.now() - start_time
    
    if stats_all is not None:
        results['all_features'] = {
            'stats': stats_all,
            'time': time_all,
            'strategy': 'All Features (No Causality)'
        }
        
        logging.info(f"\n{'='*80}")
        logging.info(f"BACKTEST 1 SUMMARY")
        logging.info(f"{'='*80}")
        logging.info(f"Backtest Time: {time_all}")
        logging.info(f"Total Return: {stats_all.get('Total Return [%]', 'N/A')}%")
        logging.info(f"Sharpe Ratio: {stats_all.get('Sharpe Ratio', 'N/A')}")
        logging.info(f"Max Drawdown: {stats_all.get('Max Drawdown [%]', 'N/A')}%")
        logging.info(f"Win Rate: {stats_all.get('Win Rate [%]', 'N/A')}%")
        logging.info(f"Total Trades: {stats_all.get('Total Trades', 'N/A')}")
    else:
        logging.warning("Backtest 1 generated no trades!")
        results['all_features'] = None
    
    # Clean up model files again before second test
    for f in model_files:
        if f.exists():
            os.remove(f)
    
    # ============================================================================
    # BACKTEST 2: CAUSAL FEATURES ONLY
    # ============================================================================
    
    logging.info(f"\n{'='*80}")
    logging.info(f"BACKTEST 2: HMM WITH CAUSAL FEATURES ONLY")
    logging.info(f"{'='*80}\n")
    
    strategy_causal = HMMStrategy(
        n_components=2,
        model_order=1,
        optimize_order=False,
        walk_forward_window=252,
        retrain_period=63
    )
    
    # Enable causality filtering
    strategy_causal.use_causality_filter = True
    strategy_causal.causality_significance = 0.05
    
    start_time = datetime.now()
    backtester_causal = Backtester(
        strategy=strategy_causal,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )
    stats_causal = backtester_causal.run(initial_cash=initial_cash, verbose=True)
    time_causal = datetime.now() - start_time
    
    if stats_causal is not None:
        results['causal_features'] = {
            'stats': stats_causal,
            'time': time_causal,
            'strategy': 'Causal Features Only'
        }
        
        logging.info(f"\n{'='*80}")
        logging.info(f"BACKTEST 2 SUMMARY")
        logging.info(f"{'='*80}")
        logging.info(f"Backtest Time: {time_causal}")
        logging.info(f"Total Return: {stats_causal.get('Total Return [%]', 'N/A')}%")
        logging.info(f"Sharpe Ratio: {stats_causal.get('Sharpe Ratio', 'N/A')}")
        logging.info(f"Max Drawdown: {stats_causal.get('Max Drawdown [%]', 'N/A')}%")
        logging.info(f"Win Rate: {stats_causal.get('Win Rate [%]', 'N/A')}%")
        logging.info(f"Total Trades: {stats_causal.get('Total Trades', 'N/A')}")
    else:
        logging.warning("Backtest 2 generated no trades!")
        results['causal_features'] = None
    
    # ============================================================================
    # COMPARISON
    # ============================================================================
    
    logging.info(f"\n{'='*80}")
    logging.info(f"PERFORMANCE COMPARISON")
    logging.info(f"{'='*80}\n")
    
    if results.get('all_features') and results.get('causal_features'):
        stats_all = results['all_features']['stats']
        stats_causal = results['causal_features']['stats']
        
        # Create comparison table
        metrics = [
            'Total Return [%]',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Max Drawdown [%]',
            'Win Rate [%]',
            'Total Trades',
            'Avg Winning Trade [%]',
            'Avg Losing Trade [%]',
            'Profit Factor'
        ]
        
        comparison_data = []
        for metric in metrics:
            val_all = stats_all.get(metric, 'N/A')
            val_causal = stats_causal.get(metric, 'N/A')
            
            # Calculate difference if both are numeric
            diff = "N/A"
            if isinstance(val_all, (int, float)) and isinstance(val_causal, (int, float)):
                diff = val_causal - val_all
                diff_str = f"{diff:+.2f}"
                
                # Add percentage for return metrics
                if '%' in metric and metric != 'Win Rate [%]':
                    diff_str += "%"
            else:
                diff_str = diff
            
            comparison_data.append({
                'Metric': metric,
                'All Features': f"{val_all:.2f}" if isinstance(val_all, float) else val_all,
                'Causal Only': f"{val_causal:.2f}" if isinstance(val_causal, float) else val_causal,
                'Difference': diff_str
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        logging.info("\n" + comparison_df.to_string(index=False))
        
        # Summary verdict
        logging.info(f"\n{'='*80}")
        logging.info(f"VERDICT")
        logging.info(f"{'='*80}")
        
        total_return_all = stats_all.get('Total Return [%]', 0)
        total_return_causal = stats_causal.get('Total Return [%]', 0)
        sharpe_all = stats_all.get('Sharpe Ratio', 0)
        sharpe_causal = stats_causal.get('Sharpe Ratio', 0)
        
        if total_return_causal > total_return_all:
            logging.info(f"✓ Causality filtering IMPROVED returns by {total_return_causal - total_return_all:.2f}%")
        elif total_return_causal < total_return_all:
            logging.info(f"✗ Causality filtering REDUCED returns by {total_return_all - total_return_causal:.2f}%")
        else:
            logging.info(f"= Causality filtering had NO EFFECT on returns")
        
        if sharpe_causal > sharpe_all:
            logging.info(f"✓ Causality filtering IMPROVED Sharpe ratio by {sharpe_causal - sharpe_all:.2f}")
        elif sharpe_causal < sharpe_all:
            logging.info(f"✗ Causality filtering REDUCED Sharpe ratio by {sharpe_all - sharpe_causal:.2f}")
        else:
            logging.info(f"= Causality filtering had NO EFFECT on Sharpe ratio")
        
        logging.info(f"\nBacktest Time:")
        logging.info(f"  All Features: {results['all_features']['time']}")
        logging.info(f"  Causal Only:  {results['causal_features']['time']}")
        logging.info(f"  Difference:   {results['causal_features']['time'] - results['all_features']['time']}")
        
    else:
        logging.warning("Cannot compare - one or both backtests failed to generate trades")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"BACKTEST COMPARISON COMPLETE")
    logging.info(f"{'='*80}\n")
    
    return results

if __name__ == "__main__":
    # Configuration
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    start_date = sys.argv[2] if len(sys.argv) > 2 else "2023-01-01"
    end_date = sys.argv[3] if len(sys.argv) > 3 else "2024-12-31"
    initial_cash = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
    
    try:
        results = run_backtest_comparison(ticker, start_date, end_date, initial_cash)
        sys.exit(0)
    except Exception as e:
        logging.error(f"Backtest comparison failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
