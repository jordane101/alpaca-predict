#!/usr/bin/env python3
"""
Simple backtest comparison - just test HMM predictions, not full portfolio simulation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from src.hmm.hmm_analysis import AnalyzeHMM, setup_logging
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv
import os

def simple_backtest_comparison(ticker="AAPL", start_date="2023-01-01", end_date="2024-12-31"):
    """
    Compare prediction accuracy and returns for HMM with/without causality filtering.
    """
    setup_logging()
    load_dotenv(".env")
    
    KEY = os.getenv("PAPER_KEY")
    SECRET = os.getenv("PAPER_SEC")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"SIMPLE BACKTEST: CAUSALITY COMPARISON")
    logging.info(f"{'='*80}")
    logging.info(f"Ticker: {ticker}")
    logging.info(f"Period: {start_date} to {end_date}")
    logging.info(f"{'='*80}\n")
    
    # Fetch data
    client = StockHistoricalDataClient(KEY, SECRET)
    logging.info(f"Fetching data for {ticker}...")
    request_params = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
        feed=DataFeed.IEX
    )
    bars = client.get_stock_bars(request_params)
    data = bars.df
    
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index(level='symbol', drop=True)
    
    logging.info(f"Data fetched: {len(data)} rows\n")
    
    # Split data: first 252 days for training, rest for testing
    train_size = 252
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    logging.info(f"Training period: {len(train_data)} days")
    logging.info(f"Test period: {len(test_data)} days\n")
    
    results = {}
    
    #==========================================================================
    # TEST 1: ALL FEATURES
    #==========================================================================
    
    logging.info(f"{'='*80}")
    logging.info(f"TEST 1: HMM WITH ALL FEATURES")
    logging.info(f"{'='*80}\n")
    
    start_time = datetime.now()
    model_all = AnalyzeHMM(
        ticker=ticker,
        n_components=2,
        model_order=1,
        bars_data=train_data,
        use_causality_filter=False,
        force_retrain=True
    )
    train_time_all = datetime.now() - start_time
    
    logging.info(f"Training time: {train_time_all}")
    logging.info(f"Features used: {model_all.features}\n")
    
    # Test predictions
    predictions_all = []
    actual_returns = []
    
    for i in range(len(test_data) - 1):
        current_data = pd.concat([train_data, test_data.iloc[:i+1]])
        
        # Update model with current data
        temp_model = AnalyzeHMM(
            ticker=f"{ticker}_test",
            n_components=2,
            model_order=1,
            bars_data=current_data,
            verbose=False,
            force_retrain=False
        )
        temp_model.model = model_all.model
        temp_model.quantizer = model_all.quantizer
        temp_model.state_means = model_all.state_means
        temp_model.state_stds = model_all.state_stds
        temp_model.state_regimes = model_all.state_regimes
        temp_model.features = model_all.features
        temp_model._predict_states_for_data()
        
        prediction = temp_model.predict_next_day_outlook()
        predictions_all.append(1 if prediction['outlook'] == 'positive' else -1)
        
        # Calculate actual next-day return
        next_return = test_data['close'].iloc[i+1] / test_data['close'].iloc[i] - 1
        actual_returns.append(next_return)
    
    predictions_all = np.array(predictions_all)
    actual_returns = np.array(actual_returns)
    
    # Calculate metrics
    correct_direction = np.sum(np.sign(actual_returns) == predictions_all)
    accuracy_all = correct_direction / len(predictions_all) * 100
    
    strategy_returns_all = predictions_all * actual_returns
    cumulative_return_all = (1 + strategy_returns_all).prod() - 1
    sharpe_all = np.mean(strategy_returns_all) / (np.std(strategy_returns_all) + 1e-9) * np.sqrt(252)
    
    results['all'] = {
        'accuracy': accuracy_all,
        'cumulative_return': cumulative_return_all * 100,
        'sharpe': sharpe_all,
        'train_time': train_time_all,
        'features': len(model_all.features)
    }
    
    logging.info(f"RESULTS (All Features):")
    logging.info(f"  Prediction Accuracy: {accuracy_all:.2f}%")
    logging.info(f"  Cumulative Return: {cumulative_return_all*100:.2f}%")
    logging.info(f"  Sharpe Ratio: {sharpe_all:.2f}")
    
    #==========================================================================
    # TEST 2: CAUSAL FEATURES ONLY
    #==========================================================================
    
    logging.info(f"\n{'='*80}")
    logging.info(f"TEST 2: HMM WITH CAUSAL FEATURES ONLY")
    logging.info(f"{'='*80}\n")
    
    start_time = datetime.now()
    model_causal = AnalyzeHMM(
        ticker=ticker,
        n_components=2,
        model_order=1,
        bars_data=train_data,
        use_causality_filter=True,
        causality_significance=0.05,
        force_retrain=True
    )
    train_time_causal = datetime.now() - start_time
    
    logging.info(f"\nTraining time: {train_time_causal}")
    logging.info(f"Features used: {model_causal.features}\n")
    
    # Test predictions
    predictions_causal = []
    
    for i in range(len(test_data) - 1):
        current_data = pd.concat([train_data, test_data.iloc[:i+1]])
        
        temp_model = AnalyzeHMM(
            ticker=f"{ticker}_test_causal",
            n_components=2,
            model_order=1,
            bars_data=current_data,
            verbose=False,
            force_retrain=False
        )
        temp_model.model = model_causal.model
        temp_model.quantizer = model_causal.quantizer
        temp_model.state_means = model_causal.state_means
        temp_model.state_stds = model_causal.state_stds
        temp_model.state_regimes = model_causal.state_regimes
        temp_model.features = model_causal.features
        temp_model._predict_states_for_data()
        
        prediction = temp_model.predict_next_day_outlook()
        predictions_causal.append(1 if prediction['outlook'] == 'positive' else -1)
    
    predictions_causal = np.array(predictions_causal)
    
    # Calculate metrics
    correct_direction = np.sum(np.sign(actual_returns) == predictions_causal)
    accuracy_causal = correct_direction / len(predictions_causal) * 100
    
    strategy_returns_causal = predictions_causal * actual_returns
    cumulative_return_causal = (1 + strategy_returns_causal).prod() - 1
    sharpe_causal = np.mean(strategy_returns_causal) / (np.std(strategy_returns_causal) + 1e-9) * np.sqrt(252)
    
    results['causal'] = {
        'accuracy': accuracy_causal,
        'cumulative_return': cumulative_return_causal * 100,
        'sharpe': sharpe_causal,
        'train_time': train_time_causal,
        'features': len(model_causal.features)
    }
    
    logging.info(f"RESULTS (Causal Features):")
    logging.info(f"  Prediction Accuracy: {accuracy_causal:.2f}%")
    logging.info(f"  Cumulative Return: {cumulative_return_causal*100:.2f}%")
    logging.info(f"  Sharpe Ratio: {sharpe_causal:.2f}")
    
    #==========================================================================
    # COMPARISON
    #==========================================================================
    
    logging.info(f"\n{'='*80}")
    logging.info(f"COMPARISON")
    logging.info(f"{'='*80}\n")
    
    comparison = pd.DataFrame({
        'Metric': ['Prediction Accuracy (%)', 'Cumulative Return (%)', 'Sharpe Ratio', 'Features Used', 'Training Time (s)'],
        'All Features': [
            f"{results['all']['accuracy']:.2f}",
            f"{results['all']['cumulative_return']:.2f}",
            f"{results['all']['sharpe']:.2f}",
            results['all']['features'],
            f"{results['all']['train_time'].total_seconds():.2f}"
        ],
        'Causal Only': [
            f"{results['causal']['accuracy']:.2f}",
            f"{results['causal']['cumulative_return']:.2f}",
            f"{results['causal']['sharpe']:.2f}",
            results['causal']['features'],
            f"{results['causal']['train_time'].total_seconds():.2f}"
        ],
        'Difference': [
            f"{results['causal']['accuracy'] - results['all']['accuracy']:+.2f}",
            f"{results['causal']['cumulative_return'] - results['all']['cumulative_return']:+.2f}",
            f"{results['causal']['sharpe'] - results['all']['sharpe']:+.2f}",
            f"{results['causal']['features'] - results['all']['features']:+d}",
            f"{(results['causal']['train_time'] - results['all']['train_time']).total_seconds():+.2f}"
        ]
    })
    
    logging.info("\n" + comparison.to_string(index=False))
    
    # Verdict
    logging.info(f"\n{'='*80}")
    logging.info(f"VERDICT")
    logging.info(f"{'='*80}")
    
    if results['causal']['accuracy'] > results['all']['accuracy']:
        logging.info(f"✓ Causality filtering IMPROVED accuracy by {results['causal']['accuracy'] - results['all']['accuracy']:.2f}%")
    elif results['causal']['accuracy'] < results['all']['accuracy']:
        logging.info(f"✗ Causality filtering REDUCED accuracy by {results['all']['accuracy'] - results['causal']['accuracy']:.2f}%")
    else:
        logging.info(f"= Same accuracy")
    
    if results['causal']['cumulative_return'] > results['all']['cumulative_return']:
        logging.info(f"✓ Causality filtering IMPROVED returns by {results['causal']['cumulative_return'] - results['all']['cumulative_return']:.2f}%")
    elif results['causal']['cumulative_return'] < results['all']['cumulative_return']:
        logging.info(f"✗ Causality filtering REDUCED returns by {results['all']['cumulative_return'] - results['causal']['cumulative_return']:.2f}%")
    else:
        logging.info(f"= Same returns")
    
    if results['causal']['sharpe'] > results['all']['sharpe']:
        logging.info(f"✓ Causality filtering IMPROVED Sharpe by {results['causal']['sharpe'] - results['all']['sharpe']:.2f}")
    elif results['causal']['sharpe'] < results['all']['sharpe']:
        logging.info(f"✗ Causality filtering REDUCED Sharpe by {results['all']['sharpe'] - results['causal']['sharpe']:.2f}")
    else:
        logging.info(f"= Same Sharpe")
    
    logging.info(f"\n{'='*80}\n")
    
    return results

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    start = sys.argv[2] if len(sys.argv) > 2 else "2023-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2024-12-31"
    
    try:
        results = simple_backtest_comparison(ticker, start, end)
        sys.exit(0)
    except Exception as e:
        logging.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
