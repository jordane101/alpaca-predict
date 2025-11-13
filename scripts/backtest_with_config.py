#!/usr/bin/env python3
"""
Configurable Tech Sector Portfolio Backtest

Reads configuration from config/backtest_config.yaml
Allows easy adjustment of parameters without modifying code.
"""

import sys
import yaml
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hmm.hmm_analysis import AnalyzeHMM
from src.utils.paths import CAUSALITY_CACHE_DIR, DATA_DIR

# Import Alpaca API
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import os

# Load environment
env_path = PROJECT_ROOT / '.env'
load_dotenv(env_path)

# Initialize Alpaca client
api_key = os.getenv('PAPER_KEY') or os.getenv('APCA_API_KEY_ID')
api_secret = os.getenv('PAPER_SEC') or os.getenv('APCA_API_SECRET_KEY')
data_client = StockHistoricalDataClient(api_key, api_secret)


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / 'config' / 'backtest_config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_bars(ticker, start_date, end_date):
    """Fetch historical bars from Alpaca API."""
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        bars = data_client.get_stock_bars(request_params)
        df = bars.df
        
        if ticker in df.index.get_level_values(0):
            df = df.xs(ticker, level=0)
        
        return df
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return pd.DataFrame()


def get_universe(config):
    """Get trading universe based on config."""
    universe_type = config['universe']['type']
    
    if universe_type == 'top_20':
        return config['universe']['top_20']
    elif universe_type == 'full_dag':
        # Load from DAG
        dag_file = CAUSALITY_CACHE_DIR / config['causality']['dag_file']
        with open(dag_file, 'rb') as f:
            data = pickle.load(f)
        
        # Filter out ETFs
        etfs = ['QQQ', 'SPY', 'VOO', 'VTI', 'VGT', 'XLK', 'IGV', 'SOXX', 'SMH', 
                'QTEC', 'DIA', 'IWM', 'ARKK', 'CIBR']
        return [t for t in data['universe'] if t not in etfs]
    else:
        raise ValueError(f"Unknown universe type: {universe_type}")


def run_backtest(strategy_config, config, stock_data, rebalance_dates):
    """
    Run backtest for a single strategy.
    
    Args:
        strategy_config: Strategy configuration dict
        config: Full configuration dict
        stock_data: Pre-loaded stock data dict
        rebalance_dates: List of rebalance dates
        
    Returns:
        Performance metrics dict
    """
    portfolio_cfg = config['portfolio']
    hmm_cfg = config['hmm']
    verbosity = config['output']['verbosity']
    
    print(f"\n{'='*70}")
    print(f"BACKTEST: {strategy_config['name']}")
    print(f"{'='*70}")
    
    # Portfolio tracking
    portfolio = {
        'cash': portfolio_cfg['initial_capital'],
        'positions': {},
        'equity_history': [],
        'dates': [],
        'trades': []
    }
    
    # Get DAG file path if needed
    dag_file = None
    if strategy_config.get('use_causal_features'):
        dag_file = str(CAUSALITY_CACHE_DIR / config['causality']['dag_file'])
    
    # Run backtest
    if verbosity != 'quiet':
        print("\nRunning backtest...")
    
    for idx, date in enumerate(rebalance_dates):
        if verbosity == 'verbose':
            print(f"\nRebalance {idx+1}/{len(rebalance_dates)}: {date.strftime('%Y-%m-%d')}")
        elif verbosity == 'normal' and idx % 2 == 0:
            print(f"  Progress: {idx+1}/{len(rebalance_dates)} rebalances...", end='\r')
        
        # Analyze stocks
        signals = []
        for ticker, full_data in stock_data.items():
            try:
                # Get historical data up to this date
                hist_data = full_data[full_data.index <= date]
                if len(hist_data) < 252:
                    continue
                
                # Create analyzer
                analyzer = AnalyzeHMM(
                    ticker=ticker,
                    n_components=strategy_config['n_components'],
                    model_order=hmm_cfg['model_order'],
                    bars_data=hist_data,
                    verbose=False,
                    force_retrain=hmm_cfg['force_retrain'],
                    use_causal_features=strategy_config.get('use_causal_features', False),
                    causal_dag_file=dag_file,
                    optimize_n_components=strategy_config.get('optimize_n_components', False),
                    n_components_range=tuple(strategy_config.get('n_components_range', [2, 4]))
                )
                
                # Get prediction
                pred = analyzer.predict_next_day_outlook()
                
                # Get current price
                current_price = hist_data['close'].iloc[-1]
                
                signals.append({
                    'ticker': ticker,
                    'outlook': pred['outlook'],
                    'confidence': pred.get('confidence', 1.0),
                    'position_size': pred.get('position_size', 1.0 if pred['outlook'] == 'positive' else 0.0),
                    'price': current_price
                })
                
            except Exception as e:
                if verbosity == 'verbose':
                    print(f"  Error analyzing {ticker}: {str(e)[:50]}")
                continue
        
        if verbosity == 'verbose':
            print(f"  Analyzed {len(signals)} stocks")
        
        # Calculate portfolio value
        portfolio_value = portfolio['cash']
        for ticker, pos in portfolio['positions'].items():
            if ticker in stock_data:
                curr_data = stock_data[ticker][stock_data[ticker].index <= date]
                if not curr_data.empty:
                    portfolio_value += pos['shares'] * curr_data['close'].iloc[-1]
        
        if verbosity == 'verbose':
            print(f"  Portfolio Value: ${portfolio_value:,.2f}")
        
        # Close all positions (rebalancing)
        for ticker in list(portfolio['positions'].keys()):
            pos = portfolio['positions'][ticker]
            curr_data = stock_data[ticker][stock_data[ticker].index <= date]
            if not curr_data.empty:
                sell_price = curr_data['close'].iloc[-1]
                proceeds = pos['shares'] * sell_price
                portfolio['cash'] += proceeds
                
                pnl = proceeds - (pos['shares'] * pos['entry_price'])
                pnl_pct = (sell_price / pos['entry_price'] - 1) * 100
                
                portfolio['trades'].append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'sell',
                    'price': sell_price,
                    'pnl_pct': pnl_pct
                })
                
                del portfolio['positions'][ticker]
        
        # Filter buy signals
        min_pos_size = hmm_cfg['min_position_size']
        buy_signals = [s for s in signals 
                      if s['position_size'] > min_pos_size
                      and s['outlook'] == 'positive']
        
        # Sort by confidence-weighted size
        buy_signals.sort(key=lambda x: abs(x['position_size']) * x['confidence'], reverse=True)
        buys = buy_signals[:portfolio_cfg['max_positions']]
        
        if buys:
            # Allocate capital
            cash_to_use = portfolio['cash'] * portfolio_cfg['cash_usage']
            per_position = cash_to_use / len(buys)
            
            for signal in buys:
                ticker = signal['ticker']
                price = signal['price']
                shares = int(per_position / price)
                
                if shares > 0:
                    cost = shares * price
                    portfolio['cash'] -= cost
                    portfolio['positions'][ticker] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': date
                    }
                    
                    portfolio['trades'].append({
                        'date': date,
                        'ticker': ticker,
                        'action': 'buy',
                        'price': price
                    })
                    
                    if verbosity == 'verbose':
                        print(f"    BUY {ticker}: {shares} shares @ ${price:.2f}")
        
        # Record equity
        portfolio['equity_history'].append(portfolio_value)
        portfolio['dates'].append(date)
        
        if verbosity == 'verbose':
            print(f"  Positions: {len(portfolio['positions'])}, Cash: ${portfolio['cash']:,.2f}")
    
    # Calculate final value
    backtest_cfg = config['backtest']
    if 'end_date' in backtest_cfg:
        end_date = pd.Timestamp(backtest_cfg['end_date'], tz='UTC')
    else:
        end_date = pd.Timestamp(datetime.now(), tz='UTC')
    
    final_value = portfolio['cash']
    for ticker, pos in portfolio['positions'].items():
        if ticker in stock_data:
            final_data = stock_data[ticker][stock_data[ticker].index <= end_date]
            if not final_data.empty:
                final_value += pos['shares'] * final_data['close'].iloc[-1]
    
    portfolio['equity_history'].append(final_value)
    portfolio['dates'].append(end_date)
    
    # Calculate metrics
    equity_series = pd.Series(portfolio['equity_history'], index=portfolio['dates'])
    returns = equity_series.pct_change().dropna()
    
    initial_capital = portfolio_cfg['initial_capital']
    total_return = (final_value / initial_capital - 1) * 100
    
    if 'start_date' in backtest_cfg:
        start = pd.Timestamp(backtest_cfg['start_date'])
        end = pd.Timestamp(backtest_cfg['end_date'])
        days = (end - start).days
    else:
        days = backtest_cfg['years'] * 365
    
    annual_return = ((final_value / initial_capital) ** (365 / days) - 1) * 100
    
    sharpe = 0
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    
    max_dd = ((equity_series / equity_series.cummax() - 1).min()) * 100
    
    closed_trades = [t for t in portfolio['trades'] if t['action'] == 'sell']
    winning_trades = [t for t in closed_trades if t['pnl_pct'] > 0]
    win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
    
    return {
        'strategy': strategy_config['name'],
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'total_trades': len(portfolio['trades']),
        'equity_curve': equity_series,
        'trades': portfolio['trades']
    }


def main(config_path=None):
    """Run backtest with config file."""
    # Load configuration
    config = load_config(config_path)
    
    print("\n" + "="*70)
    print("  TECH PORTFOLIO BACKTEST (Config-Driven)")
    print("="*70)
    print(f"\nConfig: {config_path or 'config/backtest_config.yaml'}")
    
    # Parse dates
    backtest_cfg = config['backtest']
    if 'start_date' in backtest_cfg and 'end_date' in backtest_cfg:
        start_date = datetime.strptime(backtest_cfg['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(backtest_cfg['end_date'], '%Y-%m-%d')
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * backtest_cfg['years'])
    
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Initial Capital: ${config['portfolio']['initial_capital']:,}")
    print(f"Max Positions: {config['portfolio']['max_positions']}")
    
    # Get universe
    universe = get_universe(config)
    print(f"Universe: {len(universe)} stocks - {', '.join(universe[:10])}...")
    
    # Fetch data
    print("\nFetching data...")
    stock_data = {}
    for i, ticker in enumerate(universe, 1):
        print(f"  [{i}/{len(universe)}] {ticker}...", end='\r')
        bars = get_bars(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if not bars.empty and len(bars) >= 252:
            stock_data[ticker] = bars
    
    print(f"\n✓ Loaded {len(stock_data)} stocks")
    
    # Generate rebalance dates
    rebalance_days = config['portfolio']['rebalance_days']
    rebalance_dates = []
    current = start_date + timedelta(days=252)  # Start after 1 year training
    while current < end_date:
        rebalance_dates.append(pd.Timestamp(current, tz='UTC'))
        current += timedelta(days=rebalance_days)
    
    print(f"Rebalance dates: {len(rebalance_dates)} (every {rebalance_days} days)")
    
    # Run backtests for enabled strategies
    enabled_strategies = [s for s in config['strategies'] if s.get('enabled', True)]
    print(f"\nTesting {len(enabled_strategies)} strategies...")
    
    results = []
    for strategy in enabled_strategies:
        try:
            result = run_backtest(strategy, config, stock_data, rebalance_dates)
            results.append(result)
            
            print(f"\n{'='*70}")
            print(f"RESULTS: {result['strategy']}")
            print(f"{'='*70}")
            print(f"Final Value:     ${result['final_value']:,.2f}")
            print(f"Total Return:    {result['total_return']:+.2f}%")
            print(f"Annual Return:   {result['annual_return']:+.2f}%")
            print(f"Sharpe Ratio:    {result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown:    {result['max_drawdown']:.2f}%")
            print(f"Win Rate:        {result['win_rate']:.1f}%")
            print(f"Total Trades:    {result['total_trades']}")
            
        except Exception as e:
            print(f"\n❌ Error running '{strategy['name']}': {e}")
            import traceback
            traceback.print_exc()
    
    # Get benchmark if enabled
    benchmark_result = None
    if config['benchmark']['enabled']:
        benchmark_ticker = config['benchmark']['ticker']
        print(f"\nFetching {benchmark_ticker} benchmark...")
        
        bars = get_bars(benchmark_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if not bars.empty:
            start_price = bars['close'].iloc[0]
            end_price = bars['close'].iloc[-1]
            
            total_return = (end_price / start_price - 1) * 100
            annual_return = ((end_price / start_price) ** (365 / (end_date - start_date).days) - 1) * 100
            
            returns = bars['close'].pct_change().dropna()
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            max_dd = ((bars['close'] / bars['close'].cummax() - 1).min()) * 100
            
            benchmark_result = {
                'ticker': benchmark_ticker,
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd
            }
            
            print(f"{benchmark_ticker} Total Return: {total_return:+.2f}%")
            print(f"{benchmark_ticker} Annual Return: {annual_return:+.2f}%")
    
    # Comparison table
    print(f"\n{'='*70}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"{'Strategy':<35} {'Return':>10} {'Annual':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print("-" * 70)
    
    if benchmark_result:
        print(f"{benchmark_result['ticker'] + ' (Buy & Hold)':<35} "
              f"{benchmark_result['total_return']:>9.2f}% "
              f"{benchmark_result['annual_return']:>9.2f}% "
              f"{benchmark_result['sharpe_ratio']:>7.2f} "
              f"{benchmark_result['max_drawdown']:>7.2f}%")
    
    for result in results:
        print(f"{result['strategy']:<35} {result['total_return']:>9.2f}% "
              f"{result['annual_return']:>9.2f}% {result['sharpe_ratio']:>7.2f} "
              f"{result['max_drawdown']:>7.2f}%")
    
    print("\n" + "="*70)
    
    # Save results if enabled
    if config['output']['save_results']:
        results_dir = Path(config['output']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f"backtest_{timestamp}.pkl"
        
        with open(results_file, 'wb') as f:
            pickle.dump({
                'strategies': results,
                'benchmark': benchmark_result,
                'config': config
            }, f)
        
        print(f"\n✓ Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run configurable portfolio backtest')
    parser.add_argument('--config', type=str, help='Path to config file', default=None)
    
    args = parser.parse_args()
    main(args.config)
