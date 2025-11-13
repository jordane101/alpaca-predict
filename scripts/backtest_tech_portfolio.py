#!/usr/bin/env python3
"""
Tech Sector Portfolio Backtest

Tests the HMM trading strategy on a portfolio of tech stocks from the Causal DAG.
Compares multiple strategy configurations:
1. 2-state HMM with technical indicators
2. 4-state HMM with technical indicators  
3. 4-state HMM with causal features
4. 4-state HMM with causal features + confidence sizing

Uses QQQ (Nasdaq-100 ETF) as the tech sector benchmark instead of S&P 500.
"""

import sys
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during backtest
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_bars(ticker, start_date, end_date, timeframe='1Day'):
    """Fetch historical bars from Alpaca API."""
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Day if timeframe == '1Day' else TimeFrame.Hour,
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


def get_tech_universe():
    """Load tech stocks from the Causal DAG."""
    dag_file = CAUSALITY_CACHE_DIR / "large_network_graph_dag.pkl"
    
    if not dag_file.exists():
        print(f"❌ DAG file not found: {dag_file}")
        print("   Run scripts/build_large_dag.py first")
        return []
    
    with open(dag_file, 'rb') as f:
        data = pickle.load(f)
    
    universe = data['universe']
    
    # Remove ETFs and indices to focus on individual stocks
    etfs = ['QQQ', 'SPY', 'VOO', 'VTI', 'VGT', 'XLK', 'IGV', 'SOXX', 'SMH', 
            'QTEC', 'DIA', 'IWM', 'ARKK', 'CIBR']
    
    tech_stocks = [ticker for ticker in universe if ticker not in etfs]
    
    return sorted(tech_stocks)


def analyze_stock_worker(ticker, bars_data, strategy_config):
    """
    Worker function to analyze a single stock with given strategy configuration.
    Runs in a separate process.
    """
    try:
        # Create analyzer with specified configuration
        analyzer = AnalyzeHMM(
            ticker=ticker,
            n_components=strategy_config['n_components'],
            model_order=1,
            bars_data=bars_data,
            verbose=False,
            force_retrain=True,
            use_causal_features=strategy_config.get('use_causal_features', False),
            causal_dag_file=strategy_config.get('causal_dag_file'),
            optimize_n_components=strategy_config.get('optimize_n_components', False),
            n_components_range=strategy_config.get('n_components_range', (2, 4))
        )
        
        # Get prediction
        prediction = analyzer.predict_next_day_outlook()
        
        # Extract key metrics
        result = {
            'ticker': ticker,
            'outlook': prediction['outlook'],
            'confidence': prediction.get('confidence', 1.0),
            'position_size': prediction.get('position_size', 1.0 if prediction['outlook'] == 'positive' else 0.0),
            'position_action': prediction.get('position_action', 'buy' if prediction['outlook'] == 'positive' else 'hold'),
            'predicted_return': prediction['predicted_state_mean_return'],
            'predicted_volatility': prediction['predicted_state_std_return'],
            'regime': prediction.get('regime', 'unknown')
        }
        
        return result
        
    except Exception as e:
        print(f"  Error analyzing {ticker}: {e}")
        return None


def backtest_portfolio(strategy_config, start_date, end_date, initial_capital=100000, 
                      max_positions=20, rebalance_days=21):
    """
    Backtest a portfolio strategy over a time period.
    
    Args:
        strategy_config: Dictionary with strategy parameters
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        max_positions: Maximum number of positions
        rebalance_days: Days between rebalances
        
    Returns:
        Dictionary with backtest results
    """
    print(f"\n{'='*80}")
    print(f"BACKTESTING: {strategy_config['name']}")
    print(f"{'='*80}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Max Positions: {max_positions}")
    print(f"Rebalance Period: {rebalance_days} days")
    
    # Get tech universe
    tech_universe = get_tech_universe()
    print(f"Universe: {len(tech_universe)} tech stocks")
    
    # Fetch data for all stocks
    print("\nFetching historical data...")
    stock_data = {}
    failed_tickers = []
    
    for i, ticker in enumerate(tech_universe, 1):
        print(f"  [{i}/{len(tech_universe)}] Fetching {ticker}...", end='\r')
        bars = get_bars(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if not bars.empty and len(bars) >= 252:  # Need at least 1 year of data
            stock_data[ticker] = bars
        else:
            failed_tickers.append(ticker)
    
    print(f"\n✓ Loaded data for {len(stock_data)}/{len(tech_universe)} stocks")
    if failed_tickers:
        print(f"  Failed: {', '.join(failed_tickers[:10])}{'...' if len(failed_tickers) > 10 else ''}")
    
    # Generate rebalance dates (timezone-aware for Alpaca data)
    current_date = start_date + timedelta(days=252)  # Start after 1 year for training
    rebalance_dates = []
    while current_date < end_date:
        # Make timezone-aware (UTC) to match Alpaca data
        rebalance_dates.append(pd.Timestamp(current_date, tz='UTC'))
        current_date += timedelta(days=rebalance_days)
    
    print(f"\nRebalance dates: {len(rebalance_dates)}")
    
    # Initialize portfolio tracking
    portfolio = {
        'cash': initial_capital,
        'positions': {},  # ticker -> {shares, entry_price, entry_date}
        'equity_history': [],
        'dates': [],
        'trades': []
    }
    
    # Run backtest
    print("\nRunning backtest...")
    for rebalance_idx, rebalance_date in enumerate(rebalance_dates):
        print(f"\n--- Rebalance {rebalance_idx + 1}/{len(rebalance_dates)}: {rebalance_date.strftime('%Y-%m-%d')} ---")
        
        # Get data up to rebalance date for analysis
        analysis_results = []
        
        # Analyze each stock in parallel
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {}
            for ticker, full_data in stock_data.items():
                # Get data up to rebalance date
                mask = full_data.index <= rebalance_date
                historical_data = full_data[mask]
                
                if len(historical_data) >= 252:
                    future = executor.submit(analyze_stock_worker, ticker, historical_data, strategy_config)
                    futures[future] = ticker
            
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                if result:
                    analysis_results.append(result)
        
        print(f"  Analyzed {len(analysis_results)} stocks")
        
        # Calculate current portfolio value
        portfolio_value = portfolio['cash']
        held_tickers = list(portfolio['positions'].keys())
        
        # Mark to market existing positions
        for ticker in held_tickers:
            if ticker in stock_data:
                mask = stock_data[ticker].index <= rebalance_date
                current_data = stock_data[ticker][mask]
                if not current_data.empty:
                    current_price = current_data['close'].iloc[-1]
                    position = portfolio['positions'][ticker]
                    position_value = position['shares'] * current_price
                    portfolio_value += position_value
        
        print(f"  Current Portfolio Value: ${portfolio_value:,.2f}")
        
        # Filter for actionable signals
        buy_signals = [r for r in analysis_results 
                      if r['position_action'] in ['buy', 'short'] 
                      and r['ticker'] not in held_tickers
                      and abs(r['position_size']) > 0.1]  # At least 10% position
        
        sell_signals = [ticker for ticker in held_tickers
                       if any(r['ticker'] == ticker and r['outlook'] == 'negative' 
                             for r in analysis_results)]
        
        print(f"  Buy Signals: {len(buy_signals)}, Sell Signals: {len(sell_signals)}")
        
        # Execute sells first to free up capital
        for ticker in sell_signals:
            if ticker in portfolio['positions']:
                position = portfolio['positions'][ticker]
                
                # Get current price
                mask = stock_data[ticker].index <= rebalance_date
                current_data = stock_data[ticker][mask]
                if not current_data.empty:
                    sell_price = current_data['close'].iloc[-1]
                    proceeds = position['shares'] * sell_price
                    portfolio['cash'] += proceeds
                    
                    # Calculate return
                    pnl = proceeds - (position['shares'] * position['entry_price'])
                    pnl_pct = (sell_price / position['entry_price'] - 1) * 100
                    
                    portfolio['trades'].append({
                        'date': rebalance_date,
                        'ticker': ticker,
                        'action': 'sell',
                        'shares': position['shares'],
                        'price': sell_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    })
                    
                    print(f"    SELL {ticker}: {position['shares']:.0f} shares @ ${sell_price:.2f} "
                          f"(P&L: ${pnl:,.2f}, {pnl_pct:+.1f}%)")
                    
                    del portfolio['positions'][ticker]
        
        # Execute buys
        if buy_signals:
            # Sort by confidence-weighted return
            buy_signals.sort(key=lambda x: abs(x['position_size']) * x['predicted_return'], reverse=True)
            
            # Calculate how many positions we can open
            open_slots = max_positions - len(portfolio['positions'])
            buys_to_execute = buy_signals[:open_slots]
            
            if buys_to_execute:
                # Allocate capital using waterfall with confidence weighting
                total_position_size = sum(abs(r['position_size']) for r in buys_to_execute)
                
                for result in buys_to_execute:
                    ticker = result['ticker']
                    
                    # Get current price
                    mask = stock_data[ticker].index <= rebalance_date
                    current_data = stock_data[ticker][mask]
                    if current_data.empty:
                        continue
                    
                    buy_price = current_data['close'].iloc[-1]
                    
                    # Calculate position size based on confidence
                    allocation_pct = abs(result['position_size']) / total_position_size
                    position_capital = portfolio['cash'] * allocation_pct * 0.9  # Use 90% of available cash
                    
                    if position_capital >= buy_price:
                        shares = int(position_capital / buy_price)
                        cost = shares * buy_price
                        
                        if shares > 0 and cost <= portfolio['cash']:
                            portfolio['cash'] -= cost
                            portfolio['positions'][ticker] = {
                                'shares': shares,
                                'entry_price': buy_price,
                                'entry_date': rebalance_date
                            }
                            
                            portfolio['trades'].append({
                                'date': rebalance_date,
                                'ticker': ticker,
                                'action': 'buy',
                                'shares': shares,
                                'price': buy_price,
                                'confidence': result['confidence'],
                                'position_size': result['position_size']
                            })
                            
                            action_str = "SHORT" if result['position_size'] < 0 else "BUY"
                            print(f"    {action_str} {ticker}: {shares} shares @ ${buy_price:.2f} "
                                  f"(Conf: {result['confidence']:.1%}, Size: {result['position_size']:+.1%}, "
                                  f"Cost: ${cost:,.2f})")
        
        # Record portfolio value
        portfolio['equity_history'].append(portfolio_value)
        portfolio['dates'].append(rebalance_date)
        
        print(f"  End Cash: ${portfolio['cash']:,.2f}")
        print(f"  Positions: {len(portfolio['positions'])}")
    
    # Calculate final portfolio value
    final_date = pd.Timestamp(end_date, tz='UTC')
    final_value = portfolio['cash']
    
    for ticker, position in portfolio['positions'].items():
        if ticker in stock_data:
            mask = stock_data[ticker].index <= final_date
            final_data = stock_data[ticker][mask]
            if not final_data.empty:
                final_price = final_data['close'].iloc[-1]
                final_value += position['shares'] * final_price
    
    portfolio['equity_history'].append(final_value)
    portfolio['dates'].append(final_date)
    
    # Calculate performance metrics
    equity_series = pd.Series(portfolio['equity_history'], index=portfolio['dates'])
    returns = equity_series.pct_change().dropna()
    
    total_return = (final_value / initial_capital - 1) * 100
    annual_return = ((final_value / initial_capital) ** (365 / (end_date - start_date).days) - 1) * 100
    sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    max_drawdown = ((equity_series / equity_series.cummax() - 1).min()) * 100
    
    # Win rate from closed trades
    closed_trades = [t for t in portfolio['trades'] if t['action'] == 'sell']
    winning_trades = [t for t in closed_trades if t['pnl'] > 0]
    win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
    
    results = {
        'strategy': strategy_config['name'],
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(portfolio['trades']),
        'closed_trades': len(closed_trades),
        'equity_curve': equity_series,
        'trades': portfolio['trades']
    }
    
    return results


def get_benchmark_performance(benchmark_ticker, start_date, end_date, initial_capital):
    """Calculate buy-and-hold benchmark performance."""
    print(f"\nFetching {benchmark_ticker} benchmark data...")
    
    bars = get_bars(benchmark_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if bars.empty:
        print(f"❌ Could not fetch {benchmark_ticker} data")
        return None
    
    start_price = bars['close'].iloc[0]
    end_price = bars['close'].iloc[-1]
    
    shares = initial_capital / start_price
    final_value = shares * end_price
    
    total_return = (final_value / initial_capital - 1) * 100
    annual_return = ((final_value / initial_capital) ** (365 / (end_date - start_date).days) - 1) * 100
    
    # Calculate Sharpe ratio
    returns = bars['close'].pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    # Calculate max drawdown
    cumulative = (1 + returns).cumprod()
    max_drawdown = ((cumulative / cumulative.cummax() - 1).min()) * 100
    
    print(f"✓ {benchmark_ticker} Total Return: {total_return:.2f}%")
    
    return {
        'ticker': benchmark_ticker,
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }


def main():
    """Run comprehensive backtest comparing multiple strategies."""
    print("\n" + "="*80)
    print("  TECH SECTOR PORTFOLIO BACKTEST")
    print("="*80)
    
    # Backtest parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years
    initial_capital = 100000
    max_positions = 20
    rebalance_days = 21  # Monthly rebalancing
    
    # Get DAG file path
    dag_file = str(CAUSALITY_CACHE_DIR / "large_network_graph_dag.pkl")
    
    # Define strategy configurations
    strategies = [
        {
            'name': '2-State Technical',
            'n_components': 2,
            'use_causal_features': False,
            'optimize_n_components': False
        },
        {
            'name': '4-State Technical',
            'n_components': 4,
            'use_causal_features': False,
            'optimize_n_components': False
        },
        {
            'name': 'Optimized Technical (2-4 states)',
            'n_components': 3,
            'use_causal_features': False,
            'optimize_n_components': True,
            'n_components_range': (2, 4)
        },
        {
            'name': 'Optimized Causal (2-4 states)',
            'n_components': 3,
            'use_causal_features': True,
            'causal_dag_file': dag_file,
            'optimize_n_components': True,
            'n_components_range': (2, 4)
        }
    ]
    
    # Run backtests
    all_results = []
    
    for strategy in strategies:
        try:
            results = backtest_portfolio(
                strategy,
                start_date,
                end_date,
                initial_capital,
                max_positions,
                rebalance_days
            )
            all_results.append(results)
            
            print(f"\n{'='*80}")
            print(f"RESULTS: {results['strategy']}")
            print(f"{'='*80}")
            print(f"Final Value:     ${results['final_value']:,.2f}")
            print(f"Total Return:    {results['total_return']:+.2f}%")
            print(f"Annual Return:   {results['annual_return']:+.2f}%")
            print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown:    {results['max_drawdown']:.2f}%")
            print(f"Win Rate:        {results['win_rate']:.1f}%")
            print(f"Total Trades:    {results['total_trades']}")
            print(f"Closed Trades:   {results['closed_trades']}")
            
        except Exception as e:
            print(f"\n❌ Error running strategy '{strategy['name']}': {e}")
            import traceback
            traceback.print_exc()
    
    # Get benchmark performance (QQQ for tech sector)
    benchmark = get_benchmark_performance('QQQ', start_date, end_date, initial_capital)
    
    # Print comparison
    print(f"\n{'='*80}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*80}\n")
    
    # Create comparison table
    print(f"{'Strategy':<35} {'Return':<12} {'Annual':<12} {'Sharpe':<10} {'MaxDD':<10} {'Trades':<8}")
    print("-" * 95)
    
    if benchmark:
        print(f"{'QQQ (Buy & Hold)':<35} {benchmark['total_return']:>10.2f}% "
              f"{benchmark['annual_return']:>10.2f}% {benchmark['sharpe_ratio']:>8.2f} "
              f"{benchmark['max_drawdown']:>8.2f}% {'N/A':<8}")
    
    for result in all_results:
        print(f"{result['strategy']:<35} {result['total_return']:>10.2f}% "
              f"{result['annual_return']:>10.2f}% {result['sharpe_ratio']:>8.2f} "
              f"{result['max_drawdown']:>8.2f}% {result['closed_trades']:<8}")
    
    print("\n" + "="*80)
    
    # Save results
    results_file = DATA_DIR / "backtest_results" / f"tech_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'wb') as f:
        pickle.dump({
            'strategies': all_results,
            'benchmark': benchmark,
            'parameters': {
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'max_positions': max_positions,
                'rebalance_days': rebalance_days
            }
        }, f)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    main()
