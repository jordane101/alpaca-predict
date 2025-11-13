#!/usr/bin/env python3
"""
Quick Tech Sector Portfolio Backtest

Simplified version that tests on a smaller universe with faster execution.
Focus on top 20 tech stocks with highest liquidity and market cap.
"""

import sys
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


def get_top_tech_stocks():
    """Get top 20 most liquid tech stocks from DAG universe."""
    # High market cap, high liquidity tech stocks
    top_tech = [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AMZN',
        'AMD', 'AVGO', 'ORCL', 'NFLX', 'CRM', 'ADBE', 'INTC',
        'CSCO', 'QCOM', 'TXN', 'AMAT', 'INTU', 'MU'
    ]
    return sorted(top_tech)


def run_simple_backtest(strategy_config, start_date, end_date, 
                       initial_capital=100000, max_positions=10):
    """
    Simple backtest with monthly rebalancing.
    
    Args:
        strategy_config: Strategy parameters
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        max_positions: Maximum positions
        
    Returns:
        Performance metrics
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {strategy_config['name']}")
    print(f"{'='*70}")
    
    # Get tech stocks
    universe = get_top_tech_stocks()
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
    
    # Generate quarterly rebalance dates
    rebalance_dates = []
    current = start_date + timedelta(days=252)  # Start after 1 year training
    while current < end_date:
        rebalance_dates.append(pd.Timestamp(current, tz='UTC'))
        current += timedelta(days=63)  # Quarterly
    
    print(f"Rebalance dates: {len(rebalance_dates)}")
    
    # Portfolio tracking
    portfolio = {
        'cash': initial_capital,
        'positions': {},
        'equity_history': [],
        'dates': [],
        'trades': []
    }
    
    # Run backtest
    print("\nRunning backtest...")
    for idx, date in enumerate(rebalance_dates):
        print(f"\nRebalance {idx+1}/{len(rebalance_dates)}: {date.strftime('%Y-%m-%d')}")
        
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
                    model_order=1,
                    bars_data=hist_data,
                    verbose=False,
                    force_retrain=True,
                    use_causal_features=strategy_config.get('use_causal_features', False),
                    causal_dag_file=strategy_config.get('causal_dag_file'),
                    optimize_n_components=strategy_config.get('optimize_n_components', False),
                    n_components_range=strategy_config.get('n_components_range', (2, 4))
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
                print(f"  Error analyzing {ticker}: {str(e)[:50]}")
                continue
        
        print(f"  Analyzed {len(signals)} stocks")
        
        # Calculate portfolio value
        portfolio_value = portfolio['cash']
        for ticker, pos in portfolio['positions'].items():
            if ticker in stock_data:
                curr_data = stock_data[ticker][stock_data[ticker].index <= date]
                if not curr_data.empty:
                    portfolio_value += pos['shares'] * curr_data['close'].iloc[-1]
        
        print(f"  Portfolio Value: ${portfolio_value:,.2f}")
        
        # Close all positions (monthly rebalance = full turnover)
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
        buy_signals = [s for s in signals 
                      if s['position_size'] > 0.1  # At least 10% position
                      and s['outlook'] == 'positive']
        
        # Sort by confidence-weighted size
        buy_signals.sort(key=lambda x: abs(x['position_size']) * x['confidence'], reverse=True)
        buys = buy_signals[:max_positions]
        
        if buys:
            # Equal weight allocation
            per_position = portfolio['cash'] / len(buys) * 0.95  # Use 95% of cash
            
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
                    
                    print(f"    BUY {ticker}: {shares} shares @ ${price:.2f}")
        
        # Record equity
        portfolio['equity_history'].append(portfolio_value)
        portfolio['dates'].append(date)
        
        print(f"  Positions: {len(portfolio['positions'])}, Cash: ${portfolio['cash']:,.2f}")
    
    # Final value
    final_date = pd.Timestamp(end_date, tz='UTC')
    final_value = portfolio['cash']
    for ticker, pos in portfolio['positions'].items():
        if ticker in stock_data:
            final_data = stock_data[ticker][stock_data[ticker].index <= final_date]
            if not final_data.empty:
                final_value += pos['shares'] * final_data['close'].iloc[-1]
    
    portfolio['equity_history'].append(final_value)
    portfolio['dates'].append(final_date)
    
    # Calculate metrics
    equity_series = pd.Series(portfolio['equity_history'], index=portfolio['dates'])
    returns = equity_series.pct_change().dropna()
    
    total_return = (final_value / initial_capital - 1) * 100
    days = (end_date - start_date).days
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
        'equity_curve': equity_series
    }


def main():
    """Run quick backtest."""
    print("\n" + "="*70)
    print("  QUICK TECH PORTFOLIO BACKTEST (Top 20 Stocks)")
    print("="*70)
    
    # Parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years
    initial_capital = 100000
    max_positions = 10
    
    # Get DAG file
    dag_file = str(CAUSALITY_CACHE_DIR / "large_network_graph_dag.pkl")
    
    # Test strategies
    strategies = [
        {
            'name': '2-State Technical',
            'n_components': 2,
            'use_causal_features': False,
            'optimize_n_components': False
        },
        {
            'name': '4-State Causal + Confidence',
            'n_components': 4,
            'use_causal_features': True,
            'causal_dag_file': dag_file,
            'optimize_n_components': False
        }
    ]
    
    # Run backtests
    results = []
    for strategy in strategies:
        try:
            result = run_simple_backtest(
                strategy, start_date, end_date,
                initial_capital, max_positions
            )
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
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Get QQQ benchmark
    print("\nFetching QQQ benchmark...")
    qqq_bars = get_bars('QQQ', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if not qqq_bars.empty:
        qqq_start = qqq_bars['close'].iloc[0]
        qqq_end = qqq_bars['close'].iloc[-1]
        qqq_return = (qqq_end / qqq_start - 1) * 100
        qqq_annual = ((qqq_end / qqq_start) ** (365 / (end_date - start_date).days) - 1) * 100
        
        qqq_returns = qqq_bars['close'].pct_change().dropna()
        qqq_sharpe = (qqq_returns.mean() / qqq_returns.std() * np.sqrt(252)) if qqq_returns.std() > 0 else 0
        qqq_dd = ((qqq_bars['close'] / qqq_bars['close'].cummax() - 1).min()) * 100
        
        print(f"QQQ Total Return: {qqq_return:+.2f}%")
        print(f"QQQ Annual Return: {qqq_annual:+.2f}%")
        print(f"QQQ Sharpe: {qqq_sharpe:.2f}")
        print(f"QQQ Max DD: {qqq_dd:.2f}%")
    
    # Comparison table
    print(f"\n{'='*70}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"{'Strategy':<35} {'Return':>10} {'Annual':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print("-" * 70)
    
    if not qqq_bars.empty:
        print(f"{'QQQ (Buy & Hold)':<35} {qqq_return:>9.2f}% {qqq_annual:>9.2f}% {qqq_sharpe:>7.2f} {qqq_dd:>7.2f}%")
    
    for result in results:
        print(f"{result['strategy']:<35} {result['total_return']:>9.2f}% "
              f"{result['annual_return']:>9.2f}% {result['sharpe_ratio']:>7.2f} "
              f"{result['max_drawdown']:>7.2f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
