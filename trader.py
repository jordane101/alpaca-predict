"""
A simple trading bot that uses the HMM analysis prediction
to make trade decisions with the Alpaca API.

Author - Eli Jordan
Date - 07/29/2025
"""

import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from strategies import BaseStrategy, HMMStrategy

class Trader:
    """
    A trader that scans S&P 500 stocks using a configurable strategy,
    identifies the top opportunities, and executes trades.
    """
    load_dotenv(".env")
    KEY = os.getenv("PAPER_KEY")
    SECRET = os.getenv("PAPER_SEC")

    def __init__(self, strategy: BaseStrategy, trade_qty: int = 1, max_positions: int = 10):
        """
        Initializes the trader and the Alpaca trading client.

        Args:
            strategy (BaseStrategy): The trading strategy to use for analysis.
            trade_qty (int): The number of shares to trade in a single order.
            max_positions (int): The maximum number of positions to hold at any time.
        """
        # Use paper=True for paper trading environment
        self.trading_client = TradingClient(self.KEY, self.SECRET, paper=True)
        self.data_client = StockHistoricalDataClient(self.KEY, self.SECRET)
        self.strategy = strategy
        self.trade_qty = trade_qty
        self.max_positions = max_positions
        self.sp500_tickers = self._get_sp500_tickers()
        self.held_tickers = set()

    def _get_sp500_tickers(self):
        """Fetches the list of S&P 500 tickers from Wikipedia."""
        print("Fetching S&P 500 tickers...")
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            # Add a User-Agent header to mimic a browser and avoid 403 Forbidden error
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status() # Raise an exception for bad status codes
            table = pd.read_html(response.text)
            df = table[0]
            # Per user feedback, tickers from Wikipedia with '.' should be used as-is.
            tickers = df['Symbol'].tolist()
            print(f"Found {len(tickers)} tickers.")
            return tickers
        except Exception as e:
            print(f"Could not fetch S&P 500 tickers: {e}")
            return []
    
    def _analyze_single_ticker(self, ticker: str, bars_df: pd.DataFrame):
        """
        Analyzes a single stock using pre-fetched data. Designed to be run in a parallel worker.

        Args:
            ticker (str): The stock ticker to analyze.
            bars_df (pd.DataFrame): A DataFrame of historical bar data for the ticker.

        Returns:
            tuple: A tuple containing (signal_type, data), e.g.,
                   ('positive', prediction_dict), ('negative', ticker), or ('no_action', ticker).
                   Returns (None, None) on failure.
        """
        try:
            # Data is pre-fetched. Delegate analysis to the strategy.
            outlook, data = self.strategy.analyze(ticker, bars_df)

            # 3. Determine the trade signal based on the outlook and current holdings
            is_held = ticker in self.held_tickers

            if is_held and outlook == 'negative':
                print(f"  -> SELL SIGNAL for held position {ticker}.")
                return 'negative', ticker
            elif is_held:
                print(f"  -> HOLD SIGNAL for {ticker} (Outlook: {outlook}).")
                return 'no_action', ticker
            elif not is_held and ticker in self.sp500_tickers and outlook == 'positive':
                print(f"  -> BUY SIGNAL for {ticker}. Avg Return: {data['predicted_state_mean_return']:.4f}")
                return 'positive', data
            else:
                # Covers: not held and not positive, or not in S&P500.
                return 'no_action', ticker

        except Exception as e:
            print(f"  -> Could not analyze {ticker}. Reason: {e}")
            return None, None

    def run_scanner_and_trade(self):
        """
        Scans all S&P 500 stocks, finds the top N positive predictions,
        and executes trades.
        """
        # Get list of current positions to check for sell signals
        try:
            positions = self.trading_client.get_all_positions()
            self.held_tickers = {p.symbol for p in positions if p.asset_class == AssetClass.US_EQUITY}
            if self.held_tickers:
                print(f"Currently holding positions in: {list(self.held_tickers)}")
            else:
                print("No open positions.")
        except APIError as e:
            print(f"Could not get current positions: {e}")
            self.held_tickers = set()

        # We analyze all S&P 500 tickers for buy signals, and all held tickers for sell signals.
        # A union of both sets ensures we cover everything.
        tickers_to_analyze = sorted(list(set(self.sp500_tickers) | self.held_tickers))

        if not tickers_to_analyze:
            print("No tickers to analyze. Exiting.")
            return

        # --- Batching and Parallel Analysis ---
        positive_predictions = []
        sell_signals = []
        BATCH_SIZE = 100  # Alpaca API allows up to 100 symbols per bar request
        ticker_batches = [tickers_to_analyze[i:i + BATCH_SIZE] for i in range(0, len(tickers_to_analyze), BATCH_SIZE)]

        print(f"\n--- Starting Market Scan ({len(tickers_to_analyze)} tickers in {len(ticker_batches)} batches) ---")

        for batch_num, batch in enumerate(ticker_batches):
            print(f"  -> Processing batch {batch_num + 1}/{len(ticker_batches)} ({len(batch)} tickers)...")
            try:
                # 1. Fetch data for the entire batch in one API call
                end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
                request_params = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date,
                    feed='iex'  # Explicitly specify the free IEX data feed
                )
                bars_data = self.data_client.get_stock_bars(request_params)

                if bars_data.df.empty:
                    print(f"    -> API returned no data for batch {batch_num + 1}.")
                    continue

                # Group the entire multi-index DataFrame by the 'symbol' level of the index.
                # This is a more robust way to access data for each ticker.
                grouped_data = bars_data.df.groupby('symbol')

                # 2. Analyze each ticker in the batch in parallel
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_ticker = {}
                    for ticker in batch:
                        if ticker in grouped_data.groups:
                            ticker_df = grouped_data.get_group(ticker).reset_index(level='symbol', drop=True)
                            future_to_ticker[executor.submit(self._analyze_single_ticker, ticker, ticker_df)] = ticker
                        else:
                            pass # Silently skip tickers that have no data in the response

                    for i, future in enumerate(as_completed(future_to_ticker)):
                        print(f"     Progress on batch: ({i+1}/{len(future_to_ticker)})", end='\r')
                        signal_type, data = future.result()
                        if signal_type == 'positive':
                            positive_predictions.append(data)
                        elif signal_type == 'negative':
                            sell_signals.append(data)
                print(" " * 40, end='\r') # Clear the progress line

            except APIError as e:
                print(f"    -> API Error on batch {batch_num + 1}: {e}. Skipping batch.")
            except Exception as e:
                print(f"    -> Unexpected error on batch {batch_num + 1}: {e}. Skipping batch.")
        print("--- Market Scan Complete ---")

        # --- Execute Trades based on analysis ---
        # 1. Process sells first to free up capital and position slots
        self._execute_sells(sell_signals)

        print(f"\n--- Analysis Complete ---")
        print(f"Found {len(positive_predictions)} stocks with a positive outlook.")
        print(f"Found {len(sell_signals)} stocks with a negative outlook (sell signals).")

        if not positive_predictions:
            print("No stocks with positive outlook found. No trades will be placed.")
            return

        # --- Prioritize and Execute Buys ---
        # Sort by predicted mean return, descending
        sorted_predictions = sorted(positive_predictions, key=lambda x: x['predicted_state_mean_return'], reverse=True)

        # Print the top 20 ranked signals found during the scan, regardless of trading action
        if sorted_predictions:
            print("\n--- Top 20 Ranked Positive Signals ---")
            # The strength metric is 'predicted_state_mean_return'
            for i, pick in enumerate(sorted_predictions[:20]):
                print(f"  {i+1:2d}. {pick['ticker']:<6}: Strength = {pick['predicted_state_mean_return']:.4f}")

        # Recalculate held positions after sells have been executed
        current_positions = self.trading_client.get_all_positions()
        self.held_tickers = {p.symbol for p in current_positions if p.asset_class == AssetClass.US_EQUITY} # Update held_tickers after sells
        num_held_positions = len(self.held_tickers)
        slots_to_fill = self.max_positions - num_held_positions

        if slots_to_fill <= 0:
            print(f"\nPortfolio is full ({num_held_positions}/{self.max_positions} positions). No new buy orders will be placed.")
            top_picks = []
        else:
            print(f"\nPortfolio has {num_held_positions}/{self.max_positions} positions. Looking to fill {slots_to_fill} slot(s).")
            # Filter out stocks already held and then pick the top ones
            available_for_buy = [p for p in sorted_predictions if p['ticker'] not in self.held_tickers]
            top_picks = available_for_buy[:slots_to_fill]

        if top_picks:
            print(f"\n--- Top {len(top_picks)} Picks for Trading ---")
            for pick in top_picks:
                print(f"  - {pick['ticker']}: Predicted State Avg. Return = {pick['predicted_state_mean_return']:.4f} (Last Return: {pick['last_return']:.4f})")

            print("\n--- Executing Buy Orders ---")
            for pick in top_picks:
                ticker = pick['ticker']
                print(f"Decision: BUY {ticker}. Placing market order for {self.trade_qty} share(s).")
                try:
                    market_order_data = MarketOrderRequest(
                        symbol=ticker,
                        qty=self.trade_qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    self.trading_client.submit_order(order_data=market_order_data)
                    print(f"  -> Successfully submitted BUY order for {self.trade_qty} of {ticker}.")
                except APIError as e:
                    print(f"  -> Failed to submit order for {ticker}. Reason: {e}")

    def _execute_sells(self, tickers_to_sell: list): # This function was already defined correctly.
        """Executes sell orders for the given list of tickers."""
        if not tickers_to_sell:
            return
        print("\n--- Executing Sell Orders ---")
        for ticker in tickers_to_sell:
            print(f"Decision: SELL {ticker}. Closing position.")
            try:
                # close_position liquidates the entire position by default.
                closed_order = self.trading_client.close_position(ticker)
                print(f"  -> Successfully submitted SELL order for {closed_order.qty} shares of {ticker}.")
            except APIError as e:
                print(f"  -> Failed to close position for {ticker}. Reason: {e}")

if __name__ == "__main__":
    # --- General Trader Configuration ---
    TRADE_QUANTITY = 5    # Number of shares to trade for each new position
    MAX_POSITIONS = 10    # Maximum number of positions to hold

    # --- Strategy Selection ---
    # Choose which strategy to run by uncommenting one of the blocks below.

    # == Option 1: HMM Strategy (Default) ==
    # N_HMM_COMPONENTS = 3
    # MODEL_ORDER = 1
    # OPTIMIZE_ORDER_PER_STOCK = False
    # MAX_ORDER_TO_TEST = 5
    # print("Configuring HMM Strategy...")
    # active_strategy = HMMStrategy(
    #     n_components=N_HMM_COMPONENTS,
    #     model_order=MODEL_ORDER,
    #     optimize_order=OPTIMIZE_ORDER_PER_STOCK,
    #     max_order_to_test=MAX_ORDER_TO_TEST
    # )

    # == Option 2: Donchian Breakout Strategy ==
    from strategies import DonchianBreakoutStrategy
    DONCHIAN_PERIOD = 20 # 20-day breakout is a classic
    print("Configuring Donchian Breakout Strategy...")
    active_strategy = DonchianBreakoutStrategy(period=DONCHIAN_PERIOD)


    # --- Trader Initialization and Execution ---
    # 1. Create the trader and pass the chosen strategy to it
    print(f"Starting Trader with {active_strategy.__class__.__name__}...")
    trader = Trader(strategy=active_strategy, trade_qty=TRADE_QUANTITY, max_positions=MAX_POSITIONS)

    # 2. Run the trader
    trader.run_scanner_and_trade()
    print("\nTrader finished.")
