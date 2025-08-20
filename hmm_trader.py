"""
A simple trading bot that uses the HMM analysis prediction
to make trade decisions with the Alpaca API.

Author - Eli Jordan
Date - 07/29/2025
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from hmm_analysis import AnalyzeHMM

class HMMTrader:
    """
    A trader that scans S&P 500 stocks using Hidden Markov Model predictions,
    identifies the top opportunities, and executes trades.
    """
    load_dotenv(".env")
    KEY = os.getenv("PAPER_KEY")
    SECRET = os.getenv("PAPER_SEC")

    def __init__(self, trade_qty: int = 1, max_positions: int = 10, n_components: int = 3, model_order: int = 1, optimize_order_per_stock: bool = False, max_order_to_test: int = 10):
        """
        Initializes the trader, HMM analyzer, and the Alpaca trading client.

        Args:
            trade_qty (int): The number of shares to trade in a single order.
            max_positions (int): The maximum number of positions to hold at any time.
            n_components (int): The number of hidden states for the HMM.
            model_order (int): The default order of the Markov model. Used if optimize_order_per_stock is False.
            optimize_order_per_stock (bool): If True, finds the optimal order for each stock individually.
            max_order_to_test (int): The maximum order to test when optimizing.
        """
        # Use paper=True for paper trading environment
        self.trading_client = TradingClient(self.KEY, self.SECRET, paper=True)
        self.data_client = StockHistoricalDataClient(self.KEY, self.SECRET)
        self.trade_qty = trade_qty
        self.max_positions = max_positions
        self.n_components = n_components
        self.model_order = model_order
        self.optimize_order_per_stock = optimize_order_per_stock
        self.max_order_to_test = max_order_to_test
        self.sp500_tickers = self._get_sp500_tickers()
        self.held_tickers = set()

    def _get_sp500_tickers(self):
        """Fetches the list of S&P 500 tickers from Wikipedia."""
        print("Fetching S&P 500 tickers...")
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url)
            df = table[0]
            # Alpaca uses '-' for symbols like BRK-B, while Wikipedia uses '.'
            tickers = [ticker.replace('.', '-') for ticker in df['Symbol'].tolist()]
            print(f"Found {len(tickers)} tickers.")
            return tickers
        except Exception as e:
            print(f"Could not fetch S&P 500 tickers: {e}")
            return []

    def _analyze_single_ticker(self, ticker: str):
        """
        Performs HMM analysis for a single stock. Designed to be run in a parallel worker.

        Args:
            ticker (str): The stock ticker to analyze.

        Returns:
            tuple: A tuple containing (signal_type, data), e.g.,
                   ('positive', prediction_dict), ('negative', ticker), or ('no_action', ticker).
                   Returns (None, None) on failure.
        """
        try:
            # 1. Fetch data once for this ticker
            request_params = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=TimeFrame.Day, # Assuming Day timeframe for the trader
                start="2025-01-01"
            )
            bars_data = self.data_client.get_stock_bars(request_params)

            if bars_data.df.empty:
                print(f"  -> No data for {ticker}. Skipping.")
                return None, None

            # 2. Determine the model order (either default or optimized)
            current_model_order = self.model_order
            if self.optimize_order_per_stock:
                # Create a temporary analyzer to find the optimal order
                temp_analyzer = AnalyzeHMM(ticker, n_components=self.n_components, model_order=1, bars_data=bars_data)
                current_model_order = temp_analyzer.find_optimal_order(max_order=self.max_order_to_test)

            # 3. Create the final analyzer with the determined order and pre-fetched data
            analyzer = AnalyzeHMM(
                ticker=ticker,
                n_components=self.n_components,
                model_order=current_model_order,
                bars_data=bars_data
            )

            prediction = analyzer.predict_next_day_outlook()
            outlook = prediction['outlook']

            # 4. Determine the signal based on the outlook and current holdings
            is_held = ticker in self.held_tickers

            if is_held and outlook == 'negative':
                print(f"  -> SELL SIGNAL for held position {ticker}.")
                return 'negative', ticker
            elif is_held:
                print(f"  -> HOLD SIGNAL for held position {ticker} (Outlook: {outlook}).")
                return 'no_action', ticker
            elif not is_held and ticker in self.sp500_tickers and outlook == 'positive':
                print(f"  -> BUY SIGNAL found for {ticker}. Avg Return: {prediction['predicted_state_mean_return']:.4f}")
                prediction['ticker'] = ticker
                return 'positive', prediction
            else:
                print(f"  -> Outlook for {ticker}: {outlook}. No action taken.")
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
            self.held_tickers = {p.symbol for p in positions}
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

        # --- Parallel Analysis ---
        positive_predictions = []
        sell_signals = []
        # Use max_workers to control concurrency. 10 is a good starting point.
        with ThreadPoolExecutor(max_workers=10) as executor:
            print(f"\n--- Starting Market Scan ({len(tickers_to_analyze)} tickers) with up to {executor._max_workers} parallel workers ---")
            future_to_ticker = {executor.submit(self._analyze_single_ticker, ticker): ticker for ticker in tickers_to_analyze}

            for i, future in enumerate(as_completed(future_to_ticker)):
                print(f"Progress: ({i+1}/{len(tickers_to_analyze)})", end='\r')
                signal_type, data = future.result()
                if signal_type == 'positive':
                    positive_predictions.append(data)
                elif signal_type == 'negative':
                    sell_signals.append(data)
        print("\n") # Newline after progress indicator

        # --- Execute Trades based on analysis ---
        # 1. Process sells first to free up capital and position slots
        self._execute_sells(sell_signals)

        print(f"\n--- Analysis Complete ---")
        print(f"Found {len(positive_predictions)} stocks with a positive outlook.")

        if not positive_predictions:
            print("No stocks with positive outlook found. No trades will be placed.")
            return

        # --- Prioritize and Execute Buys ---
        # Sort by predicted mean return, descending
        sorted_predictions = sorted(positive_predictions, key=lambda x: x['predicted_state_mean_return'], reverse=True)

        # Recalculate held positions after sells have been executed
        current_positions = self.trading_client.get_all_positions()
        self.held_tickers = {p.symbol for p in current_positions} # Update held_tickers after sells
        num_held_positions = len(self.held_tickers)
        slots_to_fill = self.max_positions - num_held_positions

        if slots_to_fill <= 0:
            print(f"\nPortfolio is full ({num_held_positions}/{self.max_positions} positions). No new buy orders will be placed.")
            top_picks = []
        else:
            print(f"\nPortfolio has {num_held_positions}/{self.max_positions} positions. Looking to fill {slots_to_fill} slot(s).")
            # Filter out stocks already held and then pick the top ones
            available_for_buy = [p for p in sorted_predictions if p['ticker'] not in self.held_tickers]
            top_picks = available_for_buy[:slots_to_fill+1]

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
        """Executes buy orders for the given list of stock picks."""
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
    # --- Configuration ---
    TRADE_QUANTITY = 1    # Number of shares to trade for each new position
    MAX_POSITIONS = 10    # Maximum number of positions to hold

    # Number of hidden states for the HMM.
    # 2 is often used for simple 'bull'/'bear' regimes.
    # 3 can capture 'bull'/'bear'/'neutral' or 'high/low/extreme volatility'.
    N_HMM_COMPONENTS = 3

    # The order of the model. 1 is standard. 2 or 3 will use past data as features
    # to give the model "memory", approximating a higher-order HMM.
    MODEL_ORDER = 1 # Default order if optimization is off.

    # --- Advanced Configuration ---
    # Set to True to find the optimal model order for EACH stock.
    # WARNING: This will make the script run VERY slowly.
    OPTIMIZE_ORDER_PER_STOCK = False
    # If optimizing, what is the highest order to test? (e.g., 1 through 10)
    MAX_ORDER_TO_TEST = 5

    print("Starting HMM Trader...")
    trader = HMMTrader(trade_qty=TRADE_QUANTITY,
                       max_positions=MAX_POSITIONS,
                       n_components=N_HMM_COMPONENTS,
                       model_order=MODEL_ORDER,
                       optimize_order_per_stock=OPTIMIZE_ORDER_PER_STOCK,
                       max_order_to_test=MAX_ORDER_TO_TEST)
    trader.run_scanner_and_trade()
    print("\nHMM Trader finished.")