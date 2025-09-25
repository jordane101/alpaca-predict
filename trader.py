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

    def __init__(self, strategy: BaseStrategy, max_positions: int = 10, total_allocation_pct: float = 0.5, waterfall_allocation_pcts: list = None, stop_loss_pct: float = None, take_profit_pct: float = None):
        """
        Initializes the trader and the Alpaca trading client.

        Args:
            strategy (BaseStrategy): The trading strategy to use for analysis.
            max_positions (int): The maximum number of positions to hold at any time.
            total_allocation_pct (float): The percentage of total equity to allocate to this strategy (e.g., 0.5 for 50%).
            waterfall_allocation_pcts (list[float]): A list of percentages for waterfall allocation for new buys.
                                                     The list length determines the max number of new buys in a single run.
                                                     If None, a default descending weight allocation is created.
                                                     The list should sum to 1.0.
            stop_loss_pct (float, optional): The percentage loss at which to trigger a stop-loss sell (e.g., 0.05 for 5%). Defaults to None.
            take_profit_pct (float, optional): The percentage gain at which to trigger a take-profit sell (e.g., 0.10 for 10%). Defaults to None.
        """
        # Use paper=True for paper trading environment
        self.trading_client = TradingClient(self.KEY, self.SECRET, paper=True)
        self.data_client = StockHistoricalDataClient(self.KEY, self.SECRET)
        self.strategy = strategy
        self.max_positions = max_positions
        self.total_allocation_pct = total_allocation_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.waterfall_allocation_pcts = waterfall_allocation_pcts
        self.sp500_tickers = self._get_sp500_tickers()
        self.held_tickers = set()

        if not (0 < self.total_allocation_pct <= 1.0):
            raise ValueError("total_allocation_pct must be between 0 and 1.0.")

        if self.waterfall_allocation_pcts is None:
            # Create a default descending waterfall based on max_positions
            weights = list(range(self.max_positions, 0, -1))
            total_weight = sum(weights)
            self.waterfall_allocation_pcts = [w / total_weight for w in weights]
            print(f"Using default waterfall allocation for up to {self.max_positions} positions.")

        if abs(sum(self.waterfall_allocation_pcts) - 1.0) > 1e-9:
            raise ValueError("waterfall_allocation_pcts must sum to 1.0.")

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
                print(f"  -> BUY SIGNAL for {ticker}. Ranking Strength: {data['ranking_strength']:.4f}")
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
            positions = [] # Ensure positions is a list on failure

        # --- Check for stop-loss / take-profit triggers before analysis ---
        stop_loss_sells = []
        take_profit_sells = []
        if (self.stop_loss_pct is not None or self.take_profit_pct is not None) and positions:
            print("\n--- Checking for Stop-Loss and Take-Profit Triggers ---")
            for p in positions:
                if p.asset_class != AssetClass.US_EQUITY:
                    continue

                unrealized_plpc = float(p.unrealized_plpc) # profit/loss percentage

                # Check for stop-loss
                if self.stop_loss_pct is not None and unrealized_plpc <= -self.stop_loss_pct:
                    print(f"  -> STOP-LOSS triggered for {p.symbol} (Loss: {unrealized_plpc:.2%}).")
                    stop_loss_sells.append(p.symbol)
                    continue # A position can't be both a stop-loss and take-profit sell in the same run

                # Check for take-profit
                if self.take_profit_pct is not None and unrealized_plpc >= self.take_profit_pct:
                    print(f"  -> TAKE-PROFIT triggered for {p.symbol} (Gain: {unrealized_plpc:.2%}).")
                    take_profit_sells.append(p.symbol)

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
        # Combine strategy-based sells with stop-loss/take-profit sells
        all_sells_to_make = set(sell_signals) | set(stop_loss_sells) | set(take_profit_sells)
        self._execute_sells(list(all_sells_to_make))

        print(f"\n--- Analysis Complete ---")
        print(f"Found {len(positive_predictions)} stocks with a positive outlook.")
        print(f"Found {len(sell_signals)} stocks with a negative outlook (sell signals).")
        print(f"Triggered {len(stop_loss_sells)} stop-loss sells and {len(take_profit_sells)} take-profit sells.")

        if not positive_predictions:
            print("No stocks with positive outlook found. No trades will be placed.")
            return

        # --- Prioritize and Execute Buys ---
        # Sort by the chosen ranking metric (e.g., Sharpe ratio or mean return), descending
        sorted_predictions = sorted(positive_predictions, key=lambda x: x['ranking_strength'], reverse=True)

        if sorted_predictions:
            print("\n--- Top 20 Ranked Positive Signals ---")
            # The strength metric is now 'ranking_strength'
            for i, pick in enumerate(sorted_predictions[:20]):
                print(f"  {i+1:2d}. {pick['ticker']:<6}: Strength = {pick['ranking_strength']:.4f}")

        # --- Calculate capital for new buys based on total allocation ---
        try:
            account = self.trading_client.get_account()
            total_equity = float(account.equity)
            target_portfolio_value = total_equity * self.total_allocation_pct

            current_positions = self.trading_client.get_all_positions()
            self.held_tickers = {p.symbol for p in current_positions if p.asset_class == AssetClass.US_EQUITY}
            current_positions_value = sum(float(p.market_value) for p in current_positions if p.asset_class == AssetClass.US_EQUITY)

            cash_for_new_buys = target_portfolio_value - current_positions_value
            num_held_positions = len(self.held_tickers)
            slots_to_fill = self.max_positions - num_held_positions

            print(f"\nAccount Equity: ${total_equity:,.2f}")
            print(f"Target Allocation ({self.total_allocation_pct:.0%}): ${target_portfolio_value:,.2f}")
            print(f"Current Position Value: ${current_positions_value:,.2f}")
            print(f"Cash available for new buys: ${cash_for_new_buys:,.2f}")

        except APIError as e:
            print(f"Could not get account details to calculate allocation: {e}")
            return

        if cash_for_new_buys <= 1 or slots_to_fill <= 0: # Need at least $1 for notional orders
            print(f"\nPortfolio is full or strategy is fully allocated. No new buy orders will be placed.")
            top_picks = []
        else:
            print(f"\nPortfolio has {num_held_positions}/{self.max_positions} positions. Looking to fill up to {slots_to_fill} slot(s).")
            # Filter out stocks already held
            available_for_buy = [p for p in sorted_predictions if p['ticker'] not in self.held_tickers]
            # Limit buys by available slots and waterfall definition
            num_buys_to_make = min(len(available_for_buy), slots_to_fill, len(self.waterfall_allocation_pcts))
            top_picks = available_for_buy[:num_buys_to_make]

        if top_picks:
            print(f"\n--- Top {len(top_picks)} Picks for Trading (Waterfall Allocation) ---")
            for i, pick in enumerate(top_picks):
                allocation_pct = self.waterfall_allocation_pcts[i]
                notional_value = cash_for_new_buys * allocation_pct
                print(f"  - {pick['ticker']}: Allocating {allocation_pct:.1%} of available cash (${notional_value:,.2f})")

            print("\n--- Executing Buy Orders ---")
            for i, pick in enumerate(top_picks):
                ticker = pick['ticker']
                allocation_pct = self.waterfall_allocation_pcts[i]
                notional_value = round(cash_for_new_buys * allocation_pct, 2)

                if notional_value < 1: # Alpaca requires at least $1 for notional orders
                    print(f"  -> Skipping BUY for {ticker}, allocated value ${notional_value:.2f} is less than $1.")
                    continue

                print(f"Decision: BUY {ticker}. Placing notional market order for approx. ${notional_value:.2f}.")
                try:
                    market_order_data = MarketOrderRequest(
                        symbol=ticker,
                        notional=notional_value,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    self.trading_client.submit_order(order_data=market_order_data)
                    print(f"  -> Successfully submitted BUY order for {ticker}.")
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
    # TRADE_QUANTITY = 10    # Replaced by notional, waterfall allocation
    MAX_POSITIONS = 30    # Maximum number of positions to hold
    TOTAL_ALLOCATION_PCT = 0.50 # Use 50% of total equity for this strategy
    # Example of a custom waterfall: top pick gets 25%, second 20%, etc.
    # Must sum to 1.0. The length limits the number of new buys per run.
    WATERFALL_ALLOCATION_PCTS = [0.25, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05, 0.025, 0.025]

    # --- Risk Management Configuration ---
    STOP_LOSS_PCT = 0.05 # Sell if a position drops by 5%
    TAKE_PROFIT_PCT = 0.10 # Sell if a position gains 10%

    # --- Strategy Selection ---
    # Choose which strategy to run by uncommenting one of the blocks below.

    # == Option 1: HMM Strategy (Default) ==
    N_HMM_COMPONENTS = 3
    MODEL_ORDER = 1
    OPTIMIZE_ORDER_PER_STOCK = False
    MAX_ORDER_TO_TEST = 5
    RANKING_METRIC = 'sharpe' # 'sharpe' or 'return'
    print("Configuring HMM Strategy...")
    active_strategy = HMMStrategy(
        n_components=N_HMM_COMPONENTS,
        model_order=MODEL_ORDER,
        optimize_order=OPTIMIZE_ORDER_PER_STOCK,
        max_order_to_test=MAX_ORDER_TO_TEST,
        ranking_metric=RANKING_METRIC
    )

    # == Option 2: Donchian Breakout Strategy ==
    # from strategies import DonchianBreakoutStrategy
    # DONCHIAN_PERIOD = 20 # 20-day breakout is a classic
    # print("Configuring Donchian Breakout Strategy...")
    # active_strategy = DonchianBreakoutStrategy(period=DONCHIAN_PERIOD)


    # --- Trader Initialization and Execution ---
    # 1. Create the trader and pass the chosen strategy to it
    print(f"Starting Trader with {active_strategy.__class__.__name__}...")
    trader = Trader(
        strategy=active_strategy,
        max_positions=MAX_POSITIONS,
        total_allocation_pct=TOTAL_ALLOCATION_PCT,
        waterfall_allocation_pcts=WATERFALL_ALLOCATION_PCTS,
        stop_loss_pct=STOP_LOSS_PCT,
        take_profit_pct=TAKE_PROFIT_PCT
    )

    # 2. Run the trader
    trader.run_scanner_and_trade()
    print("\nTrader finished.")