"""
A simple trading bot that uses the HMM analysis prediction
to make trade decisions with the Alpaca API.

Author - Eli Jordan
Date - 07/29/2025
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError

from hmm_analysis import AnalyzeHMM

class HMMTrader:
    """
    A trader that scans S&P 500 stocks using Hidden Markov Model predictions,
    identifies the top opportunities, and executes trades.
    """
    load_dotenv(".env")
    KEY = os.getenv("PAPER_KEY")
    SECRET = os.getenv("PAPER_SEC")

    def __init__(self, trade_qty: int = 1, top_n: int = 10):
        """
        Initializes the trader, HMM analyzer, and the Alpaca trading client.

        Args:
            trade_qty (int): The number of shares to trade in a single order.
            top_n (int): The number of top-ranked stocks to trade.
        """
        # Use paper=True for paper trading environment
        self.trading_client = TradingClient(self.KEY, self.SECRET, paper=True)
        self.trade_qty = trade_qty
        self.top_n = top_n
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

        positive_predictions = []

        print(f"\n--- Starting Market Scan ({len(tickers_to_analyze)} tickers) ---")
        for i, ticker in enumerate(tickers_to_analyze):
            print(f"\nAnalyzing ({i+1}/{len(tickers_to_analyze)}): {ticker}")
            try:
                analyzer = AnalyzeHMM(ticker=ticker, n_components=3)
                prediction = analyzer.predict_next_day_outlook()
                outlook = prediction['outlook']

                # --- Decision Logic ---
                is_held = ticker in self.held_tickers

                if is_held and outlook == 'negative':
                    print(f"  -> SELL SIGNAL for held position {ticker}. Closing position.")
                    try:
                        self.trading_client.close_position(ticker)
                        print(f"  -> Successfully submitted SELL order for {ticker}.")
                        # Remove from held_tickers so it's not considered for buying later in this run
                        self.held_tickers.remove(ticker)
                    except APIError as e:
                        print(f"  -> Failed to close position for {ticker}. Reason: {e}")
                elif is_held:
                    print(f"  -> HOLD SIGNAL for held position {ticker} (Outlook: {outlook}).")
                elif not is_held and ticker in self.sp500_tickers and outlook == 'positive':
                    print(f"  -> BUY SIGNAL found for {ticker}. Avg Return: {prediction['predicted_state_mean_return']:.4f}")
                    prediction['ticker'] = ticker
                    positive_predictions.append(prediction)
                else:
                    print(f"  -> Outlook for {ticker}: {outlook}. No action taken.")
                
                time.sleep(0.1) # Small delay to be kind to APIs

            except Exception as e:
                print(f"  -> Could not analyze {ticker}. Reason: {e}")
                continue

        print(f"\n--- Analysis Complete ---")
        print(f"Found {len(positive_predictions)} stocks with a positive outlook.")

        if not positive_predictions:
            print("No stocks with positive outlook found. No trades will be placed.")
            return
        
        # Sort by predicted mean return, descending
        sorted_predictions = sorted(positive_predictions, key=lambda x: x['predicted_state_mean_return'], reverse=True)
        top_picks = sorted_predictions[:self.top_n]

        print(f"\n--- Top {self.top_n} Picks for Trading ---")
        for pick in top_picks:
            print(f"  - {pick['ticker']}: Predicted State Avg. Return = {pick['predicted_state_mean_return']:.4f}")

        self._execute_buys(top_picks)

    def _execute_buys(self, picks_to_buy):
        """Executes buy orders for the given list of stock picks."""
        print("\n--- Executing Buy Orders ---")
        for pick in picks_to_buy:
            ticker = pick['ticker']
            if ticker in self.held_tickers:
                print(f"Decision: HOLD {ticker}. Already have a position.")
            else:
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

if __name__ == "__main__":
    # --- Configuration ---
    TRADE_QUANTITY = 1    # Number of shares to trade for each new position
    TOP_N_STOCKS = 10     # Number of top stocks to consider for buying
    # ---------------------

    print("Starting HMM Trader...")
    trader = HMMTrader(trade_qty=TRADE_QUANTITY, top_n=TOP_N_STOCKS)
    trader.run_scanner_and_trade()
    print("\nHMM Trader finished.")