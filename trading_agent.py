"""
Defines the TradingAgent, a class responsible for executing a trading strategy.

Author - Eli Jordan
Date - 07/29/2025
"""

import os
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from strategies import BaseStrategy

def _worker_analyze_ticker(strategy: BaseStrategy, tradable_universe: list, ticker: str, bars_df: pd.DataFrame, is_held: bool):
    """
    Worker function for parallel analysis. Runs in a separate process to avoid the GIL.
    This function is defined at the top level to ensure it can be pickled by ProcessPoolExecutor.

    Args:
        strategy (BaseStrategy): The strategy instance to use for analysis.
        tradable_universe (list): A list of tickers in the agent's universe (e.g., S&P 500 or crypto list).
        ticker (str): The stock ticker to analyze.
        bars_df (pd.DataFrame): A DataFrame of historical bar data for the ticker.
        is_held (bool): Whether the ticker is currently in the portfolio.

    Returns:
        tuple: A tuple containing (signal_type, data).
    """
    try:
        outlook, data = strategy.analyze(ticker, bars_df)

        if is_held and outlook == 'negative':
            print(f"  -> SELL SIGNAL for held position {ticker}.")
            return 'negative', ticker
        elif is_held:
            print(f"  -> HOLD SIGNAL for {ticker} (Outlook: {outlook}).")
            return 'no_action', ticker
        elif not is_held and ticker in tradable_universe and outlook == 'positive':
            print(f"  -> BUY SIGNAL for {ticker}. Ranking Strength: {data['ranking_strength']:.4f}")
            return 'positive', data
        else:
            # Covers neutral/negative signals for non-held stocks.
            return 'no_action', ticker

    except Exception as e:
        print(f"  -> Could not analyze {ticker}. Reason: {e}")
        return 'error', ticker

class TradingAgent:
    """
    A trading agent that scans S&P 500 stocks using a configurable strategy,
    identifies the top opportunities, and executes trades based on a
    defined capital allocation.
    """

    def __init__(self, name: str, strategy: BaseStrategy, trading_client: TradingClient, data_client: StockHistoricalDataClient, max_positions: int = 10, total_allocation_pct: float = 0.5, waterfall_allocation_pcts: list = None, stop_loss_pct: float = None, take_profit_pct: float = None, max_analysis_workers: int = 8, asset_class: str = 'us_equity'):
        """
        Initializes the agent and the Alpaca trading client.

        Args:
            name (str): A unique name for this agent instance.
            strategy (BaseStrategy): The trading strategy to use for analysis.
            trading_client (TradingClient): An authenticated Alpaca trading client.
            data_client (StockHistoricalDataClient): An authenticated Alpaca data client.
            max_positions (int): The maximum number of positions to hold at any time.
            total_allocation_pct (float): The percentage of total equity to allocate to this strategy (e.g., 0.5 for 50%).
            waterfall_allocation_pcts (list[float]): A list of percentages for waterfall allocation for new buys.
                                                     The list length determines the max number of new buys per run.
                                                     If None, a default descending weight allocation is created.
                                                     The list should sum to 1.0.
            stop_loss_pct (float, optional): The percentage loss at which to trigger a stop-loss sell (e.g., 0.05 for 5%). Defaults to None.
            take_profit_pct (float, optional): The percentage gain at which to trigger a take-profit sell (e.g., 0.10 for 10%). Defaults to None.
            max_analysis_workers (int, optional): The max number of processes for analysis. Defaults to os.cpu_count() - 1.
            asset_class (str, optional): The class of assets to trade ('us_equity' or 'crypto'). Defaults to 'us_equity'.
        """
        self.name = name
        self.trading_client = trading_client
        self.strategy = strategy
        self.max_positions = max_positions
        self.total_allocation_pct = total_allocation_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.waterfall_allocation_pcts = waterfall_allocation_pcts
        self.max_analysis_workers = max_analysis_workers
        self.asset_class = asset_class

        # Initialize the correct data client and ticker universe based on asset class
        if self.asset_class == 'crypto':
            self.data_client = CryptoHistoricalDataClient(os.getenv("PAPER_KEY"), os.getenv("PAPER_SEC"))
            self.tradable_tickers = self._get_crypto_tickers()
            print(f"Crypto agent '{self.name}' initialized. Trading: {self.tradable_tickers}")
        else:  # 'us_equity'
            self.data_client = data_client  # Use the one passed from Orchestrator
            self.tradable_tickers = self._get_sp500_tickers()
            print(f"Equity agent '{self.name}' initialized.")

        if not (0 < self.total_allocation_pct <= 1.0):
            raise ValueError("total_allocation_pct must be between 0 and 1.0.")

        if self.waterfall_allocation_pcts is None:
            weights = list(range(self.max_positions, 0, -1))
            total_weight = sum(weights)
            self.waterfall_allocation_pcts = [w / total_weight for w in weights]
            print(f"Using default waterfall allocation for up to {self.max_positions} positions.")

        if abs(sum(self.waterfall_allocation_pcts) - 1.0) > 1e-9:
            raise ValueError("waterfall_allocation_pcts must sum to 1.0.")

    def generate_trade_decisions(self, account, all_positions, owned_tickers):
        """
        Analyzes the market and generates a list of buy and sell decisions.
        This method does not execute any trades.

        Args:
            account: The Alpaca account object.
            all_positions (list): A list of all positions in the account.
            owned_tickers (set): A set of tickers this agent currently owns.

        Returns:
            dict: A dictionary containing 'buys' and 'sells' lists.
                  - 'sells': A list of tickers to sell.
                  - 'buys': A list of dictionaries, each representing a stock to buy,
                            including 'ticker' and 'notional_value'.
        """
        print(f"Agent '{self.name}' owns {len(owned_tickers)} tickers: {list(owned_tickers)}")

        # 1. Identify sell signals for positions owned by this agent
        owned_positions = [p for p in all_positions if p.symbol in owned_tickers]
        risk_management_sells = self._check_risk_management_triggers(owned_positions)
        
        # 2. Scan market for buy signals and strategy-based sell signals
        positive_signals, strategy_sells = self._scan_and_analyze_market(owned_tickers)

        # Combine all sell signals for tickers this agent owns
        all_sells_to_make = set(risk_management_sells) | set(strategy_sells)
        
        # 3. Determine buy decisions based on positive signals and capital
        buy_decisions = self._decide_buys(positive_signals, account, all_positions, owned_tickers)

        return {
            'sells': list(all_sells_to_make),
            'buys': buy_decisions
        }

    def _check_risk_management_triggers(self, positions):
        """Checks all positions for stop-loss or take-profit conditions."""
        stop_loss_sells = []
        take_profit_sells = []
        if not (self.stop_loss_pct or self.take_profit_pct) or not positions:
            return []

        print("Checking for stop-loss and take-profit triggers...")
        for p in positions:
            # Ensure agent only manages assets of its designated class
            if self.asset_class == 'us_equity' and p.asset_class != AssetClass.US_EQUITY:
                continue
            if self.asset_class == 'crypto' and p.asset_class != AssetClass.CRYPTO:
                continue

            unrealized_plpc = float(p.unrealized_plpc)

            if self.stop_loss_pct is not None and unrealized_plpc <= -self.stop_loss_pct:
                print(f"  -> STOP-LOSS triggered for {p.symbol} (Loss: {unrealized_plpc:.2%}).")
                stop_loss_sells.append(p.symbol)
                continue

            if self.take_profit_pct is not None and unrealized_plpc >= self.take_profit_pct:
                print(f"  -> TAKE-PROFIT triggered for {p.symbol} (Gain: {unrealized_plpc:.2%}).")
                take_profit_sells.append(p.symbol)

        print(f"Triggered {len(stop_loss_sells)} stop-loss and {len(take_profit_sells)} take-profit sells.")
        return stop_loss_sells + take_profit_sells

    def _scan_and_analyze_market(self, held_tickers):
        """Scans, fetches data, and analyzes tickers to generate trade signals."""
        tickers_to_analyze = sorted(list(set(self.tradable_tickers) | held_tickers))
        if not tickers_to_analyze:
            print("No tickers to analyze.")
            return [], []

        positive_predictions = []
        sell_signals = []
        BATCH_SIZE = 100
        ticker_batches = [tickers_to_analyze[i:i + BATCH_SIZE] for i in range(0, len(tickers_to_analyze), BATCH_SIZE)]

        num_workers = self.max_analysis_workers
        if num_workers is None:
            # Default to all but one core to leave resources for the main event loop and OS.
            # Ensure at least one worker.
            num_workers = max(1, os.cpu_count() - 1)

        print(f"Starting market scan for {len(tickers_to_analyze)} tickers using {num_workers} worker processes...")

        for batch_num, batch in enumerate(ticker_batches):
            print(f"  -> Processing batch {batch_num + 1}/{len(ticker_batches)} ({len(batch)} tickers)...")
            try:
                end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')

                if self.asset_class == 'crypto':
                    request_params = CryptoBarsRequest(
                        symbol_or_symbols=batch,
                        timeframe=TimeFrame.Day,
                        start=start_date,
                        end=end_date
                    )
                    bars_data = self.data_client.get_crypto_bars(request_params)
                else:  # us_equity
                    request_params = StockBarsRequest(
                        symbol_or_symbols=batch,
                        timeframe=TimeFrame.Day,
                        start=start_date,
                        end=end_date,
                        feed=DataFeed.IEX
                    )
                    bars_data = self.data_client.get_stock_bars(request_params)

                if bars_data.df.empty:
                    print(f"    -> API returned no data for batch {batch_num + 1}.")
                    continue

                grouped_data = bars_data.df.groupby('symbol')

                # Use ProcessPoolExecutor for CPU-bound tasks like HMM training.
                # This avoids Python's GIL and can fully utilize multiple CPU cores.
                # We set max_workers to the number of CPU cores for optimal performance.
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    future_to_ticker = {}
                    for ticker in batch:
                        if ticker in grouped_data.groups:
                            ticker_df = grouped_data.get_group(ticker).reset_index(level='symbol', drop=True)
                            is_held = ticker in held_tickers
                            future = executor.submit(_worker_analyze_ticker, self.strategy, self.tradable_tickers, ticker, ticker_df, is_held)
                            future_to_ticker[future] = ticker

                    for i, future in enumerate(as_completed(future_to_ticker)):
                        print(f"     Progress on batch: ({i+1}/{len(future_to_ticker)})", end='\r')
                        signal_type, data = future.result()
                        if signal_type == 'positive':
                            positive_predictions.append(data)
                        elif signal_type == 'negative' and data in held_tickers:
                            sell_signals.append(data)
                print(" " * 40, end='\r')

            except APIError as e:
                print(f"    -> API Error on batch {batch_num + 1}: {e}. Skipping batch.")
            except Exception as e:
                print(f"    -> Unexpected error on batch {batch_num + 1}: {e}. Skipping batch.")
        
        print("Market scan complete.")
        print(f"Found {len(positive_predictions)} stocks with a positive outlook.")
        print(f"Found {len(sell_signals)} stocks with a negative outlook (strategy sell signals).")
        return positive_predictions, sell_signals

    def _decide_buys(self, positive_predictions, account, all_positions, owned_tickers):
        """Prioritizes buy signals and calculates notional values for each."""
        buy_decisions = []
        if not positive_predictions:
            print("No positive signals found, no new buy decisions.")
            return buy_decisions

        sorted_predictions = sorted(positive_predictions, key=lambda x: x['ranking_strength'], reverse=True)

        total_equity = float(account.equity)
        target_portfolio_value = total_equity * self.total_allocation_pct

        # Calculate value of positions owned *by this agent*
        agent_positions_value = sum(float(p.market_value) for p in all_positions if p.symbol in owned_tickers)

        cash_for_new_buys = target_portfolio_value - agent_positions_value
        num_held_positions = len(owned_tickers)
        slots_to_fill = self.max_positions - num_held_positions

        print(f"Agent Target Allocation ({self.total_allocation_pct:.0%}): ${target_portfolio_value:,.2f}")
        print(f"Agent Current Position Value: ${agent_positions_value:,.2f}")
        print(f"Cash available for new buys: ${cash_for_new_buys:,.2f}")

        if cash_for_new_buys <= 1 or slots_to_fill <= 0:
            print(f"Agent portfolio is full or fully allocated. No new buy decisions.")
            return buy_decisions

        print(f"Agent has {num_held_positions}/{self.max_positions} positions. Looking to fill up to {slots_to_fill} slot(s).")
        # Filter out stocks already owned by this agent
        available_for_buy = [p for p in sorted_predictions if p['ticker'] not in owned_tickers]
        num_buys_to_make = min(len(available_for_buy), slots_to_fill, len(self.waterfall_allocation_pcts))
        top_picks = available_for_buy[:num_buys_to_make]

        if not top_picks:
            print("Top picks are already held by this agent or no new signals. No new buy decisions.")
            return buy_decisions

        print(f"Top {len(top_picks)} picks for buying:")
        for i, pick in enumerate(top_picks):
            allocation_pct = self.waterfall_allocation_pcts[i]
            notional_value = cash_for_new_buys * allocation_pct
            pick['notional_value'] = round(notional_value, 2)
            print(f"  - {pick['ticker']}: Strength={pick['ranking_strength']:.4f}, Notional=${pick['notional_value']:.2f}")
            if pick['notional_value'] >= 1:
                buy_decisions.append(pick)

        return buy_decisions

    def _get_crypto_tickers(self):
        """Fetches the list of crypto tickers from an environment variable."""
        print("Fetching crypto tickers from environment variable CRYPTO_TICKERS...")
        crypto_list_str = os.getenv("CRYPTO_TICKERS")
        if not crypto_list_str:
            print("Warning: CRYPTO_TICKERS environment variable not set. No crypto tickers to trade.")
            return []

        tickers = [ticker.strip() for ticker in crypto_list_str.split(',')]
        print(f"Found {len(tickers)} crypto tickers.")
        return tickers

    def _get_sp500_tickers(self):
        """Fetches the list of S&P 500 tickers from Wikipedia."""
        print("Fetching S&P 500 tickers...")
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            table = pd.read_html(response.text)
            df = table[0]
            tickers = df['Symbol'].tolist()
            print(f"Found {len(tickers)} tickers.")
            return tickers
        except Exception as e:
            print(f"Could not fetch S&P 500 tickers: {e}")
            return []