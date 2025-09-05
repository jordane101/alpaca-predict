"""
Provides backtesting functionality for trading strategies using vectorbt.

Author - Eli Jordan
Date - 07/29/2025
"""
import os
import pandas as pd
import vectorbt as vbt
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from strategies import BaseStrategy

class Backtester:
    """
    A class to backtest a trading strategy on a single stock using vectorbt.
    """
    load_dotenv(".env")
    KEY = os.getenv("PAPER_KEY")
    SECRET = os.getenv("PAPER_SEC")

    def __init__(self, strategy: BaseStrategy, ticker: str, start_date: str, end_date: str, timeframe=TimeFrame.Day):
        """
        Initializes the Backtester.

        Args:
            strategy (BaseStrategy): The strategy instance to backtest.
            ticker (str): The stock ticker to backtest.
            start_date (str): The start date for the backtest (e.g., "2022-01-01").
            end_date (str): The end date for the backtest (e.g., "2023-01-01").
            timeframe: The timeframe for the data.
        """
        self.strategy = strategy
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.data_client = StockHistoricalDataClient(self.KEY, self.SECRET)
        self.bars_df = None
        self.close_prices = None
        self.entries = None
        self.exits = None

    def _fetch_data(self):
        """Fetches historical price data for the specified ticker."""
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")
        request_params = StockBarsRequest(
            symbol_or_symbols=[self.ticker], # API requires a list
            timeframe=self.timeframe,
            start=self.start_date,
            end=self.end_date
        )
        bars = self.data_client.get_stock_bars(request_params)

        if not bars.df.empty:
            # For a single stock, the .df property gives a multi-index DataFrame.
            # We reset the index to remove the 'symbol' level.
            self.bars_df = bars.df.reset_index(level='symbol', drop=True)
            self.close_prices = self.bars_df['close']
            print(f"Data fetched. Price data shape: {self.close_prices.shape}")
        else:
            print(f"No data fetched for {self.ticker}.")
            self.bars_df = pd.DataFrame()
            self.close_prices = pd.Series(dtype='float64')

    def _generate_signals(self):
        """Generates entry and exit signals for the ticker using the strategy."""
        if self.bars_df.empty:
            print("No data available to generate signals.")
            self.entries = pd.Series(dtype='bool')
            self.exits = pd.Series(dtype='bool')
            return

        print(f"Generating signals for {self.ticker}...")
        try:
            # The strategy's generate_signals method takes a DataFrame of bar data
            # and returns two boolean Series for entries and exits.
            self.entries, self.exits = self.strategy.generate_signals(self.bars_df)
            print("Signal generation complete.")
        except Exception as e:
            print(f"\nFailed to generate signals for {self.ticker}: {e}")
            self.entries = pd.Series(dtype='bool')
            self.exits = pd.Series(dtype='bool')

    def run(self, initial_cash=100000, freq='1D', verbose=True):
        """
        Runs the backtest and prints the performance summary.
        Args:
            initial_cash (int): The starting cash for the portfolio.
            freq (str): The frequency of the data, required by vectorbt.
            verbose (bool): If True, print full stats and save plot.
        """
        self._fetch_data()
        self._generate_signals()

        if self.entries is None or self.entries.empty:
            if verbose:
                print("No signals were generated. Cannot run backtest.")
            return None

        # For a single stock, vectorbt can use Series directly.
        portfolio = vbt.Portfolio.from_signals(
            self.close_prices,
            entries=self.entries,
            exits=self.exits,
            init_cash=initial_cash,
            freq=freq,
        )

        stats = portfolio.stats()

        if verbose:
            print(f"\n--- Backtest Results for {self.ticker} ---")
            print(stats)

            print("\n--- Important Caveat: Lookahead Bias ---")
            print("The current HMMStrategy trains on the full backtest period at once.")
            print("This introduces 'lookahead bias' because the model 'knows' the future when making past 'decisions'.")
            print("For a more realistic result, a walk-forward optimization or rolling training window should be implemented.")

            print("\nGenerating and saving plot...")
            try:
                fig = portfolio.plot()
                output_dir = "outputs"
                os.makedirs(output_dir, exist_ok=True)
                strategy_name = self.strategy.__class__.__name__
                filename = f"{output_dir}/backtest_{self.ticker}_{strategy_name}_{self.start_date}_to_{self.end_date}.html"
                fig.write_html(filename)
                print(f"Plot saved to: {filename}")
            except Exception as e:
                print(f"Could not generate or save plot. Reason: {e}")

        return stats

if __name__ == '__main__':
    # --- Backtest Configuration ---
    BACKTEST_TICKER = 'TSLA' # Test a single, volatile stock
    START_DATE = "2023-01-01"
    END_DATE = "2024-12-31"
    INITIAL_CASH = 100_000

    # --- Strategy Selection ---
    # Choose which strategy to test by uncommenting one of the blocks below.

    # == Option 1: HMM Strategy (Default) ==
    # from strategies import HMMStrategy
    # active_strategy = HMMStrategy(n_components=3, model_order=1, optimize_order=False)

    # == Option 2: Donchian Breakout Strategy ==
    from strategies import DonchianBreakoutStrategy
    active_strategy = DonchianBreakoutStrategy(period=20)

    # --- Run Backtest ---
    print(f"--- Setting up Backtest for {BACKTEST_TICKER} with {active_strategy.__class__.__name__} ---")
    backtester = Backtester(strategy=active_strategy, ticker=BACKTEST_TICKER, start_date=START_DATE, end_date=END_DATE)
    backtester.run(initial_cash=INITIAL_CASH)