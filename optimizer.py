"""
Provides optimization functionality for trading strategies by running multiple backtests.

Author - Eli Jordan
Date - 07/29/2025
"""
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Import the components we need
from backtester import Backtester
from strategies import DonchianBreakoutStrategy

class Optimizer:
    """
    A class to optimize the parameters of a trading strategy on a single stock
    by running multiple backtests.
    """
    load_dotenv(".env")
    KEY = os.getenv("PAPER_KEY")
    SECRET = os.getenv("PAPER_SEC")

    def __init__(self, ticker: str, start_date: str, end_date: str):
        """
        Initializes the Optimizer.

        Args:
            ticker (str): The stock ticker to optimize on.
            start_date (str): The start date for the optimization period.
            end_date (str): The end date for the optimization period.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def run(self, param_grid: dict, metric: str = 'Total Return [%]', initial_cash=100000):
        """
        Runs the optimization by looping through parameters and backtesting each one.

        Args:
            param_grid (dict): A dictionary where keys are parameter names and values are lists of values to test.
                               Example: {'period': [10, 20, 30, 40, 50]}
            metric (str): The performance metric from vectorbt's stats to optimize for.
            initial_cash (int): The starting cash for the portfolio.
        """
        param_name = list(param_grid.keys())[0]
        param_values = param_grid[param_name]

        results = []

        print(f"--- Starting Optimization for DonchianBreakoutStrategy on {self.ticker} ---")
        print(f"Testing parameter '{param_name}' with values: {param_values}")

        for i, value in enumerate(param_values):
            print(f"  -> Testing {param_name} = {value} ({i+1}/{len(param_values)})")

            # 1. Create the strategy with the current parameter value
            strategy = DonchianBreakoutStrategy(period=value)

            # 2. Create and run the backtester for this single strategy instance
            backtester = Backtester(
                strategy=strategy,
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            # Run in non-verbose mode to avoid cluttering the console
            stats = backtester.run(initial_cash=initial_cash, verbose=False)

            # 3. Store the results
            if stats is not None and not stats.empty:
                # Add the parameter value to the stats Series for later reference
                stats['period'] = value
                results.append(stats)
            else:
                print(f"     No trades for period = {value}. Skipping.")

        if not results:
            print("\nOptimization finished with no valid results. No trades were made for any parameter.")
            return

        # 4. Analyze the collected results
        results_df = pd.DataFrame(results)
        results_df = results_df.set_index('period') # Use 'period' as the index

        best_period = results_df[metric].idxmax()
        best_stats = results_df.loc[best_period]

        print(f"\n--- Optimization Results for {self.ticker} ---")
        print(f"Optimized for: {metric}")
        print(f"\nBest '{param_name}': {best_period}")
        print("\n--- Performance with Best Parameter ---")
        print(best_stats)

        # 5. Plot and save results
        print("\nGenerating and saving optimization plot...")
        try:
            # We need to import vectorbt here for plotting
            import vectorbt as vbt
            # First, create the base plot from the vectorbt accessor
            fig = results_df[metric].vbt.plot()
            # Then, update the layout with the correct plotly property names
            fig.update_layout(
                title_text=f"Optimization of '{param_name}' for {self.ticker}",
                xaxis_title=f"'{param_name}' value",
                yaxis_title=metric
            )
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{output_dir}/optimize_{self.ticker}_DonchianBreakoutStrategy_{param_name}.html"
            fig.write_html(filename)
            print(f"Plot saved to: {filename}")
        except Exception as e:
            print(f"Could not generate or save plot. Reason: {e}")

if __name__ == '__main__':
    OPTIMIZE_TICKER = 'AAPL'
    START_DATE = "2022-01-01"
    END_DATE = "2023-12-31"
    PARAM_GRID = {'period': np.arange(10, 101, 1)} # Test periods from 10 to 100 in steps of 5
    OPTIMIZATION_METRIC = 'Sharpe Ratio'

    optimizer = Optimizer(ticker=OPTIMIZE_TICKER, start_date=START_DATE, end_date=END_DATE)
    optimizer.run(param_grid=PARAM_GRID, metric=OPTIMIZATION_METRIC)