"""
Defines the strategy interface and concrete strategy implementations for the trader.

Author - Eli Jordan
Date - 07/29/2025
"""

from abc import ABC, abstractmethod
import pandas as pd
from hmm_analysis import AnalyzeHMM

class BaseStrategy(ABC):
    """
    Abstract base class for a trading strategy.
    """
    @abstractmethod
    def analyze(self, ticker: str, bars_data: pd.DataFrame):
        """
        Analyzes a single stock and returns a trading outlook.

        Args:
            ticker (str): The stock ticker to analyze.
            bars_data (pd.DataFrame): A DataFrame of historical bar data for the ticker.

        Returns:
            tuple: A tuple containing (outlook, data).
                   - outlook (str): 'positive', 'negative', or 'similar'/'neutral'.
                   - data (dict): A dictionary containing metadata for the decision,
                                  e.g., predicted returns. Must include 'ticker'.
        """
        pass

    @abstractmethod
    def generate_signals(self, bars_data: pd.DataFrame):
        """
        Analyzes historical data and returns entry and exit signals for backtesting.

        Args:
            bars_data (pd.DataFrame): A DataFrame of historical bar data for one ticker.

        Returns:
            tuple: A tuple of two pandas Series (entries, exits) with boolean values
                   and an index matching `bars_data`.
        """
        pass


class HMMStrategy(BaseStrategy):
    """
    A trading strategy that uses a Hidden Markov Model to predict market regimes.
    """
    def __init__(self, n_components: int = 3, model_order: int = 1, optimize_order: bool = False, max_order_to_test: int = 10, ranking_metric: str = 'sharpe'):
        """
        Initializes the HMM-based strategy.

        Args:
            n_components (int): The number of hidden states for the HMM.
            model_order (int): The default order of the Markov model.
            optimize_order (bool): If True, finds the optimal order for each stock individually.
            max_order_to_test (int): The maximum order to test when optimizing.
            ranking_metric (str): The metric to rank positive signals ('sharpe' or 'return').
        """
        self.n_components = n_components
        self.model_order = model_order
        self.optimize_order = optimize_order
        self.max_order_to_test = max_order_to_test
        if ranking_metric not in ['sharpe', 'return']:
            raise ValueError("ranking_metric must be either 'sharpe' or 'return'.")
        self.ranking_metric = ranking_metric

    def analyze(self, ticker: str, bars_data: pd.DataFrame):
        """
        Performs HMM analysis for a single stock and returns an outlook.

        Args:
            ticker (str): The stock ticker to analyze.
            bars_data (pd.DataFrame): Historical bar data for the ticker.

        Returns:
            tuple: (outlook, prediction_dict)
        """
        # 1. Determine the model order (either default or optimized)
        current_model_order = self.model_order
        if self.optimize_order:
            print(f"  -> Optimizing model order for {ticker} (max: {self.max_order_to_test})...")
            # Create a temporary analyzer to find the optimal order
            temp_analyzer = AnalyzeHMM(ticker, n_components=self.n_components, model_order=1, bars_data=bars_data)
            current_model_order = temp_analyzer.find_optimal_order(max_order=self.max_order_to_test)
            print(f"  -> Using optimal order {current_model_order} for {ticker}.")

        # 2. Create the final analyzer with the determined order and pre-fetched data
        analyzer = AnalyzeHMM(
            ticker=ticker,
            n_components=self.n_components,
            model_order=current_model_order,
            bars_data=bars_data
        )

        prediction = analyzer.predict_next_day_outlook()
        outlook = prediction['outlook']
        prediction['ticker'] = ticker

        # 3. Calculate the ranking strength based on the chosen metric
        if self.ranking_metric == 'sharpe':
            mean_return = prediction['predicted_state_mean_return']
            std_return = prediction['predicted_state_std_return']
            # Add a small epsilon to avoid division by zero for states with no volatility
            prediction['ranking_strength'] = mean_return / (std_return + 1e-9)
        else: # 'return'
            prediction['ranking_strength'] = prediction['predicted_state_mean_return']

        return outlook, prediction

    def generate_signals(self, bars_data: pd.DataFrame):
        """
        Generates HMM-based entry and exit signals for a given historical dataset.

        NOTE: This implementation trains the HMM on the entire dataset at once,
        which introduces significant lookahead bias. A more rigorous backtest
        would use a walk-forward or rolling window approach.

        Args:
            bars_data (pd.DataFrame): Historical bar data for the ticker.

        Returns:
            tuple: (entries, exits) pandas Series.
        """
        if bars_data.empty or len(bars_data) < 60: # Need enough data for features
            return pd.Series(False, index=bars_data.index), pd.Series(False, index=bars_data.index)

        # 1. Analyze the entire history to find states. Run in non-verbose mode.
        analyzer = AnalyzeHMM(
            ticker="backtest",
            n_components=self.n_components,
            model_order=self.model_order,
            bars_data=bars_data,
            verbose=False
        )

        # 2. Identify positive and negative states
        # state_regimes is sorted by return, from lowest to highest.
        negative_state = analyzer.state_regimes[0]
        positive_state = analyzer.state_regimes[-1]

        # 3. Get the series of hidden states for the historical data
        states = analyzer.data['Hidden_State']

        # 4. Create entry and exit signals
        entries = (states == positive_state)
        exits = (states == negative_state)

        # Reindex to match the original bars_data index, filling missing values with False
        return entries.reindex(bars_data.index, fill_value=False), exits.reindex(bars_data.index, fill_value=False)


class DonchianBreakoutStrategy(BaseStrategy):
    """
    A strategy based on Donchian Channel breakouts.
    - Buy signal: Price closes above the upper channel of the previous period.
    - Sell signal: Price closes below the lower channel of the previous period.
    """
    def __init__(self, period: int = 20):
        """
        Initializes the Donchian Breakout strategy.
        Args:
            period (int): The lookback period for the Donchian Channels (e.g., 20 days).
        """
        if period < 1:
            raise ValueError("Period must be greater than 0.")
        self.period = period

    def _calculate_channels(self, bars_data: pd.DataFrame):
        """Helper to calculate Donchian channels."""
        data = bars_data.copy()
        data['upper'] = data['high'].rolling(self.period).max()
        data['lower'] = data['low'].rolling(self.period).min()
        return data

    def analyze(self, ticker: str, bars_data: pd.DataFrame):
        """
        Analyzes the most recent data point for a breakout signal for live trading.
        """
        # We need at least `period` days of history PLUS the current day to check for a breakout.
        if len(bars_data) < self.period + 1:
            return 'neutral', {'ticker': ticker, 'predicted_state_mean_return': 0.0, 'last_return': 0.0}

        # The most recent data point is T-1 (yesterday).
        last_bar = bars_data.iloc[-1]
        last_close = last_bar['close']

        # The lookback data is the `period` days BEFORE the last bar.
        # This defines the channel that the last bar's close can break out of.
        lookback_bars = bars_data.iloc[-(self.period + 1):-1]

        if lookback_bars.empty:
            return 'neutral', {'ticker': ticker, 'predicted_state_mean_return': 0.0, 'last_return': 0.0}

        # Find the highest high and lowest low in that lookback period.
        upper_channel = lookback_bars['high'].max()
        lower_channel = lookback_bars['low'].min()

        outlook = 'neutral'
        strength = 0.0

        if last_close > upper_channel:
            outlook = 'positive'
            strength = (last_close - upper_channel) / upper_channel
        elif last_close < lower_channel:
            outlook = 'negative'

        prediction_data = {
            'ticker': ticker,
            'last_return': bars_data['close'].pct_change().iloc[-1],
            'predicted_state_mean_return': strength
        }

        return outlook, prediction_data

    def generate_signals(self, bars_data: pd.DataFrame):
        """
        Generates historical entry and exit signals for backtesting.
        """
        if len(bars_data) < self.period:
            return pd.Series(False, index=bars_data.index), pd.Series(False, index=bars_data.index)

        data_with_channels = self._calculate_channels(bars_data)

        # A buy signal is when the close crosses above the *previous* day's upper channel.
        # The channel for a given day includes that day's high/low, so we must shift
        # the channel data by 1 to avoid looking into the future.
        entries = data_with_channels['close'] > data_with_channels['upper'].shift(1)
        exits = data_with_channels['close'] < data_with_channels['lower'].shift(1)

        # Ensure signals are boolean and align with the original index
        entries = entries.reindex(bars_data.index, fill_value=False)
        exits = exits.reindex(bars_data.index, fill_value=False)

        return entries, exits