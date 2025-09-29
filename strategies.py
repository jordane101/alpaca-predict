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
    def __init__(self, n_components: int = 3, model_order: int = 1, optimize_order: bool = False, max_order_to_test: int = 10, ranking_metric: str = 'sharpe', retrain_max_age_days: int = 30, walk_forward_window: int = 252, retrain_period: int = 63):
        """
        Initializes the HMM-based strategy.

        Args:
            n_components (int): The number of hidden states for the HMM.
            model_order (int): The default order of the Markov model.
            optimize_order (bool): If True, finds the optimal order for each stock individually.
            max_order_to_test (int): The maximum order to test when optimizing.
            ranking_metric (str): The metric to rank positive signals ('sharpe' or 'return').
            retrain_max_age_days (int): The max age in days for a cached model before it's retrained for live trading.
            walk_forward_window (int): The size of the rolling training window (in days) for backtesting.
            retrain_period (int): How often (in days) to retrain the model during a walk-forward backtest.
        """
        self.n_components = n_components
        self.model_order = model_order
        self.optimize_order = optimize_order
        self.max_order_to_test = max_order_to_test
        self.retrain_max_age_days = retrain_max_age_days
        self.walk_forward_window = walk_forward_window
        self.retrain_period = retrain_period
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
            # Create a temporary analyzer to find the optimal order.
            # Force retraining to ensure optimization runs on fresh data.
            temp_analyzer = AnalyzeHMM(
                ticker,
                n_components=self.n_components,
                model_order=1,
                bars_data=bars_data,
                force_retrain=True
            )
            current_model_order = temp_analyzer.find_optimal_order(max_order=self.max_order_to_test)
            print(f"  -> Using optimal order {current_model_order} for {ticker}.")

        # 2. Create the final analyzer with the determined order and pre-fetched data
        # This will use the cache if a valid model exists for the given order.
        analyzer = AnalyzeHMM(
            ticker=ticker,
            n_components=self.n_components,
            model_order=current_model_order,
            bars_data=bars_data,
            max_age_days=self.retrain_max_age_days
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
        Generates HMM-based entry and exit signals for a given historical dataset
        using a walk-forward approach to mitigate lookahead bias.

        The model is trained on a rolling window of past data and then used to
        predict signals for the next period. This is more realistic but slower
        than training on the full dataset at once.

        Args:
            bars_data (pd.DataFrame): Historical bar data for the ticker.

        Returns:
            tuple: (entries, exits) pandas Series with boolean values.
        """
        # Minimum data needed is one training window + one feature lookback period
        min_data_len = self.walk_forward_window + 60
        if bars_data.empty or len(bars_data) < min_data_len:
            print(f"  -> Not enough data for walk-forward backtest. Need at least {min_data_len}, have {len(bars_data)}.")
            return pd.Series(False, index=bars_data.index), pd.Series(False, index=bars_data.index)

        entries = pd.Series(False, index=bars_data.index)
        exits = pd.Series(False, index=bars_data.index)
        
        feature_lookback = 60

        print(f"  -> Starting walk-forward backtest for HMMStrategy (Window: {self.walk_forward_window}, Retrain every: {self.retrain_period} days)...")
        for i in range(self.walk_forward_window, len(bars_data), self.retrain_period):
            train_start, train_end = i - self.walk_forward_window, i
            training_data = bars_data.iloc[train_start:train_end]

            try:
                # This needs a method to predict on new data, which is not in the original AnalyzeHMM
                # We need to add `predict_states_for_new_data` to AnalyzeHMM
                # For now, we'll re-implement a simplified version here.
                analyzer = AnalyzeHMM(
                    ticker="backtest_wf",
                    n_components=self.n_components,
                    model_order=self.model_order,
                    bars_data=training_data,
                    verbose=False,
                    force_retrain=True
                )
            except Exception as e:
                print(f"  -> Warning: Walk-forward training failed for window [{train_start}:{train_end}]. Reason: {e}")
                continue

            predict_start = i
            predict_end = min(i + self.retrain_period, len(bars_data))
            prediction_data = bars_data.iloc[predict_start:predict_end]

            if prediction_data.empty: continue

            # Re-create features and predict on the new slice
            analyzer_pred = AnalyzeHMM(ticker="backtest_pred", bars_data=prediction_data, model_order=self.model_order, verbose=False)
            states = analyzer_pred.data['Hidden_State']

            negative_state, positive_state = analyzer.state_regimes[0], analyzer.state_regimes[-1]
            
            entries.update(states == positive_state)
            exits.update(states == negative_state)

        print("  -> Walk-forward backtest complete.")
        return entries, exits


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
            'predicted_state_mean_return': strength,
            'predicted_state_std_return': 0.0 # Add for consistent data structure
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