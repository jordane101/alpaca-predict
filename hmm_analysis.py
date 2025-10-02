"""
Hidden Markov Matrix analysis using the Alpaca API
Author - Eli Jordan
Date - 07/29/2025

"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import logging
from hmmlearn import hmm
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

from datetime import datetime, timedelta
import os

def setup_logging():
    """Sets up logging to file and console for standalone script execution."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Generate a filename based on the current date and time (AM/PM)
    now = datetime.now()
    # Format: hmm_training_YYYY-MM-DD_AM.log or hmm_training_YYYY-MM-DD_PM.log
    log_filename = "hmm_training_" + now.strftime("%Y-%m-%d_%p") + ".log"
    log_file = os.path.join(log_dir, log_filename)

    # Get the root logger.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates if run multiple times.
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler for logging to a file.
    file_handler = logging.FileHandler(log_file, mode='a') # Append mode
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Console handler for printing to the console.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s')) # Cleaner console output
    logger.addHandler(console_handler)

    logging.info(f"--- New Training Run at {now.strftime('%Y-%m-%d %H:%M:%S')} ---")
    logging.info(f"Logging for this run will be appended to: {log_file}")


class AnalyzeHMM:
    load_dotenv(".env")
    KEY = os.getenv("PAPER_KEY")
    SECRET = os.getenv("PAPER_SEC")
    MODEL_DIR = Path("hmm_models")

    def __init__(self,  ticker:str, timeframe=TimeFrame.Day, n_components=3, model_order=1, bars_data=None, verbose=True, force_retrain=False, max_age_days=30):
        self.client = StockHistoricalDataClient(self.KEY,self.SECRET)
        self.timeframe = timeframe
        self.ticker = ticker
        self.n_components = n_components
        self.model_order = model_order
        self.verbose = verbose
        self.model = None
        self.quantizer = None
        self.state_means = None
        self.state_stds = None
        self.state_regimes = None

        # Sanitize ticker for use in filenames (e.g., replace 'ETH/USD' with 'ETH_USD')
        safe_ticker_filename = self.ticker.replace('/', '_')
        self.model_path = self.MODEL_DIR / f"{safe_ticker_filename}_{self.n_components}_{self.model_order}.pkl"

        if self.model_order < 1:
            raise ValueError("Model order must be 1 or greater.")

        if bars_data is not None:
            self.bars = bars_data
        else:
            self.getStockByTicker(ticker)

        # Set base features based on timeframe
        self.base_features = []
        if type(timeframe) == type(TimeFrame.Day):
            self.base_features = ["Return", "Volatility", "SMA_50"]
        elif type(timeframe) == type(TimeFrame.Week):
            self.base_features = ["Return", "Volatility", "SMA_10"]

        self.features = [] # Will be populated by createFeatures()
        self.data = self.createFeatures()

        # --- Model Loading/Training Logic ---
        # For backtesting, ticker might be 'backtest'. We don't want to cache these.
        use_cache = not force_retrain and "backtest" not in self.ticker

        if use_cache and self.load_model(max_age_days):
            # Model loaded successfully. Predict states for the current data.
            self._predict_states_for_data()
        else:
            if self.verbose:
                if force_retrain:
                    logging.info(f"Forcing retrain for {ticker}.")
                elif not use_cache:
                    logging.info(f"Cache disabled for ticker '{ticker}'. Retraining.")

            # This will train and then save the model.
            self.train()

    def getStockByTicker(self, ticker: str):
        # Calculate a dynamic start date (e.g., 2 years ago) to ensure we get data.
        start_date = (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=self.timeframe,
            start=start_date,
            feed=DataFeed.IEX  # Explicitly specify the free IEX data feed
        )

        self.bars = self.client.get_stock_bars(request_params)

    def getData(self):
        return self.data

    def createFeatures(self, data_df=None):
        # The data source can be a BarSet object (from live trading) which has a .df attribute,
        # or a raw DataFrame (from the backtester). This handles both cases.
        if data_df is not None:
            data = data_df.copy()
        elif hasattr(self.bars, 'df'): # It's a BarSet object from Alpaca client
            data = self.bars.df.copy()
        else: # It's already a DataFrame, passed from the backtester
            data = self.bars.copy()
        # Calculate returns
        data['Return'] = data['close'].pct_change()

        # Calculate volatility (30-day rolling standard deviation of returns)
        data['Volatility'] = data['Return'].rolling(window=30).std()

        # Simple Moving Average
        if 'SMA_50' in self.base_features:
            data['SMA_50'] = data['close'].rolling(window=50).mean()
        if 'SMA_10' in self.base_features:
            data['SMA_10'] = data['close'].rolling(window=10).mean()

        # Create lagged features for higher-order model
        self.features = list(self.base_features) # Start with current features
        if self.model_order > 1:
            for i in range(1, self.model_order):
                for feature in self.base_features:
                    lagged_feature_name = f"{feature}_lag_{i}"
                    data[lagged_feature_name] = data[feature].shift(i)
                    self.features.append(lagged_feature_name)

        # Drop missing values resulting from the calculations
        # This also removes the initial rows that have no lagged data
        data = data.dropna()
        return data
    
    def train(self):
        """
        Trains the HMM, predicts states for the historical data,
        and analyzes the characteristics of each state.
        """
        X = self.data[self.features].values.copy()

        # QuantileTransformer to map to a uniform distribution
        self.quantizer = QuantileTransformer(n_quantiles=self.n_components, output_distribution='uniform', random_state=0)
        X_quantized = self.quantizer.fit_transform(X)

        # Train HMM
        self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="full", random_state=42)
        self.model.fit(X_quantized)  # Use the quantized features

        # Predict states for historical data
        hidden_states = self.model.predict(X_quantized)
        self.data['Hidden_State'] = hidden_states

        # Analyze state characteristics
        self.state_means = self.data.groupby('Hidden_State')[self.features].mean()
        self.state_stds = self.data.groupby('Hidden_State')[self.features].std()

        if self.verbose:
            logging.info("State Characteristics (Means):")
            logging.info("\n" + self.state_means.to_string())
            logging.info("\nState Characteristics (Standard Deviations):")
            logging.info("\n" + self.state_stds.to_string())

        # Identify state regimes based on returns. The index of this series
        # is the state number, sorted from lowest return to highest.
        sorted_returns = self.state_means['Return'].sort_values()
        self.state_regimes = sorted_returns.index.tolist()

        # Save the newly trained model if it's not a temporary backtest model
        if "backtest" not in self.ticker:
            self.save_model()

    def save_model(self):
        """Saves the trained model and its components to a file."""
        if not self.model:
            if self.verbose:
                logging.warning(f"No model to save for {self.ticker}.")
            return

        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        trained_at_dt = datetime.now()
        model_data = {
            'model': self.model,
            'quantizer': self.quantizer,
            'state_means': self.state_means,
            'state_stds': self.state_stds,
            'state_regimes': self.state_regimes,
            'features': self.features,
            'trained_at': trained_at_dt
        }
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            if self.verbose:
                logging.info(f"Saved model to {self.model_path}")

            # Also save the human-readable summary
            self.save_model_summary(trained_at_dt)

        except Exception as e:
            logging.error(f"Error saving model for {self.ticker} to {self.model_path}: {e}")

    def save_model_summary(self, trained_at_dt):
        """Saves a human-readable JSON summary of the trained model."""
        if not self.model:
            return

        summary_path = self.model_path.with_suffix('.json')

        # Helper to convert non-serializable types like numpy arrays and dataframes
        def json_converter(o):
            if isinstance(o, datetime):
                return o.isoformat()
            if isinstance(o, (np.ndarray, pd.Series)):
                return o.tolist()
            if isinstance(o, pd.DataFrame):
                # Use to_dict('index') for better readability of state means/stds
                return o.to_dict(orient='index')
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            if isinstance(o, (np.float64, np.float32)):
                return float(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        # Create a clear mapping of state number to its regime type
        regime_map = {}
        if self.state_regimes:
            regime_map[self.state_regimes[0]] = 'negative'
            regime_map[self.state_regimes[-1]] = 'positive'
            for state in self.state_regimes:
                if state not in regime_map:
                    regime_map[state] = 'neutral'

        # Sort by state number for consistent output and ensure keys are strings for JSON
        sorted_regime_map = {str(k): regime_map[k] for k in sorted(regime_map)}

        summary_data = {
            'ticker': self.ticker,
            'n_components': self.n_components,
            'model_order': self.model_order,
            'trained_at': trained_at_dt,
            'features_used': self.features,
            'state_regime_mapping': sorted_regime_map,
            'state_means': self.state_means,
            'state_stds': self.state_stds,
            'transition_matrix': self.model.transmat_,
            'start_probabilities': self.model.startprob_
        }

        try:
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, default=json_converter, indent=4)
            if self.verbose:
                logging.info(f"Saved human-readable summary to {summary_path}")
        except Exception as e:
            logging.error(f"Error saving model summary for {self.ticker} to {summary_path}: {e}")

    def load_model(self, max_age_days: int):
        """Loads a pre-trained model if it exists and is not too old."""
        if not self.model_path.exists():
            return False

        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            trained_at = model_data.get('trained_at', datetime.min)
            if (datetime.now().date() - trained_at.date()).days > max_age_days:
                if self.verbose:
                    logging.info(f"Model for {self.ticker} is older than {max_age_days} calendar days. Will retrain.")
                return False

            self.model = model_data['model']
            self.quantizer = model_data['quantizer']
            self.state_means = model_data['state_means']
            self.state_stds = model_data['state_stds']
            self.state_regimes = model_data['state_regimes']
            self.features = model_data['features']
            logging.info(f"Loaded cached model for {self.ticker} from {self.model_path}")
            return True
        except Exception as e:
            logging.error(f"Could not load model from {self.model_path}. Deleting corrupt file. Error: {e}")
            try:
                os.remove(self.model_path)
            except OSError as oe:
                logging.error(f"Could not delete corrupt model file {self.model_path}: {oe}")
            return False

    def _predict_states_for_data(self):
        """Predicts hidden states for the current self.data using the loaded model."""
        X = self.data[self.features].values.copy()
        X_quantized = self.quantizer.transform(X) # Use transform, not fit_transform
        hidden_states = self.model.predict(X_quantized)
        self.data['Hidden_State'] = hidden_states

    def find_optimal_order(self, max_order=10):
        """
        Tests different model orders to find the one that best fits unseen data.

        Args:
            max_order (int): The maximum model order to test.

        Returns:
            int: The model order with the highest log-likelihood score on the test set.
        """
        if self.verbose:
            logging.info(f"\n--- Finding Optimal Model Order (1 to {max_order}) ---")
        # We can't split before creating features, as features rely on rolling windows.
        # So we create features on the full dataset first.
        full_data_with_features = self.createFeatures()

        # Split data into training and testing sets (80/20 split)
        # We don't shuffle time series data.
        train_data, test_data = train_test_split(full_data_with_features, test_size=0.2, shuffle=False)

        scores = []
        for order in range(1, max_order + 1):
            try:
                # Create the specific lagged features for this order
                features = list(self.base_features)
                if order > 1:
                    for i in range(1, order):
                        for feature in self.base_features:
                            features.append(f"{feature}_lag_{i}")

                # Ensure all features exist in the dataframes
                train_X = train_data[features].values
                test_X = test_data[features].values

                # Train a new HMM for this order on the training data
                model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="full", random_state=42)
                model.fit(train_X)

                # Score the model on the unseen test data
                score = model.score(test_X)
                scores.append(score)
                if self.verbose:
                    logging.info(f"  Order {order}: Score = {score:.2f}")
            except Exception as e:
                logging.error(f"  Order {order}: Failed. Reason: {e}")
                scores.append(float('-inf')) # Use negative infinity for failed models

        best_order = np.argmax(scores) + 1
        if self.verbose:
            logging.info(f"--- Optimal model order found: {best_order} ---")
        return best_order

    def predict_next_day_outlook(self):
        """
        Predicts the most likely state for the next day and classifies its outlook.

        Returns:
            dict: A dictionary containing the prediction details:
                  - 'outlook': "positive", "negative", or "neutral"
                  - 'last_return': The actual return of the last day.
                  - 'predicted_state_mean_return': The historical average return of the predicted state.
                  - 'predicted_state_std_return': The historical standard deviation of returns of the predicted state.
                  - 'comparison': "higher", "lower", or "the same".
                  - 'predicted_state': The predicted hidden state for the next day.
        """
        if 'Hidden_State' not in self.data.columns or self.data.empty or pd.isna(self.data['Hidden_State'].iloc[-1]):
            return {
                'outlook': 'neutral',
                'last_return': 0,
                'predicted_state_mean_return': 0,
                'predicted_state_std_return': 0,
                'comparison': 'the same',
                'predicted_state': -1
            }

        # Get the most recent hidden state
        last_state = int(self.data['Hidden_State'].iloc[-1])
        last_return = self.data['Return'].iloc[-1]

        # Use the transition matrix to find the most likely next state
        transition_matrix = self.model.transmat_
        predicted_next_state = np.argmax(transition_matrix[last_state])
        predicted_state_mean_return = self.state_means.loc[predicted_next_state, 'Return']
        predicted_state_std_return = self.state_stds.loc[predicted_next_state, 'Return']

        # Classify the predicted state's outlook based on its historical return
        # self.state_regimes is sorted by return, from lowest to highest
        negative_state = self.state_regimes[0]
        positive_state = self.state_regimes[-1]

        if predicted_next_state == negative_state:
            outlook = "negative"
        elif predicted_next_state == positive_state:
            outlook = "positive"
        else:
            outlook = "neutral"

        # Compare returns for a more direct message
        if predicted_state_mean_return > last_return:
            comparison = "higher"
        elif predicted_state_mean_return < last_return:
            comparison = "lower"
        else:
            comparison = "the same"

        return {
            'outlook': outlook,
            'last_return': last_return,
            'predicted_state_mean_return': predicted_state_mean_return,
            'predicted_state_std_return': predicted_state_std_return,
            'comparison': comparison,
            'predicted_state': predicted_next_state
        }

if __name__ == "__main__":
    # When run as a script, set up file and console logging.
    setup_logging()

    TICKER_TO_ANALYZE = "RBLX"  # Define the ticker once
    N_COMPONENTS = 3
    MAX_ORDER_TO_TEST = 10

    start_time = datetime.now()

    # Create an analyzer instance to find the optimal order for our target stock.
    # A base model_order=1 is sufficient here as find_optimal_order handles feature creation internally.
    temp_analyzer = AnalyzeHMM(TICKER_TO_ANALYZE, n_components=N_COMPONENTS, model_order=5)
    optimal_order = temp_analyzer.find_optimal_order(max_order=MAX_ORDER_TO_TEST)

    # Now, create the final analyzer with the determined optimal order
    logging.info(f"\n--- Analyzing {TICKER_TO_ANALYZE} with optimal order: {optimal_order} ---")
    ah = AnalyzeHMM(TICKER_TO_ANALYZE, n_components=N_COMPONENTS, model_order=optimal_order)
    end_time = datetime.now()
    stopwatch = end_time - start_time
    last_state = ah.data['Hidden_State'].iloc[-1]
    last_return = ah.data['Return'].iloc[-1]

    logging.info(f"Time to run: {stopwatch}")
    logging.info(f"\nToday's Hidden State: {last_state}")
    logging.info(f"Today's Return: {last_return:.4f}")

    prediction = ah.predict_next_day_outlook()

    logging.info(f"\nPredicted Next State: {prediction['predicted_state']} (Regime Outlook: {prediction['outlook'].upper()})")
    logging.info(f"Tomorrow's return is predicted to be {prediction['comparison']} than today's.")
    logging.info(f" -> Today's Actual Return: {prediction['last_return']:.4f}")
    logging.info(f" -> Predicted State's Avg. Return: {prediction['predicted_state_mean_return']:.4f}")
    logging.info(f" -> Predicted State's Return Std. Dev.: {prediction['predicted_state_std_return']:.4f}")