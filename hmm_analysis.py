"""
Hidden Markov Matrix analysis using the Alpaca API
Author - Eli Jordan
Date - 07/29/2025

"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

from datetime import datetime


class AnalyzeHMM:
    load_dotenv(".env")
    KEY = os.getenv("PAPER_KEY")
    SECRET = os.getenv("PAPER_SEC")
    def __init__(self,  ticker:str, timeframe=TimeFrame.Day, n_components=3, model_order=1, bars_data=None):
        self.client = StockHistoricalDataClient(self.KEY,self.SECRET)
        self.timeframe = timeframe
        self.ticker = ticker
        self.n_components = n_components
        self.model_order = model_order
        self.model = None
        self.quantizer = None
        self.state_means = None
        self.state_regimes = None

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
        self.train() # Train the model upon initialization

    def getStockByTicker(self, ticker: str):
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=self.timeframe,
            start="2025-01-01"
        )

        self.bars = self.client.get_stock_bars(request_params)

    def getData(self):
        return self.data

    def createFeatures(self, data_df=None):
        data = self.bars.df.copy() if data_df is None else data_df.copy()
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
        print("State Characteristics (Means):")
        print(self.state_means)

        # Identify state regimes based on returns. The index of this series
        # is the state number, sorted from lowest return to highest.
        sorted_returns = self.state_means['Return'].sort_values()
        self.state_regimes = sorted_returns.index.tolist()

    def find_optimal_order(self, max_order=10):
        """
        Tests different model orders to find the one that best fits unseen data.

        Args:
            max_order (int): The maximum model order to test.

        Returns:
            int: The model order with the highest log-likelihood score on the test set.
        """
        print(f"\n--- Finding Optimal Model Order (1 to {max_order}) ---")
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
                print(f"  Order {order}: Score = {score:.2f}")
            except Exception as e:
                print(f"  Order {order}: Failed. Reason: {e}")
                scores.append(float('-inf')) # Use negative infinity for failed models

        best_order = np.argmax(scores) + 1
        print(f"--- Optimal model order found: {best_order} ---")
        return best_order

    def predict_next_day_outlook(self):
        """
        Predicts the most likely state for the next day and classifies its outlook.

        Returns:
            dict: A dictionary containing the prediction details:
                  - 'outlook': "positive", "negative", or "similar"
                  - 'last_return': The actual return of the last day.
                  - 'predicted_state_mean_return': The historical average return of the predicted state.
                  - 'comparison': "higher", "lower", or "the same".
                  - 'predicted_state': The predicted hidden state for the next day.
        """
        # Get the most recent hidden state
        last_state = self.data['Hidden_State'].iloc[-1]
        last_return = self.data['Return'].iloc[-1]

        # Use the transition matrix to find the most likely next state
        transition_matrix = self.model.transmat_
        predicted_next_state = np.argmax(transition_matrix[last_state])
        predicted_state_mean_return = self.state_means.loc[predicted_next_state, 'Return']

        # Classify the predicted state's outlook based on its historical return
        # self.state_regimes is sorted by return, from lowest to highest
        negative_state = self.state_regimes[0]
        positive_state = self.state_regimes[-1]

        if predicted_next_state == negative_state:
            outlook = "negative"
        elif predicted_next_state == positive_state:
            outlook = "positive"
        else:
            outlook = "similar"

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
            'comparison': comparison,
            'predicted_state': predicted_next_state
        }

if __name__ == "__main__":
    TICKER_TO_ANALYZE = "TTD"  # Define the ticker once
    N_COMPONENTS = 3
    MAX_ORDER_TO_TEST = 10

    start_time = datetime.now()

    # Create a temporary analyzer instance to find the optimal order for our target stock.
    # A base model_order=1 is sufficient here as find_optimal_order handles feature creation internally.
    temp_analyzer = AnalyzeHMM(TICKER_TO_ANALYZE, n_components=N_COMPONENTS, model_order=5)
    optimal_order = temp_analyzer.find_optimal_order(max_order=MAX_ORDER_TO_TEST)

    # Now, create the final analyzer with the determined optimal order
    print(f"\n--- Analyzing {TICKER_TO_ANALYZE} with optimal order: {optimal_order} ---")
    ah = AnalyzeHMM(TICKER_TO_ANALYZE, n_components=N_COMPONENTS, model_order=optimal_order)
    end_time = datetime.now()
    stopwatch = end_time - start_time
    last_state = ah.data['Hidden_State'].iloc[-1]
    last_return = ah.data['Return'].iloc[-1]

    print(f"Time to run: {stopwatch}")
    print(f"\nToday's Hidden State: {last_state}")
    print(f"Today's Return: {last_return:.4f}")

    prediction = ah.predict_next_day_outlook()

    print(f"\nPredicted Next State: {prediction['predicted_state']} (Regime Outlook: {prediction['outlook'].upper()})")
    print(f"Tomorrow's return is predicted to be {prediction['comparison']} than today's.")
    print(f" -> Today's Actual Return: {prediction['last_return']:.4f}")
    print(f" -> Predicted State's Avg. Return: {prediction['predicted_state_mean_return']:.4f}")