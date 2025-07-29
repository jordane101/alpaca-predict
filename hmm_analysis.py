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

class AnalyzeHMM:
    load_dotenv(".env")
    KEY = os.getenv("PAPER_KEY")
    SECRET = os.getenv("PAPER_SEC")
    def __init__(self,  ticker:str, timeframe=TimeFrame.Day, n_components=3):
        self.client = StockHistoricalDataClient(self.KEY,self.SECRET)
        self.timeframe = timeframe
        self.n_components = n_components
        self.model = None
        self.quantizer = None
        self.state_means = None
        self.state_regimes = None

        self.getStockByTicker(ticker)

        # set features based on timeframe
        if type(timeframe) == type(TimeFrame.Day):
            self.features = ["Return", "Volatility", "SMA_50"]
        elif type(timeframe) == type(TimeFrame.Week):
            self.features = ["Return", "Volatility", "SMA_10"]

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

    def createFeatures(self):
        data = self.bars.df
        # Calculate returns
        data['Return'] = data['close'].pct_change()

        # Calculate volatility (30-day rolling standard deviation of returns)
        data['Volatility'] = data['Return'].rolling(window=30).std()

        # Simple Moving Average
        if type(self.timeframe) == type(TimeFrame.Day):
            data['SMA_50'] = data['close'].rolling(window=50).mean()
        elif type(self.timeframe) == type(TimeFrame.Week):
            data['SMA_10'] = data['close'].rolling(window=10).mean()

        # Drop missing values resulting from the calculations
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
    ah = AnalyzeHMM("NVDA",n_components=3)
    last_state = ah.data['Hidden_State'].iloc[-1]
    last_return = ah.data['Return'].iloc[-1]
    print(f"\nToday's Hidden State: {last_state}")
    print(f"Today's Return: {last_return:.4f}")

    prediction = ah.predict_next_day_outlook()

    print(f"\nPredicted Next State: {prediction['predicted_state']} (Regime Outlook: {prediction['outlook'].upper()})")
    print(f"Tomorrow's return is predicted to be {prediction['comparison']} than today's.")
    print(f" -> Today's Actual Return: {prediction['last_return']:.4f}")
    print(f" -> Predicted State's Avg. Return: {prediction['predicted_state_mean_return']:.4f}")