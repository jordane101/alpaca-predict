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

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.quantile_regression import QuantReg

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
    
    # Use the centralized path from utils
    from src.utils.paths import HMM_MODELS_DIR
    MODEL_DIR = HMM_MODELS_DIR

    def __init__(self,  ticker:str, timeframe=TimeFrame.Day, n_components=3, model_order=1, bars_data=None, verbose=True, force_retrain=False, max_age_days=30, sp500_data=None, use_causality_filter=False, causality_significance=0.05, use_causal_features=False, causal_dag_file=None, optimize_n_components=True, n_components_range=(2, 4)):
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
        self.sp500_data = sp500_data
        self.use_causality_filter = use_causality_filter
        self.causality_significance = causality_significance
        self.causality_results = None
        
        # Causal DAG feature integration
        self.use_causal_features = use_causal_features
        self.causal_dag_file = causal_dag_file
        self.causal_engine = None
        
        # Auto-optimization of number of states
        self.optimize_n_components = optimize_n_components
        self.n_components_range = n_components_range
        self.optimal_n_components = n_components
        self.model_selection_results = None

        # Sanitize ticker for use in filenames (e.g., replace 'ETH/USD' with 'ETH_USD')
        safe_ticker_filename = self.ticker.replace('/', '_')
        causal_suffix = "_causal" if self.use_causal_features else ""
        self.model_path = self.MODEL_DIR / f"{safe_ticker_filename}_{self.n_components}_{self.model_order}{causal_suffix}.pkl"

        if self.model_order < 1:
            raise ValueError("Model order must be 1 or greater.")
        
        # Initialize causal feature engine if enabled
        if self.use_causal_features:
            try:
                from src.causality.causal_feature_engine import CausalFeatureEngine
                self.causal_engine = CausalFeatureEngine(dag_file=causal_dag_file)
                if self.verbose:
                    logging.info(f"✓ Causal Feature Engine initialized with DAG: {self.causal_engine.dag_file}")
            except Exception as e:
                logging.warning(f"⚠️  Could not initialize Causal Feature Engine: {e}")
                logging.warning("Falling back to technical indicators only")
                self.use_causal_features = False
                self.causal_engine = None

        if bars_data is not None:
            self.bars = bars_data
        else:
            self.getStockByTicker(ticker)
        
        # Get S&P 500 data if not provided and ticker is not SPY itself
        if self.sp500_data is None and ticker != "SPY" and "backtest" not in ticker:
            self.sp500_data = self._get_sp500_data()

        # Get S&P 500 data if not provided and ticker is not SPY itself
        if self.sp500_data is None and ticker != "SPY" and "backtest" not in ticker:
            self.sp500_data = self._get_sp500_data()

        # Set base features based on timeframe and causal feature usage
        self.base_features = []
        if self.use_causal_features and self.causal_engine:
            # Use causal parents from DAG + minimal technical indicators
            # Include Vol_Adjusted_Return for better state detection across volatility regimes
            if type(timeframe) == type(TimeFrame.Day):
                self.base_features = ["Return", "Volatility", "Vol_Adjusted_Return"]  # Start with core features
            elif type(timeframe) == type(TimeFrame.Week):
                self.base_features = ["Return", "Volatility", "Vol_Adjusted_Return"]
            # Causal features will be added in createFeatures()
        else:
            # Traditional technical indicators
            # Include Vol_Adjusted_Return for better state detection across volatility regimes
            if type(timeframe) == type(TimeFrame.Day):
                self.base_features = ["Return", "Volatility", "Vol_Adjusted_Return", "SMA_50", "SP500_Return"]
            elif type(timeframe) == type(TimeFrame.Week):
                self.base_features = ["Return", "Volatility", "Vol_Adjusted_Return", "SMA_10", "SP500_Return"]

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
    
    def _get_sp500_data(self):
        """Fetch S&P 500 (SPY) data for the same time period as the stock."""
        try:
            start_date = (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
            request_params = StockBarsRequest(
                symbol_or_symbols=["SPY"],
                timeframe=self.timeframe,
                start=start_date,
                feed=DataFeed.IEX
            )
            spy_bars = self.client.get_stock_bars(request_params)
            if hasattr(spy_bars, 'df'):
                spy_df = spy_bars.df.copy()
            else:
                spy_df = spy_bars.copy()
            
            # Reset index to remove symbol level, keep only timestamp
            if isinstance(spy_df.index, pd.MultiIndex):
                spy_df = spy_df.reset_index(level='symbol', drop=True)
            
            # Calculate SPY returns
            spy_df['SP500_Return'] = spy_df['close'].pct_change()
            
            # Keep only the return column
            spy_df = spy_df[['SP500_Return']].copy()
            
            if self.verbose:
                logging.info(f"Loaded S&P 500 data: {len(spy_df)} rows")
            
            return spy_df
        except Exception as e:
            logging.warning(f"Could not fetch S&P 500 data: {e}. Will proceed without it.")
            return None

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
        
        # Reset index if MultiIndex (remove symbol level, keep timestamp)
        if isinstance(data.index, pd.MultiIndex) and 'symbol' in data.index.names:
            data = data.reset_index(level='symbol', drop=True)
            
        # Calculate returns
        data['Return'] = data['close'].pct_change()

        # Calculate volatility (30-day rolling standard deviation of returns)
        data['Volatility'] = data['Return'].rolling(window=30).std()
        
        # Calculate volatility-adjusted return (return normalized by rolling volatility)
        # This helps HMM identify states more consistently across different volatility regimes
        # Add small epsilon to avoid division by zero
        data['Vol_Adjusted_Return'] = data['Return'] / (data['Volatility'] + 1e-8)

        # Simple Moving Average
        if 'SMA_50' in self.base_features:
            data['SMA_50'] = data['close'].rolling(window=50).mean()
        if 'SMA_10' in self.base_features:
            data['SMA_10'] = data['close'].rolling(window=10).mean()
        
        # Merge S&P 500 returns if available
        if self.sp500_data is not None and 'SP500_Return' in self.base_features:
            # Align the S&P 500 data with stock data by date index
            data = data.join(self.sp500_data, how='left')
            # Forward fill any missing S&P 500 values (for days when market is closed differently)
            data['SP500_Return'] = data['SP500_Return'].ffill()
        elif 'SP500_Return' in self.base_features:
            # If S&P 500 data is not available, create a dummy column with zeros
            data['SP500_Return'] = 0.0
            if self.verbose:
                logging.warning("S&P 500 data not available. Using zeros for SP500_Return feature.")
        
        # Add causal features from DAG if enabled
        if self.use_causal_features and self.causal_engine:
            try:
                # Get causal parents from DAG
                parents = self.causal_engine.get_causal_parents(
                    ticker=self.ticker,
                    top_k=5,  # Use top 5 causal parents
                    max_p_value=0.01  # Only strong relationships
                )
                
                if parents and len(parents) > 0:
                    if self.verbose:
                        logging.info(f"\n📊 Adding {len(parents)} causal features for {self.ticker}:")
                        for parent, p_val, lag in parents:
                            logging.info(f"  • {parent} (p={p_val:.4f}, lag={lag})")
                    
                    # Fetch returns for causal parents
                    for parent_ticker, p_value, lag in parents:
                        try:
                            # Use same date range as main data
                            start_date = data.index.min()
                            end_date = data.index.max()
                            
                            parent_returns = self.causal_engine.fetch_returns(
                                ticker=parent_ticker,
                                start_date=start_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d'),
                                use_cache=True
                            )
                            
                            if parent_returns is not None and len(parent_returns) > 0:
                                # Create lagged feature name
                                feature_name = f"{parent_ticker}_Return_Lag{lag}"
                                
                                # Shift by the causal lag (parent_returns is already a Series)
                                parent_series = parent_returns.shift(lag)
                                
                                # Merge with main data
                                data[feature_name] = parent_series
                                
                                # Add to base_features for lagging
                                self.base_features.append(feature_name)
                                
                        except Exception as e:
                            if self.verbose:
                                logging.warning(f"  ⚠️  Could not fetch {parent_ticker}: {e}")
                else:
                    if self.verbose:
                        logging.info(f"No causal parents found for {self.ticker} in DAG")
                        
            except Exception as e:
                logging.warning(f"Error adding causal features: {e}")

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
    
    def quantile_granger_causality_test(self, target='Return', quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], maxlag=5, significance_level=0.05):
        """
        Test Granger causality from each feature to the target variable at different quantiles.
        
        This method tests whether past values of each feature provide statistically significant
        information about future values of the target variable at different quantiles of the
        conditional distribution. Features that show causality at ANY quantile are considered
        potentially useful for prediction.
        
        Args:
            target (str): The target variable to test causality towards (default: 'Return')
            quantiles (list): List of quantiles to test (default: [0.1, 0.25, 0.5, 0.75, 0.9])
            maxlag (int): Maximum number of lags to test (default: 5)
            significance_level (float): P-value threshold for significance (default: 0.05)
            
        Returns:
            dict: Dictionary with causality test results for each feature and quantile:
                {
                    'feature_name': {
                        'quantile': quantile_value,
                        'lag': best_lag,
                        'p_value': min_p_value,
                        'is_causal': True/False,
                        'quantile_results': {quantile: {'lag': lag, 'p_value': p_value}, ...}
                    }
                }
        """
        if self.verbose:
            logging.info(f"\n{'='*70}")
            logging.info(f"QUANTILE GRANGER CAUSALITY ANALYSIS")
            logging.info(f"{'='*70}")
            logging.info(f"Target: {target}")
            logging.info(f"Quantiles: {quantiles}")
            logging.info(f"Max lag: {maxlag}")
            logging.info(f"Significance level: {significance_level}")
        
        # Get clean data without NaNs
        test_data = self.data[self.base_features].dropna().copy()
        
        if len(test_data) < maxlag * 3:
            logging.warning(f"Insufficient data for causality testing. Need at least {maxlag * 3} samples, have {len(test_data)}")
            return {}
        
        causality_results = {}
        
        # Test each feature (except the target itself)
        features_to_test = [f for f in self.base_features if f != target]
        
        for feature in features_to_test:
            if self.verbose:
                logging.info(f"\n--- Testing {feature} -> {target} ---")
            
            feature_results = {
                'quantile_results': {},
                'is_causal': False,
                'min_p_value': 1.0,
                'best_quantile': None,
                'best_lag': None
            }
            
            # Test at each quantile
            for q in quantiles:
                quantile_best_p = 1.0
                quantile_best_lag = None
                
                # Test different lags
                for lag in range(1, maxlag + 1):
                    try:
                        # Create lagged data
                        Y = test_data[target].iloc[lag:].values
                        X_target = test_data[target].iloc[:-lag].values.reshape(-1, 1)
                        X_feature = test_data[feature].iloc[:-lag].values.reshape(-1, 1)
                        
                        # Quantile regression with only lagged target (restricted model)
                        model_restricted = QuantReg(Y, np.hstack([np.ones((len(Y), 1)), X_target]))
                        res_restricted = model_restricted.fit(q=q)
                        
                        # Quantile regression with lagged target and lagged feature (unrestricted model)
                        model_unrestricted = QuantReg(Y, np.hstack([np.ones((len(Y), 1)), X_target, X_feature]))
                        res_unrestricted = model_unrestricted.fit(q=q)
                        
                        # Likelihood ratio test approximation
                        # For quantile regression, we use the pseudo R-squared improvement
                        # as a measure of additional explanatory power
                        
                        # Get residuals
                        resid_restricted = Y - res_restricted.predict(np.hstack([np.ones((len(Y), 1)), X_target]))
                        resid_unrestricted = Y - res_unrestricted.predict(np.hstack([np.ones((len(Y), 1)), X_target, X_feature]))
                        
                        # Calculate sum of absolute residuals (quantile loss)
                        def quantile_loss(resid, q):
                            return np.sum(np.where(resid >= 0, q * resid, (q - 1) * resid))
                        
                        loss_restricted = quantile_loss(resid_restricted, q)
                        loss_unrestricted = quantile_loss(resid_unrestricted, q)
                        
                        # Improvement in fit
                        improvement = (loss_restricted - loss_unrestricted) / loss_restricted if loss_restricted > 0 else 0
                        
                        # Convert to approximate p-value using chi-square distribution
                        # This is an approximation; for more rigorous testing, use bootstrap
                        from scipy import stats
                        test_stat = len(Y) * improvement
                        p_value = 1 - stats.chi2.cdf(test_stat, df=1)
                        
                        # Track best result for this quantile
                        if p_value < quantile_best_p:
                            quantile_best_p = p_value
                            quantile_best_lag = lag
                            
                    except Exception as e:
                        if self.verbose:
                            logging.debug(f"  Quantile {q}, lag {lag}: Error - {str(e)}")
                        continue
                
                # Store results for this quantile
                feature_results['quantile_results'][q] = {
                    'lag': quantile_best_lag,
                    'p_value': quantile_best_p
                }
                
                if self.verbose:
                    sig_marker = "***" if quantile_best_p < significance_level else ""
                    logging.info(f"  Q={q:.2f}: p-value={quantile_best_p:.4f} (lag={quantile_best_lag}) {sig_marker}")
                
                # Update overall best result
                if quantile_best_p < feature_results['min_p_value']:
                    feature_results['min_p_value'] = quantile_best_p
                    feature_results['best_quantile'] = q
                    feature_results['best_lag'] = quantile_best_lag
            
            # Determine if feature is causal at any quantile
            feature_results['is_causal'] = feature_results['min_p_value'] < significance_level
            
            if self.verbose:
                if feature_results['is_causal']:
                    logging.info(f"  ✓ {feature} IS CAUSAL (best: q={feature_results['best_quantile']}, p={feature_results['min_p_value']:.4f})")
                else:
                    logging.info(f"  ✗ {feature} NOT CAUSAL (best p={feature_results['min_p_value']:.4f})")
            
            causality_results[feature] = feature_results
        
        if self.verbose:
            logging.info(f"\n{'='*70}")
            causal_features = [f for f, r in causality_results.items() if r['is_causal']]
            logging.info(f"SUMMARY: {len(causal_features)}/{len(features_to_test)} features show Granger causality")
            if causal_features:
                logging.info(f"Causal features: {', '.join(causal_features)}")
            logging.info(f"{'='*70}\n")
        
        return causality_results
    
    def _create_regime_labels(self):
        """
        Create meaningful regime labels based on number of states and their characteristics.
        States are already sorted from lowest to highest return in self.state_regimes.
        
        Returns:
            list: Labels for each state (in sorted order)
        """
        n_states = len(self.state_regimes)
        
        if n_states == 2:
            # Binary classification
            return ['Bear', 'Bull']
        elif n_states == 3:
            # Three-regime classification
            return ['Bear', 'Neutral', 'Bull']
        elif n_states == 4:
            # Four-regime classification with granular sentiment
            return ['Strong Bear', 'Mild Bear', 'Mild Bull', 'Strong Bull']
        else:
            # Fallback for other numbers of states
            return [f'State_{i}' for i in range(n_states)]
    
    def select_optimal_n_components(self, X, n_components_range=(2, 4)):
        """
        Select optimal number of HMM states using AIC and BIC.
        
        Args:
            X: Feature matrix
            n_components_range: Tuple of (min, max) components to test
            
        Returns:
            dict: Results with optimal_n (by AIC), models, and scores
        """
        if self.verbose:
            logging.info(f"\n{'='*70}")
            logging.info("MODEL SELECTION: Testing different numbers of states")
            logging.info(f"{'='*70}\n")
        
        results = {
            'n_components': [],
            'aic': [],
            'bic': [],
            'log_likelihood': [],
            'models': {}
        }
        
        min_n, max_n = n_components_range
        
        for n in range(min_n, max_n + 1):
            try:
                # Train model with n components
                model = hmm.GaussianHMM(n_components=n, covariance_type="full", random_state=42, n_iter=100)
                model.fit(X)
                
                # Calculate information criteria
                log_likelihood = model.score(X)
                n_params = (n * n) + (n * X.shape[1]) + (n * X.shape[1] * (X.shape[1] + 1) / 2)  # transition + means + covariances
                
                aic = -2 * log_likelihood + 2 * n_params
                bic = -2 * log_likelihood + n_params * np.log(len(X))
                
                results['n_components'].append(n)
                results['aic'].append(aic)
                results['bic'].append(bic)
                results['log_likelihood'].append(log_likelihood)
                results['models'][n] = model
                
                if self.verbose:
                    logging.info(f"  n_components={n}: AIC={aic:,.2f}, BIC={bic:,.2f}, LogLik={log_likelihood:,.2f}")
                    
            except Exception as e:
                logging.warning(f"  Failed to fit model with {n} components: {e}")
        
        if not results['n_components']:
            raise ValueError("Could not fit any models in the specified range")
        
        # Select best model (lowest BIC - more conservative than AIC)
        best_idx_bic = np.argmin(results['bic'])
        best_idx_aic = np.argmin(results['aic'])
        
        optimal_n_bic = results['n_components'][best_idx_bic]
        optimal_n_aic = results['n_components'][best_idx_aic]
        
        # Use BIC by default (more conservative, penalizes complexity more)
        optimal_n = optimal_n_bic
        
        if self.verbose:
            logging.info(f"\n  📊 Best by BIC: {optimal_n_bic} states (BIC={results['bic'][best_idx_bic]:,.2f})")
            logging.info(f"  📊 Best by AIC: {optimal_n_aic} states (AIC={results['aic'][best_idx_aic]:,.2f})")
            logging.info(f"  ✓ Selected: {optimal_n} states (using BIC)")
            logging.info(f"{'='*70}\n")
        
        return {
            'optimal_n': optimal_n,
            'optimal_n_bic': optimal_n_bic,
            'optimal_n_aic': optimal_n_aic,
            'results': results,
            'best_model': results['models'][optimal_n]
        }

    def train(self):
        """
        Trains the HMM, predicts states for the historical data,
        and analyzes the characteristics of each state.
        
        If use_causality_filter is enabled, performs quantile Granger causality
        testing and filters features to only include those with significant
        predictive power for returns.
        """
        # Perform causality testing if enabled
        if self.use_causality_filter:
            if self.verbose:
                logging.info(f"\n{'='*70}")
                logging.info("STEP 1: QUANTILE GRANGER CAUSALITY TESTING")
                logging.info(f"{'='*70}\n")
            
            # Test causality for base features
            self.causality_results = self.quantile_granger_causality_test(
                target='Return',
                significance_level=self.causality_significance
            )
            
            # Filter features based on causality
            causal_base_features = ['Return']  # Always include the target
            for feature, result in self.causality_results.items():
                if result['is_causal']:
                    causal_base_features.append(feature)
            
            if len(causal_base_features) == 1:
                logging.warning("No features passed causality test! Using all base features.")
                causal_base_features = list(self.base_features)
            
            # Update base_features to only causal ones
            original_base_features = self.base_features.copy()
            self.base_features = causal_base_features
            
            # Recreate features with filtered base features
            if self.verbose:
                logging.info(f"\nFiltered features: {self.base_features}")
                logging.info(f"Removed: {[f for f in original_base_features if f not in self.base_features]}")
            
            self.data = self.createFeatures()
            
            if self.verbose:
                logging.info(f"\n{'='*70}")
                logging.info("STEP 2: HMM TRAINING WITH FILTERED FEATURES")
                logging.info(f"{'='*70}\n")
        
        X = self.data[self.features].values.copy()

        # QuantileTransformer to map to a uniform distribution
        # Use more quantiles than states for better distribution mapping
        # Ensure at least 10 quantiles for reasonable distribution
        n_quantiles = max(10, min(len(X), 100))  # Use 10-100 quantiles
        self.quantizer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform', random_state=0)
        X_quantized = self.quantizer.fit_transform(X)

        # Model selection or direct training
        if self.optimize_n_components:
            # Select optimal number of components
            selection_results = self.select_optimal_n_components(X_quantized, self.n_components_range)
            self.optimal_n_components = selection_results['optimal_n']
            self.model_selection_results = selection_results['results']
            self.model = selection_results['best_model']
            
            # Update n_components to reflect the optimal selection
            self.n_components = self.optimal_n_components
            
            if self.verbose:
                logging.info(f"✓ Using optimal model with {self.n_components} states")
        else:
            # Train HMM with specified n_components
            self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="full", random_state=42)
            self.model.fit(X_quantized)  # Use the quantized features

        # Predict states for historical data
        hidden_states = self.model.predict(X_quantized)
        self.data['Hidden_State'] = hidden_states

        # Analyze state characteristics
        self.state_means = self.data.groupby('Hidden_State')[self.features].mean()
        self.state_stds = self.data.groupby('Hidden_State')[self.features].std()

        if self.verbose:
            logging.info("\nState Characteristics (Means):")
            logging.info("\n" + self.state_means.to_string())
            logging.info("\nState Characteristics (Standard Deviations):")
            logging.info("\n" + self.state_stds.to_string())

        # Identify state regimes based on returns. The index of this series
        # is the state number, sorted from lowest return to highest.
        sorted_returns = self.state_means['Return'].sort_values()
        self.state_regimes = sorted_returns.index.tolist()
        
        # Create regime labels based on number of states
        self.regime_labels = self._create_regime_labels()
        
        if self.verbose:
            logging.info("\nRegime Classification:")
            for state_idx, regime_label in zip(self.state_regimes, self.regime_labels):
                mean_return = self.state_means.loc[state_idx, 'Return']
                volatility = self.state_stds.loc[state_idx, 'Return']
                state_pct = (self.data['Hidden_State'] == state_idx).sum() / len(self.data) * 100
                logging.info(f"  State {state_idx} ({regime_label}): Return={mean_return:.4f}, Vol={volatility:.4f}, Freq={state_pct:.1f}%")

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
            'trained_at': trained_at_dt,
            'causality_results': self.causality_results,
            'use_causality_filter': self.use_causality_filter
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
            if isinstance(o, (np.bool_, bool)):
                return bool(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        # Create a clear mapping of state number to its regime type
        regime_map = {}
        if self.state_regimes and hasattr(self, 'regime_labels'):
            for state_idx, label in zip(self.state_regimes, self.regime_labels):
                regime_map[state_idx] = label.lower().replace(' ', '_')
        elif self.state_regimes:
            # Fallback for backward compatibility
            regime_map[self.state_regimes[0]] = 'negative'
            regime_map[self.state_regimes[-1]] = 'positive'

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
        
        # Add causality results if available
        if self.causality_results is not None:
            causality_summary = {}
            for feature, result in self.causality_results.items():
                causality_summary[feature] = {
                    'is_causal': result['is_causal'],
                    'min_p_value': result['min_p_value'],
                    'best_quantile': result['best_quantile'],
                    'best_lag': result['best_lag'],
                    'quantile_results': result['quantile_results']
                }
            summary_data['causality_analysis'] = {
                'enabled': self.use_causality_filter,
                'significance_level': self.causality_significance,
                'results': causality_summary
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
            # Load causality results if available (for backward compatibility)
            self.causality_results = model_data.get('causality_results', None)
            saved_use_causality = model_data.get('use_causality_filter', False)
            # If saved model used causality filtering but current request doesn't, warn user
            if saved_use_causality and not self.use_causality_filter:
                logging.warning(f"Model was trained with causality filtering but current settings don't use it.")
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
    
    def get_state_probabilities(self):
        """
        Calculate the probability distribution over states for the most recent observation
        using the forward algorithm.
        
        Returns:
            dict: State probabilities and confidence metrics
                  - 'probabilities': Array of probabilities for each state
                  - 'most_likely_state': Index of most likely state
                  - 'confidence': Probability of most likely state
                  - 'state_probs_dict': Dictionary mapping state indices to probabilities
        """
        if not self.model or self.data.empty:
            return {
                'probabilities': None,
                'most_likely_state': -1,
                'confidence': 0.0,
                'state_probs_dict': {}
            }
        
        # Get the most recent features
        X = self.data[self.features].values[-1:].copy()
        
        # Transform using the same quantizer used in training
        X_quantized = self.quantizer.transform(X)
        
        # Use the forward algorithm to get state probabilities
        # The score_samples method returns log probabilities
        log_prob, posteriors = self.model.score_samples(X_quantized)
        
        # posteriors shape: (n_samples, n_states)
        # We want the last (most recent) observation
        state_probs = posteriors[-1]
        
        # Get most likely state and its probability
        most_likely_state = np.argmax(state_probs)
        confidence = state_probs[most_likely_state]
        
        # Create dictionary mapping state index to probability
        state_probs_dict = {i: prob for i, prob in enumerate(state_probs)}
        
        if self.verbose:
            logging.info(f"\nState Probabilities (Forward Algorithm):")
            for state_idx in self.state_regimes:
                prob = state_probs_dict[state_idx]
                label = self.regime_labels[self.state_regimes.index(state_idx)] if hasattr(self, 'regime_labels') else f"State {state_idx}"
                logging.info(f"  {label} (State {state_idx}): {prob:.4f} ({prob*100:.1f}%)")
        
        return {
            'probabilities': state_probs,
            'most_likely_state': most_likely_state,
            'confidence': confidence,
            'state_probs_dict': state_probs_dict
        }
    
    def calculate_position_size(self, min_confidence: float = 0.5, 
                                max_position: float = 1.0,
                                allow_shorts: bool = True,
                                short_confidence_threshold: float = 0.7):
        """
        Calculate position size based on state confidence and regime.
        
        Args:
            min_confidence: Minimum confidence to take any position (default 0.5)
            max_position: Maximum position size as fraction of portfolio (default 1.0 = 100%)
            allow_shorts: Whether to allow negative (short) positions (default True)
            short_confidence_threshold: Minimum confidence required for short positions (default 0.7)
            
        Returns:
            dict: Position sizing information
                  - 'position_size': Suggested position size (-1.0 to 1.0)
                  - 'confidence': Confidence level for the position
                  - 'regime': Regime classification (bearish/neutral/bullish)
                  - 'action': Recommended action (buy/sell/short/hold)
                  - 'reasoning': Explanation of the decision
        """
        # Get state probabilities
        prob_info = self.get_state_probabilities()
        
        if prob_info['probabilities'] is None:
            return {
                'position_size': 0.0,
                'confidence': 0.0,
                'regime': 'unknown',
                'action': 'hold',
                'reasoning': 'No model available'
            }
        
        # Calculate expected return weighted by state probabilities
        expected_return = 0.0
        for state_idx, prob in prob_info['state_probs_dict'].items():
            state_return = self.state_means.loc[state_idx, 'Return']
            expected_return += prob * state_return
        
        # Determine regime based on expected return and state probabilities
        # Weight probabilities by position in regime hierarchy
        n_states = len(self.state_regimes)
        
        # Calculate bullish vs bearish sentiment score
        # States are sorted from most bearish to most bullish
        sentiment_score = 0.0
        for i, state_idx in enumerate(self.state_regimes):
            prob = prob_info['state_probs_dict'][state_idx]
            # Score from -1 (most bearish) to +1 (most bullish)
            position_score = (2 * i / (n_states - 1)) - 1 if n_states > 1 else 0
            sentiment_score += prob * position_score
        
        # Classify regime
        if sentiment_score < -0.3:
            regime = 'bearish'
        elif sentiment_score > 0.3:
            regime = 'bullish'
        else:
            regime = 'neutral'
        
        # Get confidence for the dominant regime
        most_likely_state = prob_info['most_likely_state']
        confidence = prob_info['confidence']
        
        # Determine position size and action
        if confidence < min_confidence:
            # Not confident enough for any position
            return {
                'position_size': 0.0,
                'confidence': confidence,
                'regime': regime,
                'action': 'hold',
                'reasoning': f'Confidence {confidence:.2%} below minimum {min_confidence:.2%}'
            }
        
        # Calculate base position size scaled by confidence
        # Map confidence from [min_confidence, 1.0] to [0, max_position]
        confidence_scaled = (confidence - min_confidence) / (1.0 - min_confidence)
        base_position = confidence_scaled * max_position
        
        # Determine direction and action
        if regime == 'bullish':
            position_size = base_position
            action = 'buy'
            reasoning = f'Bullish regime (sentiment={sentiment_score:.2f}) with {confidence:.2%} confidence'
            
        elif regime == 'bearish':
            if allow_shorts and confidence >= short_confidence_threshold:
                # Short position (negative size)
                position_size = -base_position
                action = 'short'
                reasoning = f'Bearish regime (sentiment={sentiment_score:.2f}) with {confidence:.2%} confidence (>= {short_confidence_threshold:.2%} threshold)'
            else:
                # Not confident enough for shorts, or shorts not allowed
                position_size = 0.0
                action = 'hold'
                if allow_shorts:
                    reasoning = f'Bearish but confidence {confidence:.2%} below short threshold {short_confidence_threshold:.2%}'
                else:
                    reasoning = f'Bearish but shorts not allowed'
        
        else:  # neutral
            # Scale down position for neutral regimes
            position_size = base_position * 0.5 if expected_return > 0 else 0.0
            action = 'hold' if position_size == 0 else 'buy'
            reasoning = f'Neutral regime (sentiment={sentiment_score:.2f}), reduced position'
        
        if self.verbose:
            logging.info(f"\nPosition Sizing:")
            logging.info(f"  Regime: {regime} (sentiment score: {sentiment_score:.3f})")
            logging.info(f"  Confidence: {confidence:.2%}")
            logging.info(f"  Expected Return: {expected_return:.4f}")
            logging.info(f"  Position Size: {position_size:.2%} of portfolio")
            logging.info(f"  Action: {action}")
            logging.info(f"  Reasoning: {reasoning}")
        
        return {
            'position_size': position_size,
            'confidence': confidence,
            'regime': regime,
            'action': action,
            'reasoning': reasoning,
            'sentiment_score': sentiment_score,
            'expected_return': expected_return
        }

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
        # For multi-state models, use the regime labels for classification
        n_states = len(self.state_regimes)
        state_position = self.state_regimes.index(predicted_next_state)
        
        if n_states == 2:
            # Binary: bear/bull
            outlook = "negative" if state_position == 0 else "positive"
        elif n_states == 3:
            # Three states: bear/neutral/bull
            if state_position == 0:
                outlook = "negative"
            elif state_position == 1:
                outlook = "neutral"
            else:
                outlook = "positive"
        elif n_states == 4:
            # Four states: strong_bear/mild_bear/mild_bull/strong_bull
            if state_position <= 1:  # strong_bear or mild_bear
                outlook = "negative"
            else:  # mild_bull or strong_bull
                outlook = "positive"
        else:
            # Fallback: bottom half = negative, top half = positive
            middle = n_states // 2
            outlook = "negative" if state_position < middle else "positive"

        # Compare returns for a more direct message
        if predicted_state_mean_return > last_return:
            comparison = "higher"
        elif predicted_state_mean_return < last_return:
            comparison = "lower"
        else:
            comparison = "the same"

        # Get state probabilities for confidence
        prob_info = self.get_state_probabilities()
        confidence = prob_info['confidence']
        
        # Get position sizing recommendation
        position_info = self.calculate_position_size()
        
        return {
            'outlook': outlook,
            'last_return': last_return,
            'predicted_state_mean_return': predicted_state_mean_return,
            'predicted_state_std_return': predicted_state_std_return,
            'comparison': comparison,
            'predicted_state': predicted_next_state,
            'confidence': confidence,
            'state_probabilities': prob_info['state_probs_dict'],
            'position_size': position_info['position_size'],
            'position_action': position_info['action'],
            'regime': position_info['regime']
        }

if __name__ == "__main__":
    # When run as a script, set up file and console logging.
    setup_logging()

    TICKER_TO_ANALYZE = "AAPL"  # Define the ticker once
    N_COMPONENTS = 2  # Changed to 2 states
    MAX_ORDER_TO_TEST = 5
    USE_CAUSALITY = True  # Enable quantile Granger causality filtering

    start_time = datetime.now()

    logging.info(f"\n{'='*70}")
    logging.info(f"QUANTILE GRANGER CAUSALITY HMM ANALYSIS")
    logging.info(f"Ticker: {TICKER_TO_ANALYZE}")
    logging.info(f"N Components: {N_COMPONENTS}")
    logging.info(f"Causality Filtering: {'ENABLED' if USE_CAUSALITY else 'DISABLED'}")
    logging.info(f"{'='*70}\n")

    # Create an analyzer with causality filtering enabled
    # This will automatically test causality and filter features before training
    ah = AnalyzeHMM(
        TICKER_TO_ANALYZE, 
        n_components=N_COMPONENTS, 
        model_order=1,
        use_causality_filter=USE_CAUSALITY,
        causality_significance=0.05,
        force_retrain=True  # Force retrain to see causality analysis
    )
    
    end_time = datetime.now()
    stopwatch = end_time - start_time
    last_state = ah.data['Hidden_State'].iloc[-1]
    last_return = ah.data['Return'].iloc[-1]

    logging.info(f"\n{'='*70}")
    logging.info(f"RESULTS")
    logging.info(f"{'='*70}")
    logging.info(f"Time to run: {stopwatch}")
    logging.info(f"Features used: {ah.features}")
    logging.info(f"\nToday's Hidden State: {last_state}")
    logging.info(f"Today's Return: {last_return:.4f}")

    prediction = ah.predict_next_day_outlook()

    logging.info(f"\nPredicted Next State: {prediction['predicted_state']} (Regime Outlook: {prediction['outlook'].upper()})")
    logging.info(f"Tomorrow's return is predicted to be {prediction['comparison']} than today's.")
    
    # Print causality summary if available
    if ah.causality_results:
        logging.info(f"\n{'='*70}")
        logging.info(f"CAUSALITY SUMMARY")
        logging.info(f"{'='*70}")
        for feature, result in ah.causality_results.items():
            status = "✓ CAUSAL" if result['is_causal'] else "✗ NOT CAUSAL"
            logging.info(f"{feature:20s}: {status:15s} (p={result['min_p_value']:.4f}, q={result['best_quantile']})")
    
    logging.info(f"\n{'='*70}\n")
    logging.info(f" -> Today's Actual Return: {prediction['last_return']:.4f}")
    logging.info(f" -> Predicted State's Avg. Return: {prediction['predicted_state_mean_return']:.4f}")
    logging.info(f" -> Predicted State's Return Std. Dev.: {prediction['predicted_state_std_return']:.4f}")