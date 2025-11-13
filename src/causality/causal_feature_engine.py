"""
Causal Feature Engine - Extract features from the Market Causality DAG.

This module provides functionality to use causal parent returns as features
for HMM training instead of traditional technical indicators.
"""

import numpy as np
import pandas as pd
import pickle
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


class CausalFeatureEngine:
    """
    Extract causal features from the Market Causality DAG for HMM training.
    """
    
    load_dotenv(".env")
    KEY = os.getenv("PAPER_KEY")
    SECRET = os.getenv("PAPER_SEC")
    
    def __init__(self, dag_file: str = None):
        """
        Initialize the Causal Feature Engine.
        
        Args:
            dag_file: Path to the pickled DAG file (defaults to data/causality_cache/large_network_graph_dag.pkl)
        """
        if dag_file is None:
            from src.utils.paths import DEFAULT_DAG_FILE
            dag_file = str(DEFAULT_DAG_FILE)
        
        self.dag_file = dag_file
        self.graph = None
        self.universe = None
        self.causality_matrix = None
        self.p_value_matrix = None
        self.lag_matrix = None
        self.is_dag = None
        
        # Alpaca client
        self.client = StockHistoricalDataClient(self.KEY, self.SECRET)
        
        # Cache
        self.returns_cache = {}
        
        self._load_dag()
        
        logging.info(f"CausalFeatureEngine initialized")
        logging.info(f"  DAG: {len(self.universe)} stocks, {self.graph.number_of_edges()} edges")
        logging.info(f"  Is DAG: {self.is_dag}")
    
    def _load_dag(self):
        """Load the DAG from file."""
        with open(self.dag_file, 'rb') as f:
            data = pickle.load(f)
        
        self.graph = data['graph']
        self.universe = data['universe']
        self.causality_matrix = data['causality_matrix']
        self.p_value_matrix = data['p_value_matrix']
        self.lag_matrix = data['lag_matrix']
        self.is_dag = data.get('is_dag', False)
    
    def get_causal_parents(self, ticker: str, top_k: int = None, 
                          max_p_value: float = None) -> List[Tuple[str, float, int]]:
        """
        Get the causal parents of a ticker.
        
        Args:
            ticker: Target ticker
            top_k: Return only top k strongest parents (by p-value)
            max_p_value: Filter parents with p-value above this threshold
            
        Returns:
            List of (parent_ticker, p_value, lag) tuples
        """
        if ticker not in self.graph:
            logging.warning(f"{ticker} not in DAG")
            return []
        
        parents = []
        for parent in self.graph.predecessors(ticker):
            edge_data = self.graph[parent][ticker]
            p_value = edge_data['p_value']
            lag = edge_data['lag']
            
            if max_p_value is None or p_value <= max_p_value:
                parents.append((parent, p_value, lag))
        
        # Sort by p-value (strongest first)
        parents.sort(key=lambda x: x[1])
        
        if top_k is not None:
            parents = parents[:top_k]
        
        return parents
    
    def fetch_returns(self, ticker: str, start_date: str, end_date: str, 
                     use_cache: bool = True) -> pd.Series:
        """
        Fetch returns for a ticker.
        
        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Use cached returns if available
            
        Returns:
            Series of returns indexed by date
        """
        cache_key = f"{ticker}_{start_date}_{end_date}"
        
        if use_cache and cache_key in self.returns_cache:
            return self.returns_cache[cache_key]
        
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
                feed=DataFeed.IEX
            )
            bars = self.client.get_stock_bars(request_params)
            
            if bars.df.empty:
                logging.warning(f"No data for {ticker}")
                return pd.Series(dtype=float)
            
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level='symbol', drop=True)
            
            returns = df['close'].pct_change().dropna()
            
            if use_cache:
                self.returns_cache[cache_key] = returns
            
            return returns
            
        except Exception as e:
            logging.error(f"Error fetching {ticker}: {e}")
            return pd.Series(dtype=float)
    
    def create_causal_features(self, ticker: str, start_date: str, end_date: str,
                               top_k: int = 10, max_p_value: float = 0.01,
                               include_self: bool = True) -> pd.DataFrame:
        """
        Create causal features for a ticker based on its causal parents.
        
        Args:
            ticker: Target ticker
            start_date: Start date for data
            end_date: End date for data
            top_k: Use top k strongest parents (default: 10)
            max_p_value: Maximum p-value for parent inclusion (default: 0.01)
            include_self: Include the ticker's own return (default: True)
            
        Returns:
            DataFrame with causal features
        """
        logging.info(f"\nCreating causal features for {ticker}")
        
        # Get causal parents
        parents = self.get_causal_parents(ticker, top_k=top_k, max_p_value=max_p_value)
        
        if not parents:
            logging.warning(f"  No causal parents found for {ticker}")
            # Fall back to just own returns
            if include_self:
                returns = self.fetch_returns(ticker, start_date, end_date)
                return pd.DataFrame({'Return': returns})
            else:
                return pd.DataFrame()
        
        logging.info(f"  Found {len(parents)} causal parents:")
        for parent, p_val, lag in parents[:5]:  # Show top 5
            logging.info(f"    {parent}: p={p_val:.4f}, lag={lag}")
        
        # Fetch returns for target and all parents
        all_returns = {}
        
        if include_self:
            target_returns = self.fetch_returns(ticker, start_date, end_date)
            all_returns['Return'] = target_returns
        
        for parent, p_val, lag in parents:
            parent_returns = self.fetch_returns(parent, start_date, end_date)
            if not parent_returns.empty:
                # Shift by optimal lag
                all_returns[f'{parent}_Return_Lag{lag}'] = parent_returns.shift(lag)
        
        # Combine into DataFrame
        df = pd.DataFrame(all_returns)
        df = df.dropna()
        
        logging.info(f"  Created {len(df.columns)} features with {len(df)} observations")
        
        return df
    
    def create_hybrid_features(self, ticker: str, start_date: str, end_date: str,
                               top_k: int = 5, max_p_value: float = 0.01,
                               include_technical: bool = True) -> pd.DataFrame:
        """
        Create hybrid features combining causal parents and technical indicators.
        
        Args:
            ticker: Target ticker
            start_date: Start date for data
            end_date: End date for data
            top_k: Use top k strongest parents (default: 5)
            max_p_value: Maximum p-value for parent inclusion (default: 0.01)
            include_technical: Include technical indicators (default: True)
            
        Returns:
            DataFrame with hybrid features
        """
        logging.info(f"\nCreating hybrid features for {ticker}")
        
        # Get base price data
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
                feed=DataFeed.IEX
            )
            bars = self.client.get_stock_bars(request_params)
            
            if bars.df.empty:
                logging.error(f"No data for {ticker}")
                return pd.DataFrame()
            
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level='symbol', drop=True)
            
        except Exception as e:
            logging.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()
        
        # Calculate base features
        features = pd.DataFrame(index=df.index)
        features['Return'] = df['close'].pct_change()
        
        if include_technical:
            # Volatility (20-day rolling std)
            features['Volatility'] = features['Return'].rolling(window=20).std()
            
            # SMA crossover (50-day)
            features['SMA_50'] = df['close'].rolling(window=50).mean()
            features['SMA_50'] = (df['close'] - features['SMA_50']) / features['SMA_50']
            
            # RSI (14-day)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['RSI'] = 100 - (100 / (1 + rs))
            features['RSI'] = (features['RSI'] - 50) / 50  # Normalize around 0
        
        # Add causal parent returns
        parents = self.get_causal_parents(ticker, top_k=top_k, max_p_value=max_p_value)
        
        logging.info(f"  Adding {len(parents)} causal parents")
        
        for parent, p_val, lag in parents:
            parent_returns = self.fetch_returns(parent, start_date, end_date)
            if not parent_returns.empty:
                features[f'{parent}_Return'] = parent_returns.shift(lag)
        
        # Drop NaN
        features = features.dropna()
        
        logging.info(f"  Created {len(features.columns)} features with {len(features)} observations")
        
        return features
    
    def get_feature_importance(self, ticker: str) -> pd.DataFrame:
        """
        Get feature importance based on causal relationships.
        
        Args:
            ticker: Target ticker
            
        Returns:
            DataFrame with parent stocks ranked by strength
        """
        parents = self.get_causal_parents(ticker)
        
        if not parents:
            return pd.DataFrame()
        
        importance = []
        for parent, p_val, lag in parents:
            importance.append({
                'Parent': parent,
                'P_Value': p_val,
                'Lag': lag,
                'Strength': 1 - p_val  # Higher strength = lower p-value
            })
        
        df = pd.DataFrame(importance)
        df = df.sort_values('Strength', ascending=False)
        
        return df
    
    def visualize_causal_tree(self, ticker: str, max_depth: int = 2, 
                             top_k: int = 5) -> None:
        """
        Print a tree visualization of causal relationships.
        
        Args:
            ticker: Root ticker
            max_depth: Maximum depth to traverse
            top_k: Show top k parents at each level
        """
        print(f"\nCausal Tree for {ticker}:")
        print("="*60)
        
        def print_tree(node, depth=0, max_depth=max_depth):
            if depth > max_depth:
                return
            
            indent = "  " * depth
            if depth == 0:
                print(f"{indent}📊 {node}")
            else:
                print(f"{indent}└─ {node}")
            
            parents = self.get_causal_parents(node, top_k=top_k)
            for parent, p_val, lag in parents:
                print(f"{indent}   └─ {parent} (p={p_val:.4f}, lag={lag})")
                if depth < max_depth:
                    print_tree(parent, depth + 1, max_depth)
        
        print_tree(ticker)
        print("="*60)
    
    def save_features(self, features: pd.DataFrame, ticker: str, 
                     output_dir: str = "features") -> str:
        """
        Save features to CSV.
        
        Args:
            features: Feature DataFrame
            ticker: Ticker symbol
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename = output_path / f"{ticker}_causal_features.csv"
        features.to_csv(filename)
        
        logging.info(f"Saved features to {filename}")
        
        return str(filename)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\n" + "="*80)
    print("CAUSAL FEATURE ENGINE - DEMO")
    print("="*80)
    
    # Initialize engine
    engine = CausalFeatureEngine()
    
    # Test with a stock
    test_ticker = "AAPL"
    
    print(f"\n\n{'='*80}")
    print(f"ANALYZING {test_ticker}")
    print("="*80)
    
    # Show causal parents
    print(f"\nCausal Parents of {test_ticker}:")
    parents = engine.get_causal_parents(test_ticker, top_k=10)
    for i, (parent, p_val, lag) in enumerate(parents, 1):
        print(f"{i:2d}. {parent:6s}: p={p_val:.4f}, lag={lag}")
    
    # Create causal features
    print(f"\n\n{'='*80}")
    print(f"CREATING CAUSAL FEATURES FOR {test_ticker}")
    print("="*80)
    
    features = engine.create_causal_features(
        ticker=test_ticker,
        start_date="2024-01-01",
        end_date="2024-12-31",
        top_k=10,
        max_p_value=0.01
    )
    
    print(f"\nFeature Summary:")
    print(features.describe())
    
    # Create hybrid features
    print(f"\n\n{'='*80}")
    print(f"CREATING HYBRID FEATURES FOR {test_ticker}")
    print("="*80)
    
    hybrid = engine.create_hybrid_features(
        ticker=test_ticker,
        start_date="2024-01-01",
        end_date="2024-12-31",
        top_k=5,
        max_p_value=0.01,
        include_technical=True
    )
    
    print(f"\nHybrid Feature Summary:")
    print(hybrid.describe())
    
    # Visualize causal tree
    print(f"\n\n{'='*80}")
    print("CAUSAL TREE VISUALIZATION")
    print("="*80)
    engine.visualize_causal_tree(test_ticker, max_depth=2, top_k=3)
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80 + "\n")
