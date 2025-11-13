"""
Market Causality DAG - Build a directed acyclic graph of causal relationships
between stocks/sectors to use for feature engineering in HMM models.

Author: Eli Jordan
Date: November 12, 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import warnings

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats

# Suppress iteration warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')


class MarketCausalityDAG:
    """
    Builds and manages a Directed Acyclic Graph of Granger causality relationships
    between stocks in a universe.
    """
    
    load_dotenv(".env")
    KEY = os.getenv("PAPER_KEY")
    SECRET = os.getenv("PAPER_SEC")
    
    def __init__(self, universe: List[str], start_date: str = None, end_date: str = None, 
                 quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
                 max_lag: int = 5, significance: float = 0.05):
        """
        Initialize the Market Causality DAG.
        
        Args:
            universe: List of stock tickers to analyze
            start_date: Start date for historical data (default: 2 years ago)
            end_date: End date for historical data (default: today)
            quantiles: Quantiles to test for causality
            max_lag: Maximum lag to test for Granger causality
            significance: P-value threshold for significance
        """
        self.universe = sorted(universe)
        self.quantiles = quantiles
        self.max_lag = max_lag
        self.significance = significance
        
        # Date range
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        
        self.start_date = start_date
        self.end_date = end_date
        
        # Data storage
        self.returns_data = None  # DataFrame with returns for all stocks
        self.causality_matrix = None  # NxN matrix of causality relationships
        self.p_value_matrix = None  # NxN matrix of minimum p-values
        self.lag_matrix = None  # NxN matrix of optimal lags
        self.graph = None  # NetworkX DiGraph
        
        # Alpaca client
        self.client = StockHistoricalDataClient(self.KEY, self.SECRET)
        
        # Cache directory
        self.cache_dir = Path("causality_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        logging.info(f"Initialized MarketCausalityDAG with {len(universe)} stocks")
        logging.info(f"Period: {start_date} to {end_date}")
    
    def fetch_returns_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch historical returns data for all stocks in the universe.
        
        Args:
            force_refresh: Force re-fetch even if cached data exists
            
        Returns:
            DataFrame with returns for all stocks (index=date, columns=tickers)
        """
        universe_hash = hash(tuple(self.universe))
        cache_file = self.cache_dir / f"returns_{self.start_date}_{self.end_date}_{len(self.universe)}_{abs(universe_hash)}.pkl"
        
        if not force_refresh and cache_file.exists():
            logging.info(f"Loading cached returns data from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.returns_data = pickle.load(f)
            return self.returns_data
        
        logging.info(f"Fetching returns data for {len(self.universe)} stocks...")
        
        returns_dict = {}
        
        for ticker in self.universe:
            try:
                logging.info(f"  Fetching {ticker}...")
                request_params = StockBarsRequest(
                    symbol_or_symbols=[ticker],
                    timeframe=TimeFrame.Day,
                    start=self.start_date,
                    end=self.end_date,
                    feed=DataFeed.IEX
                )
                bars = self.client.get_stock_bars(request_params)
                
                if bars.df.empty:
                    logging.warning(f"  No data for {ticker}")
                    continue
                
                # Extract close prices and calculate returns
                df = bars.df
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index(level='symbol', drop=True)
                
                returns = df['close'].pct_change().dropna()
                returns_dict[ticker] = returns
                
            except Exception as e:
                logging.error(f"  Error fetching {ticker}: {e}")
                continue
        
        # Combine into single DataFrame with aligned dates
        self.returns_data = pd.DataFrame(returns_dict)
        self.returns_data = self.returns_data.dropna()  # Only keep dates with all stocks
        
        logging.info(f"Fetched {len(self.returns_data)} days of returns for {len(self.returns_data.columns)} stocks")
        
        # Cache the data
        with open(cache_file, 'wb') as f:
            pickle.dump(self.returns_data, f)
        
        return self.returns_data
    
    def test_pairwise_causality(self, cause_ticker: str, effect_ticker: str) -> Dict:
        """
        Test if cause_ticker Granger-causes effect_ticker using quantile regression.
        
        Args:
            cause_ticker: The potential cause
            effect_ticker: The potential effect
            
        Returns:
            Dict with causality test results
        """
        if self.returns_data is None:
            raise ValueError("Must fetch returns data first")
        
        if cause_ticker not in self.returns_data.columns or effect_ticker not in self.returns_data.columns:
            return {'is_causal': False, 'min_p_value': 1.0, 'best_lag': None, 'best_quantile': None}
        
        cause_returns = self.returns_data[cause_ticker].values
        effect_returns = self.returns_data[effect_ticker].values
        
        result = {
            'is_causal': False,
            'min_p_value': 1.0,
            'best_lag': None,
            'best_quantile': None,
            'quantile_results': {}
        }
        
        # Test at each quantile
        for q in self.quantiles:
            quantile_best_p = 1.0
            quantile_best_lag = None
            
            # Test different lags
            for lag in range(1, self.max_lag + 1):
                try:
                    # Create lagged data
                    Y = effect_returns[lag:]
                    X_effect = effect_returns[:-lag].reshape(-1, 1)  # Lagged effect (restricted)
                    X_cause = cause_returns[:-lag].reshape(-1, 1)  # Lagged cause (unrestricted)
                    
                    # Quantile regression - restricted model (only lagged effect)
                    model_restricted = QuantReg(Y, np.hstack([np.ones((len(Y), 1)), X_effect]))
                    res_restricted = model_restricted.fit(q=q, max_iter=5000)
                    
                    # Quantile regression - unrestricted model (lagged effect + lagged cause)
                    model_unrestricted = QuantReg(Y, np.hstack([np.ones((len(Y), 1)), X_effect, X_cause]))
                    res_unrestricted = model_unrestricted.fit(q=q, max_iter=5000)
                    
                    # Calculate quantile loss
                    def quantile_loss(resid, q):
                        return np.sum(np.where(resid >= 0, q * resid, (q - 1) * resid))
                    
                    resid_restricted = Y - res_restricted.predict(np.hstack([np.ones((len(Y), 1)), X_effect]))
                    resid_unrestricted = Y - res_unrestricted.predict(np.hstack([np.ones((len(Y), 1)), X_effect, X_cause]))
                    
                    loss_restricted = quantile_loss(resid_restricted, q)
                    loss_unrestricted = quantile_loss(resid_unrestricted, q)
                    
                    # Improvement in fit
                    improvement = (loss_restricted - loss_unrestricted) / loss_restricted if loss_restricted > 0 else 0
                    
                    # Convert to approximate p-value
                    test_stat = len(Y) * improvement
                    p_value = 1 - stats.chi2.cdf(test_stat, df=1)
                    
                    if p_value < quantile_best_p:
                        quantile_best_p = p_value
                        quantile_best_lag = lag
                        
                except Exception as e:
                    continue
            
            # Store quantile results
            result['quantile_results'][q] = {
                'p_value': quantile_best_p,
                'lag': quantile_best_lag
            }
            
            # Update overall best
            if quantile_best_p < result['min_p_value']:
                result['min_p_value'] = quantile_best_p
                result['best_quantile'] = q
                result['best_lag'] = quantile_best_lag
        
        # Determine if causal
        result['is_causal'] = result['min_p_value'] < self.significance
        
        return result
    
    @staticmethod
    def _test_causality_worker(args):
        """
        Worker function for parallel causality testing.
        This is a static method to avoid pickling issues with multiprocessing.
        """
        i, j, cause, effect, returns_data, quantiles, max_lag, significance = args
        
        if i == j:  # Skip self-causality
            return (i, j, {'is_causal': False, 'min_p_value': 1.0, 'best_lag': None, 'best_quantile': None})
        
        # Extract returns
        cause_returns = returns_data[cause].values
        effect_returns = returns_data[effect].values
        
        result = {
            'is_causal': False,
            'min_p_value': 1.0,
            'best_lag': None,
            'best_quantile': None,
            'quantile_results': {}
        }
        
        # Test at each quantile
        for q in quantiles:
            quantile_best_p = 1.0
            quantile_best_lag = None
            
            # Test different lags
            for lag in range(1, max_lag + 1):
                try:
                    # Create lagged data
                    Y = effect_returns[lag:]
                    X_effect = effect_returns[:-lag].reshape(-1, 1)
                    X_cause = cause_returns[:-lag].reshape(-1, 1)
                    
                    # Quantile regression - restricted model
                    model_restricted = QuantReg(Y, np.hstack([np.ones((len(Y), 1)), X_effect]))
                    res_restricted = model_restricted.fit(q=q, max_iter=5000)
                    
                    # Quantile regression - unrestricted model
                    model_unrestricted = QuantReg(Y, np.hstack([np.ones((len(Y), 1)), X_effect, X_cause]))
                    res_unrestricted = model_unrestricted.fit(q=q, max_iter=5000)
                    
                    # Calculate quantile loss
                    def quantile_loss(resid, q):
                        return np.sum(np.where(resid >= 0, q * resid, (q - 1) * resid))
                    
                    resid_restricted = Y - res_restricted.predict(np.hstack([np.ones((len(Y), 1)), X_effect]))
                    resid_unrestricted = Y - res_unrestricted.predict(np.hstack([np.ones((len(Y), 1)), X_effect, X_cause]))
                    
                    loss_restricted = quantile_loss(resid_restricted, q)
                    loss_unrestricted = quantile_loss(resid_unrestricted, q)
                    
                    # Improvement in fit
                    improvement = (loss_restricted - loss_unrestricted) / loss_restricted if loss_restricted > 0 else 0
                    
                    # Convert to approximate p-value
                    test_stat = len(Y) * improvement
                    p_value = 1 - stats.chi2.cdf(test_stat, df=1)
                    
                    if p_value < quantile_best_p:
                        quantile_best_p = p_value
                        quantile_best_lag = lag
                        
                except Exception as e:
                    continue
            
            # Store quantile results
            result['quantile_results'][q] = {
                'p_value': quantile_best_p,
                'lag': quantile_best_lag
            }
            
            # Update overall best
            if quantile_best_p < result['min_p_value']:
                result['min_p_value'] = quantile_best_p
                result['best_quantile'] = q
                result['best_lag'] = quantile_best_lag
        
        # Determine if causal
        result['is_causal'] = result['min_p_value'] < significance
        
        return (i, j, result)
    
    def build_causality_matrix(self, force_recompute: bool = False, n_jobs: int = None) -> np.ndarray:
        """
        Build the full causality matrix by testing all pairs using parallel processing.
        
        Args:
            force_recompute: Force recomputation even if cached
            n_jobs: Number of parallel jobs (default: CPU count - 1)
            
        Returns:
            NxN boolean matrix where matrix[i,j]=True means i causes j
        """
        universe_hash = hash(tuple(self.universe))
        cache_file = self.cache_dir / f"causality_matrix_{self.start_date}_{self.end_date}_{self.significance}_{len(self.universe)}_{abs(universe_hash)}.pkl"
        
        if not force_recompute and cache_file.exists():
            logging.info(f"Loading cached causality matrix from {cache_file}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.causality_matrix = data['causality']
                self.p_value_matrix = data['p_values']
                self.lag_matrix = data['lags']
            return self.causality_matrix
        
        if self.returns_data is None:
            self.fetch_returns_data()
        
        n = len(self.universe)
        self.causality_matrix = np.zeros((n, n), dtype=bool)
        self.p_value_matrix = np.ones((n, n))
        self.lag_matrix = np.zeros((n, n), dtype=int)
        
        # Determine number of workers
        if n_jobs is None:
            n_jobs = max(1, cpu_count() - 1)
        
        logging.info(f"\nBuilding causality matrix ({n}x{n} = {n*n} tests)...")
        logging.info(f"Using {n_jobs} parallel workers")
        logging.info("This may take a while...\n")
        
        total_tests = n * (n - 1)  # Don't test self-causality
        
        # Prepare all test arguments
        test_args = []
        for i, cause in enumerate(self.universe):
            for j, effect in enumerate(self.universe):
                if i != j:  # Skip self-causality
                    test_args.append((
                        i, j, cause, effect, 
                        self.returns_data, 
                        self.quantiles, 
                        self.max_lag, 
                        self.significance
                    ))
        
        # Run tests in parallel
        completed = 0
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                future_to_args = {
                    executor.submit(self._test_causality_worker, args): args 
                    for args in test_args
                }
                
                # Process results as they complete
                for future in as_completed(future_to_args):
                    try:
                        i, j, result = future.result()
                        
                        self.causality_matrix[i, j] = result['is_causal']
                        self.p_value_matrix[i, j] = result['min_p_value']
                        if result['best_lag'] is not None:
                            self.lag_matrix[i, j] = result['best_lag']
                        
                        completed += 1
                        if completed % 100 == 0 or completed == total_tests:
                            logging.info(f"  Progress: {completed}/{total_tests} tests ({100*completed/total_tests:.1f}%)")
                    except Exception as e:
                        logging.warning(f"  Error processing result: {e}")
                        completed += 1
        except Exception as e:
            logging.error(f"Error in parallel processing: {e}")
            raise
        
        logging.info(f"\nCausality matrix complete!")
        logging.info(f"Total causal relationships found: {np.sum(self.causality_matrix)}")
        
        # Cache the results
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'causality': self.causality_matrix,
                'p_values': self.p_value_matrix,
                'lags': self.lag_matrix,
                'universe': self.universe,
                'date_range': (self.start_date, self.end_date),
                'significance': self.significance
            }, f)
        
        return self.causality_matrix
    
    def build_graph(self, break_cycles: bool = False, max_cycle_detection: int = 10) -> nx.DiGraph:
        """
        Build NetworkX directed graph from causality matrix.
        
        Args:
            break_cycles: If True, remove weakest edges to make graph acyclic
            max_cycle_detection: Maximum number of cycles to detect (for large graphs)
        
        Returns:
            NetworkX DiGraph object
        """
        if self.causality_matrix is None:
            self.build_causality_matrix()
        
        self.graph = nx.DiGraph()
        
        # Add nodes
        for ticker in self.universe:
            self.graph.add_node(ticker)
        
        # Add edges
        n = len(self.universe)
        for i in range(n):
            for j in range(n):
                if self.causality_matrix[i, j]:
                    cause = self.universe[i]
                    effect = self.universe[j]
                    self.graph.add_edge(
                        cause, 
                        effect,
                        p_value=self.p_value_matrix[i, j],
                        lag=int(self.lag_matrix[i, j]),
                        weight=1.0 - self.p_value_matrix[i, j]  # Higher weight = stronger causality
                    )
        
        logging.info(f"\nGraph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        # Check for cycles (skip expensive detection for large graphs)
        if self.graph.number_of_edges() > 1000:
            logging.info("Skipping full cycle detection (large graph)")
            # Just do a quick check
            try:
                nx.find_cycle(self.graph)
                logging.warning("✗ Graph has cycles (quick check detected at least one)")
                has_cycles = True
            except nx.NetworkXNoCycle:
                logging.info("✓ Graph is acyclic (DAG)")
                has_cycles = False
        else:
            is_dag = nx.is_directed_acyclic_graph(self.graph)
            has_cycles = not is_dag
            if is_dag:
                logging.info("✓ Graph is acyclic (DAG)")
            else:
                # For smaller graphs, try to count cycles
                try:
                    cycles = list(nx.simple_cycles(self.graph))
                    if len(cycles) > max_cycle_detection:
                        logging.warning(f"✗ Graph has many cycles (detected {max_cycle_detection}+)")
                        logging.warning(f"  Example cycle: {cycles[0] if cycles else 'None'}")
                    else:
                        logging.warning(f"✗ Graph has {len(cycles)} cycles")
                        logging.warning(f"  Example cycle: {cycles[0] if cycles else 'None'}")
                except:
                    logging.warning(f"✗ Graph has cycles (too many to count efficiently)")
        
        if has_cycles and break_cycles:
            logging.info("\nBreaking cycles to create DAG...")
            self._break_cycles()
        
        return self.graph
    
    def _break_cycles_fast(self):
        """
        Fast cycle breaking using greedy heuristic based on node ordering.
        This is much faster than iteratively finding cycles.
        """
        initial_edges = self.graph.number_of_edges()
        logging.info(f"  Initial graph: {initial_edges} edges")
        
        # Greedy heuristic: order nodes by (out-degree - in-degree)
        # Remove backward edges relative to this ordering
        nodes = list(self.graph.nodes())
        
        # Calculate scores for ordering
        scores = {}
        for node in nodes:
            out_deg = self.graph.out_degree(node)
            in_deg = self.graph.in_degree(node)
            scores[node] = out_deg - in_deg
        
        # Sort nodes by score (descending)
        ordered_nodes = sorted(nodes, key=lambda n: scores[n], reverse=True)
        node_order = {node: i for i, node in enumerate(ordered_nodes)}
        
        logging.info(f"  Computed node ordering")
        
        # Remove edges that go backward in the ordering
        edges_to_remove = []
        for source, target, data in self.graph.edges(data=True):
            if node_order[source] > node_order[target]:
                # This is a backward edge
                edges_to_remove.append((source, target, data['p_value']))
        
        logging.info(f"  Found {len(edges_to_remove)} backward edges to remove")
        
        # Remove edges
        for source, target, p_value in edges_to_remove:
            self.graph.remove_edge(source, target)
        
        edges_removed = len(edges_to_remove)
        
        # Check if we succeeded
        if nx.is_directed_acyclic_graph(self.graph):
            logging.info(f"\n✓ Graph is now acyclic!")
        else:
            logging.warning(f"\n⚠ Graph still has cycles, applying iterative removal...")
            # Fall back to iterative removal for remaining cycles
            edges_removed += self._break_cycles_iterative(max_iterations=500)
        
        logging.info(f"  Removed {edges_removed} edges ({100*edges_removed/initial_edges:.1f}% of total)")
        logging.info(f"  Remaining edges: {self.graph.number_of_edges()}/{initial_edges}")
        
        # Update causality matrix to reflect removed edges
        for i, cause in enumerate(self.universe):
            for j, effect in enumerate(self.universe):
                if self.causality_matrix[i, j] and not self.graph.has_edge(cause, effect):
                    self.causality_matrix[i, j] = False
    
    def _break_cycles_iterative(self, max_iterations: int = 500):
        """
        Iterative cycle breaking (slower, used as fallback).
        Returns number of edges removed.
        """
        edges_removed = 0
        iteration = 0
        
        while not nx.is_directed_acyclic_graph(self.graph) and iteration < max_iterations:
            iteration += 1
            
            try:
                cycle = nx.find_cycle(self.graph, orientation='original')
                
                # Find the weakest edge in this cycle
                weakest_edge = None
                weakest_p_value = -1
                
                for edge in cycle:
                    source, target = edge[0], edge[1]
                    p_value = self.graph[source][target]['p_value']
                    
                    if p_value > weakest_p_value:
                        weakest_p_value = p_value
                        weakest_edge = (source, target)
                
                if weakest_edge:
                    if edges_removed % 50 == 0 and edges_removed > 0:
                        logging.info(f"    Iterative removal: {edges_removed} edges...")
                    self.graph.remove_edge(*weakest_edge)
                    edges_removed += 1
                else:
                    break
                    
            except nx.NetworkXNoCycle:
                break
            except Exception as e:
                logging.warning(f"  Error during iterative cycle breaking: {e}")
                break
        
        if nx.is_directed_acyclic_graph(self.graph):
            logging.info(f"  ✓ Iterative removal succeeded!")
        else:
            logging.warning(f"  ⚠ Could not fully remove cycles after {iteration} iterations")
        
        return edges_removed
    
    def _break_cycles(self):
        """
        Break cycles using fast greedy heuristic.
        """
        self._break_cycles_fast()
    
    def get_causal_parents(self, ticker: str) -> List[str]:
        """
        Get all stocks that Granger-cause the given ticker.
        
        Args:
            ticker: Target stock
            
        Returns:
            List of parent tickers
        """
        if self.graph is None:
            self.build_graph()
        
        return list(self.graph.predecessors(ticker))
    
    def get_causal_children(self, ticker: str) -> List[str]:
        """
        Get all stocks that are Granger-caused by the given ticker.
        
        Args:
            ticker: Source stock
            
        Returns:
            List of child tickers
        """
        if self.graph is None:
            self.build_graph()
        
        return list(self.graph.successors(ticker))
    
    def visualize_graph(self, output_file: str = "outputs/causality_graph.html", 
                       highlight_ticker: str = None, show_edge_weights: bool = True):
        """
        Create an interactive visualization of the causality graph.
        
        Args:
            output_file: Path to save HTML visualization
            highlight_ticker: Ticker to highlight (shows its parents/children)
            show_edge_weights: If True, show p-values as edge labels
        """
        if self.graph is None:
            self.build_graph()
        
        try:
            import plotly.graph_objects as go
            
            # Get positions using spring layout with weight consideration
            pos = nx.spring_layout(self.graph, k=2, iterations=50, weight='weight')
            
            # Create edge traces with weights
            edge_traces = []
            edge_annotations = []
            
            for edge in self.graph.edges(data=True):
                source, target, data = edge
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                
                p_value = data['p_value']
                lag = data['lag']
                weight = data['weight']
                
                # Edge color and width based on strength (lower p-value = darker/thicker)
                # Map p-value to opacity and width
                opacity = 0.3 + 0.7 * (1 - p_value / self.significance)  # 0.3 to 1.0
                width = 0.5 + 3.5 * (1 - p_value / self.significance)  # 0.5 to 4.0
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=width, color=f'rgba(100, 100, 100, {opacity})'),
                    hovertext=f"{source} → {target}<br>p-value: {p_value:.4f}<br>lag: {lag}",
                    hoverinfo='text',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
                
                # Add edge label if requested
                if show_edge_weights:
                    # Position label at midpoint
                    mid_x = (x0 + x1) / 2
                    mid_y = (y0 + y1) / 2
                    
                    edge_annotations.append(
                        dict(
                            x=mid_x,
                            y=mid_y,
                            text=f"{p_value:.3f}",
                            showarrow=False,
                            font=dict(size=8, color='gray'),
                            opacity=0.7
                        )
                    )
            
            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            for node in self.graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Node info
                parents = self.get_causal_parents(node)
                children = self.get_causal_children(node)
                
                text = f"{node}<br>"
                text += f"Parents: {len(parents)}<br>"
                text += f"Children: {len(children)}"
                node_text.append(text)
                
                # Color based on degree
                degree = len(parents) + len(children)
                node_color.append(degree)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[n for n in self.graph.nodes()],
                textposition="top center",
                hovertext=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=15,
                    color=node_color,
                    colorbar=dict(
                        thickness=15,
                        title=dict(text='Degree', side='right'),
                        xanchor='left'
                    ),
                    line_width=2
                )
            )
            
            # Create figure
            is_dag = nx.is_directed_acyclic_graph(self.graph)
            title_text = f'Market Causality {"DAG" if is_dag else "Network"}<br>'
            title_text += f'{len(self.universe)} stocks, {self.graph.number_of_edges()} causal relationships<br>'
            title_text += f'<span style="font-size:12px">Edge labels show p-values (lower = stronger causality)</span>'
            
            fig = go.Figure(data=edge_traces + [node_trace],
                          layout=go.Layout(
                              title=title_text,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=0,l=0,r=0,t=80),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              annotations=edge_annotations
                          )
                          )
            
            # Save
            os.makedirs(Path(output_file).parent, exist_ok=True)
            fig.write_html(output_file)
            logging.info(f"Saved graph visualization to {output_file}")
            
        except ImportError:
            logging.warning("plotly not installed, using matplotlib instead")
            self._visualize_matplotlib(output_file.replace('.html', '.png'), highlight_ticker)
    
    def _visualize_matplotlib(self, output_file: str, highlight_ticker: str = None):
        """Fallback visualization using matplotlib."""
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, arrows=True, arrowsize=20)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=500, node_color='lightblue')
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10)
        
        plt.title(f"Market Causality DAG\n{len(self.universe)} stocks, {self.graph.number_of_edges()} causal relationships")
        plt.axis('off')
        plt.tight_layout()
        
        os.makedirs(Path(output_file).parent, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Saved graph visualization to {output_file}")
        plt.close()
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics about the causality network."""
        if self.graph is None:
            self.build_graph()
        
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        
        stats = {
            'num_stocks': len(self.universe),
            'num_relationships': self.graph.number_of_edges(),
            'is_dag': nx.is_directed_acyclic_graph(self.graph),
            'avg_in_degree': np.mean(list(in_degrees.values())),
            'avg_out_degree': np.mean(list(out_degrees.values())),
            'most_influential': max(out_degrees, key=out_degrees.get),
            'most_influenced': max(in_degrees, key=in_degrees.get),
            'isolated_nodes': [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        }
        
        return stats
    
    def save(self, filepath: str = None):
        """Save the DAG to a file."""
        if filepath is None:
            from src.utils.paths import CAUSALITY_CACHE_DIR
            filepath = str(CAUSALITY_CACHE_DIR / "market_dag.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f"DAG saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = None):
        """Load a DAG from a file."""
        if filepath is None:
            from src.utils.paths import CAUSALITY_CACHE_DIR
            filepath = str(CAUSALITY_CACHE_DIR / "market_dag.pkl")
        with open(filepath, 'rb') as f:
            dag = pickle.load(f)
        logging.info(f"DAG loaded from {filepath}")
        return dag


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Define universe (major indices + tech stocks)
    universe = ['SPY', 'QQQ', 'DIA', 'IWM',  # Market indices
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Big tech
                'TSLA', 'META']  # Other influential stocks
    
    print("\n" + "="*80)
    print("MARKET CAUSALITY DAG BUILDER")
    print("="*80)
    print(f"\nUniverse: {', '.join(universe)}")
    print(f"Total pairs to test: {len(universe) * (len(universe) - 1)}")
    print("="*80 + "\n")
    
    # Build DAG
    dag = MarketCausalityDAG(
        universe=universe,
        start_date="2023-01-01",
        end_date="2024-12-31",
        significance=0.05
    )
    
    # Fetch data
    dag.fetch_returns_data()
    
    # Build causality matrix
    dag.build_causality_matrix()
    
    # Build graph with cycle breaking
    dag.build_graph(break_cycles=True)
    
    # Get stats
    stats = dag.get_summary_stats()
    
    print("\n" + "="*80)
    print("NETWORK STATISTICS")
    print("="*80)
    print(f"Stocks: {stats['num_stocks']}")
    print(f"Causal relationships: {stats['num_relationships']}")
    print(f"Is DAG: {stats['is_dag']}")
    print(f"Average in-degree: {stats['avg_in_degree']:.2f}")
    print(f"Average out-degree: {stats['avg_out_degree']:.2f}")
    print(f"Most influential (causes most stocks): {stats['most_influential']}")
    print(f"Most influenced (caused by most stocks): {stats['most_influenced']}")
    if stats['isolated_nodes']:
        print(f"Isolated nodes: {', '.join(stats['isolated_nodes'])}")
    
    # Show causal relationships for each stock
    print("\n" + "="*80)
    print("CAUSAL RELATIONSHIPS")
    print("="*80)
    for ticker in universe:
        parents = dag.get_causal_parents(ticker)
        children = dag.get_causal_children(ticker)
        print(f"\n{ticker}:")
        if parents:
            print(f"  Caused by: {', '.join(parents)}")
        if children:
            print(f"  Causes: {', '.join(children)}")
        if not parents and not children:
            print(f"  No causal relationships")
    
    # Visualize
    dag.visualize_graph()
    
    # Save
    dag.save()
    
    print("\n" + "="*80)
    print("DAG CONSTRUCTION COMPLETE")
    print("="*80 + "\n")
