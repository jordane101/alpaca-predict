"""
Analyze a pre-built causality DAG.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import networkx as nx
from src.causality.market_causality_dag import MarketCausalityDAG


def analyze_dag():
    """Load and analyze the market causality DAG."""
    
    # Load the DAG
    dag_path = "causality_cache/market_dag.pkl"
    
    with open(dag_path, 'rb') as f:
        dag = pickle.load(f)
    
    print("\n" + "="*80)
    print("MARKET CAUSALITY DAG ANALYSIS")
    print("="*80)
    
    # Basic stats
    print(f"\nUniverse: {', '.join(dag.universe)}")
    print(f"Date Range: {dag.start_date} to {dag.end_date}")
    print(f"Significance Level: {dag.significance}")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(dag.graph)}")
    
    # Network metrics
    print("\n" + "-"*80)
    print("NETWORK METRICS")
    print("-"*80)
    print(f"Nodes: {dag.graph.number_of_nodes()}")
    print(f"Edges: {dag.graph.number_of_edges()}")
    
    in_degrees = dict(dag.graph.in_degree())
    out_degrees = dict(dag.graph.out_degree())
    
    print(f"Average in-degree: {sum(in_degrees.values()) / len(in_degrees):.2f}")
    print(f"Average out-degree: {sum(out_degrees.values()) / len(out_degrees):.2f}")
    
    # Topological order (only works for DAG)
    if nx.is_directed_acyclic_graph(dag.graph):
        print("\n" + "-"*80)
        print("TOPOLOGICAL ORDER (Causal Hierarchy)")
        print("-"*80)
        topo_order = list(nx.topological_sort(dag.graph))
        
        # Group by levels
        levels = {}
        for node in topo_order:
            # Level = longest path from node with no predecessors
            if dag.graph.in_degree(node) == 0:
                level = 0
            else:
                level = max(levels.get(pred, 0) for pred in dag.graph.predecessors(node)) + 1
            levels[node] = level
        
        # Print by level
        max_level = max(levels.values())
        for level in range(max_level + 1):
            nodes_at_level = [node for node, l in levels.items() if l == level]
            print(f"\nLevel {level}: {', '.join(nodes_at_level)}")
            
            if level == 0:
                print("  (Root causes - not influenced by others)")
            elif level == max_level:
                print("  (End effects - don't influence others, or influence back to roots)")
    
    # Most influential stocks
    print("\n" + "-"*80)
    print("TOP INFLUENCERS (by out-degree)")
    print("-"*80)
    sorted_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
    for i, (stock, degree) in enumerate(sorted_out[:5], 1):
        children = list(dag.graph.successors(stock))
        print(f"{i}. {stock}: causes {degree} stocks")
        print(f"   Influences: {', '.join(children)}")
    
    # Most influenced stocks
    print("\n" + "-"*80)
    print("TOP INFLUENCED (by in-degree)")
    print("-"*80)
    sorted_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
    for i, (stock, degree) in enumerate(sorted_in[:5], 1):
        parents = list(dag.graph.predecessors(stock))
        print(f"{i}. {stock}: caused by {degree} stocks")
        print(f"   Influenced by: {', '.join(parents)}")
    
    # Strongest causal relationships
    print("\n" + "-"*80)
    print("STRONGEST CAUSAL RELATIONSHIPS (lowest p-values)")
    print("-"*80)
    
    edges_with_pvalues = []
    for source, target, data in dag.graph.edges(data=True):
        edges_with_pvalues.append((source, target, data['p_value'], data['lag']))
    
    edges_with_pvalues.sort(key=lambda x: x[2])
    
    for i, (source, target, p_value, lag) in enumerate(edges_with_pvalues[:10], 1):
        print(f"{i}. {source} → {target}: p={p_value:.4f}, lag={lag}")
    
    # Isolated or weakly connected nodes
    print("\n" + "-"*80)
    print("CONNECTIVITY ANALYSIS")
    print("-"*80)
    
    for stock in dag.universe:
        in_deg = dag.graph.in_degree(stock)
        out_deg = dag.graph.out_degree(stock)
        total_deg = in_deg + out_deg
        
        if total_deg == 0:
            print(f"{stock}: ISOLATED (no causal connections)")
        elif total_deg <= 2:
            print(f"{stock}: Weakly connected (in={in_deg}, out={out_deg})")
    
    # Path analysis
    print("\n" + "-"*80)
    print("CAUSAL PATHS (Examples)")
    print("-"*80)
    
    # Find some interesting paths
    roots = [n for n in dag.graph.nodes() if dag.graph.in_degree(n) == 0]
    leaves = [n for n in dag.graph.nodes() if dag.graph.out_degree(n) == 0]
    
    if roots and leaves:
        print("\nLongest causal chains:")
        for root in roots[:3]:
            for leaf in leaves[:3]:
                if nx.has_path(dag.graph, root, leaf):
                    paths = list(nx.all_simple_paths(dag.graph, root, leaf))
                    if paths:
                        longest = max(paths, key=len)
                        if len(longest) > 2:
                            print(f"  {' → '.join(longest)}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    analyze_dag()
