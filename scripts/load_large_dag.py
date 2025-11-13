"""
Load the cached large DAG and analyze it (skip computation).
"""

import pickle
import logging
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(message)s')

print("\n" + "="*80)
print("LOADING LARGE CAUSALITY NETWORK FROM CACHE")
print("="*80 + "\n")

# Load the most recent causality matrix
cache_file = "causality_cache/causality_matrix_2023-01-01_2024-12-31_0.05_95_8867242579514336506.pkl"

logging.info(f"Loading from: {cache_file}")

with open(cache_file, 'rb') as f:
    data = pickle.load(f)

causality = data['causality']
p_values = data['p_values']
lags = data['lags']
universe = data['universe']

logging.info(f"\nLoaded causality matrix:")
logging.info(f"  Universe size: {len(universe)}")
logging.info(f"  Total relationships: {causality.sum()}")
logging.info(f"  Density: {100 * causality.sum() / (len(universe) * (len(universe) - 1)):.1f}%")

# Build graph
logging.info("\nBuilding graph...")
graph = nx.DiGraph()

for ticker in universe:
    graph.add_node(ticker)

n = len(universe)
for i in range(n):
    for j in range(n):
        if causality[i, j]:
            cause = universe[i]
            effect = universe[j]
            graph.add_edge(
                cause,
                effect,
                p_value=p_values[i, j],
                lag=int(lags[i, j]),
                weight=1.0 - p_values[i, j]
            )

logging.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

# Quick cycle check
logging.info("\nChecking for cycles (quick test)...")
try:
    cycle = nx.find_cycle(graph)
    logging.info(f"✗ Graph has cycles. Example: {' → '.join([c[0] for c in cycle[:5]])}")
    has_cycles = True
except nx.NetworkXNoCycle:
    logging.info("✓ Graph is acyclic")
    has_cycles = False

# Statistics
logging.info("\n" + "="*80)
logging.info("NETWORK STATISTICS")
logging.info("="*80)

in_degrees = dict(graph.in_degree())
out_degrees = dict(graph.out_degree())

logging.info(f"Average in-degree: {sum(in_degrees.values()) / len(in_degrees):.2f}")
logging.info(f"Average out-degree: {sum(out_degrees.values()) / len(out_degrees):.2f}")

# Top influencers
logging.info("\n" + "-"*80)
logging.info("TOP 20 INFLUENCERS (by out-degree)")
logging.info("-"*80)
sorted_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
for i, (stock, degree) in enumerate(sorted_out[:20], 1):
    logging.info(f"{i:2d}. {stock:6s}: causes {degree:3d} stocks")

# Top influenced
logging.info("\n" + "-"*80)
logging.info("TOP 20 INFLUENCED (by in-degree)")
logging.info("-"*80)
sorted_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
for i, (stock, degree) in enumerate(sorted_in[:20], 1):
    logging.info(f"{i:2d}. {stock:6s}: caused by {degree:3d} stocks")

# Strongest relationships
logging.info("\n" + "-"*80)
logging.info("STRONGEST CAUSAL RELATIONSHIPS (lowest p-values)")
logging.info("-"*80)

edges_with_pvalues = []
for source, target, data in graph.edges(data=True):
    edges_with_pvalues.append((source, target, data['p_value'], data['lag']))

edges_with_pvalues.sort(key=lambda x: x[2])

for i, (source, target, p_value, lag) in enumerate(edges_with_pvalues[:20], 1):
    logging.info(f"{i:2d}. {source:6s} → {target:6s}: p={p_value:.4f}, lag={lag}")

# Save as NetworkX graph for easy loading
logging.info("\n" + "="*80)
logging.info("SAVING")
logging.info("="*80)

import pickle
with open("causality_cache/large_network_graph.pkl", 'wb') as f:
    pickle.dump({
        'graph': graph,
        'universe': universe,
        'causality_matrix': causality,
        'p_value_matrix': p_values,
        'lag_matrix': lags,
        'has_cycles': has_cycles
    }, f)

logging.info("Saved to: causality_cache/large_network_graph.pkl")
logging.info("\n" + "="*80)
logging.info("COMPLETE!")
logging.info("="*80 + "\n")
