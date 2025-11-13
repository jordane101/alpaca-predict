"""
Apply fast cycle breaking to the large network.
"""

import pickle
import logging
import networkx as nx
import time

logging.basicConfig(level=logging.INFO, format='%(message)s')

print("\n" + "="*80)
print("APPLYING FAST CYCLE BREAKING TO LARGE NETWORK")
print("="*80 + "\n")

# Load the network
logging.info("Loading network...")
with open("causality_cache/large_network_graph.pkl", 'rb') as f:
    data = pickle.load(f)

graph = data['graph']
universe = data['universe']
causality_matrix = data['causality_matrix']
p_value_matrix = data['p_value_matrix']
lag_matrix = data['lag_matrix']

initial_edges = graph.number_of_edges()
logging.info(f"Loaded: {graph.number_of_nodes()} nodes, {initial_edges} edges")

# Fast cycle breaking algorithm
logging.info("\n" + "="*80)
logging.info("FAST CYCLE BREAKING (Greedy Heuristic)")
logging.info("="*80)

start_time = time.time()

# Step 1: Order nodes by (out-degree - in-degree)
logging.info("\nStep 1: Computing node ordering...")
nodes = list(graph.nodes())
scores = {}
for node in nodes:
    out_deg = graph.out_degree(node)
    in_deg = graph.in_degree(node)
    scores[node] = out_deg - in_deg

# Sort nodes by score (descending)
ordered_nodes = sorted(nodes, key=lambda n: scores[n], reverse=True)
node_order = {node: i for i, node in enumerate(ordered_nodes)}

logging.info(f"  Computed ordering for {len(nodes)} nodes")
logging.info(f"  Top 5 nodes: {ordered_nodes[:5]}")
logging.info(f"  Bottom 5 nodes: {ordered_nodes[-5:]}")

# Step 2: Remove backward edges
logging.info("\nStep 2: Identifying backward edges...")
edges_to_remove = []
for source, target, data in graph.edges(data=True):
    if node_order[source] > node_order[target]:
        # This is a backward edge
        edges_to_remove.append((source, target, data['p_value']))

logging.info(f"  Found {len(edges_to_remove)} backward edges ({100*len(edges_to_remove)/initial_edges:.1f}%)")

# Sort by p-value (remove weakest first if we need to prioritize)
edges_to_remove.sort(key=lambda x: x[2], reverse=True)

logging.info("\nStep 3: Removing backward edges...")
for i, (source, target, p_value) in enumerate(edges_to_remove):
    graph.remove_edge(source, target)
    if (i + 1) % 500 == 0:
        logging.info(f"  Removed {i + 1}/{len(edges_to_remove)} edges...")

edges_removed = len(edges_to_remove)

# Check if we succeeded
logging.info("\nStep 4: Verifying graph is acyclic...")
try:
    nx.find_cycle(graph)
    logging.warning("  Graph still has cycles! Applying iterative removal...")
    is_dag = False
except nx.NetworkXNoCycle:
    logging.info("  ✓ Graph is now acyclic!")
    is_dag = True

# If still has cycles, apply iterative removal
if not is_dag:
    logging.info("\nStep 5: Iterative cycle removal (cleanup)...")
    iteration = 0
    max_iterations = 1000
    
    while iteration < max_iterations:
        iteration += 1
        
        try:
            cycle = nx.find_cycle(graph, orientation='original')
            
            # Find the weakest edge in this cycle
            weakest_edge = None
            weakest_p_value = -1
            
            for edge in cycle:
                source, target = edge[0], edge[1]
                p_value = graph[source][target]['p_value']
                
                if p_value > weakest_p_value:
                    weakest_p_value = p_value
                    weakest_edge = (source, target)
            
            if weakest_edge:
                if edges_removed % 50 == 0:
                    logging.info(f"    Removed {edges_removed} edges so far...")
                graph.remove_edge(*weakest_edge)
                edges_removed += 1
            else:
                break
                
        except nx.NetworkXNoCycle:
            logging.info("  ✓ Graph is now acyclic!")
            is_dag = True
            break
        except Exception as e:
            logging.warning(f"  Error: {e}")
            break

elapsed = time.time() - start_time

# Final statistics
logging.info("\n" + "="*80)
logging.info("CYCLE BREAKING COMPLETE")
logging.info("="*80)
logging.info(f"Time taken: {elapsed:.1f} seconds")
logging.info(f"Is DAG: {is_dag}")
logging.info(f"Edges removed: {edges_removed} ({100*edges_removed/initial_edges:.1f}%)")
logging.info(f"Remaining edges: {graph.number_of_edges()}/{initial_edges}")

# Update causality matrix
logging.info("\nUpdating causality matrix...")
for i, cause in enumerate(universe):
    for j, effect in enumerate(universe):
        if causality_matrix[i, j] and not graph.has_edge(cause, effect):
            causality_matrix[i, j] = False

# Get new statistics
logging.info("\n" + "="*80)
logging.info("DAG STATISTICS")
logging.info("="*80)

in_degrees = dict(graph.in_degree())
out_degrees = dict(graph.out_degree())

logging.info(f"Nodes: {graph.number_of_nodes()}")
logging.info(f"Edges: {graph.number_of_edges()}")
logging.info(f"Average in-degree: {sum(in_degrees.values()) / len(in_degrees):.2f}")
logging.info(f"Average out-degree: {sum(out_degrees.values()) / len(out_degrees):.2f}")

# Top influencers after cycle breaking
logging.info("\n" + "-"*80)
logging.info("TOP 10 INFLUENCERS (after cycle breaking)")
logging.info("-"*80)
sorted_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
for i, (stock, degree) in enumerate(sorted_out[:10], 1):
    logging.info(f"{i:2d}. {stock:6s}: causes {degree:3d} stocks")

# Top influenced after cycle breaking
logging.info("\n" + "-"*80)
logging.info("TOP 10 INFLUENCED (after cycle breaking)")
logging.info("-"*80)
sorted_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
for i, (stock, degree) in enumerate(sorted_in[:10], 1):
    logging.info(f"{i:2d}. {stock:6s}: caused by {degree:3d} stocks")

# Check topological order
if is_dag:
    logging.info("\n" + "-"*80)
    logging.info("TOPOLOGICAL ORDER (First 10 and Last 10)")
    logging.info("-"*80)
    topo_order = list(nx.topological_sort(graph))
    logging.info(f"First 10 (root causes): {topo_order[:10]}")
    logging.info(f"Last 10 (end effects): {topo_order[-10:]}")

# Save the DAG
logging.info("\n" + "="*80)
logging.info("SAVING DAG")
logging.info("="*80)

with open("causality_cache/large_network_graph_dag.pkl", 'wb') as f:
    pickle.dump({
        'graph': graph,
        'universe': universe,
        'causality_matrix': causality_matrix,
        'p_value_matrix': p_value_matrix,
        'lag_matrix': lag_matrix,
        'is_dag': is_dag,
        'edges_removed': edges_removed,
        'initial_edges': initial_edges
    }, f)

logging.info("Saved to: causality_cache/large_network_graph_dag.pkl")

logging.info("\n" + "="*80)
logging.info("SUCCESS!")
logging.info("="*80 + "\n")
