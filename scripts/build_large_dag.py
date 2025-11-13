"""
Build a large market causality DAG with comprehensive stock coverage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from src.causality.market_causality_dag import MarketCausalityDAG


# Comprehensive tech and market universe
UNIVERSE = [
    # Major Market Indices
    'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO',
    
    # Magnificent 7 + FAANG
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META',
    
    # Other Big Tech
    'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'CSCO',
    'IBM', 'TXN', 'AVGO', 'NOW', 'INTU', 'AMAT', 'LRCX', 'KLAC',
    'MU', 'SNPS', 'CDNS', 'MCHP', 'ADI', 'NXPI',
    
    # Cloud & SaaS
    'SNOW', 'DDOG', 'NET', 'ZS', 'CRWD', 'OKTA', 'TEAM', 'WDAY',
    'VEEV', 'PANW', 'FTNT', 'SPLK', 'HUBS', 'ZM', 'DOCU',
    
    # Semiconductors
    'TSM', 'ASML', 'ON', 'MPWR', 'SWKS', 'QRVO', 'MRVL',
    
    # E-commerce & Fintech
    'SHOP', 'PYPL', 'SQ', 'COIN', 'MELI', 'EBAY', 'BKNG',
    
    # Social Media & Communication
    'SNAP', 'PINS', 'SPOT', 'RBLX', 'U', 'MTCH',
    
    # Gaming & Entertainment
    'EA', 'TTWO', 'ATVI', 'RBLX',
    
    # Streaming & Media
    'DIS', 'ROKU', 'PARA',
    
    # Automotive & EV
    'F', 'GM', 'RIVN', 'LCID', 'NIO',
    
    # Other Growth Tech
    'UBER', 'LYFT', 'ABNB', 'DASH', 'MRVL', 'PLTR', 'ARKK',
    
    # Tech ETFs
    'XLK', 'VGT', 'SOXX', 'SMH', 'QTEC', 'IGV', 'CIBR',
]

# Remove duplicates and sort
UNIVERSE = sorted(list(set(UNIVERSE)))

print("\n" + "="*80)
print("BUILDING LARGE MARKET CAUSALITY DAG")
print("="*80)
import multiprocessing
n_cpus = multiprocessing.cpu_count()

print(f"\nTotal stocks/ETFs: {len(UNIVERSE)}")
print(f"Total pairwise tests: {len(UNIVERSE) * (len(UNIVERSE) - 1):,}")
print(f"CPU cores available: {n_cpus}")
print(f"\nEstimated time (sequential): ~{len(UNIVERSE) * (len(UNIVERSE) - 1) * 0.5 / 60:.1f} minutes")
print(f"Estimated time (parallel, {n_cpus-1} workers): ~{len(UNIVERSE) * (len(UNIVERSE) - 1) * 0.5 / 60 / (n_cpus-1):.1f} minutes")
print("\nUniverse includes:")
print(f"  - Market indices (SPY, QQQ, DIA, etc.)")
print(f"  - Magnificent 7 (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META)")
print(f"  - Major tech companies")
print(f"  - Semiconductors")
print(f"  - Cloud & SaaS")
print(f"  - E-commerce & Fintech")
print(f"  - Social media & Communication")
print(f"  - Gaming & Entertainment")
print(f"  - EV & Automotive")
print(f"  - Tech sector ETFs")
print("\n" + "="*80 + "\n")

response = input("This will take a while. Continue? (yes/no): ")
if response.lower() not in ['yes', 'y']:
    print("Aborted.")
    exit(0)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('logs/dag_build_large.log'),
        logging.StreamHandler()
    ]
)

# Build the DAG
dag = MarketCausalityDAG(
    universe=UNIVERSE,
    start_date="2023-01-01",
    end_date="2024-12-31",
    significance=0.05,
    max_lag=5,
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
)

# Fetch data
logging.info("\n" + "="*80)
logging.info("STEP 1: FETCHING RETURNS DATA")
logging.info("="*80)
dag.fetch_returns_data(force_refresh=True)

# Build causality matrix with parallel processing
logging.info("\n" + "="*80)
logging.info("STEP 2: TESTING PAIRWISE CAUSALITY (PARALLEL)")
logging.info("="*80)
dag.build_causality_matrix(force_recompute=True, n_jobs=None)  # Use all available CPUs

# Save intermediate results before cycle breaking
logging.info("\n" + "="*80)
logging.info("SAVING INTERMEDIATE RESULTS")
logging.info("="*80)
dag.save("causality_cache/market_dag_large_with_cycles.pkl")
logging.info("Saved graph with cycles (before breaking)")

# Build graph with cycle breaking
logging.info("\n" + "="*80)
logging.info("STEP 3: BUILDING DAG (WITH CYCLE BREAKING)")
logging.info("="*80)
try:
    dag.build_graph(break_cycles=True)
except Exception as e:
    logging.error(f"Error during graph building: {e}")
    logging.info("Saving partial results...")
    dag.save("causality_cache/market_dag_large_partial.pkl")
    raise

# Get statistics
stats = dag.get_summary_stats()

logging.info("\n" + "="*80)
logging.info("LARGE DAG STATISTICS")
logging.info("="*80)
logging.info(f"Stocks/ETFs: {stats['num_stocks']}")
logging.info(f"Causal relationships: {stats['num_relationships']}")
logging.info(f"Is DAG: {stats['is_dag']}")
logging.info(f"Average in-degree: {stats['avg_in_degree']:.2f}")
logging.info(f"Average out-degree: {stats['avg_out_degree']:.2f}")
logging.info(f"Most influential: {stats['most_influential']} (out-degree: {dag.graph.out_degree(stats['most_influential'])})")
logging.info(f"Most influenced: {stats['most_influenced']} (in-degree: {dag.graph.in_degree(stats['most_influenced'])})")

# Show top influencers
logging.info("\n" + "-"*80)
logging.info("TOP 10 INFLUENCERS")
logging.info("-"*80)
out_degrees = dict(dag.graph.out_degree())
sorted_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
for i, (stock, degree) in enumerate(sorted_out[:10], 1):
    logging.info(f"{i:2d}. {stock:6s}: causes {degree:3d} stocks")

# Show top influenced
logging.info("\n" + "-"*80)
logging.info("TOP 10 INFLUENCED")
logging.info("-"*80)
in_degrees = dict(dag.graph.in_degree())
sorted_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
for i, (stock, degree) in enumerate(sorted_in[:10], 1):
    logging.info(f"{i:2d}. {stock:6s}: caused by {degree:3d} stocks")

# Visualize (skip if too large to avoid memory issues)
logging.info("\n" + "="*80)
logging.info("STEP 4: CREATING VISUALIZATION")
logging.info("="*80)
if dag.graph.number_of_edges() < 1000:
    dag.visualize_graph(
        output_file="outputs/causality_graph_large.html",
        show_edge_weights=False  # Too many edges for labels
    )
else:
    logging.info(f"  Skipping visualization ({dag.graph.number_of_edges()} edges too large)")
    logging.info(f"  Use analyze_causality_dag.py to explore the network")

# Save
logging.info("\n" + "="*80)
logging.info("STEP 5: SAVING DAG")
logging.info("="*80)
dag.save("causality_cache/market_dag_large.pkl")

logging.info("\n" + "="*80)
logging.info("LARGE DAG BUILD COMPLETE!")
logging.info("="*80)
logging.info(f"\nVisualization: outputs/causality_graph_large.html")
logging.info(f"DAG file: causality_cache/market_dag_large.pkl")
logging.info(f"Log file: logs/dag_build_large.log")
logging.info("\n")
