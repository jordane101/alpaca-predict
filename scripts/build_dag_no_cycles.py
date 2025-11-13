"""
Build large DAG without cycle breaking (faster).
Use this to get the causality network quickly.
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('logs/dag_build_no_cycles.log'),
        logging.StreamHandler()
    ]
)

print("\n" + "="*80)
print("BUILDING LARGE CAUSALITY NETWORK (NO CYCLE BREAKING)")
print("="*80)
print(f"\nTotal stocks/ETFs: {len(UNIVERSE)}")
print(f"This will build the network but NOT break cycles (much faster)")
print("="*80 + "\n")

# Build the DAG
dag = MarketCausalityDAG(
    universe=UNIVERSE,
    start_date="2023-01-01",
    end_date="2024-12-31",
    significance=0.05,
    max_lag=5,
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
)

# Fetch data (will use cache if available)
logging.info("\n" + "="*80)
logging.info("STEP 1: FETCHING RETURNS DATA")
logging.info("="*80)
dag.fetch_returns_data()

# Build causality matrix (will use cache if available)
logging.info("\n" + "="*80)
logging.info("STEP 2: TESTING PAIRWISE CAUSALITY (PARALLEL)")
logging.info("="*80)
dag.build_causality_matrix(n_jobs=None)

# Build graph WITHOUT cycle breaking
logging.info("\n" + "="*80)
logging.info("STEP 3: BUILDING GRAPH (NO CYCLE BREAKING)")
logging.info("="*80)
dag.build_graph(break_cycles=False)

# Get statistics
stats = dag.get_summary_stats()

logging.info("\n" + "="*80)
logging.info("NETWORK STATISTICS")
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

# Save (without cycle breaking)
logging.info("\n" + "="*80)
logging.info("SAVING NETWORK")
logging.info("="*80)
dag.save("causality_cache/market_dag_large_no_cycles.pkl")

logging.info("\n" + "="*80)
logging.info("NETWORK BUILD COMPLETE!")
logging.info("="*80)
logging.info(f"\nNetwork file: causality_cache/market_dag_large_no_cycles.pkl")
logging.info(f"Log file: logs/dag_build_no_cycles.log")
logging.info(f"\nNote: Network contains cycles. Use for analysis or apply cycle breaking separately.")
logging.info("\n")
