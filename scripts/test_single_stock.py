#!/usr/bin/env python3
"""
Quick test to see if HMM with causal features works for a single stock.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hmm.hmm_analysis import AnalyzeHMM
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Alpaca client
api_key = os.getenv('PAPER_KEY') or os.getenv('APCA_API_KEY_ID')
api_secret = os.getenv('PAPER_SEC') or os.getenv('APCA_API_SECRET_KEY')
data_client = StockHistoricalDataClient(api_key, api_secret)

print("Testing HMM with Causal Features")
print("="*70)

# Test ticker
ticker = "AAPL"
print(f"\nTesting {ticker}...")

# Fetch data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)

try:
    request_params = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame.Day,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d')
    )
    
    bars = data_client.get_stock_bars(request_params)
    df = bars.df
    
    if ticker in df.index.get_level_values(0):
        df = df.xs(ticker, level=0)
    
    print(f"  Fetched {len(df)} bars")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Index type: {type(df.index)}")
    print(f"  First row:\n{df.head(1)}")
    
    # Test with causal features
    print(f"\n1. Testing WITH causal features...")
    try:
        analyzer = AnalyzeHMM(
            ticker=ticker,
            n_components=4,
            model_order=1,
            bars_data=df,
            verbose=False,
            force_retrain=True,
            use_causal_features=True,
            causal_dag_file=None,  # Use default
            optimize_n_components=True,
            n_components_range=(2, 4)
        )
        
        prediction = analyzer.predict_next_day_outlook()
        print(f"  ✓ SUCCESS!")
        print(f"    Outlook: {prediction['outlook']}")
        print(f"    Confidence: {prediction.get('confidence', 'N/A')}")
        print(f"    Position Size: {prediction.get('position_size', 'N/A')}")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test without causal features
    print(f"\n2. Testing WITHOUT causal features (technical only)...")
    try:
        analyzer = AnalyzeHMM(
            ticker=ticker,
            n_components=4,
            model_order=1,
            bars_data=df,
            verbose=False,
            force_retrain=True,
            use_causal_features=False,
            optimize_n_components=True,
            n_components_range=(2, 4)
        )
        
        prediction = analyzer.predict_next_day_outlook()
        print(f"  ✓ SUCCESS!")
        print(f"    Outlook: {prediction['outlook']}")
        print(f"    Confidence: {prediction.get('confidence', 'N/A')}")
        print(f"    Position Size: {prediction.get('position_size', 'N/A')}")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"✗ Error fetching data: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
