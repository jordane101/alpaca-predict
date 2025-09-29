"""
A standalone script to test the Alpaca WebSocket connection logic.

This script helps isolate and debug connection issues without running the full
orchestrator and agent framework.
"""

import os
from pathlib import Path
import asyncio
import traceback
from dotenv import load_dotenv
import alpaca
from alpaca.data.live.stock import StockDataStream
from alpaca.data.enums import DataFeed

# Load API keys from the .env file in the project root.
# This makes the script runnable from any directory.
project_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=project_root / ".env")
KEY = os.getenv("PAPER_KEY")
SECRET = os.getenv("PAPER_SEC")

async def on_trade(trade):
    """A simple callback function to print received trades."""
    print(f"Trade received: {trade.symbol} @ ${trade.price:.2f} (Size: {trade.size})")

# The main function should be synchronous because stream.run() is a blocking call
def main():
    """Main function to connect to the WebSocket and run the stream."""
    print("--- Starting WebSocket Connection Test ---")
    if not KEY or not SECRET:
        print("Error: PAPER_KEY and PAPER_SEC must be set in the .env file.")
        return

    # Let's print the exact version of the library being used to be 100% certain.
    print(f"Using alpaca-py version: {alpaca.__version__}")
    print(f"Loaded Key ID: {KEY[:4]}...{KEY[-4:]}")

    stream = StockDataStream(KEY, SECRET, feed=DataFeed.IEX)

    # Subscribe to a high-volume ticker to ensure we get data
    ticker_to_test = "SPY"
    print(f"Subscribing to trades for {ticker_to_test}...")
    stream.subscribe_trades(on_trade, ticker_to_test)

    try:
        print("Connection established. Listening for trades... (Press Ctrl+C to stop)")
        # The .run() method is a blocking call that starts its own asyncio loop.
        # It should be called directly from a synchronous context.
        stream.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nInterruption received, shutting down.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during stream run. See traceback below:")
        traceback.print_exc()

if __name__ == "__main__":
    main()