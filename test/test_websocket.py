"""
A standalone script to test the Alpaca WebSocket connection logic.

This script helps isolate and debug connection issues without running the full
orchestrator and agent framework.
"""

import os
import asyncio
from dotenv import load_dotenv
from alpaca.data.live.stock import StockDataStream
from alpaca.data.enums import DataFeed

# Load API keys from the .env file in the project root.
# The path is relative to where you run the script from.
load_dotenv("../.env")
KEY = os.getenv("PAPER_KEY")
SECRET = os.getenv("PAPER_SEC")

async def on_trade(trade):
    """A simple callback function to print received trades."""
    print(f"Trade received: {trade.symbol} @ ${trade.price:.2f} (Size: {trade.size})")

async def main():
    """Main function to connect to the WebSocket and run the stream."""
    print("--- Starting WebSocket Connection Test ---")
    if not KEY or not SECRET:
        print("Error: PAPER_KEY and PAPER_SEC must be set in the .env file.")
        return

    # Instantiate the WebSocket client
    stream = StockDataStream(KEY, SECRET, feed=DataFeed.IEX)

    # Subscribe to a high-volume ticker to ensure we get data
    ticker_to_test = "SPY"
    print(f"Subscribing to trades for {ticker_to_test}...")
    stream.subscribe_trades(on_trade, ticker_to_test)

    try:
        print("Connection established. Listening for trades... (Press Ctrl+C to stop)")
        # The .run() method is a coroutine that will run until the connection is closed
        # or an unrecoverable error occurs.
        await stream.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nInterruption received, proceeding to shutdown.")
    except Exception as e:
        # This will catch other errors, like the 'NoneType' has no attribute 'is_running'
        print(f"\nAn unexpected error occurred during stream run: {e}")
    finally:
        print("Closing WebSocket connection...")
        await stream.close()
        print("Connection closed.")

if __name__ == "__main__":
    asyncio.run(main())

