from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from dotenv import load_dotenv
import os


load_dotenv("../.env")

key = os.getenv("PAPER_KEY")
sec = os.getenv("PAPER_SEC")

trading_client = TradingClient("PKP6NX8RZUI9HJR5VLZ3","OHgpNCENWdYXEdD8fyBbyDW7hqXUTkDmyfhAiZ14", paper=True) # ENV keys don't work for some reason
# trading_client = TradingClient(key,sec, paper=True)

search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)

assets = trading_client.get_all_assets(search_params)

# account = trading_client.get_account()

print(assets)