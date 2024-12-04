from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

client = StockHistoricalDataClient("PKP6NX8RZUI9HJR5VLZ3","OHgpNCENWdYXEdD8fyBbyDW7hqXUTkDmyfhAiZ14")

request_params = StockBarsRequest(
    symbol_or_symbols=["AAPL","TSLA","AMZN"],
    timeframe=TimeFrame.Day,
    start="2024-11-01"

)

bars = client.get_stock_bars(request_params)

print(bars.df)