from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd

client = StockHistoricalDataClient("PKJHI0WM4W283UN7RD0O","Zb1AoNcu2Bc5JE62PqarFzr8s7Fhkc8BOGM8E3dW")

symbol = "NVDA"

request_params = StockBarsRequest(
    symbol_or_symbols=[symbol],
    timeframe=TimeFrame.Day,
    start="2023-01-01"
)

bars = client.get_stock_bars(request_params)

dataframe = pd.DataFrame(bars.df)
print(dataframe.columns)
dataframe.to_csv(f"./csv/{symbol}_from_2021.csv")

