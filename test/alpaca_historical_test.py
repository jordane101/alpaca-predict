from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd

client = StockHistoricalDataClient("PKP6NX8RZUI9HJR5VLZ3","OHgpNCENWdYXEdD8fyBbyDW7hqXUTkDmyfhAiZ14")

request_params = StockBarsRequest(
    symbol_or_symbols=["AAPL"],
    timeframe=TimeFrame.Day,
    start="2020-01-01"

)

bars = client.get_stock_bars(request_params)

dataframe = pd.DataFrame(bars.df)
print(dataframe.columns)
dataframe.to_csv("./csv/AAPL_from_2021.csv")