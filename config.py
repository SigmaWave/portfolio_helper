import pandas as pd

# Date format: YYYY-MM-DD
START = pd.Timestamp("2025-01-02")
END = pd.Timestamp("2025-08-19")
DATA_DIR = "./StockData"


STOCKS = ['SPY',
          'AAPL',
          'TSLA',
          'AMZN',
          'NVDA',
          'JPM',
          'GLD', #Gold
          'TLT' #iShares 20+ Year Treasury Bond ETF
          ]