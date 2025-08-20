import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

import openbb
import pytimetk as tk

from colorama import Fore, Back, Style
import yfinance as yf

plt.rcdefaults()




def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{Fore.GREEN}Function '{func.__name__}': {elapsed_time:.2f}s{Fore.RESET}\n")
        return result
    return wrapper


stocks = ['SPY',
          'AAPL',
          'TSLA',
          'AMZN',
          'NVDA',
          'JPM']

START = "2022-09-20"
END = "2024-01-01"

two_points  = [('2024-05-17',72),('2024-06-14',58)]

DATA_DIR = "./StockData"
os.makedirs(DATA_DIR, exist_ok=True)

def yf_to_mpf(df, ticker=None):
    # If MultiIndex columns (e.g., ('Price','Open'),('Price','High') or ('Open','BNP.PA') etc.)
    if isinstance(df.columns, pd.MultiIndex):
        # If ticker is present in last level, select it
        if ticker and ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        # If first level is something like "Price", drop it
        elif 'Open' in df.columns.get_level_values(-1):
            df = df.droplevel(0, axis=1)
        else:
            # fallback: keep last level
            df = df.droplevel(0, axis=1)

    # Standardize column names
    df = df.rename(columns=lambda c: str(c).strip().title())

    # Keep only what mplfinance needs
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]]

    # Ensure numeric dtypes
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows missing OHLC
    df = df.dropna(subset=[c for c in ["Open","High","Low","Close"] if c in df.columns])

    # Sort and ensure DateTimeIndex
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def import_data(symbol, start_date, end_date):
    """
    Download stock data with yfinance and save to CSV.
    """
    df = yf.download(symbol, start=start_date, end=end_date, progress=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}. Check ticker and dates.")
    print(df.head())
    df = df.droplevel(1, axis=1)
    # yfinance already returns a DataFrame with OHLCV columns
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    print(df.head())
    out_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    df.to_csv(out_path)
    print(f"{Fore.GREEN}Saved {out_path}{Style.RESET_ALL}")
    return df


def load_data(symbol, START, END):
    """
    Load data from CSV if available, otherwise download it.
    """
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        print(f"{Fore.RED}File not found — importing now…{Style.RESET_ALL}")
        return import_data(symbol, START, END)

    data = pd.read_csv(path, index_col="Date", parse_dates=True)
    print(f"{Fore.GREEN}Loaded {symbol} from {path}{Style.RESET_ALL}")
    return data



if __name__ == "__main__":
    data = import_data('BNP.PA', START, END)

    print(data.columns)
    mpf.plot(data, type='candle', volume=True,
    hlines=dict(hlines=[53.5,73],colors=['g','r'],linestyle='-.'), 
    vlines=dict(vlines='2024-03-30',linewidths=120,alpha=0.4, colors='g'),
    alines=two_points
    # addplot=dojis_plot
    )