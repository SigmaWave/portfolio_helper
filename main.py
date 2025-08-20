import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

import openbb
import pytimetk as tk

from colorama import Fore, Back, Style
import yfinance as yf

plt.rcdefaults()
from helpers import load_data, import_data, timer

START = "2022-01-01"
END = "2025-01-01"
DATA_DIR = "./StockData"


stocks = ['SPY',
          'AAPL',
          'TSLA',
          'AMZN',
          'NVDA',
          'JPM']

two_points  = [('2024-05-17',72),('2024-06-14',58)]


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


if __name__ == "__main__":
    data = import_data('BNP.PA', START, END, DATA_DIR)
    print(data.columns)
    print(data.head())
    print(data.tail())

    data = load_data('BNP.PA', START, END, DATA_DIR)
    print(data.columns)
    print(data.head())
    print(data.tail())
    mpf.plot(data, type='candle', volume=True,
    hlines=dict(hlines=[53.5,73],colors=['g','r'],linestyle='-.'), 
    vlines=dict(vlines='2024-03-30',linewidths=120,alpha=0.4, colors='g'),
    alines=two_points
    # addplot=dojis_plot
    )