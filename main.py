import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

import openbb
import pytimetk as tk
import seaborn as sns
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
          'JPM',
          'GLD', #Gold
          'TLT' #iShares 20+ Year Treasury Bond ETF
          ]

two_points  = [('2024-05-17',72),('2024-06-14',58)]


if __name__ == "__main__":
    returns = pd.concat(
    {ticker: load_data(ticker, START, END, DATA_DIR)["Close"].pct_change() for ticker in stocks},
    axis=1
    )

    # drop the NaN from first row
    returns = returns.dropna()
    # correlation matrix
    corr_matrix = returns.corr()

    # ax = sns.heatmap(corr_matrix, cmap='RdYlGn', linewidths=.1)
    # plt.show()

    data = load_data('BNP.PA', START, END, DATA_DIR)

    mc = mpf.make_marketcolors(up='g',down='r',
                           edge='black',
                           )
    s  = mpf.make_mpf_style(marketcolors=mc)


    candle_kwargs = dict(type='candle',mav=(50,100,200), volume=True, )
    mc = mpf.make_marketcolors(up='g',down='r',
                            edge='black')
    s  = mpf.make_mpf_style(marketcolors=mc)

    mpf.plot(data, type='candle', volume=True, mav = (50,100,200),
    hlines=dict(hlines=[53.5,73],colors=['g','r'],linestyle='-.'), 
    vlines=dict(vlines='2024-03-30',linewidths=120,alpha=0.4, colors='g'),
    alines=two_points,
    title='BNP.PA',   # color volume red/green automatically
    style="yahoo"
    # addplot=dojis_plot
    )