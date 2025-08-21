import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

import seaborn as sns
from colorama import Fore

plt.rcdefaults()

from helpers import load_data, plot_breakouts, support_resistance_fills
from config import STOCKS, START, END, DATA_DIR
from indicators import mav_plot, support_resistance_breakout




two_points  = [('2024-05-17',72),('2024-06-14',58)]


if __name__ == "__main__":
    returns = pd.concat(
    {ticker: load_data(ticker, START, END, DATA_DIR)["Close"].pct_change() for ticker in STOCKS},
    axis=1
    )

    # drop the NaN from first row
    returns = returns.dropna()
    # correlation matrix
    corr_matrix = returns.corr()

    # ax = sns.heatmap(corr_matrix, cmap='RdYlGn', linewidths=.1)
    # plt.show()

    data = load_data('BNP.PA', START, END, DATA_DIR)

    data = load_data('BNP.PA', START, END, DATA_DIR)
view_data = data.loc[START:END]
addplots = []
fills = []

breakout_output = support_resistance_breakout(view_data, prd=5, bo_len=100, cwidthu=0.04, mintest=2)

addplots.extend(plot_breakouts(view_data, breakout_output))

fills.extend(support_resistance_fills(view_data, breakout_output["bull_boxes"], 'g'))
fills.extend(support_resistance_fills(view_data, breakout_output["bear_boxes"], 'r'))

addplots.extend(mav_plot(data, lengths=(50, 100, 200)))

kwargs = {}
if fills:
    kwargs["fill_between"] = fills

mpf.plot(
    view_data,
    type="candle",
    volume=True,
    style="yahoo",
    addplot=addplots,
    title='BNP.PA with Breakout Signals',
    figratio=(14, 8),
    **kwargs
)

mpf.show()


mpf.plot(data, type='candle', volume=True, mav = (50,100,200),
hlines=dict(hlines=[53.5,73],colors=['g','r'],linestyle='-.'), 
vlines=dict(vlines='2024-03-30',linewidths=120,alpha=0.4, colors='g'),
alines=two_points,
title='BNP.PA',   # color volume red/green automatically
style="yahoo"
# addplot=dojis_plot
)

out = support_resistance_breakout(data, prd=5, bo_len=200, cwidthu=0.03, mintest=2)
print(out)