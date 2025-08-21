import mplfinance as mpf
from colorama import Fore

from helpers import load_data,  support_resistance_fills, breakouts_plot
from config import STOCKS, START, END, DATA_DIR
from indicators import mav_plot, support_resistance_breakout, show_corr_matrix




two_points  = [('2024-05-17',72),('2024-06-14',58)]


if __name__ == "__main__":
    data = load_data('BNP.PA', START, END, DATA_DIR)
    view_data = data.loc[START:END]

    show_corr_matrix()
    # Add plots and fills
    addplots = []
    fills = []

    breakout_output = support_resistance_breakout(view_data, prd=5, bo_len=100, cwidthu=0.04, mintest=2)

    # Add plots
    addplots.extend(breakouts_plot(view_data, breakout_output))
    addplots.extend(mav_plot(data, lengths=(50, 100, 200)))

    # Fills
    fills.extend(support_resistance_fills(view_data, breakout_output["bull_boxes"], 'g'))
    fills.extend(support_resistance_fills(view_data, breakout_output["bear_boxes"], 'r'))

    kwargs = {}
    kwargs["addplot"] = addplots
    if fills:
        kwargs["fill_between"] = fills # Only add fills if they are not empty, otherwise throws an error

    mpf.plot(
        view_data,
        type="candle",
        volume=True,
        style="yahoo",
        title='BNP.PA with Breakout Signals',
        figratio=(14, 8),
        **kwargs
    )

    mpf.show()

#hlines=dict(hlines=[53.5,73],colors=['g','r'],linestyle='-.'), 
#vlines=dict(vlines='2024-03-30',linewidths=120,alpha=0.4, colors='g'),

out = support_resistance_breakout(data, prd=5, bo_len=200, cwidthu=0.03, mintest=2)
print(out)