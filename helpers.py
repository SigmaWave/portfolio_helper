import yfinance as yf
import pandas as pd
import os, time
from colorama import Fore
import numpy as np
import mplfinance as mpf
from config import START, END, DATA_DIR

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{Fore.GREEN}Function '{func.__name__}': {elapsed_time:.2f}s{Fore.RESET}\n")
        return result
    return wrapper

@timer
def import_data(symbol, start_date, end_date, data_dir):
    """
    Download stock data with yfinance and save to CSV.

    The data is extracted with the following format:

                        Close       High        Low       Open   Volume
    Date                                                           
    2022-09-20  39.866329  41.026765  39.866329  40.808670  3029883
    2022-09-21  39.475403  39.672925  39.010407  39.504208  2577821
    2022-09-22  39.685272  40.249028  38.627712  38.693550  2696429
    2022-09-23  38.603020  39.763456  38.257359  39.763456  3892872
    2022-09-26  38.031033  38.672978  37.594841  38.162713  2675534

    """
    # Downloading data
    df = yf.download(symbol, start=start_date, end=end_date, progress=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}. Check ticker and dates.")
    df = df.dropna()
    df = df.sort_index()
    df.columns = df.columns.droplevel('Ticker')

    # Saving data
    os.makedirs(data_dir, exist_ok=True) # creating folder if it doesn't exist
    out_path = os.path.join(data_dir, f"{symbol}.csv")
    df.to_csv(out_path)

    print(f"{Fore.GREEN} Imported data for {symbol} and saved to {out_path}{Fore.RESET}")
    return df

@timer
def load_data(symbol, START, END, data_dir):
    """
    Load data from CSV if available, otherwise download it.
    """
    path = os.path.join(data_dir, f"{symbol}.csv")
    # No data found: we download it
    if not os.path.exists(path):
        print(f"{Fore.RED}File {symbol}.csv not found — importing now …{Fore.RESET}")
        return import_data(symbol, START, END, data_dir)
    
    # data found
    data = pd.read_csv(path, index_col="Date", parse_dates=True)
    if START not in data.index or END - pd.Timedelta(days=1) not in data.index:  # end is exclusive
        print(f"{Fore.RED}Data for {symbol} is outdated — importing new data …{Fore.RESET}")
        os.remove(path)
        return import_data(symbol, START, END, data_dir)

    print(f"{Fore.GREEN}Loaded {symbol} from {path}{Fore.RESET}")
    return data


def support_resistance_fills(df, box, color):
    fills = []
    n = len(df.index)
    print(f"{Fore.BLUE}Boxes: {box}{Fore.RESET}")
    if box:
        for b in box:
            where = (df.index >= pd.Timestamp(b["bar_start"])) & (df.index <= pd.Timestamp(b["bar_end"]))
            y1 = np.full(n, float(b["top"]),    dtype=float)
            y2 = np.full(n, float(b["bottom"]), dtype=float)
        if not np.isnan(y1).all() and not np.isnan(y2).all():
            fills = [dict(y1=y1, y2=y2, where=where, alpha=0.25, color=color)] #more efficient than appends that builds dynamically
    return fills


def breakouts_plots(df, breakout_output):
    bull_signals = breakout_output["bull_signals"]
    bear_signals = breakout_output["bear_signals"]
    
    signals_df = pd.DataFrame(index=df.index, data={"BullPrice": np.nan, "BearPrice": np.nan})
    for sig in bull_signals:
        if sig["when"] in signals_df.index:
            signals_df.loc[sig["when"], "BullPrice"] = sig["price"]

    for sig in bear_signals:
        if sig["when"] in signals_df.index:
            signals_df.loc[sig["when"], "BearPrice"] = sig["price"]

    addplots = []
    print(f"{Fore.BLUE}Found {len(bull_signals)} bullish and {len(bear_signals)} bearish breakouts.{Fore.RESET}")
    if not signals_df["BullPrice"].isna().all():
        addplots.append(mpf.make_addplot(signals_df["BullPrice"], type="scatter", marker="^",
                                         markersize=100, color="green"))
    if not signals_df["BearPrice"].isna().all():
        addplots.append(mpf.make_addplot(signals_df["BearPrice"], type="scatter", marker="v",
                                         markersize=100, color="red"))
    print(f"{Fore.BLUE}Addplots for signals: {addplots}{Fore.RESET}")
    return addplots