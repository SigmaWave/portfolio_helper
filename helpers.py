import yfinance as yf
import pandas as pd
import os, time
from colorama import Fore, Back, Style
import numpy as np
import mplfinance as mpf

START = "2023-01-01"
END = "2025-01-19"
DATA_DIR = "./StockData"

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

    print(f"{Fore.GREEN} Imported data for {symbol} and saved to {out_path}{Style.RESET_ALL}")
    return df

@timer
def load_data(symbol, START, END, data_dir):
    """
    Load data from CSV if available, otherwise download it.
    """
    path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(path):
        print(f"{Fore.RED}File {symbol}.csv not found — importing now …{Style.RESET_ALL}")
        return import_data(symbol, START, END, data_dir)

    data = pd.read_csv(path, index_col="Date", parse_dates=True)
    print(f"{Fore.GREEN}Loaded {symbol} from {path}{Style.RESET_ALL}")
    return data

@timer
def breakout_finder(
    df: pd.DataFrame,
    prd: int = 5,
    bo_len: int = 200,
    cwidthu: float = 3.0/100,  # “Threshold Rate %” in fraction (e.g., 3% -> 0.03)
    mintest: int = 2,
):
    """
    df: pandas DataFrame with columns: Open, High, Low, Close (Volume optional).
    Returns:
        {
            "bull_signals": list of dicts,
            "bear_signals": list of dicts,
            "bull_boxes":   list of dicts,  # levels for the drawn box/lines
            "bear_boxes":   list of dicts,

            {'bull_signals': [{'when': Timestamp, 'price': np.float64}, ...],
            'bear_signals': [{'when': Timestamp, 'price': np.float64}, ...],
            'bull_boxes': [{'bar_end': Timestamp, 'bar_start': Timestamp, 'top': float, 'bottom': float}, ...],
            'bear_boxes': [{'bar_end': Timestamp, 'bar_start': Timestamp, 'top': float, 'bottom': float}, ...]}
        }
    """

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(df)
    idx = df.index

    # --- Helpers: Pine-like pivothigh/pivotlow (value at center bar when confirmed) ---
    # pivothigh(left=prd,right=prd): bar t is a pivot high if it's the max over [t-prd .. t+prd]
    # We return at position t (confirmed only when we reach t+prd), else np.nan.
    def pivot_extrema(series, left, right, mode="high"):
        arr = series.values
        out = np.full_like(arr, np.nan, dtype=float)
        for t in range(left, len(arr) - right):
            window = arr[t - left : t + right + 1]
            if mode == "high":
                if arr[t] == np.max(window):
                    out[t] = arr[t]
            else:
                if arr[t] == np.min(window):
                    out[t] = arr[t]
        return out

    ph_arr = pivot_extrema(df["High"], prd, prd, mode="high")  # pivot highs at t
    pl_arr = pivot_extrema(df["Low"],  prd, prd, mode="low")   # pivot lows  at t

    # Track pivot highs/lows like Pine arrays (most-recent first)
    phval, phloc = [], []  # values and integer indices
    plval, plloc = [], []

    bull_signals, bear_signals = [], []
    bull_boxes,    bear_boxes  = [], []

    for i in range(n):
        # --- width calc like Pine ---
        lll = max(min(i, 300), 1)                # cap recent lookback to <=300 bars
        h_  = np.max(h[i-lll:i+1]) if i-lll >= 0 else np.max(h[:i+1])
        l_  = np.min(l[i-lll:i+1]) if i-lll >= 0 else np.min(l[:i+1])
        chwidth = (h_ - l_) * cwidthu

        # --- register pivot highs/lows at their confirmed bar (i) ---
        if not np.isnan(ph_arr[i]):
            # Pine stores location at (bar_index - prd) for pivothigh(prd,prd)
            phval.insert(0, float(ph_arr[i]))
            phloc.insert(0, i)  # using confirmed index; close enough for Python port

            # cleanup anything older than bo_len bars
            j = 1
            while j < len(phloc):
                if i - phloc[j] > bo_len:
                    phloc.pop(j); phval.pop(j)
                else:
                    j += 1

        if not np.isnan(pl_arr[i]):
            plval.insert(0, float(pl_arr[i]))
            plloc.insert(0, i)

            j = 1
            while j < len(plloc):
                if i - plloc[j] > bo_len:
                    plloc.pop(j); plval.pop(j)
                else:
                    j += 1

        # --- Highest/lowest of last prd bars, previous value (shifted by 1) ---
        # In Pine: hgst = highest(prd)[1], lwst = lowest(prd)[1]
        if i >= 1:
            start = max(0, i - prd)
            hgst = np.max(h[start:i]) if i - start > 0 else -np.inf
            lwst = np.min(l[start:i]) if i - start > 0 else  np.inf
        else:
            hgst, lwst = -np.inf, np.inf

        # =========================
        #   Bullish "cup" breakout
        # =========================
        bomax = np.nan
        bostart = i
        num = 0
        if len(phval) >= mintest and c[i] > o[i] and c[i] > hgst:
            bomax = phval[0]
            xx = 0
            # grow xx while prior pivot highs are below current close
            for x in range(len(phval)):
                if phval[x] >= c[i]:
                    break
                xx = x
                bomax = max(bomax, phval[x])

            if xx >= mintest and o[i] <= bomax:
                # count tests within [bomax - chwidth, bomax]
                for x in range(xx + 1):
                    if (phval[x] <= bomax) and (phval[x] >= bomax - chwidth):
                        num += 1
                        bostart = phloc[x]
                if num < mintest or hgst >= bomax:
                    bomax = np.nan

        if not np.isnan(bomax) and num >= mintest:
            # emit a bull signal and the “box” (top=bomax, bottom=bomax-chwidth) from bostart..i
            bull_signals.append({"when": idx[i], "price": c[i]})
            bull_boxes.append({
                "bar_end":   idx[i],
                "bar_start": idx[bostart],
                "top":       float(bomax),
                "bottom":    float(bomax - chwidth),
            })

        # =========================
        #   Bearish "cup" breakdown
        # =========================
        bomin = np.nan
        bostart = i
        num1 = 0
        if len(plval) >= mintest and c[i] < o[i] and c[i] < lwst:
            bomin = plval[0]
            xx = 0
            for x in range(len(plval)):
                if plval[x] <= c[i]:
                    break
                xx = x
                bomin = min(bomin, plval[x])

            if xx >= mintest and o[i] >= bomin:
                for x in range(xx + 1):
                    if (plval[x] >= bomin) and (plval[x] <= bomin + chwidth):
                        num1 += 1
                        bostart = plloc[x]
                if num1 < mintest or lwst <= bomin:
                    bomin = np.nan

        if not np.isnan(bomin) and num1 >= mintest:
            bear_signals.append({"when": idx[i], "price": c[i]})
            bear_boxes.append({
                "bar_end":   idx[i],
                "bar_start": idx[bostart],
                "top":       float(bomin + chwidth),
                "bottom":    float(bomin),
            })

    return {
        "bull_signals": bull_signals,
        "bear_signals": bear_signals,
        "bull_boxes":   bull_boxes,
        "bear_boxes":   bear_boxes,
    }


def plot_breakouts(df, breakout_output):
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



# --- Plot the candlestick chart with all overlays ---
data = load_data('BNP.PA', START, END, DATA_DIR)
breakout_output = breakout_finder(data, prd=5, bo_len=100, cwidthu=0.04, mintest=2)
addplots = plot_breakouts(data, breakout_output)

fills = []
n = len(data.index)
print(f"{Fore.BLUE}Bullish boxes: {breakout_output['bull_boxes']}{Fore.RESET}")
if breakout_output["bull_boxes"]:
    for b in breakout_output["bull_boxes"]:
        where = (data.index >= pd.Timestamp(b["bar_start"])) & (data.index <= pd.Timestamp(b["bar_end"]))
        y1 = np.full(n, float(b["top"]),    dtype=float)
        y2 = np.full(n, float(b["bottom"]), dtype=float)
        if not np.isnan(y1).all() and not np.isnan(y2).all():
            fills.append(dict(y1=y1, y2=y2, where=where, alpha=0.25, color='g'))

print(f"{Fore.BLUE}Bearish boxes: {breakout_output['bear_boxes']}{Fore.RESET}")
if breakout_output["bear_boxes"]:
    for b in breakout_output["bear_boxes"]:
        where = (data.index >= pd.Timestamp(b["bar_start"])) & (data.index <= pd.Timestamp(b["bar_end"]))
        y1 = np.full(n, float(b["top"]),    dtype=float)
        y2 = np.full(n, float(b["bottom"]), dtype=float)
        if not np.isnan(y1).all() and not np.isnan(y2).all():
            fills.append(dict(y1=y1, y2=y2, where=where, alpha=0.25, color='r'))

print(f"{Fore.BLUE}Fills: {fills}{Fore.RESET}")

mpf.plot(
    data,
    type="candle",
    volume=True,
    mav=(50, 100, 200),
    style="yahoo",
    addplot=addplots,
    fill_between=fills,
    title='BNP.PA with Breakout Signals',
    figratio=(14, 8),
)

mpf.show()


'''

[{'bar_end': Timestamp('2023-01-02 00:00:00'), 'bar_start': Timestamp('2022-05-30 00:00:00'), 'top': 45.059489098220304, 'bottom': 44.484412783206885}

dates_df     = pd.DataFrame(daily.index)
buy_date     = pd.Timestamp('2019-11-06')
sell_date    = pd.Timestamp('2019-11-19')

where_values = pd.notnull(dates_df[ (dates_df>=buy_date) & (dates_df <= sell_date) ])['Date'].values

y1values = daily['Close'].values
y2value  = daily['Low'].min()

mpf.plot(daily,figscale=0.7,
         fill_between=dict(y1=y1values,y2=y2value,where=where_values,alpha=0.5,color='g')
        )
'''


# TODO: Merge overlapping boxes into a bigger box