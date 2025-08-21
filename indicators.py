import pandas as pd
import mplfinance as mpf
import numpy as np
from config import START, END

def mav_plot(df, lengths=(50, 100, 200)):
    addplots = []
    mav_colors = ['blue', 'orange', 'green']
    for i, length in enumerate(lengths):
        mav = df['Close'].rolling(window=length).mean()
        addplots.append(mpf.make_addplot(mav.loc[START:END], color=mav_colors[i%3]))
    return addplots



def support_resistance_breakout(df,
    prd = 5,
    bo_len = 200,
    cwidthu = 3.0/100,  # “Threshold Rate %” in fraction (e.g., 3% -> 0.03)
    mintest = 2,
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
