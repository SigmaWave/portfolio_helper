import yfinance as yf
import pandas as pd
import os, time
from colorama import Fore, Back, Style

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
