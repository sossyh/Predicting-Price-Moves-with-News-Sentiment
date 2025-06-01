# stock_data.py

import pandas as pd
import os

def load_stock_file(filepath):
    """
    Load a single stock CSV file, parse dates, compute daily returns.
    """
    df = pd.read_csv(filepath, parse_dates=['Date'])
    # Normalize to date only
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    # Compute daily returns from 'Close' price
    df['daily_return'] = df['Close'].pct_change()
    # Extract stock symbol from filename (assumes format SYMBOL_historical_data.csv)
    symbol = os.path.basename(filepath).split('_')[0]
    df['stock'] = symbol
    # Drop rows with NaN returns (first row or missing data)
    df = df.dropna(subset=['daily_return']).reset_index(drop=True)
    return df

def load_stock_data(filepaths):
    """
    Load multiple stock CSV files and concatenate into one DataFrame.
    """
    all_frames = []
    for fp in filepaths:
        all_frames.append(load_stock_file(fp))
    combined = pd.concat(all_frames, ignore_index=True)
    # Ensure sorting by date for each stock
    combined = combined.sort_values(['stock','Date']).reset_index(drop=True)
    return combined
