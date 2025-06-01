# src/pynance_analysis.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import talib

def load_stock_data(file_path):
    """
    Loads stock data from a CSV file.
    Assumes the file has columns: Date, Open, High, Low, Close, Adj Close, Volume
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def add_technical_indicators(df):
    """
    Adds SMA, RSI, and MACD indicators to the dataframe.
    """
    close = df['Close'].values

    # Moving Averages
    df['SMA_20'] = talib.SMA(close, timeperiod=20)
    df['SMA_50'] = talib.SMA(close, timeperiod=50)

    # RSI
    df['RSI'] = talib.RSI(close, timeperiod=14)

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(close)
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist

    return df

def plot_price_with_indicators(df, symbol):
    """
    Plots Close price, SMAs, RSI, and MACD in subplots.
    """
    plt.figure(figsize=(14, 10))

    # Price + SMA
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['SMA_20'], label='SMA 20', linestyle='--')
    plt.plot(df.index, df['SMA_50'], label='SMA 50', linestyle='--')
    plt.title(f"{symbol} Price with SMAs")
    plt.legend()
    plt.grid()

    # RSI
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['RSI'], color='purple')
    plt.axhline(70, linestyle='--', color='red')
    plt.axhline(30, linestyle='--', color='green')
    plt.title("RSI (14)")
    plt.grid()

    # MACD
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['MACD'], label='MACD', color='black')
    plt.plot(df.index, df['MACD_Signal'], label='Signal', color='orange')
    plt.bar(df.index, df['MACD_Hist'], label='Hist', color='gray', alpha=0.5)
    plt.title("MACD")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
