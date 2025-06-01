# src/pynance_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import pynance as pn
import os


def load_stock_data(file_path):
    """
    Loads historical stock data from a CSV file.
    """
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    return df


def add_technical_indicators(df):
    """
    Adds simple financial metrics using PyNance.
    """
    # Moving Averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # RSI (simplified version using price changes)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def plot_price_with_indicators(df, symbol):
    """
    Plots the closing price with SMA and RSI.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot Closing Price with SMAs
    ax1.plot(df.index, df["Close"], label="Close", color="black")
    ax1.plot(df.index, df["SMA_20"], label="SMA 20", color="blue", linestyle="--")
    ax1.plot(df.index, df["SMA_50"], label="SMA 50", color="orange", linestyle="--")
    ax1.set_title(f"{symbol} Price & Moving Averages")
    ax1.legend()
    ax1.grid(True)

    # Plot RSI
    ax2.plot(df.index, df["RSI"], label="RSI", color="green")
    ax2.axhline(70, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(30, color="blue", linestyle="--", alpha=0.5)
    ax2.set_title(f"{symbol} RSI")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
