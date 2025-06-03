import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

class DailyReturnCalculator:
    def __init__(self, stock_files):
        self.stock_files = stock_files
        self.stock_df = self._load_and_clean_data()

    def _load_and_clean_data(self):
        dfs = []
        for file in self.stock_files:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date  # Keep only date
            df = df[['Date', 'Close']].dropna()
            ticker = os.path.basename(file).split("_")[0].upper()
            df['ticker'] = ticker
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def calculate_daily_returns(self):
        df = self.stock_df.copy()
        df = df.sort_values(['ticker', 'Date'])
        df['daily_return'] = df.groupby('ticker')['Close'].pct_change()
        return df.dropna(subset=['daily_return'])

    def show_sample(self, n=5):
        df = self.calculate_daily_returns()
        print(df.head(n))
        return df
