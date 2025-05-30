# src/technical_indicators.py

import pandas as pd
import talib
import os

class TechnicalIndicatorAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.symbol = os.path.basename(file_path).split("_")[0]
        self.df = pd.read_csv(file_path)
        self.prepare_data()

    def prepare_data(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date')
        self.df.set_index('Date', inplace=True)

    def add_indicators(self):
        close = self.df['Close'].values

        # Simple Moving Averages
        self.df['SMA_20'] = talib.SMA(close, timeperiod=20)
        self.df['SMA_50'] = talib.SMA(close, timeperiod=50)

        # Relative Strength Index
        self.df['RSI'] = talib.RSI(close, timeperiod=14)

        # MACD
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.df['MACD'] = macd
        self.df['MACD_Signal'] = macdsignal
        self.df['MACD_Hist'] = macdhist

    def get_indicated_data(self):
        self.add_indicators()
        return self.df
