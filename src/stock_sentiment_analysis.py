# src/stock_sentiment_analysis.py

import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import chardet


class StockSentimentAnalyzer:
    def __init__(self, stock_files: List[str], news_file: str):
        self.stock_files = stock_files
        self.news_file = news_file
        
        self.stock_df = self._load_stock_data()
        self.sentiment_df = self._analyze_news_sentiment()
    
    def _detect_encoding(self, file_path):
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        return result['encoding']
    
    def _load_stock_data(self):
        dfs = []
        for file in self.stock_files:
            # Detect encoding first
            encoding = self._detect_encoding(file)
            df = pd.read_csv(file, encoding=encoding)
            # Convert to datetime and extract just the date part
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
            df = df[['Date', 'Close']].dropna()
            ticker = file.split("/")[-1].split("_")[0].upper()
            df['ticker'] = ticker
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        return combined

    def _analyze_news_sentiment(self):
        # Detect encoding first
        encoding = self._detect_encoding(self.news_file)
        df = pd.read_csv(self.news_file, encoding=encoding)
        df = df[['date', 'headline']].dropna()
        # Convert to datetime, handle timezone, then extract just the date part
        df['Date'] = pd.to_datetime(df['date'], errors='coerce')
        # Remove timezone if present
        if pd.api.types.is_datetime64tz_dtype(df['Date']):
            df['Date'] = df['Date'].dt.tz_convert(None)
        # Extract just the date component
        df['Date'] = df['Date'].dt.date
        df.dropna(subset=['Date'], inplace=True)

        # Handle potential encoding issues in text processing
        df['polarity'] = df['headline'].apply(
            lambda x: TextBlob(str(x).encode('ascii', errors='ignore').decode('ascii')).sentiment.polarity
        )
        daily_sentiment = df.groupby('Date')['polarity'].mean().reset_index()
        daily_sentiment.rename(columns={'polarity': 'sentiment_score'}, inplace=True)
        return daily_sentiment

    def calculate_daily_returns(self):
        df = self.stock_df.copy()
        df = df.sort_values(['ticker', 'Date'])
        df['daily_return'] = df.groupby('ticker')['Close'].pct_change()
        return df.dropna(subset=['daily_return'])

    def merge_and_correlate(self):
        returns_df = self.calculate_daily_returns()
        # Ensure both Date columns are date objects (not datetime)
        returns_df['Date'] = pd.to_datetime(returns_df['Date']).dt.date
        self.sentiment_df['Date'] = pd.to_datetime(self.sentiment_df['Date']).dt.date
        
        merged_df = pd.merge(returns_df, self.sentiment_df, on='Date', how='inner')
        correlations = merged_df.groupby('ticker').apply(
            lambda group: group['daily_return'].corr(group['sentiment_score'])
        ).to_dict()
        return correlations, merged_df

    def plot_results(self):
        correlations, merged_df = self.merge_and_correlate()
        print("Correlation coefficients between stock returns and sentiment scores:\n")
        for ticker, corr in correlations.items():
            print(f"{ticker}: {corr:.4f}")

        # Convert back to datetime for plotting
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        
        tickers = merged_df['ticker'].unique()
        n = len(tickers)
        fig, axs = plt.subplots(n, 2, figsize=(14, 4 * n), sharex=True)

        for i, ticker in enumerate(tickers):
            df = merged_df[merged_df['ticker'] == ticker]
            sns.lineplot(data=df, x='Date', y='daily_return', ax=axs[i, 0])
            axs[i, 0].set_title(f'{ticker} Daily Return')
            axs[i, 0].set_ylabel('Return')

            sns.lineplot(data=df, x='Date', y='sentiment_score', ax=axs[i, 1], color='orange')
            axs[i, 1].set_title(f'{ticker} Sentiment Score')
            axs[i, 1].set_ylabel('Sentiment Score')

        plt.tight_layout()
        plt.show()