import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

class StockSentimentAnalyzer:
    def __init__(self, stock_files: List[str], sentiment_file: str):
        self.stock_files = stock_files
        self.sentiment_file = sentiment_file
        
        self.stock_df = self._load_stocks()
        self.sentiment_df = self._load_sentiment()

    def _load_stocks(self):
        dfs = []
        for f in self.stock_files:
            df = pd.read_csv(f)
            ticker = f.split('/')[-1].split('_')[0].upper()
            df['ticker'] = ticker
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        combined.dropna(subset=['Date', 'Close'], inplace=True)
        return combined

    def _load_sentiment(self):
        df = pd.read_csv(self.sentiment_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date', 'sentiment_score'], inplace=True)
        return df

    def calculate_daily_returns(self):
        self.stock_df = self.stock_df.sort_values(['ticker', 'Date'])
        self.stock_df['daily_return'] = self.stock_df.groupby('ticker')['Close'].pct_change()
        return self.stock_df[['Date', 'ticker', 'daily_return']].dropna()

    def merge_returns_sentiment(self):
        returns = self.calculate_daily_returns()
        merged = pd.merge(returns, self.sentiment_df, on='Date', how='inner')
        return merged

    def correlation_per_ticker(self):
        merged = self.merge_returns_sentiment()
        correlations = merged.groupby('ticker').apply(
            lambda df: df['daily_return'].corr(df['sentiment_score'])
        ).to_dict()
        return correlations, merged

    def plot_results(self):
        correlations, merged = self.correlation_per_ticker()
        print("Correlation coefficients between stock daily returns and sentiment scores:")
        for ticker, corr in correlations.items():
            print(f"{ticker}: {corr:.4f}")

        tickers = merged['ticker'].unique()
        n = len(tickers)
        fig, axs = plt.subplots(n, 2, figsize=(14, 4 * n), sharex=True)

        for i, ticker in enumerate(tickers):
            df = merged[merged['ticker'] == ticker]
            sns.lineplot(data=df, x='Date', y='daily_return', ax=axs[i,0])
            axs[i,0].set_title(f"{ticker} Daily Returns")
            axs[i,0].set_ylabel("Return")

            sns.lineplot(data=df, x='Date', y='sentiment_score', ax=axs[i,1], color='orange')
            axs[i,1].set_title(f"{ticker} Sentiment Scores")
            axs[i,1].set_ylabel("Sentiment Score")

        plt.tight_layout()
        plt.show()
