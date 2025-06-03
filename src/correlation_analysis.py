import os
import pandas as pd
from textblob import TextBlob
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def compute_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

def preprocess_news(news_filepath):
    news_df = pd.read_csv(news_filepath)
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce').dt.date
    news_df.dropna(subset=['date'], inplace=True)
    news_df['sentiment'] = news_df['headline'].apply(compute_sentiment)

    # Map 'stock' column to full stock ticker names for alignment
    stock_map = {
        'A': 'AAPL', 'AMZN': 'AMZN', 'GOOG': 'GOOG', 'META': 'META',
        'MSFT': 'MSFT', 'NVDA': 'NVDA', 'TSLA': 'TSLA'
    }
    news_df['stock'] = news_df['stock'].map(stock_map)
    return news_df

def preprocess_stock(stock_filepath):
    stock_df = pd.read_csv(stock_filepath)
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
    stock_df = stock_df[['Date', 'Adj Close']].sort_values('Date')
    stock_df['Return'] = stock_df['Adj Close'].pct_change()
    stock_df.dropna(inplace=True)
    return stock_df

def merge_news_and_stock(news_df, stock_df, ticker):
    news_ticker_df = news_df[news_df['stock'] == ticker]
    sentiment_df = news_ticker_df.groupby('date')['sentiment'].mean().reset_index()
    sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

    merged = pd.merge(stock_df, sentiment_df, on='Date', how='inner')
    return merged

def analyze_correlation(merged_df, ticker):
    if len(merged_df) < 2:
        print(f"[{ticker}] Not enough data to compute correlation.")
        return None

    corr, pval = pearsonr(merged_df['Return'], merged_df['sentiment'])
    print(f"[{ticker}] Pearson Correlation: {corr:.4f}, p-value: {pval:.4e}")

    # Optional: plot
    plt.scatter(merged_df['sentiment'], merged_df['Return'])
    plt.title(f"Sentiment vs Return for {ticker}")
    plt.xlabel("Average Daily Sentiment")
    plt.ylabel("Daily Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return corr, pval

def run_analysis(stock_files, news_filepath):
    news_df = preprocess_news(news_filepath)

    for stock_file in stock_files:
        ticker = os.path.basename(stock_file).split('_')[0]
        stock_df = preprocess_stock(stock_file)
        merged_df = merge_news_and_stock(news_df, stock_df, ticker)
        analyze_correlation(merged_df, ticker)
