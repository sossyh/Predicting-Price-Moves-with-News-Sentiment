import pandas as pd
from textblob import TextBlob

def load_news_data(file_path):
    df = pd.read_csv(file_path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df

def compute_sentiment(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

def analyze_news_sentiment(df, stock):
    df_stock = df[df["stock"] == stock].copy()
    df_stock["Sentiment"] = df_stock["headline"].apply(compute_sentiment)
    daily_sentiment = df_stock.groupby("date")["Sentiment"].mean().reset_index()
    daily_sentiment.columns = ["date", "Avg_Sentiment"]
    return daily_sentiment
