import pandas as pd

class DateAligner:
    def __init__(self, news_df: pd.DataFrame, stock_df: pd.DataFrame):
        self.news_df = news_df.copy()
        self.stock_df = stock_df.copy()

    def preprocess(self):
        # Ensure datetime format
        self.news_df["date"] = pd.to_datetime(self.news_df["date"], errors='coerce').dt.date
        self.stock_df["Date"] = pd.to_datetime(self.stock_df["Date"], errors='coerce').dt.date

        # Drop rows with missing dates
        self.news_df.dropna(subset=["date"], inplace=True)
        self.stock_df.dropna(subset=["Date"], inplace=True)

    def aggregate_news_by_date(self):
        # Aggregate all headlines per day
        self.news_df = self.news_df.groupby("date")["headline"].apply(lambda x: " ".join(x)).reset_index()

    def align(self):
        self.preprocess()
        self.aggregate_news_by_date()

        # Merge on date
        merged = pd.merge(self.news_df, self.stock_df, left_on="date", right_on="Date", how="inner")

        return merged
