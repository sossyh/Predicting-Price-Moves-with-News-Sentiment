import pandas as pd
import matplotlib.pyplot as plt

class TimeSeriesAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.df["date"]):
            self.df["date"] = pd.to_datetime(self.df["date"], errors='coerce')
        self.df = self.df.dropna(subset=["date"])

    def articles_per_day(self):
        daily_counts = self.df.groupby(self.df["date"].dt.date).size()
        return daily_counts

    def plot_articles_per_day(self):
        daily_counts = self.articles_per_day()
        plt.figure(figsize=(14, 5))
        daily_counts.plot()
        plt.title("Articles Published per Day")
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def publishing_hours_distribution(self):
        self.df["hour"] = self.df["date"].dt.hour
        return self.df["hour"].value_counts().sort_index()

    def plot_publishing_hours_distribution(self):
        hour_counts = self.publishing_hours_distribution()
        plt.figure(figsize=(10, 4))
        hour_counts.plot(kind='bar')
        plt.title("Distribution of Publishing Hours")
        plt.xlabel("Hour of Day")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
