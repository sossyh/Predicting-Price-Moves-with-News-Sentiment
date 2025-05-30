import pandas as pd

class HistoricalStockAnalyzer:
    def init(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path, parse_dates=["Date"])
        return self.df

    def get_summary_stats(self):
        return self.df.describe()

    def get_price_trends(self):
        return self.df[["Date", "Close"]].set_index("Date")

    def get_volume_trends(self):
        return self.df[["Date", "Volume"]].set_index("Date")