from src.historical import HistoricalStockAnalyzer
from src.news import NewsAnalyzer
import os

# Run historical stock analysis
stock_files = [
    "data/yfinance/AAPL_historical_data.csv",
    "data/yfinance/AMZN_historical_data.csv",
    "data/yfinance/MSFT_historical_data.csv",
    "data/yfinance/GOOGL_historical_data.csv",
    "data/yfinance/TSLA_historical_data.csv"
]

for stock_file in stock_files:
    stock_name = os.path.basename(stock_file).split("_")[0]
    analyzer = HistoricalStockAnalyzer(stock_file)
    analyzer.load_data()
    print(f"=== {stock_name} Summary ===")
    print(analyzer.get_summary_stats())

# Run news analysis
news_file = "data/raw_analyst_ratings.csv"
news_analyzer = NewsAnalyzer(news_file)

print("\n=== Headline Length Stats ===")
print(news_analyzer.basic_stats())

print("\n=== Articles per Publisher ===")
print(news_analyzer.articles_per_publisher().head())

print("\n=== Topics in News Headlines ===")
topics = news_analyzer.topic_modeling()
for tid, words in topics:
    print(f"Topic {tid}: {', '.join(words)}")
