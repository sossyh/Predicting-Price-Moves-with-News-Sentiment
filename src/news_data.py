import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from textblob import TextBlob

class NewsAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

        # Parse all dates first (coerce errors)
        self.df["date"] = pd.to_datetime(self.df["date"], errors='coerce')

        # Remove timezone info (convert all to timezone-naive)
        self.df["date"] = self.df["date"].apply(
            lambda dt: dt.tz_convert(None) if pd.notna(dt) and dt.tzinfo is not None else dt
        )

        # Report conversion issues
        na_count = self.df["date"].isna().sum()
        if na_count > 0:
            print(f"Warning: {na_count} dates couldn't be converted and were set to NaT")
            print("Problematic rows:")
            print(self.df[self.df["date"].isna()])

    def check_dates(self):
        print("Date column type:", self.df["date"].dtype)
        print("First 5 dates:")
        print(self.df["date"].head())
        print("Number of null dates:", self.df["date"].isna().sum())
        print("Sample problematic dates:")
        print(self.df[self.df["date"].isna()]["date"].head())

    def basic_stats(self):
        self.df["headline_length"] = self.df["headline"].str.len()
        return self.df["headline_length"].describe()

    def check_date_column(self):
        print("Type of 'date' column:", self.df["date"].dtype)
        print("First few entries:")
        print(self.df["date"].head())

    def articles_per_publisher(self):
        return self.df["publisher"].value_counts()

    def articles_per_day(self):
        valid_dates = self.df.dropna(subset=['date'])
        if len(valid_dates) == 0:
            raise ValueError("No valid dates found in the dataset")
        return valid_dates.groupby(valid_dates["date"].dt.date).size()

    def publishing_hours_distribution(self):
        self.df["hour"] = self.df["date"].dt.hour
        return self.df["hour"].value_counts().sort_index()

    def topic_modeling(self, n_topics=5):
        self.df["cleaned"] = self.df["headline"].astype(str).str.lower()
        self.df["cleaned"] = self.df["cleaned"].apply(lambda x: re.sub(r"[^a-z\s]", "", x))

        vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=10)
        X = vectorizer.fit_transform(self.df["cleaned"])

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)

        topics = []
        for idx, topic in enumerate(lda.components_):
            top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
            topics.append((idx+1, top_words))
        return topics

    def get_domain_stats(self):
        self.df["domain"] = self.df["publisher"].str.extract(r"@([\w\.-]+)")
        return self.df["domain"].value_counts()

    def sentiment_analysis(self):
        def get_sentiment(text):
            analysis = TextBlob(str(text))
            polarity = analysis.sentiment.polarity
            if polarity > 0.1:
                return "positive"
            elif polarity < -0.1:
                return "negative"
            else:
                return "neutral"

        self.df["sentiment"] = self.df["headline"].apply(get_sentiment)
        return self.df["sentiment"].value_counts()

    def plot_sentiment_distribution(self):
        sentiment_counts = self.sentiment_analysis()
        sentiment_counts.plot(kind="bar", color=["green", "red", "gray"])
        plt.title("Sentiment Distribution of News Headlines")
        plt.xlabel("Sentiment")
        plt.ylabel("Number of Headlines")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()
