# sentiment_analysis.py

import pandas as pd
from textblob import TextBlob

class HeadlineSentimentAnalyzer:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.df = self._load_and_prepare()

    def _load_and_prepare(self):
        df = pd.read_csv(self.csv_file)
        if 'headline' not in df.columns:
            raise ValueError("The input file must contain a 'headline' column.")
        df.dropna(subset=['headline'], inplace=True)
        return df

    def analyze_sentiment(self):
        def get_polarity(text):
            return TextBlob(text).sentiment.polarity
        
        def get_sentiment_label(polarity):
            if polarity > 0.1:
                return 'positive'
            elif polarity < -0.1:
                return 'negative'
            else:
                return 'neutral'

        self.df['polarity'] = self.df['headline'].apply(get_polarity)
        self.df['sentiment'] = self.df['polarity'].apply(get_sentiment_label)
        return self.df[['headline', 'polarity', 'sentiment']]

    def sentiment_distribution(self):
        return self.df['sentiment'].value_counts()
