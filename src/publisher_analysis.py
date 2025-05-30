# src/publisher_analysis.py

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class PublisherAnalyzer:
    def __init__(self, data, publisher_col="publisher", headline_col="headline"):
        self.data = data
        self.publisher_col = publisher_col
        self.headline_col = headline_col
        self.publisher_stats = None

    def top_publishers(self, top_n=10):
        counts = self.data[self.publisher_col].value_counts().head(top_n)
        self.publisher_stats = counts
        return counts

    def plot_top_publishers(self, top_n=10):
        top_publishers = self.top_publishers(top_n)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_publishers.values, y=top_publishers.index, palette="viridis")
        plt.title("Top Publishers by Article Count")
        plt.xlabel("Number of Articles")
        plt.ylabel("Publisher")
        plt.tight_layout()
        plt.show()

    def analyze_email_domains(self):
        publishers = self.data[self.publisher_col].dropna().astype(str)
        email_publishers = publishers[publishers.str.contains("@")]
        domains = email_publishers.str.extract(r'@(.+)$')[0]
        domain_counts = domains.value_counts()
        return domain_counts

    def compare_publisher_content(self, sample_size=1000):
        """
        Returns a DataFrame with publishers and their most frequent keywords
        to examine if their content differs.
        """
        from sklearn.feature_extraction.text import CountVectorizer

        sample = self.data.dropna(subset=[self.publisher_col, self.headline_col]).sample(n=min(sample_size, len(self.data)), random_state=42)
        grouped = sample.groupby(self.publisher_col)[self.headline_col].apply(lambda x: " ".join(x)).reset_index()

        vectorizer = CountVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(grouped[self.headline_col])
        features = vectorizer.get_feature_names_out()
        keywords_df = pd.DataFrame(X.toarray(), columns=features, index=grouped[self.publisher_col])

        top_keywords = keywords_df.apply(lambda row: row.nlargest(5).index.tolist(), axis=1)
        return top_keywords.reset_index(name='Top Keywords')
