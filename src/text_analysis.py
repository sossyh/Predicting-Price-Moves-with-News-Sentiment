# src/text_analysis.py

import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

class TextAnalyzer:
    def __init__(self, texts):
        self.raw_texts = texts
        self.cleaned_texts = self._clean_texts(texts)
        self.vectorizer = None
        self.tfidf_matrix = None

    def _clean_texts(self, texts):
        stop_words = set(stopwords.words('english'))
        cleaned = []
        for text in texts:
            text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # Remove punctuation/numbers
            tokens = text.lower().split()
            tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
            cleaned.append(" ".join(tokens))
        return cleaned

    def extract_keywords(self, top_n=20):
        self.vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, max_features=1000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_texts)
        feature_array = self.vectorizer.get_feature_names_out()
        tfidf_sorting = self.tfidf_matrix.toarray().sum(axis=0).argsort()[::-1]

        top_keywords = [feature_array[i] for i in tfidf_sorting[:top_n]]
        return top_keywords

    def perform_topic_modeling(self, num_topics=5, num_words=10):
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=10, max_features=5000)
        X = self.vectorizer.fit_transform(self.cleaned_texts)

        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, learning_method='batch')
        lda.fit(X)

        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_features = topic.argsort()[::-1][:num_words]
            topic_keywords = [self.vectorizer.get_feature_names_out()[i] for i in top_features]
            topics.append(f"Topic #{topic_idx + 1}: " + ", ".join(topic_keywords))

        return topics