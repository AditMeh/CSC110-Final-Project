import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from string import punctuation
from typing import List, Set
import os.path
from os import path


class DataLoader:
    def __init__(self, FILEPATH: str):
        self.FILEPATH = FILEPATH

        if not path.exists("nltk_data"):
            print("Downloading corpus")
            nltk.data.path.append('nltk_data/')
            nltk.download('stopwords', download_dir="nltk_data/")
            nltk.download('punkt', download_dir="nltk_data/")
        else:
            print("Corpus already exists")
            nltk.data.path.append('nltk_data/')

        self.stop_words = set(stopwords.words('english'))

    def prepare_data(self, num_tweets: int):
        df = pd.read_csv(self.FILEPATH)

        labels = df.iloc[:, 0]
        tweets = df.iloc[:, 1]

        train_x = []
        train_y = []
        for i in range(num_tweets):
            filtered_tweet = self.filter_tweet(tweets[i])
            train_x.append(filtered_tweet)
            train_y.append(int(labels[i]))

        return train_x, train_y

    def filter_tweet(self, tweet: List[str]):
        tokenized_tweet = self.tokenize_tweet(tweet)
        link_filtered = self.filter_links(tokenized_tweet)
        periods_filtered = self.filter_periods(link_filtered)
        at_sign_filtered = self.filter_at_signs(periods_filtered)
        strip_punctuation = self.strip_punctuation(at_sign_filtered)
        non_ascii_filtered = self.filter_nonascii(strip_punctuation)
        stopwords_filtered = self.filter_stopwords(non_ascii_filtered)
        tweet_to_lowercase = self.tweet_to_lowercase(stopwords_filtered)
        removed_numbers = self.remove_numbers(tweet_to_lowercase)
        return removed_numbers

    def tokenize_tweet(self, tweet: str):
        return tweet.split()

    def filter_links(self, tweet: List[str]):
        copy_tweet = tweet.copy()

        for element in tweet:
            if "http" in element:
                copy_tweet.remove(element)
        return copy_tweet

    def filter_periods(self, tweet: List[str]):
        # replace all periods with spaces, double spaces will be split anyways
        for i in range(len(tweet)):
            if tweet[i] in {".", ","}:
                tweet[i] = " "
        return " ".join(tweet).split()

    def strip_punctuation(self, tweet: List[str]):

        copy_tweet = tweet.copy()

        for i in range(len(tweet)):
            copy_tweet[i] = ''.join(char for char in tweet[i] if char not in punctuation or char == "#")
        return copy_tweet

    def filter_at_signs(self, tweet: List[str]):
        copy_tweet = tweet.copy()

        for element in tweet:
            if "@" in element:
                copy_tweet.remove(element)
        return copy_tweet

    def filter_nonascii(self, tweet: List[str]):
        copy_tweet = tweet.copy()

        for element in tweet:
            if not self.is_ascii(element):
                copy_tweet.remove(element)
        return copy_tweet

    def filter_stopwords(self, tweet: List[str]):
        self.stop_words.add("RT")
        copy_tweet = tweet.copy()

        for element in tweet:
            if element in self.stop_words:
                copy_tweet.remove(element)
        return copy_tweet

    def is_ascii(self, word: str):
        return all(ord(c) < 128 for c in word)

    def tweet_to_lowercase(self, tweet: List[str]):
        return [str.lower(word) for word in tweet if word != ""]

    def remove_numbers(self, tweet: List[str]):
        return [word for word in tweet if all([str.isalpha(char) for char in word])]
