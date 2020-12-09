import pandas as pd
import nltk
from nltk.corpus import stopwords
from string import punctuation
from typing import List, Set
from os import path


class DataLoader:
    def __init__(self, FILEPATH: str) -> None:
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

    def prepare_data(self) -> List[List[str]]:
        df = pd.read_csv(self.FILEPATH)
        labels = df.iloc[:, 0]
        tweets = df.iloc[:, 1]

        iter_length = len(df)

        train_x = []
        train_y = []
        for i in range(iter_length):
            filtered_tweet = self._filter_tweet(tweets[i])
            train_x.append(filtered_tweet)
            train_y.append(int(labels[i]))

        return train_x, train_y

    def _filter_tweet(self, tweet: List[str]) -> List[str]:
        tokenized_tweet = self._tokenize_tweet(tweet)
        link_filtered = self._filter_links(tokenized_tweet)
        periods_filtered = self._filter_periods(link_filtered)
        at_sign_filtered = self._filter_at_signs(periods_filtered)
        strip_punctuation = self._strip_punctuation(at_sign_filtered)
        non_ascii_filtered = self._filter_nonascii(strip_punctuation)
        stopwords_filtered = self._filter_stopwords(non_ascii_filtered)
        tweet_to_lowercase = self._tweet_to_lowercase(stopwords_filtered)
        removed_numbers = self._remove_numbers(tweet_to_lowercase)
        return removed_numbers

    def _tokenize_tweet(self, tweet: str) -> List[str]:
        return tweet.split()

    def _filter_links(self, tweet: List[str]) -> List[str]:
        copy_tweet = tweet.copy()

        for element in tweet:
            if "http" in element:
                copy_tweet.remove(element)
        return copy_tweet

    def _filter_periods(self, tweet: List[str]) -> List[str]:
        # replace all periods with spaces, double spaces will be split anyways
        for i in range(len(tweet)):
            if tweet[i] in {".", ","}:
                tweet[i] = " "
        return " ".join(tweet).split()

    def _strip_punctuation(self, tweet: List[str]) -> List[str]:

        copy_tweet = tweet.copy()

        for i in range(len(tweet)):
            copy_tweet[i] = ''.join(char for char in tweet[i] if char not in punctuation or char == "#")
        return copy_tweet

    def _filter_at_signs(self, tweet: List[str]) -> List[str]:
        copy_tweet = tweet.copy()

        for element in tweet:
            if "@" in element:
                copy_tweet.remove(element)
        return copy_tweet

    def _filter_nonascii(self, tweet: List[str]) -> List[str]:
        copy_tweet = tweet.copy()

        for element in tweet:
            if not self._is_ascii(element):
                copy_tweet.remove(element)
        return copy_tweet

    def _filter_stopwords(self, tweet: List[str]) -> List[str]:
        self.stop_words.add("RT")
        copy_tweet = tweet.copy()

        for element in tweet:
            if element in self.stop_words:
                copy_tweet.remove(element)
        return copy_tweet

    def _is_ascii(self, word: str) -> bool:
        return all(ord(c) < 128 for c in word)

    def _tweet_to_lowercase(self, tweet: List[str]) -> List[str]:
        return [str.lower(word) for word in tweet if word != ""]

    def _remove_numbers(self, tweet: List[str]) -> List[str]:
        return [word for word in tweet if all([str.isalpha(char) or char == "#" for char in word])]
