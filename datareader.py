import pandas as pd
import nltk
from nltk.corpus import stopwords
from string import punctuation
from typing import List, Tuple
from os import path


class DataLoader:
    def __init__(self, filepath: str) -> None:
        self.FILEPATH = filepath

        if not path.exists("nltk_data"):
            print("Downloading corpus")
            nltk.data.path.append('nltk_data/')
            nltk.download('stopwords', download_dir="nltk_data/")
            nltk.download('punkt', download_dir="nltk_data/")
        else:
            print("Corpus already exists")
            nltk.data.path.append('nltk_data/')

        self.stop_words = set(stopwords.words('english'))

    def prepare_data(self) -> Tuple[List[List[str]], List[int]]:
        """
        This function reads the dataset CSV, converts it to a dataframe and creates
        a list of training tweets and labels by preprocessing each individual tweet

        :return:
            train_x: A list of preprocessed tweets
            train_y: A list of labels, where the ith label corresponds to the ith
            element of train_x
        """
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

    def _filter_tweet(self, tweet: str) -> List[str]:
        """
        This function preprocesses a tweet using a set of pre-defined filter operations

        :param tweet:
            A tweet represented by a string
        :return:
            A list of words representing the filtered words in the original tweet

        >>> tweet = "RT: @jeff Check out this cool link! https://google.com #awesome"
        >>> loader = DataLoader("twitter_sentiment_data.csv")
        >>> loader._filter_tweet(tweet)
        ["check", "this", "cool", "link", "#awesome"]

        """
        tweet = self._tokenize_tweet(tweet)
        tweet = self._filter_links(tweet)
        tweet = self._filter_periods(tweet)
        tweet = self._filter_at_signs(tweet)
        tweet = self._strip_punctuation(tweet)
        tweet = self._filter_nonascii(tweet)
        tweet = self._tweet_to_lowercase(tweet)
        tweet = self._filter_stopwords(tweet)
        tweet = self._remove_numbers(tweet)
        tweet = self._remove_empty_strings(tweet)
        return tweet

    def _tokenize_tweet(self, tweet: str) -> List[str]:
        """
        This function splits the tweet up into a list

        :param tweet:
            A string representing a tweet
        :return:
            A list consisting of each element of the tweet that was separated by spaces
        """
        return tweet.split()

    def _filter_links(self, tweet: List[str]) -> List[str]:
        """
        This function removes every element with "http" from the tweet. This is
        to filter out links

        :param tweet:
            A tweet represented by a list where each word/phrase is an element
        :return:
            A tweet without any links inside of it
        """
        for element in tweet:
            if "http" in element:
                tweet.remove(element)
        return tweet

    def _filter_periods(self, tweet: List[str]) -> List[str]:
        """
        This function replaces all elements which consist of only commas or periods with
        empty strings. These empty strings will be removed later on using another function.

        :param tweet:
            A tweet represented by a list where each word/phrase is an element
        :return:
            A tweet with no elements that consist of only commas or periods
        """
        for i in range(len(tweet)):
            if tweet[i] in {".", ","}:
                tweet[i] = ""
        return tweet

    def _strip_punctuation(self, tweet: List[str]) -> List[str]:
        """
        This function strips each individual string in the tweet for any punctuation characters
        excluding "#" symbol

        :param tweet:
            A tweet represented by a list where each word/phrase is an element
        :return:
            A tweet with no elements that have any sort of punctuation except for the "#" symbol
        """
        for i in range(len(tweet)):
            tweet[i] = ''.join(char for char in tweet[i] if char not in punctuation or char == "#")
        return tweet

    def _filter_at_signs(self, tweet: List[str]) -> List[str]:
        """
        This function removes all elements which have "@" symbols in them

        :param tweet:
            A tweet represented by a list where each word/phrase is an element
        :return:
            A tweet with no elements that have "@" in them
        """
        for element in tweet:
            if "@" in element:
                tweet.remove(element)
        return tweet

    def _filter_nonascii(self, tweet: List[str]) -> List[str]:
        """
        This function removes all words which contain non-ascii characters from the tweet

        :param tweet:
            A tweet represented by a list where each word/phrase is an element
        :return:
            A tweet with no elements that have non-ascii characters

        >>> tweet = ["my", "是hello是", "well", "是", "不"]
        >>> loader = DataLoader("twitter_sentiment_data.csv")
        >>> loader._filter_nonascii(tweet)
        ["my", "well"]
        """
        for element in tweet:
            if not self._is_ascii(element):
                tweet.remove(element)
        return tweet

    def _filter_stopwords(self, tweet: List[str]) -> List[str]:
        """
        This function filters all the words that appear in the nltk.stopwords set and
        also removes "rt", which is the lowercase form of the twitter retweet phrase

        :param tweet:
            A tweet represented by a list where each word/phrase is an element
        :return:
            A tweet with no stopwords in it


        >>> tweet = ["my", "name", "is", "a", "synonym", "for", "the", "king"]
        >>> loader = DataLoader("twitter_sentiment_data.csv")
        >>> self._filter_stopwords(tweet)
        ["name", "synonym", "king"]
        """
        self.stop_words.add("rt")

        for element in tweet:
            if element in self.stop_words:
                tweet.remove(element)
        return tweet

    def _is_ascii(self, word: str) -> bool:
        """
        This function checks if the word has only ascii characters

        :param word:
            A string which represents a word
        :return:
            A boolean which indicates whether all characters in the string are ascii characters
        """
        return all(ord(c) < 128 for c in word)

    def _tweet_to_lowercase(self, tweet: List[str]) -> List[str]:
        """
        This function converts all strings in the tweet to lowercase letters

        :param tweet:
            A tweet represented by a list where each word/phrase is an element
        :return:
            A list of strings, where each element has no lowercase characters
        """
        return [str.lower(word) for word in tweet]

    def _remove_numbers(self, tweet: List[str]) -> List[str]:
        """
        This function removes all elements which have a number or a symbol in them from the list.
        (not including the # character)

        :param tweet:
            A tweet represented by a list where each word/phrase is an element
        :return:
            A list with no string elements that have special characters (except #) or numbers

        >>> tweet = ["@My@", "#oh", "my", "what", "1do", "we", "have", "here!"]
        >>> loader = DataLoader("twitter_sentiment_data.csv")
        >>> self._remove_numbers(tweet)
        ["#oh", "my", "what", "we", "have"]
        """

        return [word for word in tweet if all([str.isalpha(char) or char == "#" for char in word])]

    def _remove_empty_strings(self, tweet: List[str]) -> List[str]:
        """
        This function removes all empty strings from the list and returns the filtered list

        :param tweet:
            A tweet represented by a list where each word/phrase is an element
        :return:
            A filtered list with empty strings
        """

        return [string for string in tweet if string != ""]


