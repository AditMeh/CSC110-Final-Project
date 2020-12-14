"""
datareader Module
===============================

The functions/classes defined in this module are responsible for preprocessing tweet data.
Specifically, the Dataloader class houses a series of functions which are responsible for reading in
and filtering the data.

PLEASE NOTE: all elements in the dataloader class will not pass their doctests,
this is fine. Those are just examples to show how those functions work.

===============================
This file is Copyright (c) 2020 Aditya Mehrotra.
"""
import pandas as pd
import nltk
from nltk.corpus import stopwords
from string import punctuation
from typing import List, Tuple, Set
import os
from os import path

NLTK_FILEPATH = os.path.join(os.getcwd(), 'nltk_data')


def _filter_links(tweet: List[str]) -> List[str]:
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


def _tokenize_tweet(tweet: str) -> List[str]:
    """
    This function splits the tweet up into a list

    :param tweet:
        A string representing a tweet
    :return:
        A list consisting of each element of the tweet that was separated by spaces
    """
    return tweet.split()


def _filter_periods(tweet: List[str]) -> List[str]:
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


def _strip_punctuation(tweet: List[str]) -> List[str]:
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


def _filter_at_signs(tweet: List[str]) -> List[str]:
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


def _is_ascii(word: str) -> bool:
    """
    This function checks if the word has only ascii characters

    :param word:
        A string which represents a word
    :return:
        A boolean which indicates whether all characters in the string are ascii characters
    """
    return all(ord(c) < 128 for c in word)


def _tweet_to_lowercase(tweet: List[str]) -> List[str]:
    """
    This function converts all strings in the tweet to lowercase letters

    :param tweet:
        A tweet represented by a list where each word/phrase is an element
    :return:
        A list of strings, where each element has no lowercase characters
    """
    return [str.lower(word) for word in tweet]


def _remove_numbers(tweet: List[str]) -> List[str]:
    """
    This function removes all elements which have a number or a symbol in them from the list.
    (not including the # character)

    :param tweet:
        A tweet represented by a list where each word/phrase is an element
    :return:
        A list with no string elements that have special characters (except #) or numbers

    >>> tweet = ["@My@", "#oh", "my", "what", "1do", "we", "have", "here!"]
    >>> _remove_numbers(tweet)
    ["#oh", "my", "what", "we", "have"]
    """

    return [word for word in tweet if all([str.isalpha(char) or char == "#" for char in word])]


def _remove_empty_strings(tweet: List[str]) -> List[str]:
    """
    This function removes all empty strings from the list and returns the filtered list

    :param tweet:
        A tweet represented by a list where each word/phrase is an element
    :return:
        A filtered list with empty strings
    """

    return [string for string in tweet if string != ""]


def _remove_ampersands(tweet: str) -> List[str]:
    """
    The dataset writes ampersands as "amp", so this function filters these out

    :param tweet:
        A tweet represented by a list of strings
    :return:
        A list without any elements that are equal to "amp"
    """
    return [word for word in tweet if word != "amp"]


def _filter_nonascii(tweet: List[str]) -> List[str]:
    """
    This function removes all words which contain non-ascii characters from the tweet

    :param tweet:
        A tweet represented by a list where each word/phrase is an element
    :return:
        A tweet with no elements that have non-ascii characters

    >>> tweet = ["my", "是hello是", "well", "是", "不"]
    >>> _filter_nonascii(tweet)
    ["my", "well"]
    """
    return [word for word in tweet if _is_ascii(word)]


class DataLoader:
    """A custom class that is responsible for reading tweets and labels from our dataset,
    preprocessing the tweet by applying a series of filters and returning
    the preprocessed dataset"""

    # Private Instance Attributes:
    #   - _stop_words: A set of stop words to be used for filtration
    #   - _FILEPATH: Filepath to the dataset
    filepath: str
    _stop_words: Set[str]
    _FILEPATH: str

    def __init__(self, filepath: str) -> None:
        """
        This initalizer creates a set of stopwords from the nltk_data file
        (it downloads it if it doesn't already exist) and allow the rest
        of the function to use it.

        :param filepath:
            The filepath to the dataset, this is defined in main.py and should never change.
        """
        self._FILEPATH = filepath

        # check if the NLTK file are downloaded
        if not path.exists(NLTK_FILEPATH):
            print("Downloading corpus")
            nltk.data.path.append(NLTK_FILEPATH)
            nltk.download('stopwords', download_dir=NLTK_FILEPATH)
        else:
            print("Corpus already exists")
            nltk.data.path.append(NLTK_FILEPATH)

        self._stop_words = set(stopwords.words('english'))

        self._stop_words.add("rt")

    def prepare_data(self) -> Tuple[List[List[str]], List[int]]:
        """
        This function reads the dataset CSV, converts it to a dataframe and creates
        a list of training tweets and labels by preprocessing each individual tweet

        :return:
            train_x: A list of preprocessed tweets
            train_y: A list of labels, where the ith label corresponds to the ith
            element of train_x
        """
        df = pd.read_csv(self._FILEPATH)
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

        Note that the following doctest will not pass, it is just to show how this function works

        >>> tweet = "RT: @jeff Check out this cool link! https://google.com #awesome"
        >>> loader = DataLoader("twitter_sentiment_data.csv")
        >>> loader._filter_tweet(tweet)
        ['check', 'cool', 'link', '#awesome']

        """
        tweet = _tokenize_tweet(tweet)
        tweet = _filter_links(tweet)
        tweet = _filter_periods(tweet)
        tweet = _filter_at_signs(tweet)
        tweet = _strip_punctuation(tweet)
        tweet = _filter_nonascii(tweet)
        tweet = _tweet_to_lowercase(tweet)
        tweet = self._filter_stopwords(tweet)
        tweet = _remove_numbers(tweet)
        tweet = _remove_empty_strings(tweet)
        tweet = _remove_ampersands(tweet)
        return tweet

    def _filter_stopwords(self, tweet: List[str]) -> List[str]:
        """
        This function filters all the words that appear in the nltk.stopwords set and
        also removes "rt", which is the lowercase form of the twitter retweet phrase

        :param tweet:
            A tweet represented by a list where each word/phrase is an element
        :return:
            A tweet with no stopwords in it

        Note that the following doctest will not pass, it is just to show how this function works

        >>> tweet = ["my", "name", "is", "a", "synonym", "for", "the", "king"]
        >>> loader = DataLoader("twitter_sentiment_data.csv")
        >>> loader._filter_stopwords(tweet)
        ['name', 'synonym', 'king']
        """
        return [word for word in tweet if word not in self._stop_words]
