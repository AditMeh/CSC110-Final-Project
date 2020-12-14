"""
Dictionary generator module
===============================

This module is responsible for creating all the dictionaries that are used throughout
this python application.

===============================

This file is Copyright (c) 2020 Aditya Mehrotra.
"""

import pickle
from typing import List, Dict
from os import path
from math import log


def generate_idf_dictionary(processed_dataset: List[List[str]]) -> Dict[str, float]:
    """
    Generates a dictionary that maps each word to its IDF score

    Preconditions:
        - all(all(word != "" for word in tweet) for tweet in processed_dataset)

    :param processed_dataset:
        A list of preprocessed tweets that is going to be used for training the
        language model
    :return:
        A dictionary mapping of words to IDF scores
    """

    # Generating the dictionary and storing in a pickle if it doesn't exist
    if not path.exists("IDF.pickle"):
        print("generating IDF pickle")

        train_set_length = len(processed_dataset)
        counts_dict = compute_word_frequency_dict(processed_dataset)
        idf_dict = {}

        for word in counts_dict:
            idf_dict[word] = 1 + log(train_set_length / counts_dict[word], 2)

        with open('IDF.pickle', 'wb') as handle:
            pickle.dump(idf_dict, handle)

        return idf_dict

    # Using pre generated pickle dictionary to save computational power
    else:
        print("IDF.pickle already exists, skipping generation")

        with open('IDF.pickle', 'rb') as handle:
            idf_dict = pickle.load(handle)
        return idf_dict


def compute_word_frequency_dict(processed_dataset: List[List[str]]) -> Dict[str, int]:
    """
    This function creates a dictionary which has the mapping of each word
    in the dataset to how many sentences it appears in.

    Preconditions:
        - all(all(word != "" for word in tweet) for tweet in processed_dataset)

    :param processed_dataset:
        A list of preprocessed tweets that is going to be used for training the
        language model
    :return:
        A dictionary mapping of a word to how many sentences that word appears in

        >>> dataset = [['a', 'a', 'a'], ['b', 'b', 'a']]
        >>> compute_word_frequency_dict(dataset)
        {'a': 2, 'b': 1}
    """

    # word accumulator
    counts_dict = {}
    for sample in processed_dataset:

        # For every word in the sentence, keep track of which ones have already been visited
        word_accumulator = set()
        for word in sample:

            # Check if a word has already been seen in the sentence elsewhere and
            # if it does not exist in our count dict
            if word not in counts_dict and word not in word_accumulator:
                counts_dict[word] = 1
                word_accumulator.add(word)

            # Check if a word has already been seen in the sentence elsewhere
            # and if it exists in our count dict
            if word in counts_dict and word not in word_accumulator:
                counts_dict[word] += 1
                word_accumulator.add(word)

    return counts_dict


def compute_class_word_frequency_dicts(processed_dataset: List[List[str]],
                                       labels: List[int],
                                       chosen_label: int) -> Dict[str, int]:
    """
    This function outputs a that dictionary contains the frequency
    of each word in the samples which correspond to the class of the key.

    Preconditions:
        - all(all(word != "" for word in tweet) for tweet in processed_dataset)
        - all(item in {-1, 0, 1, 2} for item in labels)

    :param chosen_label:
        The label that we want to compute the word frequency mapping of
    :param labels:
        The list of labels for each of the elements in the processed_dataset
    :param processed_dataset:
        A list of preprocessed sentences
    :return:
        A dictionary mapping of a word to how many times it appears
        in samples of the given class

        >>> dataset = [['a', 'a', 'a'], ['b', 'b', 'a']]
        >>> dataset_classes = [1, 0]
        >>> compute_class_word_frequency_dicts(dataset, dataset_classes, 1)
        {'a': 3}
    """

    # word counts accumulator
    class_counts_dict = {}

    for i in range(len(processed_dataset)):

        # check if the label of the sample corresponds to the class we want
        if labels[i] == chosen_label:

            for word in processed_dataset[i]:
                if word not in class_counts_dict:
                    class_counts_dict[word] = 1
                elif word in class_counts_dict:
                    class_counts_dict[word] += 1

    return class_counts_dict
