import pickle
from typing import List, Dict
from os import path
from math import log


def generate_idf_dictionary(processed_dataset: List[List[str]]) -> Dict[str, float]:
    """
    Generates a dictionary that maps each word to its IDF score

    :param processed_dataset:
        A list of preprocessed tweets that is going to be used for training the
        language model
    :return:
        A dictionary mapping of words to IDF scores
    """
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
    else:
        print("IDF.pickle already exists, skipping generation")

        with open('IDF.pickle', 'rb') as handle:
            idf_dict = pickle.load(handle)
        return idf_dict


def compute_word_frequency_dict(processed_dataset: List[List[str]]) -> Dict[str, int]:
    """
    This function computes a dictionary which stores the mapping of each word
    in the dataset to its frequency

    :param processed_dataset:
        A list of preprocessed tweets that is going to be used for training the
        language model
    :return:
        A dictionary mapping of words to frequency
    """
    counts_dict = {}
    for sample in processed_dataset:
        word_accumulator = set()
        for word in sample:
            if word not in counts_dict and word not in word_accumulator:
                counts_dict[word] = 1
                word_accumulator.add(word)
            if word in counts_dict and word not in word_accumulator:
                counts_dict[word] += 1
                word_accumulator.add(word)

    return counts_dict
