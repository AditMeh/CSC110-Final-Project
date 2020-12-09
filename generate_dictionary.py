import pickle
from typing import List, Dict
from os import path
from math import log


def generate_idf_dictionary(processed_dataset: List[List[str]]):
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


def compute_word_frequency_dict(processed_dataset: List[List[str]]):
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
