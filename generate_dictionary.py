import pickle
from typing import List
from os import path


def generate_dictionary(processed_dataset: List[List[str]]):
    if not path.exists("embed_indexes.pickle"):
        print("generating pickle")
        embed_dict = {}
        counter = 0

        for sample in processed_dataset:
            for word in sample:
                if word not in embed_dict:
                    embed_dict[word] = counter
                    counter += 1

        with open('embed_indexes.pickle', 'wb') as handle:
            pickle.dump(embed_dict, handle)
    else:
        print("embedding_indexes.pickle already exists, skipping generation")
