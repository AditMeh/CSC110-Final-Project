from dataclasses import dataclass
import pickle
import numpy as np
from typing import List

with open('embed_indexes.pickle', 'rb') as handle:
    indexes = pickle.load(handle)


@dataclass
class Node:
    words: List[str]
    label: int


class VectorGraph:

    def __init__(self, train_x, train_y, idf_dict):
        self._graph = []

    def add_node(self, node: Node):
        self._graph.append(node)

    def _compute_cosine_similarity(self):
        raise NotImplementedError

    def _compute_term_frequency(self, sentence: List[str]):
        word_count_mapping = {}
        length = len(sentence)
        for word in sentence:
            if word not in word_count_mapping:
                word_count_mapping[word] = 1
            else:
                word_count_mapping[word] += 1
        normalized_term_frequency = [word_count_mapping[word]/length for word in sentence]
        return normalized_term_frequency

    def _compute_tf_idf(self):
        #TODO


