from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass
class Node:
    words: List[str]
    label: int


class VectorGraph:

    def __init__(self, train_x, train_y, idf_dict):
        self._graph = []
        self.idf_dict = idf_dict
        assert len(train_x) == len(train_y)

        for i in range(len(train_x)):
            self._add_node(Node(train_x[i], train_y[i]))

    def _add_node(self, node: Node):
        self._graph.append(node)

    def compute_max_similar_node(self, input_text):
        max_similarity_score = -1
        max_sim_label = None
        for node in self._graph:
            node_idf = self._compute_idf_term_frequency(node.words)
            node_idf_filtered = np.asarray(self._remove_missing_words(input_text, node.words, node_idf))
            input_idf = np.asarray(self._compute_idf_term_frequency(input_text))

            score = self._compute_cosine_similarity(input_idf, node_idf_filtered)

            if max_similarity_score < score:
                max_sim_label = node.label
                max_similarity_score = score

        return max_sim_label

    def _compute_cosine_similarity(self, d1, d2):

        assert d1.shape == d2.shape

        if (np.linalg.norm(d1) * np.linalg.norm(d2)) == 0:
            return 0

        similarity = (np.dot(d1, d2)) / (np.linalg.norm(d1) * np.linalg.norm(d2))
        return similarity

    def _compute_idf_term_frequency(self, sentence: List[str]):
        word_count_mapping = {}
        length = len(sentence)
        for word in sentence:
            if word not in word_count_mapping:
                word_count_mapping[word] = 1
            else:
                word_count_mapping[word] += 1
        normalized_idf_term_frequency = [(word_count_mapping[word] / length) * self.idf_dict[word]
                                         for word in sentence]
        return normalized_idf_term_frequency

    def _remove_missing_words(self, word_input, word_node, node_idf):
        filtered_idf = []
        assert len(node_idf) == len(word_node)

        for i in range(len(word_input)):
            if word_input[i] in word_node:
                filtered_idf.append(node_idf[word_node.index(word_input[i])])
            else:
                filtered_idf.append(0)

        return filtered_idf
