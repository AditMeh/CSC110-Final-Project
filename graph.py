from dataclasses import dataclass
import numpy as np
from typing import List, Dict


@dataclass
class Node:
    words: List[str]
    label: int


class VectorGraph:

    def __init__(self, train_x: List[List[str]], train_y: List[int], idf_dict: Dict[str, float]) -> None:
        self._graph = []
        self.idf_dict = idf_dict
        assert len(train_x) == len(train_y)

        for i in range(len(train_x)):
            self._add_node(Node(train_x[i], train_y[i]))

    def _add_node(self, node: Node) -> None:
        """
        Adds a Node object to the _graph list

        :param node:
            A Node object
        :return:
            Nothing
        """
        self._graph.append(node)

    def compute_max_similar_node(self, input_text: List[str]) -> int:
        """
        Computes the node which is the most similar to the input_text and return
        the class of that node. Similarity is computed using cosine_similarity.
        
        :param input_text: 
            A cleaned/preprocessed tweet represented by a list of words
        :return: 
            The label of the node which is most similar to the input tweet
        """
        max_similarity_score = -1
        max_sim_label = None
        for node in self._graph:
            node_tf_idf = self._compute_tf_idf(node.words)
            node_idf_filtered = np.asarray(self._remove_missing_words(input_text, node.words, node_tf_idf))
            input_tf_idf = np.asarray(self._compute_tf_idf(input_text))

            score = self._compute_cosine_similarity(input_tf_idf, node_idf_filtered)

            if max_similarity_score < score:
                max_sim_label = node.label
                max_similarity_score = score

        return max_sim_label

    def _compute_cosine_similarity(self, d1: np.ndarray, d2: np.ndarray) -> float:
        """
        This function computes the cosine similarity between two vectors d1 and d2

        :param d1:
            A numpy array of shape (n,)
        :param d2:
            A numpy array of shape (n,)
        :return:
            returns the cosine similarity of the two vectors
        """
        assert d1.shape == d2.shape

        if (np.linalg.norm(d1) * np.linalg.norm(d2)) == 0:
            return 0

        similarity = (np.dot(d1, d2)) / (np.linalg.norm(d1) * np.linalg.norm(d2))
        return similarity

    def _compute_tf_idf(self, sentence: List[str]) -> List[float]:
        """
        This function computes the tf*idf for a given set of words.

        :param sentence:
            A list of words of length n
        :return:
            A list of floats of length n, where each item is the tf*idf score of
            the words at the corresponding index in the initial sentence
        """
        word_count_mapping = {}
        length = len(sentence)
        for word in sentence:
            if word not in word_count_mapping:
                word_count_mapping[word] = 1
            else:
                word_count_mapping[word] += 1

        #print("ma" in word_count_mapping)
        tf_idf = [(word_count_mapping[word] / length) * self.idf_dict[word]
                  for word in sentence]
        return tf_idf

    def _remove_missing_words(self, word_input: List[str], word_node: List[str],
                              node_idf: List[float]) -> List[float]:
        """
        This function does the following:
            1. Create a empty list called filtered_idf
            2. Iterates through the words in word_input and checks if
            the current word is also in the words of the node.
            3. In the case that the current word is in the words of the node,
            Append the idf score to filtered_idf. If the current word is not
            in the words of the node, append 0 to filtered_idf

        For each word in word_input, the corresponding index at filtered_tf_idf
        is the tf_idf score for that word from the words in words_node

        :param word_input:
            A list of words that represent the input tweet
        :param word_node:
            A list of words that represent the words in the Node
        :param node_idf:
            A list of floats which represent the tf*idf scores for each\
            words in the Node
        :return:
            A list of tf*tf idf scores which represent the tf*idf score
            for each word in word_input using word_node.
        """
        filtered_tf_idf = []
        assert len(node_idf) == len(word_node)

        for i in range(len(word_input)):
            if word_input[i] in word_node:
                filtered_tf_idf.append(node_idf[word_node.index(word_input[i])])
            else:
                filtered_tf_idf.append(0)

        return filtered_tf_idf
