"""
Graph module
===============================

This module and its contained functions/classes are responsible for creating
and fitting a cosine similarity model with TF-IDF scores. Additionally,
the VectorGraph class in this model is responsible for using the fitted
model to predict the sentiment of any given tweet in the right format.

===============================

This file is Copyright (c) 2020 Aditya Mehrotra.
"""

from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Tuple


@dataclass
class Node:
    """
    A custom data type that represents a node in our graph. Simplified, this data type
    just represents a training set sample, by containing the tweet and its label. This is used
    in the VectorGraph class to find the item with the closest cosine similarity to a query tweet.

    Representation Invariants:
        - all(word != "" for word in words)
        - all(sentiment in {-1, 0, 1, 2} for sentiment in label)
    """
    words: List[str]
    label: int


def _remove_missing_words(word_input: List[str], word_node: List[str],
                          node_tf_idf: List[float]) -> List[float]:
    """
    This function does the following:
        1. Create a empty list called filtered_idf
        2. Iterates through the words in word_input and checks if
        the current word is also in the words of the node.
        3. In the case that the current word is in the words of the node,
        Append the TF*IDF score of current word to filtered_idf. If the current word is not
        in the words of the node, append 0 to filtered_idf

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

    >>> word_input_test = ["merry", "go", "round"]
    >>> node_word_test = ["merry", "boom", "loud", "round", "chicken"]
    >>> node_idf_test =  [-1, -2, -3, -4, -5]
    >>> _remove_missing_words(word_input_test, node_word_test, node_idf_test)
    [-1, 0, -4]
    """
    filtered_tf_idf = []
    assert len(node_tf_idf) == len(word_node)

    for i in range(len(word_input)):
        if word_input[i] in word_node:
            # Word in the query is in the words of the node
            filtered_tf_idf.append(node_tf_idf[word_node.index(word_input[i])])
        else:
            # Case when the word in the query is not in the node
            filtered_tf_idf.append(0)

    return filtered_tf_idf


def _compute_cosine_similarity(d1: np.ndarray, d2: np.ndarray) -> float:
    """
    This function computes the cosine similarity between two vectors d1 and d2

    :param d1:
        A numpy array of shape (n,)
    :param d2:
        A numpy array of shape (n,)
    :return:
        Returns the cosine similarity of the two vectors
    """
    assert d1.shape == d2.shape

    # To avoid dividing by zero. This edge case occurs when both vectors share
    # no common elements
    if (np.linalg.norm(d1) * np.linalg.norm(d2)) == 0:
        return 0

    # Computing cosine similarity between both vectors, refer to report for explicit forumla
    similarity = (np.dot(d1, d2)) / (np.linalg.norm(d1) * np.linalg.norm(d2))
    return similarity


class VectorGraph:
    """A custom class that represents a graph, which holds a collection of
    nodes (defined above). This class defines methods that allow the
    user to fit a model using a training set and generate the a prediction
    for the sentiment of any given tweet."""

    # Private Instance Attributes:
    #   - _graph: A List of nodes that will be used for the prediction algorithm
    #   - _idf_dict: Dict used to convert words to their IDF scores

    train_x: List[List[str]]
    train_y: List[int]
    idf_dict: Dict[str, float]
    _graph: List[Node]
    _idf_dict: Dict[str, float]

    def __init__(self, train_x: List[List[str]], train_y: List[int],
                 idf_dict: Dict[str, float]) -> None:
        """
        This function creates a list of nodes (graph) using the training set. It
        does this by iterating through all the samples in the training set and
        using the tweet + the label to create a node object, that it stores
        in the list.

        :param train_x:
            The training set, this consists of many preprocessed tweets
        :param train_y:
            The training set labels, where the element at index i is the label for the
            element at index i in train_x
        :param idf_dict:
            A dictionary which has the mapping of words to IDF scores
        """
        print("Creating graph")

        # This list contains all the nodes
        self._graph = []
        self._idf_dict = idf_dict
        assert len(train_x) == len(train_y)

        for i in range(len(train_x)):
            # Create a Node for each training sample and append it to the graph list
            self._add_node(Node(train_x[i], train_y[i]))

    def _add_node(self, node: Node) -> None:
        """
        Adds a Node object to the _graph list

        Preconditions:
            - all(word != "" for word in node.words)
            - node.label in {-1, 0, 1, 2}

        :param node:
            A Node object
        :return:
            Nothing
        """
        # Add a node to the graph
        self._graph.append(node)

    def compute_max_similar_node(self, input_text: List[str]) -> Tuple[int, List[str]]:
        """
        Computes the node which is the most similar to the input_text and return
        the class of that node. Similarity is computed using cosine_similarity.

        :param input_text:
            A cleaned/preprocessed tweet represented by a list of words
        :return:
            The label of the node which is most similar to the input
            tweet along with the tweet stored in the node
        """
        max_similarity_score = -1
        max_sim_label = None
        max_sim_node = None

        # iterate through all the nodes in the graph
        for node in self._graph:
            node_tf_idf = self._compute_tf_idf(node.words)

            # Compute the tf*idf scores for both the current node and the query, then typecast
            # into numpy array
            node_idf_filtered = np.asarray(_remove_missing_words(input_text,
                                                                 node.words, node_tf_idf))
            input_tf_idf = np.asarray(self._compute_tf_idf(input_text))

            # Cosine similarity of the current node and query
            score = _compute_cosine_similarity(input_tf_idf, node_idf_filtered)

            # If the cosine similarity of a node is higher than the current
            # maximum cosine similarity, update max_sim_label and max_sim_node
            # to store the sentiment and tweet of that node
            if max_similarity_score < score:
                max_sim_label = node.label
                max_similarity_score = score
                max_sim_node = node.words

        return max_sim_label, max_sim_node

    def _compute_tf_idf(self, sentence: List[str]) -> List[float]:
        """
        This function computes the tf*idf for a given set of words.

        :param sentence:
            A list of words of length n
        :return:
            A list of floats of length n, where each item is the tf*idf score of
            the words at the corresponding index in the initial sentence
        """

        # mapping of words to counts
        word_count_mapping = {}
        length = len(sentence)
        for word in sentence:
            if word not in word_count_mapping:
                word_count_mapping[word] = 1
            else:
                word_count_mapping[word] += 1

        # Convert each word to a TF*IDF score and store it in a list
        tf_idf = []
        for word in sentence:
            if word in self._idf_dict:
                tf_idf.append(word_count_mapping[word] / length * self._idf_dict[word])
            else:
                # if the word never existed in the training data, append 0
                tf_idf.append(0)

        return tf_idf
