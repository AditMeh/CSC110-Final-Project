"""
Metrics module
===============================

This module and its contained functions/classes are responsible for outputting data to be fit to a cosine
similarity model and data used to test the model. It also is responsible for testing the fitted model on the
test set.

===============================

This file is Copyright (c) 2020 Aditya Mehrotra.
"""

from typing import List, Tuple
import random
from graph import VectorGraph


def train_test_split(samples: List[List[str]], labels: List[int],
                     percent_train: float, num_test: int) -> Tuple[
    List[List[str]], List[int], List[List[str]], List[int]]:
    """
    This function splits the data into training and testing sets

    Preconditions:
        - all(all(word != "" for word in tweet) for tweet in samples)
        - all(item in {-1, 0, 1, 2} for item in labels)
        - percent_train < 1

    :param samples:
        The preprocessed dataset samples
    :param labels:
        The dataset labels
    :param percent_train:
        The percentage of dataset samples that will be used for the training set.
        This is capped at 90%.
    :param num_test:
        The number of samples that will be returned for testing purposes. This is capped at
        100 because it takes a long time for each prediction.
    :return:
        A list of training samples and labels along with a list of testing samples and labels.
        It is guaranteed that no two samples/label pairs in the testing and training sets are the same.
    """

    # Putting a limit on how large our parameter values can be
    if percent_train > 0.90:
        percent_train = 0.90

    if num_test > 100:
        num_test = 100

    # computing the full dataset length and computing percent_train% of that value
    full_dataset_length = len(samples)
    dataset_train_length = round((len(samples) - 1) * percent_train)

    # taking 80% of the dataset and storing in our training lists
    train_x = samples[0: dataset_train_length]
    train_y = labels[0: dataset_train_length]

    # sampling num_tweet times from the set of the dataset that was not used for training. This gives us
    # a list of indexes for samples that belong in the test set
    test_idx = random.sample(
        list(range(dataset_train_length, full_dataset_length)), num_test)

    test_x = [samples[i] for i in test_idx]
    test_y = [labels[i] for i in test_idx]

    return train_x, train_y, test_x, test_y


def compute_accuracy(test_x: List[List[str]], test_y: List[int], graph: VectorGraph) -> float:
    """
    This function iterates through the testing set and computes the accuracy of the model

    Preconditions:
        - all(all(word != "" for word in tweet) for tweet in test_x)
        - all(item in {-1, 0, 1, 2} for item in test_y)

    :param test_x:
        The testing set samples
    :param test_y:
        The testing set labels
    :param graph:
        A graph object, this is the model that is trained on the training set that can output
        predictions given a sentence
    :return:
        The accuracy of the graph on the testing set.
    """
    num_correct = 0
    num_total = 0

    # Iterate through all testing samples, computing the accuracy at each iteration
    for i in range(len(test_x)):
        print("Test sample number " + str(i + 1))
        pred, words = graph.compute_max_similar_node(test_x[i])

        # If the model predicted the sample to the correct class
        if pred == test_y[i]:
            num_correct += 1
        num_total += 1

        print("test data: " + str(test_x[i]))
        print("closest sentence: " + str(words))
        print("Total accuracy = " + str(num_correct / num_total) + "\n")

    return "Final accuracy on a random set of " + str(len(test_x)) + " samples: " + str(num_correct / num_total)
