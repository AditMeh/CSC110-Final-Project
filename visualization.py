"""
Visualization module
===============================

This module and its contained functions/classes are responsible for creating visualisations using the data set.

===============================

This file is Copyright (c) 2020 Aditya Mehrotra.
"""
from generate_dictionary import compute_class_word_frequency_dicts
from typing import List
import matplotlib.pyplot as plt
import numpy as np

FILEPATH = "twitter_sentiment_data.csv"


def visualize_class_words(plot_class: int, samples: List[List[str]], labels: List[int]) -> None:
    """
    This function plots a histogram that shows the top 20 words most frequently seen words
    from samples in the dataset of a certain class

    Preconditions:
        - all(all(word != "" for word in tweet) for tweet in samples)
        - all(item in {-1, 0, 1, 2} for item in labels)
        - plot_class in {-1, 0, 1, 2}

    :param plot_class:
        The class from which the most frequent words will be plotted
    :param samples:
        The dataset samples
    :param labels:
        The dataset labels
    :return:
        It shows the matplotlib graph on the screen, nothing is returned
    """

    # Compute the frequency dictionary
    freq_dict = compute_class_word_frequency_dicts(samples, labels, plot_class)

    label_to_text = {
        -1: "Most common words for climate change non-believers",
        0: "Most common words for neutral tweets",
        1: "Most common words for climate change supporters",
        2: "Most common words for factual news about climate change"
    }

    # This line uses a lambda and the sorted function to create a list of tuples, which correspond to
    # (key, value) pairs sorted by the values.
    sorted_dict_key_values = sorted(freq_dict.items(), key=lambda x: x[1])

    # Get the 20 words which have the highest frequency in the sorted_dict_keys_values list
    count_words = [sorted_dict_key_values[-i][1] for i in range(1, 20)]
    words_sorted = [sorted_dict_key_values[-i][0] for i in range(1, 20)]

    y_pos = np.arange(len(count_words))

    # Plotting the histogram using matplotlib
    plt.bar(y_pos, count_words, align='center', alpha=0.5)
    plt.xticks(y_pos, words_sorted, fontsize=5)
    plt.ylabel('Count')
    plt.title(label_to_text[plot_class])

    plt.show()
