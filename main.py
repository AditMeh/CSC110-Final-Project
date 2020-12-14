"""
Program main
===============================

This is the main file of my project, which is responsible for all of the main computations.
Please read the instructions given below to see how to use this program.

===============================

This file is Copyright (c) 2020 Aditya Mehrotra.
"""

from datareader import DataLoader
from generate_dictionary import generate_idf_dictionary
from graph import VectorGraph
from visualization import visualize_class_words
from model_metrics import train_test_split, compute_accuracy

FILEPATH = "twitter_sentiment_data.csv"

"""
INSTRUCTIONS:

I have two parts, the first is data visualization, the second is model testing. 


VISUALIZATION:

This block consists of three functions. The first two functions don't need to be changed.
The third function plots a histogram for the 20 most common words for a given class. 
This function takes the desired class as the first argument. In order to try out 
different classes, try changing the first argument into a number in the set {-1, 0, 1, 2} 
to see what the barplots for that class look like. You can find what each class number 
represents in the report.

visualize_class_words(2, samples, labels)
                      ^
                      |
                      change this!

TESTING THE MODEL:

The function that can be changed here is the train_test_split function. 

  - The second last argument corresponds to what percentage of the dataset will be 
    allocated towards training. Try and change this number to something between 0 - 0.90 
    to see how well the model performs on the testing set. 
    
  - The last argument for this function is the number of testing samples, change this 
    to a number between 1-100 to change the number of samples that are used to test the model. 
    I restricted the number of testing samples to a number between 0 and 100 because it 
    takes too long for higher values.
  
  - When you run this block of code, you should see the the the model 
    evaluating itself in the console using the test set. You will see the nodes it 
    finds that are the most similar to given tweet in the test set, and also the 
    accuracy of the model over the testing samples that it has evaluated so far.


    train_x, train_y, test_x, test_y = train_test_split(samples, labels, 0.90, 3)
                                                                            ^  ^
                                                                            |  |
                                                                        Change these!
                                                                        
NOTE: when running the code, if you are running one block, please comment out the other.
Ex: if you run the visualization block, then comment out the modelling one.
"""


def main():
    """
    # VISUALIZATION
    loader = DataLoader(FILEPATH)
    samples, labels = loader.prepare_data()
    visualize_class_words(2, samples, labels)
    """

    # TESTING THE MODEL
    loader = DataLoader(FILEPATH)
    samples, labels = loader.prepare_data()
    train_x, train_y, test_x, test_y = train_test_split(samples, labels, 0.90, 3)
    idf_dict = generate_idf_dictionary(train_x)
    graph = VectorGraph(train_x, train_y, idf_dict)
    print(compute_accuracy(test_x, test_y, graph))


if __name__ == '__main__':
    main()
