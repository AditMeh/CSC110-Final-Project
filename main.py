from datareader import DataLoader
from generate_dictionary import generate_idf_dictionary
from graph import VectorGraph
from visualization import visualize_class_words
from model_metrics import train_test_split, compute_accuracy

FILEPATH = "twitter_sentiment_data.csv"


def main():
    """
    # VISUALIZATION
    loader = DataLoader(FILEPATH)
    samples, labels = loader.prepare_data()
    visualize_class_words(1, samples, labels)

    """
    # TESTING THE MODEL
    loader = DataLoader(FILEPATH)
    samples, labels = loader.prepare_data()
    train_x, train_y, test_x, test_y = train_test_split(samples, labels, 0.80, 20)
    idf_dict = generate_idf_dictionary(train_x)
    graph = VectorGraph(train_x, train_y, idf_dict)
    final_score = compute_accuracy(test_x, test_y, graph)
    print(final_score)


if __name__ == '__main__':
    main()
