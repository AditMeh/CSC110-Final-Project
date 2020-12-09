from datareader import DataLoader
from generate_dictionary import generate_idf_dictionary
from graph import VectorGraph

FILEPATH = "twitter_sentiment_data.csv"


def main():
    loader = DataLoader(FILEPATH)
    train_x, train_y = loader.prepare_data()
    idf_dict = generate_idf_dictionary(train_x)
    graph = VectorGraph(train_x, train_y, idf_dict)
    print(graph.compute_max_similar_node(train_x[0]))


if __name__ == '__main__':
    main()
