<<<<<<< HEAD
from datareader import DataLoader
from generate_dictionary import generate_dictionary

FILEPATH = "twitter_sentiment_data.csv"


def main():
    loader = DataLoader(FILEPATH)
    train_x, train_y = loader.prepare_data()
    generate_dictionary(train_x)


if __name__ == '__main__':
    main()
=======
if __name__ == '__main__':
    #TODO
>>>>>>> 592d60d54761fd9e42651a1fe8074847cc9495db
