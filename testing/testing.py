from datareader import DataLoader


def test_remove_numbers() -> None:
    tweet = ["@My@", "#oh", "my", "what", "1do", "we", "have", "here!"]
    loader = DataLoader("twitter_sentiment_data.csv")
    assert loader._remove_numbers(tweet) == ["#oh", "my", "what", "we", "have"]


def test_filter_stopwords() -> None:
    tweet = ["my", "name", "is", "a", "synonym", "for", "the", "king"]
    loader = DataLoader("twitter_sentiment_data.csv")
    assert loader._filter_stopwords(tweet) == ["name", "synonym", "king"]


def test_filter_nonascii() -> None:
    tweet = ["my", "是hello是", "well", "是", "不"]
    loader = DataLoader("twitter_sentiment_data.csv")
    assert loader._filter_nonascii(tweet) == ["my", "well"]


def test_filter_tweet() -> None:
    tweet = "RT: @jeff Check out this cool link! https://google.com #awesome"
    loader = DataLoader("twitter_sentiment_data.csv")
    assert loader._filter_tweet(tweet) == ["check", "this", "cool", "link", "#awesome"]

