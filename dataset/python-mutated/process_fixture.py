import pytest
from textblob import TextBlob

def extract_sentiment(text: str):
    if False:
        return 10
    'Extract sentiment using textblob. \n        Polarity is within range [-1, 1]'
    text = TextBlob(text)
    return text.sentiment.polarity

def text_contain_word(word: str, text: str):
    if False:
        for i in range(10):
            print('nop')
    'Find whether the text contains a particular word'
    return word in text

@pytest.fixture
def example_data():
    if False:
        return 10
    return 'Today I found a duck and I am happy'

def test_extract_sentiment(example_data):
    if False:
        print('Hello World!')
    sentiment = extract_sentiment(example_data)
    assert sentiment > 0

def test_text_contain_word(example_data):
    if False:
        i = 10
        return i + 15
    word = 'duck'
    assert text_contain_word(word, example_data) == True