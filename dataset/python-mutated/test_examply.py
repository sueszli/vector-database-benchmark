from textblob import TextBlob

def extract_sentiment(text: str):
    if False:
        i = 10
        return i + 15
    'Extract sentiment using textblob. \n        Polarity is within range [-1, 1]'
    text = TextBlob(text)
    print(text.sentiment)
    return text.sentiment.polarity

def test_extract_sentiment():
    if False:
        i = 10
        return i + 15
    text = 'I think today will be a great day'
    sentiment = extract_sentiment(text)
    assert sentiment > 0