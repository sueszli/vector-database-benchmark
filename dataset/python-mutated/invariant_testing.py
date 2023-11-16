from behave import given, then, when
from textblob import TextBlob

def get_sentiment(sent: str):
    if False:
        i = 10
        return i + 15
    return TextBlob(sent).sentiment.polarity

@given('a text')
def step_given_positive_sentiment(context):
    if False:
        for i in range(10):
            print('nop')
    context.original = 'The hotel room was great! It was spacious, clean and had a nice view of the city.'

@when('the text is paraphrased')
def step_when_paraphrased(context):
    if False:
        for i in range(10):
            print('nop')
    context.paraphrased = "The hotel room wasn't bad. It wasn't cramped, dirty, and had a decent view of the city."

@then('both text should have the same sentiment')
def step_then_sentiment_analysis(context):
    if False:
        return 10
    sentiment_original = get_sentiment(context.original)
    sentiment_paraphrased = get_sentiment(context.paraphrased)
    print(f'Sentiment of the original text: {sentiment_original:.2f}')
    print(f'Sentiment of the paraphrased sentence: {sentiment_paraphrased:.2f}')
    both_positive = sentiment_original > 0 and sentiment_paraphrased > 0
    both_negative = sentiment_original < 0 and sentiment_paraphrased < 0
    assert both_positive or both_negative