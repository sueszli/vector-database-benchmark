import language_sentiment_text

def test_analyze_sentiment_text_positive(capsys):
    if False:
        while True:
            i = 10
    language_sentiment_text.sample_analyze_sentiment('Happy Happy Joy Joy')
    (out, _) = capsys.readouterr()
    assert 'Score: 0.' in out

def test_analyze_sentiment_text_negative(capsys):
    if False:
        print('Hello World!')
    language_sentiment_text.sample_analyze_sentiment('Angry Angry Sad Sad')
    (out, _) = capsys.readouterr()
    assert 'Score: -0.' in out