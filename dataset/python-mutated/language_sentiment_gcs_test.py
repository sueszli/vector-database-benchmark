import os
import language_sentiment_gcs

def test_sample_analyze_sentiment_gcs(capsys: ...) -> None:
    if False:
        while True:
            i = 10
    assert os.environ['GOOGLE_CLOUD_PROJECT'] != ''
    language_sentiment_gcs.sample_analyze_sentiment()
    captured = capsys.readouterr()
    assert 'Document sentiment score: ' in captured.out
    assert 'Document sentiment magnitude: ' in captured.out
    assert 'Sentence text: ' in captured.out
    assert 'Sentence sentiment score: ' in captured.out
    assert 'Sentence sentiment magnitude: ' in captured.out
    assert 'Language of the text: ' in captured.out