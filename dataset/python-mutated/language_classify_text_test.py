import os
import language_classify_text

def test_sample_classify_text(capsys: ...) -> None:
    if False:
        i = 10
        return i + 15
    assert os.environ['GOOGLE_CLOUD_PROJECT'] != ''
    language_classify_text.sample_classify_text()
    captured = capsys.readouterr()
    assert 'Category name: ' in captured.out
    assert 'Confidence: ' in captured.out