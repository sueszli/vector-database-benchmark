import os
import language_classify_gcs

def test_sample_classify_text_gcs(capsys: ...) -> None:
    if False:
        print('Hello World!')
    assert os.environ['GOOGLE_CLOUD_PROJECT'] != ''
    language_classify_gcs.sample_classify_text()
    captured = capsys.readouterr()
    assert 'Category name: ' in captured.out
    assert 'Confidence: ' in captured.out