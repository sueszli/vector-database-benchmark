import os
import language_entities_gcs

def test_sample_analyze_entities_gcs(capsys: ...) -> None:
    if False:
        i = 10
        return i + 15
    assert os.environ['GOOGLE_CLOUD_PROJECT'] != ''
    language_entities_gcs.sample_analyze_entities()
    captured = capsys.readouterr()
    assert 'Representative name for the entity: ' in captured.out
    assert 'Entity type: ' in captured.out
    assert 'Mention text: ' in captured.out
    assert 'Mention type: ' in captured.out
    assert 'Probability score: ' in captured.out
    assert 'Language of the text: ' in captured.out