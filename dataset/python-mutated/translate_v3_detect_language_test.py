import os
import pytest
import translate_v3_detect_language
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_detect_language(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    translate_v3_detect_language.detect_language(PROJECT_ID)
    (out, _) = capsys.readouterr()
    assert 'en' in out