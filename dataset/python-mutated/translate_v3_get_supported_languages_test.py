import os
import pytest
import translate_v3_get_supported_languages
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_list_languages(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        print('Hello World!')
    response = translate_v3_get_supported_languages.get_supported_languages(PROJECT_ID)
    (out, _) = capsys.readouterr()
    assert 'zh' in out
    assert response is not None