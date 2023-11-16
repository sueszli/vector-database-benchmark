import os
import pytest
import translate_v3_get_supported_languages_with_target as get_supported_langs
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_list_languages_with_target(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        while True:
            i = 10
    response = get_supported_langs.get_supported_languages_with_target(PROJECT_ID)
    (out, _) = capsys.readouterr()
    assert 'Language Code: sq' in out
    assert 'Display Name: albanska' in out
    assert response is not None