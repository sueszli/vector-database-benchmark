import os
import pytest
import translate_v3_translate_text_with_glossary
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
GLOSSARY_ID = 'DO_NOT_DELETE_TEST_GLOSSARY'

def test_translate_text_with_glossary(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    response = translate_v3_translate_text_with_glossary.translate_text_with_glossary('account', PROJECT_ID, GLOSSARY_ID)
    (out, _) = capsys.readouterr()
    assert 'アカウント' or '口座' in out
    assert response is not None