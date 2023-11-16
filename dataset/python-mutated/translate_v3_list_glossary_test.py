import os
import pytest
import translate_v3_list_glossary
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
GLOSSARY_ID = 'DO_NOT_DELETE_TEST_GLOSSARY'

def test_list_glossary(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        while True:
            i = 10
    glossary = translate_v3_list_glossary.list_glossaries(PROJECT_ID)
    (out, _) = capsys.readouterr()
    assert 'gs://cloud-samples-data/translation/glossary_ja.csv' in out
    assert glossary is not None