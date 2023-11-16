import os
import pytest
import translate_v3_get_glossary
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
GLOSSARY_ID = 'DO_NOT_DELETE_TEST_GLOSSARY'

def test_get_glossary(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    response = translate_v3_get_glossary.get_glossary(PROJECT_ID, GLOSSARY_ID)
    (out, _) = capsys.readouterr()
    assert 'gs://cloud-samples-data/translation/glossary_ja.csv' in out
    assert 'gs' in response.input_config.gcs_source.input_uri