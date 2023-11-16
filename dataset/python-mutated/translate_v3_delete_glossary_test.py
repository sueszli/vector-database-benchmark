import os
import uuid
import pytest
import translate_v3_create_glossary
import translate_v3_delete_glossary
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
GLOSSARY_INPUT_URI = 'gs://cloud-samples-data/translation/glossary_ja.csv'

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_delete_glossary(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        return 10
    glossary_id = f'test-{uuid.uuid4()}'
    translate_v3_create_glossary.create_glossary(PROJECT_ID, GLOSSARY_INPUT_URI, glossary_id)
    result = translate_v3_delete_glossary.delete_glossary(PROJECT_ID, glossary_id)
    (out, _) = capsys.readouterr()
    assert 'Deleted:' in out
    assert glossary_id in result.name