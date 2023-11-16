import os
import uuid
import backoff
from google.api_core.exceptions import DeadlineExceeded, GoogleAPICallError
from google.cloud.exceptions import NotFound
import pytest
import translate_v3_create_glossary
import translate_v3_delete_glossary
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
GLOSSARY_INPUT_URI = 'gs://cloud-samples-data/translation/glossary_ja.csv'

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_create_glossary(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    try:
        glossary_id = f'test-{uuid.uuid4()}'
        result = translate_v3_create_glossary.create_glossary(PROJECT_ID, GLOSSARY_INPUT_URI, glossary_id)
        (out, _) = capsys.readouterr()
        assert 'gs://cloud-samples-data/translation/glossary_ja.csv' in result.input_config.gcs_source.input_uri
    finally:

        @backoff.on_exception(backoff.expo, (DeadlineExceeded, GoogleAPICallError), max_time=60)
        def delete_glossary() -> None:
            if False:
                return 10
            try:
                translate_v3_delete_glossary.delete_glossary(PROJECT_ID, glossary_id)
            except NotFound as e:
                print(f'Got NotFound, detail: {str(e)}')
        delete_glossary()