import uuid
import google.auth
from google.cloud import storage
import pytest
import async_api
TEST_UUID = uuid.uuid4()
BUCKET = f'optimization-ai-{TEST_UUID}'
OUTPUT_PREFIX = f'code_snippets_test_output_{TEST_UUID}'
INPUT_URI = 'gs://cloud-samples-data/optimization-ai/async_request_model.json'
BATCH_OUTPUT_URI_PREFIX = f'gs://{BUCKET}/{OUTPUT_PREFIX}/'

@pytest.fixture(autouse=True)
def setup_teardown() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Create a temporary bucket to store optimization output.'
    storage_client = storage.Client()
    bucket = storage_client.create_bucket(BUCKET)
    yield
    bucket.delete(force=True)

def test_call_async_api(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        print('Hello World!')
    (_, project_id) = google.auth.default()
    async_api.call_async_api(project_id, INPUT_URI, BATCH_OUTPUT_URI_PREFIX)
    (out, _) = capsys.readouterr()
    assert 'operations' in out