from typing import Iterable, List, Optional
from google.api_core.exceptions import InternalServerError, ServiceUnavailable, TooManyRequests
from google.cloud import storage
import pytest
from test_utils.retry import RetryErrors
from test_utils.system import unique_resource_id
from . import create_migration_workflow
retry_storage_errors = RetryErrors((TooManyRequests, InternalServerError, ServiceUnavailable))
storage_client = storage.Client()
PROJECT_ID = storage_client.project

def _create_bucket(bucket_name: str, location: Optional[str]=None) -> storage.Bucket:
    if False:
        print('Hello World!')
    bucket = storage_client.bucket(bucket_name)
    retry_storage_errors(storage_client.create_bucket)(bucket_name, location=location)
    return bucket

@pytest.fixture
def buckets_to_delete() -> Iterable[List]:
    if False:
        i = 10
        return i + 15
    doomed = []
    yield doomed
    for item in doomed:
        if isinstance(item, storage.Bucket):
            retry_storage_errors(item.delete)(force=True)

def test_create_migration_workflow(capsys: pytest.CaptureFixture, buckets_to_delete: List[storage.Bucket]) -> None:
    if False:
        while True:
            i = 10
    bucket_name = 'bq_migration_create_workflow_test' + unique_resource_id()
    path = f'gs://{PROJECT_ID}/{bucket_name}'
    bucket = _create_bucket(bucket_name)
    buckets_to_delete.extend([bucket])
    create_migration_workflow.create_migration_workflow(path, path, PROJECT_ID)
    (out, _) = capsys.readouterr()
    assert 'demo-workflow-python-example-Teradata2BQ' in out
    assert 'Current state:' in out