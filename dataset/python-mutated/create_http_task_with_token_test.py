import os
from typing import Generator
import uuid
from google.api_core.retry import Retry
from google.cloud import tasks_v2
import pytest
from create_http_task_with_token import create_http_task_with_token
TEST_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
TEST_LOCATION = os.getenv('TEST_QUEUE_LOCATION', 'us-central1')
TEST_QUEUE_ID = f'my-queue-{uuid.uuid4().hex}'
TEST_SERVICE_ACCOUNT = 'test-run-invoker@python-docs-samples-tests.iam.gserviceaccount.com'

@pytest.fixture()
def test_queue() -> Generator[tasks_v2.Queue, None, None]:
    if False:
        i = 10
        return i + 15
    client = tasks_v2.CloudTasksClient()
    queue = client.create_queue(tasks_v2.CreateQueueRequest(parent=client.common_location_path(TEST_PROJECT_ID, TEST_LOCATION), queue=tasks_v2.Queue(name=client.queue_path(TEST_PROJECT_ID, TEST_LOCATION, TEST_QUEUE_ID))))
    yield queue
    client.delete_queue(tasks_v2.DeleteQueueRequest(name=queue.name))

@Retry()
def test_create_http_task_with_token(test_queue: tasks_v2.Queue) -> None:
    if False:
        while True:
            i = 10
    task = create_http_task_with_token(TEST_PROJECT_ID, TEST_LOCATION, TEST_QUEUE_ID, 'https://example.com/task_handler', b'my-payload', TEST_SERVICE_ACCOUNT)
    assert task.name.startswith(test_queue.name)
    assert task.http_request.url == 'https://example.com/task_handler'
    assert task.http_request.oidc_token.service_account_email == TEST_SERVICE_ACCOUNT