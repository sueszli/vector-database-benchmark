import os
from typing import Generator
import uuid
from google.api_core.retry import Retry
from google.cloud import tasks_v2
import pytest
import create_http_task
TEST_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
TEST_LOCATION = os.getenv('TEST_QUEUE_LOCATION', 'us-central1')
TEST_QUEUE_ID = f'my-queue-{uuid.uuid4().hex}'

@pytest.fixture()
def test_queue() -> Generator[tasks_v2.Queue, None, None]:
    if False:
        while True:
            i = 10
    client = tasks_v2.CloudTasksClient()
    queue = client.create_queue(tasks_v2.CreateQueueRequest(parent=client.common_location_path(TEST_PROJECT_ID, TEST_LOCATION), queue=tasks_v2.Queue(name=client.queue_path(TEST_PROJECT_ID, TEST_LOCATION, TEST_QUEUE_ID))))
    yield queue
    client.delete_queue(request={'name': queue.name})

@Retry()
def test_create_http_task(test_queue: tasks_v2.Queue) -> None:
    if False:
        print('Hello World!')
    task = create_http_task.create_http_task(TEST_PROJECT_ID, TEST_LOCATION, TEST_QUEUE_ID, 'https://example.com/task_handler', json_payload={'greeting': 'hola'}, scheduled_seconds_from_now=180, task_id=uuid.uuid4().hex, deadline_in_seconds=900)
    assert task.name.startswith(test_queue.name)
    assert task.http_request.url == 'https://example.com/task_handler'