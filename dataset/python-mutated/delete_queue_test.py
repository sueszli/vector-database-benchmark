import os
import uuid
from google.api_core import exceptions
from google.cloud import tasks_v2
import pytest
import delete_queue
TEST_PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
TEST_LOCATION = os.getenv('TEST_QUEUE_LOCATION', 'us-central1')
TEST_QUEUE_ID = f'my-queue-{uuid.uuid4().hex}'

def test_delete_queue() -> None:
    if False:
        while True:
            i = 10
    client = tasks_v2.CloudTasksClient()
    client.create_queue(tasks_v2.CreateQueueRequest(parent=client.common_location_path(TEST_PROJECT_ID, TEST_LOCATION), queue=tasks_v2.Queue(name=client.queue_path(TEST_PROJECT_ID, TEST_LOCATION, TEST_QUEUE_ID))))
    delete_queue.delete_queue(TEST_PROJECT_ID, TEST_LOCATION, TEST_QUEUE_ID)
    with pytest.raises(exceptions.NotFound):
        delete_queue.delete_queue(TEST_PROJECT_ID, TEST_LOCATION, TEST_QUEUE_ID)