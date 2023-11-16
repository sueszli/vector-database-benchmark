import os
import uuid
from google.api_core.retry import Retry
from google.cloud import tasks_v2
import create_queue
TEST_PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
TEST_LOCATION = os.getenv('TEST_QUEUE_LOCATION', 'us-central1')
TEST_QUEUE_ID = f'my-queue-{uuid.uuid4().hex}'

@Retry()
def test_create_queue() -> None:
    if False:
        while True:
            i = 10
    client = tasks_v2.CloudTasksClient()
    queue = create_queue.create_queue(TEST_PROJECT_ID, TEST_LOCATION, TEST_QUEUE_ID)
    assert queue.name == client.queue_path(TEST_PROJECT_ID, TEST_LOCATION, TEST_QUEUE_ID)
    client.delete_queue(tasks_v2.DeleteQueueRequest(name=queue.name))