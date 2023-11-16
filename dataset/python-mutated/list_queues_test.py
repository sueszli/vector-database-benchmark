import os
import uuid
from google.api_core.retry import Retry
from google.cloud import tasks_v2
import list_queues
TEST_PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
TEST_LOCATION = os.getenv('TEST_QUEUE_LOCATION', 'us-central1')

@Retry()
def test_list_queues() -> None:
    if False:
        return 10
    client = tasks_v2.CloudTasksClient()
    queue = client.create_queue(tasks_v2.CreateQueueRequest(parent=client.common_location_path(TEST_PROJECT_ID, TEST_LOCATION), queue=tasks_v2.Queue(name=client.queue_path(TEST_PROJECT_ID, TEST_LOCATION, f'my-queue-{uuid.uuid4().hex}'))))
    assert queue.name in list_queues.list_queues(TEST_PROJECT_ID, TEST_LOCATION)
    client.delete_queue(tasks_v2.DeleteQueueRequest(name=queue.name))
    assert queue.name not in list_queues.list_queues(TEST_PROJECT_ID, TEST_LOCATION)