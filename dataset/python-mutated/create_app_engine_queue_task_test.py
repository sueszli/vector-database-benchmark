import os
import create_app_engine_queue_task
TEST_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
TEST_LOCATION = os.getenv('TEST_QUEUE_LOCATION', 'us-central1')
TEST_QUEUE_NAME = os.getenv('TEST_QUEUE_NAME', 'my-appengine-queue')

def test_create_task():
    if False:
        return 10
    result = create_app_engine_queue_task.create_task(TEST_PROJECT_ID, TEST_QUEUE_NAME, TEST_LOCATION)
    assert TEST_QUEUE_NAME in result.name