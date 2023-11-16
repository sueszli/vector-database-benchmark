import uuid
import google.auth
from google.cloud import tasks_v2beta3 as tasks
import pytest
import get_http_queue
HOST = 'example.com'
LOCATION = 'us-central1'

@pytest.fixture
def q():
    if False:
        i = 10
        return i + 15
    (_, project) = google.auth.default()
    name = 'tests-tasks-' + uuid.uuid4().hex
    http_target = {'uri_override': {'host': HOST, 'uri_override_enforce_mode': 2}}
    client = tasks.CloudTasksClient()
    queue = client.create_queue(tasks.CreateQueueRequest(parent=client.common_location_path(project, LOCATION), queue={'name': f'projects/{project}/locations/{LOCATION}/queues/{name}', 'http_target': http_target}))
    yield queue
    try:
        client.delete_queue(name=queue.name)
    except Exception as e:
        print(f'Tried my best to clean up, but could not: {e}')

def test_get_http_queue(q) -> None:
    if False:
        while True:
            i = 10
    (_, project, _, location, _, name) = q.name.split('/')
    q2 = get_http_queue.get_http_queue(project, location, name)
    assert q2 is not None
    assert q2 == q