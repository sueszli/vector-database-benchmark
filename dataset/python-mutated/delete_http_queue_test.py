import uuid
from google.api_core.exceptions import NotFound
import google.auth
from google.cloud import tasks_v2beta3 as tasks
import pytest
import delete_http_queue
HOST = 'example.com'
LOCATION = 'us-central1'

@pytest.fixture
def q():
    if False:
        return 10
    (_, project) = google.auth.default()
    name = 'tests-tasks-' + uuid.uuid4().hex
    http_target = {'uri_override': {'host': HOST, 'uri_override_enforce_mode': 2}}
    client = tasks.CloudTasksClient()
    queue = client.create_queue(tasks.CreateQueueRequest(parent=client.common_location_path(project, LOCATION), queue={'name': f'projects/{project}/locations/{LOCATION}/queues/{name}', 'http_target': http_target}))
    yield queue
    try:
        client.delete_queue(name=queue.name)
    except Exception as e:
        if type(e) == NotFound:
            pass
        else:
            print(f'Tried my best to clean up, but could not: {e}')

def test_delete_http_queue(q) -> None:
    if False:
        while True:
            i = 10
    name = q.name
    delete_http_queue.delete_http_queue(q)
    client = tasks.CloudTasksClient()
    with pytest.raises(Exception) as exc_info:
        client.get_queue(name=name)
    assert exc_info.typename == 'NotFound'