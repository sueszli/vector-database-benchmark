import uuid
import google.auth
from google.cloud import tasks_v2beta3 as tasks
import pytest
import update_http_queue
HOST = 'example.com'
LOCATION = 'us-central1'

@pytest.fixture
def q():
    if False:
        while True:
            i = 10
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

def test_update_http_queue(q) -> None:
    if False:
        return 10
    print(f'Queue name is {q.name}')
    q = update_http_queue.update_http_queue(q, uri='https://example.com/somepath')
    assert q.http_target.uri_override.scheme != 1
    assert q.http_target.uri_override.path_override.path == '/somepath'
    print(f'Queue name is {q.name}')
    q = update_http_queue.update_http_queue(q, max_per_second=5.0, max_attempts=2)
    assert q.rate_limits is not None
    assert q.rate_limits.max_dispatches_per_second == 5.0
    assert q.retry_config.max_attempts == 2