import uuid
import google.auth
from google.cloud import tasks_v2beta3 as tasks
import create_http_queue

def test_create() -> None:
    if False:
        i = 10
        return i + 15
    (_, project) = google.auth.default()
    name = 'tests-tasks-' + uuid.uuid4().hex
    q = create_http_queue.create_http_queue(project, 'us-central1', name, 'http://example.com/')
    assert q is not None
    assert q.http_target.uri_override is not None
    assert q.http_target.uri_override.host == 'example.com'
    assert q.http_target.uri_override.scheme == 1
    try:
        client = tasks.Client()
        client.delete_queue(name=q.name)
    except Exception as e:
        print(f'Tried my best to clean up, but could not: {e}')