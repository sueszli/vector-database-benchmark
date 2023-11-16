import json
import subprocess
import uuid
import google.auth
from google.cloud import tasks_v2beta3 as tasks
import pytest
import send_task_to_http_queue
HOST = 'example.com'
LOCATION = 'us-central1'

@pytest.fixture
def q():
    if False:
        print('Hello World!')
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

def get_access_token():
    if False:
        for i in range(10):
            print('nop')
    output = subprocess.run('gcloud auth application-default print-access-token --quiet --format=json', capture_output=True, shell=True, check=True)
    entries = json.loads(output.stdout)
    return entries['token']

def test_send_task_to_http_queue(q) -> None:
    if False:
        i = 10
        return i + 15
    token = get_access_token()
    result = send_task_to_http_queue.send_task_to_http_queue(q, body='something', token=token)
    assert result < 400