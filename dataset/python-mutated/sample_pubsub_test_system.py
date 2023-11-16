from datetime import datetime
from os import getenv
import subprocess
import time
import uuid
from google.cloud import pubsub_v1
import pytest
PROJECT = getenv('GCP_PROJECT')
TOPIC = getenv('FUNCTIONS_TOPIC')
assert PROJECT is not None
assert TOPIC is not None

@pytest.fixture(scope='module')
def publisher_client():
    if False:
        for i in range(10):
            print('nop')
    yield pubsub_v1.PublisherClient()

def test_print_name(publisher_client):
    if False:
        for i in range(10):
            print('nop')
    start_time = datetime.utcnow().isoformat()
    topic_path = publisher_client.topic_path(PROJECT, TOPIC)
    name = uuid.uuid4()
    data = str(name).encode('utf-8')
    publisher_client.publish(topic_path, data=data).result()
    time.sleep(15)
    log_process = subprocess.Popen(['gcloud', 'alpha', 'functions', 'logs', 'read', 'hello_pubsub', '--start-time', start_time], stdout=subprocess.PIPE)
    logs = str(log_process.communicate()[0])
    assert f'Hello {name}!' in logs