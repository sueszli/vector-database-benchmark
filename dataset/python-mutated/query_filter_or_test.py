import os
from google.cloud import datastore
import pytest
from query_filter_or import query_filter_or
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

@pytest.fixture()
def entities():
    if False:
        for i in range(10):
            print('nop')
    client = datastore.Client(project=PROJECT_ID)
    task_key = client.key('Task')
    task1 = datastore.Entity(key=task_key)
    task1['description'] = 'Buy milk'
    client.put(task1)
    task_key2 = client.key('Task')
    task2 = datastore.Entity(key=task_key2)
    task2['description'] = 'Feed cats'
    client.put(task2)
    yield entities
    client.delete(task1)
    client.delete(task2)

def test_query_filter_or(capsys, entities):
    if False:
        while True:
            i = 10
    query_filter_or(project_id=PROJECT_ID)
    (out, _) = capsys.readouterr()
    assert 'Feed cats' in out