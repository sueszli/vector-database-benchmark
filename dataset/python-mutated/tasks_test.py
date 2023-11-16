import os
import uuid
import backoff
from google.cloud import datastore
import pytest
import tasks
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

@pytest.fixture
def client():
    if False:
        return 10
    namespace = uuid.uuid4().hex
    client = datastore.Client(PROJECT, namespace=namespace)
    with client.batch():
        client.delete_multi([x.key for x in client.query(kind='Task').fetch()])
    yield client
    with client.batch():
        client.delete_multi([x.key for x in client.query(kind='Task').fetch()])

@pytest.mark.flaky
def test_create_client():
    if False:
        print('Hello World!')
    tasks.create_client(PROJECT)

@pytest.mark.flaky
def test_add_task(client):
    if False:
        for i in range(10):
            print('nop')
    task_key = tasks.add_task(client, 'Test task')
    task = client.get(task_key)
    assert task
    assert task['description'] == 'Test task'

@pytest.mark.flaky
def test_mark_done(client):
    if False:
        print('Hello World!')
    task_key = tasks.add_task(client, 'Test task')
    tasks.mark_done(client, task_key.id)
    task = client.get(task_key)
    assert task
    assert task['done']

@pytest.mark.flaky
def test_list_tasks(client):
    if False:
        print('Hello World!')
    task1_key = tasks.add_task(client, 'Test task 1')
    task2_key = tasks.add_task(client, 'Test task 2')

    @backoff.on_exception(backoff.expo, AssertionError, max_time=120)
    def _():
        if False:
            return 10
        task_list = tasks.list_tasks(client)
        assert [x.key for x in task_list] == [task1_key, task2_key]

@pytest.mark.flaky
def test_delete_task(client):
    if False:
        return 10
    task_key = tasks.add_task(client, 'Test task 1')
    tasks.delete_task(client, task_key.id)
    assert client.get(task_key) is None

@pytest.mark.flaky
def test_format_tasks(client):
    if False:
        i = 10
        return i + 15
    task1_key = tasks.add_task(client, 'Test task 1')
    tasks.add_task(client, 'Test task 2')
    tasks.mark_done(client, task1_key.id)

    @backoff.on_exception(backoff.expo, AssertionError, max_time=120)
    def run_sample():
        if False:
            return 10
        output = tasks.format_tasks(tasks.list_tasks(client))
        assert 'Test task 1' in output
        assert 'Test task 2' in output
        assert 'done' in output
        assert 'created' in output
    run_sample()