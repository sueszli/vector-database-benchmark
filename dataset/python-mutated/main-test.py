import os
import pytest
import uuid
os.environ['LOCATION'] = 'us-central1'
os.environ['QUEUE'] = str(uuid.uuid4())
import main
TEST_NAME = 'taskqueue-migration-' + os.environ['QUEUE']
TEST_TASKS = {'alpha': 2, 'beta': 1, 'gamma': 3}

@pytest.fixture(scope='module')
def queue():
    if False:
        i = 10
        return i + 15
    project = main.project
    location = main.location
    parent = 'projects/{}/locations/{}'.format(project, location)
    queue = main.client.create_queue(parent=parent, queue={'name': parent + '/queues/' + TEST_NAME})
    yield queue
    main.client.delete_queue(name='{}/queues/{}'.format(parent, TEST_NAME))

@pytest.fixture(scope='module')
def entity_kind():
    if False:
        print('Hello World!')
    yield TEST_NAME
    datastore_client = main.datastore_client
    query = datastore_client.query(kind=TEST_NAME)
    keys = [entity.key for entity in query.fetch()]
    datastore_client.delete_multi(keys)

def test_get_home_page(queue, entity_kind):
    if False:
        while True:
            i = 10
    save_queue = main.queue_name
    save_entity_kind = main.entity_kind
    main.queue = queue.name
    main.entity_kind = entity_kind
    main.app.testing = True
    client = main.app.test_client()
    r = client.get('/')
    assert r.status_code == 200
    assert 'Counters' in r.data.decode('utf-8')
    assert '<li>' not in r.data.decode('utf-8')
    main.queue_name = save_queue
    main.entity_kind = save_entity_kind

def test_enqueuetasks(queue):
    if False:
        return 10
    save_queue = main.queue
    main.queue = queue.name
    main.app.testing = True
    client = main.app.test_client()
    for task in TEST_TASKS:
        for i in range(TEST_TASKS[task]):
            r = client.post('/', data={'key': task})
            assert r.status_code == 302
            assert r.headers.get('location').count('/') == 3
    counters_found = {}
    tasks = main.client.list_tasks(parent=queue.name)
    for task in tasks:
        details = main.client.get_task(request={'name': task.name, 'response_view': main.tasks.Task.View.FULL})
        key = details.app_engine_http_request.body.decode()
        if key not in counters_found:
            counters_found[key] = 0
        counters_found[key] += 1
    for key in TEST_TASKS:
        assert key in counters_found
        assert TEST_TASKS[key] == counters_found[key]
    for key in counters_found:
        assert key in TEST_TASKS
        assert counters_found[key] == TEST_TASKS[key]
    main.queue = save_queue

def test_processtasks(entity_kind):
    if False:
        while True:
            i = 10
    save_entity_kind = main.entity_kind
    main.entity_kind = entity_kind
    main.app.testing = True
    client = main.app.test_client()
    for key in TEST_TASKS:
        for i in range(TEST_TASKS[key]):
            r = client.post('/push-task', data=key, content_type='text/plain', headers=[('X-AppEngine-QueueName', main.queue_name)])
        assert r.status_code == 200
        assert r.data == b'OK'
    r = client.post('/push-task', data=key, content_type='text/plain', headers=[('X-AppEngine-QueueName', 'WRONG-NAME')])
    assert r.status_code == 200
    assert r.data == b'REJECTED'
    r = client.post('/push-task', data=key, content_type='text/plain')
    assert r.status_code == 200
    assert r.data == b'REJECTED'
    r = client.get('/')
    assert r.status_code == 200
    assert 'Counters' in r.data.decode('utf-8')
    for key in TEST_TASKS:
        assert '{}: {}'.format(key, TEST_TASKS[key]) in r.data.decode('utf-8')
    main.entity_kind = save_entity_kind