import datetime
from sacred.observers.queue import QueueObserver
import mock
import pytest
import sys
if sys.version_info >= (3, 10):
    pytest.skip("Skip pymongo tests for Python 3.10 because mongomock doesn't support Python 3.10", allow_module_level=True)
from sacred.metrics_logger import ScalarMetricLogEntry, linearize_metrics
pymongo = pytest.importorskip('pymongo')
mongomock = pytest.importorskip('mongomock')
import gridfs
from mongomock.gridfs import enable_gridfs_integration
enable_gridfs_integration()
import pymongo.errors
from sacred.dependencies import get_digest
from sacred.observers.mongo import QueuedMongoObserver, MongoObserver, QueueCompatibleMongoObserver
from .failing_mongo_mock import ReconnectingMongoClient
T1 = datetime.datetime(1999, 5, 4, 3, 2, 1)
T2 = datetime.datetime(1999, 5, 5, 5, 5, 5)

@pytest.fixture
def mongo_obs(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    client = ReconnectingMongoClient(max_calls_before_reconnect=10, max_calls_before_failure=1, exception_to_raise=pymongo.errors.ServerSelectionTimeoutError)
    fs = gridfs.GridFS(client.sacred)
    monkeypatch.setattr(pymongo, 'MongoClient', lambda *args, **kwargs: client)
    monkeypatch.setattr(gridfs, 'GridFS', lambda _: fs)
    return QueuedMongoObserver(interval=0.01, retry_interval=0.01)

@pytest.fixture()
def sample_run():
    if False:
        return 10
    exp = {'name': 'test_exp', 'sources': [], 'doc': '', 'base_dir': '/tmp'}
    host = {'hostname': 'test_host', 'cpu_count': 1, 'python_version': '3.4'}
    config = {'config': 'True', 'foo': 'bar', 'answer': 42}
    command = 'run'
    meta_info = {'comment': 'test run'}
    return {'_id': 'FEDCBA9876543210', 'ex_info': exp, 'command': command, 'host_info': host, 'start_time': T1, 'config': config, 'meta_info': meta_info}

def test_mongo_observer_started_event_creates_run(mongo_obs, sample_run):
    if False:
        return 10
    sample_run['_id'] = None
    _id = mongo_obs.started_event(**sample_run)
    mongo_obs.join()
    assert _id is not None
    assert mongo_obs.runs.count_documents({}) == 1
    db_run = mongo_obs.runs.find_one()
    assert db_run == {'_id': _id, 'experiment': sample_run['ex_info'], 'format': mongo_obs.VERSION, 'command': sample_run['command'], 'host': sample_run['host_info'], 'start_time': sample_run['start_time'], 'heartbeat': None, 'info': {}, 'captured_out': '', 'artifacts': [], 'config': sample_run['config'], 'meta': sample_run['meta_info'], 'status': 'RUNNING', 'resources': []}

def test_mongo_observer_started_event_uses_given_id(mongo_obs, sample_run):
    if False:
        print('Hello World!')
    _id = mongo_obs.started_event(**sample_run)
    mongo_obs.join()
    assert _id == sample_run['_id']
    assert mongo_obs.runs.count_documents({}) == 1
    db_run = mongo_obs.runs.find_one()
    assert db_run['_id'] == sample_run['_id']

def test_mongo_observer_equality(mongo_obs):
    if False:
        print('Hello World!')
    runs = mongo_obs.runs
    mongo_obs.join()
    fs = mock.MagicMock()
    m = MongoObserver.create_from(runs, fs)
    assert mongo_obs == m
    assert not mongo_obs != m
    assert not mongo_obs == 'foo'
    assert mongo_obs != 'foo'

def test_mongo_observer_heartbeat_event_updates_run(mongo_obs, sample_run):
    if False:
        return 10
    mongo_obs.started_event(**sample_run)
    info = {'my_info': [1, 2, 3], 'nr': 7}
    outp = 'some output'
    mongo_obs.heartbeat_event(info=info, captured_out=outp, beat_time=T2, result=1337)
    mongo_obs.join()
    assert mongo_obs.runs.count_documents({}) == 1
    db_run = mongo_obs.runs.find_one()
    assert db_run['heartbeat'] == T2
    assert db_run['result'] == 1337
    assert db_run['info'] == info
    assert db_run['captured_out'] == outp

def test_mongo_observer_completed_event_updates_run(mongo_obs, sample_run):
    if False:
        for i in range(10):
            print('nop')
    mongo_obs.started_event(**sample_run)
    mongo_obs.completed_event(stop_time=T2, result=42)
    assert mongo_obs.runs.count_documents({}) == 1
    db_run = mongo_obs.runs.find_one()
    assert db_run['stop_time'] == T2
    assert db_run['result'] == 42
    assert db_run['status'] == 'COMPLETED'

def test_mongo_observer_interrupted_event_updates_run(mongo_obs, sample_run):
    if False:
        return 10
    mongo_obs.started_event(**sample_run)
    mongo_obs.interrupted_event(interrupt_time=T2, status='INTERRUPTED')
    assert mongo_obs.runs.count_documents({}) == 1
    db_run = mongo_obs.runs.find_one()
    assert db_run['stop_time'] == T2
    assert db_run['status'] == 'INTERRUPTED'

def test_mongo_observer_failed_event_updates_run(mongo_obs, sample_run):
    if False:
        while True:
            i = 10
    mongo_obs.started_event(**sample_run)
    fail_trace = 'lots of errors and\nso\non...'
    mongo_obs.failed_event(fail_time=T2, fail_trace=fail_trace)
    assert mongo_obs.runs.count_documents({}) == 1
    db_run = mongo_obs.runs.find_one()
    assert db_run['stop_time'] == T2
    assert db_run['status'] == 'FAILED'
    assert db_run['fail_trace'] == fail_trace

def test_mongo_observer_artifact_event(mongo_obs, sample_run):
    if False:
        while True:
            i = 10
    mongo_obs.started_event(**sample_run)
    filename = 'setup.py'
    name = 'mysetup'
    mongo_obs.artifact_event(name, filename)
    mongo_obs.join()
    [file] = mongo_obs.fs.list()
    assert file.endswith(name)
    db_run = mongo_obs.runs.find_one()
    assert db_run['artifacts']

def test_mongo_observer_resource_event(mongo_obs, sample_run):
    if False:
        i = 10
        return i + 15
    mongo_obs.started_event(**sample_run)
    filename = 'setup.py'
    md5 = get_digest(filename)
    mongo_obs.resource_event(filename)
    info = {'my_info': [1, 2, 3], 'nr': 7}
    outp = 'some output'
    mongo_obs.heartbeat_event(info=info, captured_out=outp, beat_time=T2, result=1337)
    mongo_obs.join()
    db_run = mongo_obs.runs.find_one()
    assert db_run['resources'][0] == [filename, md5]

@pytest.fixture
def logged_metrics():
    if False:
        while True:
            i = 10
    return [ScalarMetricLogEntry('training.loss', 10, datetime.datetime.utcnow(), 1), ScalarMetricLogEntry('training.loss', 20, datetime.datetime.utcnow(), 2), ScalarMetricLogEntry('training.loss', 30, datetime.datetime.utcnow(), 3), ScalarMetricLogEntry('training.accuracy', 10, datetime.datetime.utcnow(), 100), ScalarMetricLogEntry('training.accuracy', 20, datetime.datetime.utcnow(), 200), ScalarMetricLogEntry('training.accuracy', 30, datetime.datetime.utcnow(), 300), ScalarMetricLogEntry('training.loss', 40, datetime.datetime.utcnow(), 10), ScalarMetricLogEntry('training.loss', 50, datetime.datetime.utcnow(), 20), ScalarMetricLogEntry('training.loss', 60, datetime.datetime.utcnow(), 30)]

def test_log_metrics(mongo_obs, sample_run, logged_metrics):
    if False:
        return 10
    "\n    Test storing scalar measurements\n\n    Test whether measurements logged using _run.metrics.log_scalar_metric\n    are being stored in the 'metrics' collection\n    and that the experiment 'info' dictionary contains a valid reference\n    to the metrics collection for each of the metric.\n\n    Metrics are identified by name (e.g.: 'training.loss') and by the\n    experiment run that produced them. Each metric contains a list of x values\n    (e.g. iteration step), y values (measured values) and timestamps of when\n    each of the measurements was taken.\n    "
    mongo_obs.started_event(**sample_run)
    info = {'my_info': [1, 2, 3], 'nr': 7}
    outp = 'some output'
    mongo_obs.log_metrics(linearize_metrics(logged_metrics[:6]), info)
    mongo_obs.heartbeat_event(info=info, captured_out=outp, beat_time=T1, result=0)
    mongo_obs.log_metrics(linearize_metrics(logged_metrics[6:]), info)
    mongo_obs.heartbeat_event(info=info, captured_out=outp, beat_time=T2, result=0)
    mongo_obs.join()
    assert mongo_obs.runs.count_documents({}) == 1
    db_run = mongo_obs.runs.find_one()
    assert 'metrics' in db_run['info']
    assert mongo_obs.metrics.count_documents({}) == 2
    loss = mongo_obs.metrics.find_one({'name': 'training.loss', 'run_id': db_run['_id']})
    assert {'name': 'training.loss', 'id': str(loss['_id'])} in db_run['info']['metrics']
    assert loss['steps'] == [10, 20, 30, 40, 50, 60]
    assert loss['values'] == [1, 2, 3, 10, 20, 30]
    for i in range(len(loss['timestamps']) - 1):
        assert loss['timestamps'][i] <= loss['timestamps'][i + 1]
    accuracy = mongo_obs.metrics.find_one({'name': 'training.accuracy', 'run_id': db_run['_id']})
    assert {'name': 'training.accuracy', 'id': str(accuracy['_id'])} in db_run['info']['metrics']
    assert accuracy['steps'] == [10, 20, 30]
    assert accuracy['values'] == [100, 200, 300]
    sample_run['_id'] = 'NEWID'
    mongo_obs.started_event(**sample_run)
    mongo_obs.log_metrics(linearize_metrics(logged_metrics[:4]), info)
    mongo_obs.heartbeat_event(info=info, captured_out=outp, beat_time=T1, result=0)
    mongo_obs.join()
    assert mongo_obs.runs.count_documents({}) == 2
    assert mongo_obs.metrics.count_documents({}) == 4

def test_mongo_observer_artifact_event_content_type_added(mongo_obs, sample_run):
    if False:
        print('Hello World!')
    'Test that the detected content_type is added to other metadata.'
    mongo_obs.started_event(**sample_run)
    filename = 'setup.py'
    name = 'mysetup'
    mongo_obs.artifact_event(name, filename)
    mongo_obs.join()
    file = mongo_obs.fs.find_one({})
    assert file.content_type == 'text/x-python'
    db_run = mongo_obs.runs.find_one()
    assert db_run['artifacts']

def test_mongo_observer_artifact_event_content_type_not_overwritten(mongo_obs, sample_run):
    if False:
        return 10
    'Test that manually set content_type is not overwritten by automatic detection.'
    mongo_obs.started_event(**sample_run)
    filename = 'setup.py'
    name = 'mysetup'
    mongo_obs.artifact_event(name, filename, content_type='application/json')
    mongo_obs.join()
    file = mongo_obs.fs.find_one({})
    assert file.content_type == 'application/json'
    db_run = mongo_obs.runs.find_one()
    assert db_run['artifacts']

def test_mongo_observer_artifact_event_metadata(mongo_obs, sample_run):
    if False:
        while True:
            i = 10
    'Test that the detected content-type is added to other metadata.'
    mongo_obs.started_event(**sample_run)
    filename = 'setup.py'
    name = 'mysetup'
    mongo_obs.artifact_event(name, filename, metadata={'comment': 'the setup file'})
    mongo_obs.join()
    file = mongo_obs.fs.find_one({})
    assert file.metadata['comment'] == 'the setup file'
    db_run = mongo_obs.runs.find_one()
    assert db_run['artifacts']