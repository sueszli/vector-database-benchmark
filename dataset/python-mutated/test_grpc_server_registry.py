import sys
import threading
import time
from unittest import mock
import pytest
from dagster import file_relative_path, job, repository
from dagster._core.errors import DagsterUserCodeProcessError
from dagster._core.host_representation.code_location import GrpcServerCodeLocation
from dagster._core.host_representation.grpc_server_registry import GrpcServerRegistry
from dagster._core.host_representation.origin import ManagedGrpcPythonEnvCodeLocationOrigin, RegisteredCodeLocationOrigin
from dagster._core.test_utils import instance_for_test
from dagster._core.types.loadable_target_origin import LoadableTargetOrigin
from dagster._grpc.server import GrpcServerProcess

@job
def noop_job():
    if False:
        while True:
            i = 10
    pass

@repository
def repo():
    if False:
        print('Hello World!')
    return [noop_job]

@repository
def other_repo():
    if False:
        for i in range(10):
            print('nop')
    return [noop_job]

def _can_connect(origin, endpoint):
    if False:
        print('Hello World!')
    try:
        with GrpcServerCodeLocation(origin=origin, server_id=endpoint.server_id, port=endpoint.port, socket=endpoint.socket, host=endpoint.host, watch_server=False):
            return True
    except Exception:
        return False

@pytest.fixture
def instance():
    if False:
        return 10
    with instance_for_test() as instance:
        yield instance

def test_error_repo_in_registry(instance):
    if False:
        return 10
    error_origin = ManagedGrpcPythonEnvCodeLocationOrigin(loadable_target_origin=LoadableTargetOrigin(executable_path=sys.executable, attribute='error_repo', python_file=file_relative_path(__file__, 'error_repo.py')))
    with GrpcServerRegistry(instance_ref=instance.get_ref(), reload_interval=5, heartbeat_ttl=10, startup_timeout=5, wait_for_processes_on_shutdown=True) as registry:
        endpoint = registry.get_grpc_endpoint(error_origin)
        with pytest.raises(DagsterUserCodeProcessError, match='object is not callable'):
            with GrpcServerCodeLocation(origin=error_origin, server_id=endpoint.server_id, port=endpoint.port, socket=endpoint.socket, host=endpoint.host, watch_server=False):
                pass
        with pytest.raises(DagsterUserCodeProcessError, match='object is not callable'):
            with GrpcServerCodeLocation(origin=error_origin, server_id=endpoint.server_id, port=endpoint.port, socket=endpoint.socket, host=endpoint.host, watch_server=False):
                pass

def test_server_registry(instance):
    if False:
        for i in range(10):
            print('nop')
    origin = ManagedGrpcPythonEnvCodeLocationOrigin(loadable_target_origin=LoadableTargetOrigin(executable_path=sys.executable, attribute='repo', python_file=file_relative_path(__file__, 'test_grpc_server_registry.py')))
    with GrpcServerRegistry(instance_ref=instance.get_ref(), reload_interval=5, heartbeat_ttl=10, startup_timeout=5, wait_for_processes_on_shutdown=True) as registry:
        endpoint_one = registry.get_grpc_endpoint(origin)
        endpoint_two = registry.get_grpc_endpoint(origin)
        assert endpoint_two == endpoint_one
        assert _can_connect(origin, endpoint_one)
        assert _can_connect(origin, endpoint_two)
        start_time = time.time()
        while True:
            endpoint_three = registry.get_grpc_endpoint(origin)
            if endpoint_three.server_id != endpoint_one.server_id:
                break
            if time.time() - start_time > 15:
                raise Exception('Server ID never changed')
            time.sleep(1)
        assert _can_connect(origin, endpoint_three)
        start_time = time.time()
        while True:
            if not _can_connect(origin, endpoint_one):
                break
            if time.time() - start_time > 30:
                raise Exception('Old Server never died after process manager released it')
            time.sleep(1)
        while True:
            endpoint_four = registry.get_grpc_endpoint(origin)
            if endpoint_four.server_id != endpoint_three.server_id:
                assert _can_connect(origin, endpoint_four)
                break
    assert not _can_connect(origin, endpoint_three)
    assert not _can_connect(origin, endpoint_four)

def _registry_thread(origin, registry, endpoint, event):
    if False:
        i = 10
        return i + 15
    if registry.get_grpc_endpoint(origin) == endpoint:
        event.set()

def test_registry_multithreading(instance):
    if False:
        print('Hello World!')
    origin = ManagedGrpcPythonEnvCodeLocationOrigin(loadable_target_origin=LoadableTargetOrigin(executable_path=sys.executable, attribute='repo', python_file=file_relative_path(__file__, 'test_grpc_server_registry.py')))
    with GrpcServerRegistry(instance_ref=instance.get_ref(), reload_interval=300, heartbeat_ttl=600, startup_timeout=30, wait_for_processes_on_shutdown=True) as registry:
        endpoint = registry.get_grpc_endpoint(origin)
        threads = []
        success_events = []
        for _index in range(5):
            event = threading.Event()
            thread = threading.Thread(target=_registry_thread, args=(origin, registry, endpoint, event))
            threads.append(thread)
            success_events.append(event)
            thread.start()
        for thread in threads:
            thread.join()
        for event in success_events:
            assert event.is_set()
        assert _can_connect(origin, endpoint)
    assert not _can_connect(origin, endpoint)

class TestMockProcessGrpcServerRegistry(GrpcServerRegistry):

    def __init__(self, instance):
        if False:
            print('Hello World!')
        self.mocked_loadable_target_origin = None
        super(TestMockProcessGrpcServerRegistry, self).__init__(instance_ref=instance.get_ref(), reload_interval=300, heartbeat_ttl=600, startup_timeout=30, wait_for_processes_on_shutdown=True)

    def supports_origin(self, code_location_origin):
        if False:
            i = 10
            return i + 15
        return isinstance(code_location_origin, RegisteredCodeLocationOrigin)

    def _get_loadable_target_origin(self, code_location_origin):
        if False:
            for i in range(10):
                print('nop')
        return self.mocked_loadable_target_origin

def test_custom_loadable_target_origin(instance):
    if False:
        for i in range(10):
            print('nop')
    first_loadable_target_origin = LoadableTargetOrigin(executable_path=sys.executable, attribute='repo', python_file=file_relative_path(__file__, 'test_grpc_server_registry.py'))
    second_loadable_target_origin = LoadableTargetOrigin(executable_path=sys.executable, attribute='other_repo', python_file=file_relative_path(__file__, 'test_grpc_server_registry.py'))
    origin = RegisteredCodeLocationOrigin('test_location')
    with TestMockProcessGrpcServerRegistry(instance) as registry:
        registry.mocked_loadable_target_origin = first_loadable_target_origin
        endpoint_one = registry.get_grpc_endpoint(origin)
        assert registry.get_grpc_endpoint(origin).server_id == endpoint_one.server_id
        registry.mocked_loadable_target_origin = second_loadable_target_origin
        endpoint_two = registry.get_grpc_endpoint(origin)
        assert endpoint_two.server_id != endpoint_one.server_id
    registry.wait_for_processes()
    assert not _can_connect(origin, endpoint_one)
    assert not _can_connect(origin, endpoint_two)

def test_failure_on_open_server_process(instance):
    if False:
        for i in range(10):
            print('nop')
    loadable_target_origin = LoadableTargetOrigin(executable_path=sys.executable, attribute='repo', python_file=file_relative_path(__file__, 'test_grpc_server_registry.py'))
    with mock.patch('dagster._grpc.server.open_server_process') as mock_open_server_process:
        mock_open_server_process.side_effect = Exception('OOPS')
        with pytest.raises(Exception, match='OOPS'):
            with GrpcServerProcess(instance_ref=instance.get_ref(), loadable_target_origin=loadable_target_origin):
                pass