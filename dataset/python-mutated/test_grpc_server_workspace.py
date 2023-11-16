from contextlib import ExitStack
import pytest
import yaml
from dagster import _seven
from dagster._check import CheckError
from dagster._core.errors import DagsterUserCodeUnreachableError
from dagster._core.host_representation import GrpcServerCodeLocationOrigin
from dagster._core.test_utils import environ, instance_for_test
from dagster._core.workspace.load import location_origins_from_config
from dagster._grpc.server import GrpcServerProcess
from dagster._utils import file_relative_path

@pytest.fixture
def instance():
    if False:
        print('Hello World!')
    with instance_for_test() as instance:
        yield instance

@pytest.mark.skipif(_seven.IS_WINDOWS, reason='no named sockets on Windows')
def test_grpc_socket_workspace(instance):
    if False:
        print('Hello World!')
    with GrpcServerProcess(instance_ref=instance.get_ref(), wait_on_exit=True) as first_server_process:
        first_server = first_server_process.create_client()
        with GrpcServerProcess(instance_ref=instance.get_ref(), wait_on_exit=True) as second_server_process:
            second_server = second_server_process.create_client()
            first_socket = first_server.socket
            second_socket = second_server.socket
            workspace_yaml = f"\nload_from:\n- grpc_server:\n    host: localhost\n    socket: {first_socket}\n- grpc_server:\n    socket: {second_socket}\n    location_name: 'local_port_default_host'\n                "
            origins = location_origins_from_config(yaml.safe_load(workspace_yaml), file_relative_path(__file__, 'not_a_real.yaml'))
            with ExitStack() as stack:
                code_locations = {name: stack.enter_context(origin.create_location()) for (name, origin) in origins.items()}
                assert len(code_locations) == 2
                default_location_name = f'grpc:localhost:{first_socket}'
                assert code_locations.get(default_location_name)
                local_port = code_locations.get(default_location_name)
                assert local_port.socket == first_socket
                assert local_port.host == 'localhost'
                assert local_port.port is None
                assert code_locations.get('local_port_default_host')
                local_port_default_host = code_locations.get('local_port_default_host')
                assert local_port_default_host.socket == second_socket
                assert local_port_default_host.host == 'localhost'
                assert local_port_default_host.port is None
                assert all(map(lambda x: x.name, code_locations.values()))

def test_grpc_server_env_vars():
    if False:
        while True:
            i = 10
    with environ({'FOO_PORT': '1234', 'FOO_SOCKET': 'barsocket', 'FOO_HOST': 'barhost'}):
        valid_yaml = "\n    load_from:\n        - grpc_server:\n            host:\n              env: FOO_HOST\n            port:\n              env: FOO_PORT\n            location_name: 'my_grpc_server_port'\n        - grpc_server:\n            host:\n              env: FOO_HOST\n            socket:\n              env: FOO_SOCKET\n            location_name: 'my_grpc_server_socket'\n    "
        origins = location_origins_from_config(yaml.safe_load(valid_yaml), file_relative_path(__file__, 'not_a_real.yaml'))
        assert len(origins) == 2
        port_origin = origins['my_grpc_server_port']
        assert isinstance(origins['my_grpc_server_port'], GrpcServerCodeLocationOrigin)
        assert port_origin.port == 1234
        assert port_origin.host == 'barhost'
        socket_origin = origins['my_grpc_server_socket']
        assert isinstance(origins['my_grpc_server_socket'], GrpcServerCodeLocationOrigin)
        assert socket_origin.socket == 'barsocket'
        assert socket_origin.host == 'barhost'

def test_ssl_grpc_server_workspace(instance):
    if False:
        print('Hello World!')
    with GrpcServerProcess(instance_ref=instance.get_ref(), force_port=True, wait_on_exit=True) as server_process:
        client = server_process.create_client()
        assert client.heartbeat(echo='Hello')
        port = server_process.port
        ssl_yaml = f'\nload_from:\n- grpc_server:\n    host: localhost\n    port: {port}\n    ssl: true\n'
        origins = location_origins_from_config(yaml.safe_load(ssl_yaml), file_relative_path(__file__, 'not_a_real.yaml'))
        origin = next(iter(origins.values()))
        assert origin.use_ssl
        try:
            with origin.create_location():
                assert False
        except DagsterUserCodeUnreachableError:
            pass

def test_grpc_server_workspace(instance):
    if False:
        i = 10
        return i + 15
    with GrpcServerProcess(instance_ref=instance.get_ref(), force_port=True, wait_on_exit=True) as first_server_process:
        first_server = first_server_process.create_client()
        with GrpcServerProcess(instance_ref=instance.get_ref(), force_port=True, wait_on_exit=True) as second_server_process:
            second_server = second_server_process.create_client()
            first_port = first_server.port
            second_port = second_server.port
            workspace_yaml = f"\nload_from:\n- grpc_server:\n    host: localhost\n    port: {first_port}\n- grpc_server:\n    port: {second_port}\n    location_name: 'local_port_default_host'\n                "
            origins = location_origins_from_config(yaml.safe_load(workspace_yaml), file_relative_path(__file__, 'not_a_real.yaml'))
            with ExitStack() as stack:
                code_locations = {name: stack.enter_context(origin.create_location()) for (name, origin) in origins.items()}
                assert len(code_locations) == 2
                default_location_name = f'grpc:localhost:{first_port}'
                assert code_locations.get(default_location_name)
                local_port = code_locations.get(default_location_name)
                assert local_port.port == first_port
                assert local_port.host == 'localhost'
                assert local_port.socket is None
                assert code_locations.get('local_port_default_host')
                local_port_default_host = code_locations.get('local_port_default_host')
                assert local_port_default_host.port == second_port
                assert local_port_default_host.host == 'localhost'
                assert local_port_default_host.socket is None
                assert all(map(lambda x: x.name, code_locations.values()))

def test_cannot_set_socket_and_port():
    if False:
        return 10
    workspace_yaml = '\nload_from:\n  - grpc_server:\n      socket: myname\n      port: 5678\n    '
    with pytest.raises(CheckError, match='must supply either a socket or a port'):
        location_origins_from_config(yaml.safe_load(workspace_yaml), file_relative_path(__file__, 'not_a_real.yaml'))