import os
import sys
import unittest.mock
import signal
import subprocess
import grpc
import pytest
import ray
import ray._private.services
from ray.client_builder import ClientContext
from ray.cluster_utils import Cluster
from ray.util.client.common import ClientObjectRef
from ray.util.client.ray_client_helpers import ray_start_client_server
from ray.util.client.worker import Worker
from ray._private.test_utils import wait_for_condition, enable_external_redis

@pytest.mark.skipif(os.environ.get('CI') and sys.platform == 'win32', reason='Flaky when run on windows CI')
@pytest.mark.parametrize('input', [None, 'auto'])
def test_ray_address(input, call_ray_start):
    if False:
        print('Hello World!')
    address = call_ray_start
    with unittest.mock.patch.dict(os.environ, {'RAY_ADDRESS': address}):
        res = ray.init(input)
        assert not isinstance(res, ClientContext)
        assert res.address_info['gcs_address'] == address
        ray.shutdown()
    addr = 'localhost:{}'.format(address.split(':')[-1])
    with unittest.mock.patch.dict(os.environ, {'RAY_ADDRESS': addr}):
        res = ray.init(input)
        assert not isinstance(res, ClientContext)
        assert res.address_info['gcs_address'] == address
        ray.shutdown()

@pytest.mark.parametrize('address', [None, 'auto'])
def test_ray_init_no_local_instance(shutdown_only, address):
    if False:
        i = 10
        return i + 15
    if address is None:
        ray.init(address=address)
    else:
        with pytest.raises(ConnectionError):
            ray.init(address=address)

@pytest.mark.skipif(os.environ.get('CI') and sys.platform == 'win32', reason='Flaky when run on windows CI')
@pytest.mark.parametrize('address', [None, 'auto'])
def test_ray_init_existing_instance(call_ray_start, address):
    if False:
        for i in range(10):
            print('nop')
    ray_address = call_ray_start
    res = ray.init(address=address)
    assert res.address_info['gcs_address'] == ray_address
    ray.shutdown()
    try:
        subprocess.check_output('ray start --head', shell=True)
        res = ray.init(address=address)
        assert res.address_info['gcs_address'] != ray_address
        ray.shutdown()
        with unittest.mock.patch.dict(os.environ, {'RAY_ADDRESS': ray_address}):
            res = ray.init(address=address)
            assert res.address_info['gcs_address'] == ray_address
    finally:
        ray.shutdown()
        subprocess.check_output('ray stop --force', shell=True)

@pytest.mark.skipif(os.environ.get('CI') and sys.platform == 'win32', reason='Flaky when run on windows CI')
def test_ray_init_existing_instance_via_blocked_ray_start():
    if False:
        return 10
    blocked = subprocess.Popen(['ray', 'start', '--head', '--block', '--num-cpus', '1999'])

    def _connect_to_existing_instance():
        if False:
            i = 10
            return i + 15
        while True:
            try:
                ray.init()
                if ray.cluster_resources().get('CPU', 0) == 1999:
                    return True
                else:
                    return False
            except Exception:
                return False
            finally:
                ray.shutdown()
    try:
        wait_for_condition(_connect_to_existing_instance, timeout=30, retry_interval_ms=1000)
    finally:
        blocked.terminate()
        blocked.wait()
        subprocess.check_output('ray stop --force', shell=True)

@pytest.mark.skipif(os.environ.get('CI') and sys.platform == 'win32', reason='Flaky when run on windows CI')
@pytest.mark.parametrize('address', [None, 'auto'])
def test_ray_init_existing_instance_crashed(address):
    if False:
        print('Hello World!')
    ray._private.utils.write_ray_address('localhost:6379')
    try:
        ray._private.node.NUM_REDIS_GET_RETRIES = 1
        with pytest.raises(ConnectionError):
            ray.init(address=address)
    finally:
        ray._private.utils.reset_ray_address()

class Credentials(grpc.ChannelCredentials):

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.name = name

class Stop(Exception):

    def __init__(self, credentials):
        if False:
            for i in range(10):
                print('nop')
        self.credentials = credentials

def test_ray_init_credentials_with_client(monkeypatch):
    if False:
        i = 10
        return i + 15

    def mock_init(self, conn_str='', secure=False, metadata=None, connection_retries=3, _credentials=None):
        if False:
            print('Hello World!')
        raise Stop(_credentials)
    monkeypatch.setattr(Worker, '__init__', mock_init)
    with pytest.raises(Stop) as stop:
        with ray_start_client_server(_credentials=Credentials('test')):
            pass
    assert stop.value.credentials.name == 'test'

def test_ray_init_credential(monkeypatch):
    if False:
        print('Hello World!')

    def mock_secure_channel(conn_str, credentials, options=None, compression=None):
        if False:
            print('Hello World!')
        raise Stop(credentials)
    monkeypatch.setattr(grpc, 'secure_channel', mock_secure_channel)
    with pytest.raises(Stop) as stop:
        ray.init('ray://127.0.0.1', _credentials=Credentials('test'))
    ray.util.disconnect()
    assert stop.value.credentials.name == 'test'

def test_auto_init_non_client(call_ray_start):
    if False:
        return 10
    address = call_ray_start
    with unittest.mock.patch.dict(os.environ, {'RAY_ADDRESS': address}):
        res = ray.put(300)
        assert not isinstance(res, ClientObjectRef)
        ray.shutdown()
    addr = 'localhost:{}'.format(address.split(':')[-1])
    with unittest.mock.patch.dict(os.environ, {'RAY_ADDRESS': addr}):
        res = ray.put(300)
        assert not isinstance(res, ClientObjectRef)

@pytest.mark.parametrize('call_ray_start', ['ray start --head --ray-client-server-port 25036 --port 0'], indirect=True)
@pytest.mark.parametrize('function', [lambda : ray.put(300), lambda : ray.remote(ray.nodes).remote()])
def test_auto_init_client(call_ray_start, function):
    if False:
        while True:
            i = 10
    address = call_ray_start.split(':')[0]
    with unittest.mock.patch.dict(os.environ, {'RAY_ADDRESS': f'ray://{address}:25036'}):
        res = function()
        assert isinstance(res, ClientObjectRef)
        ray.shutdown()
    with unittest.mock.patch.dict(os.environ, {'RAY_ADDRESS': 'ray://localhost:25036'}):
        res = function()
        assert isinstance(res, ClientObjectRef)

@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='This test is only run on linux CI machines.')
def test_ray_init_using_hostname(ray_start_cluster):
    if False:
        print('Hello World!')
    import socket
    hostname = socket.gethostname()
    cluster = Cluster(initialize_head=True, head_node_args={'node_ip_address': hostname})
    ray.init(address=cluster.address, _node_ip_address=hostname)
    node_table = cluster.global_state.node_table()
    assert len(node_table) == 1
    assert node_table[0].get('NodeManagerHostname', '') == hostname

def test_new_ray_instance_new_session_dir(shutdown_only):
    if False:
        i = 10
        return i + 15
    ray.init()
    session_dir = ray._private.worker._global_node.get_session_dir_path()
    ray.shutdown()
    ray.init()
    if enable_external_redis():
        assert ray._private.worker._global_node.get_session_dir_path() == session_dir
    else:
        assert ray._private.worker._global_node.get_session_dir_path() != session_dir

def test_new_cluster_new_session_dir(ray_start_cluster):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster
    cluster.add_node()
    ray.init(address=cluster.address)
    session_dir = ray._private.worker._global_node.get_session_dir_path()
    ray.shutdown()
    cluster.shutdown()
    cluster.add_node()
    ray.init(address=cluster.address)
    if enable_external_redis():
        assert ray._private.worker._global_node.get_session_dir_path() == session_dir
    else:
        assert ray._private.worker._global_node.get_session_dir_path() != session_dir
    ray.shutdown()
    cluster.shutdown()

@pytest.mark.skipif(sys.platform == 'win32', reason='SIGTERM only on posix')
def test_ray_init_sigterm_handler():
    if False:
        i = 10
        return i + 15
    TEST_FILENAME = 'sigterm.txt'

    def sigterm_handler_cmd(ray_init=False):
        if False:
            return 10
        return f'''\nimport os\nimport sys\nimport signal\ndef sigterm_handler(signum, frame):\n    f = open("{TEST_FILENAME}", "w")\n    sys.exit(0)\nsignal.signal(signal.SIGTERM, sigterm_handler)\n\nimport ray\n{('ray.init()' if ray_init else '')}\nos.kill(os.getpid(), signal.SIGTERM)\n'''
    test_child = subprocess.run(['python', '-c', sigterm_handler_cmd()])
    assert test_child.returncode == 0 and os.path.exists(TEST_FILENAME)
    os.remove(TEST_FILENAME)
    test_child = subprocess.run(['python', '-c', sigterm_handler_cmd(ray_init=True)])
    assert test_child.returncode == signal.SIGTERM and (not os.path.exists(TEST_FILENAME))
if __name__ == '__main__':
    import sys
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))