import os
import subprocess
import sys
import warnings
from unittest.mock import Mock, patch
import pytest
import ray
import ray.client_builder as client_builder
import ray.util.client.server.server as ray_client_server
from ray._private.test_utils import run_string_as_driver, run_string_as_driver_nonblocking, wait_for_condition
from ray.util.state import list_workers

@pytest.mark.parametrize('address', ['localhost:1234', 'localhost:1234/url?params', '1.2.3.4/cluster-1?test_param=param1?', ''])
def test_split_address(address):
    if False:
        i = 10
        return i + 15
    assert client_builder._split_address(address) == ('ray', address)
    specified_module = f'ray://{address}'
    assert client_builder._split_address(specified_module) == ('ray', address)
    specified_other_module = f'module://{address}'
    assert client_builder._split_address(specified_other_module) == ('module', address)
    non_url_compliant_module = f'module_test://{address}'
    assert client_builder._split_address(non_url_compliant_module) == ('module_test', address)

@pytest.mark.parametrize('address', ['localhost', '1.2.3.4:1200', 'ray://1.2.3.4:1200', 'local', None])
def test_client(address):
    if False:
        while True:
            i = 10
    builder = client_builder.client(address)
    assert isinstance(builder, client_builder.ClientBuilder)
    if address in ('local', None):
        assert isinstance(builder, client_builder._LocalClientBuilder)
    else:
        assert type(builder) == client_builder.ClientBuilder
        assert builder.address == address.replace('ray://', '')

def test_namespace(ray_start_cluster):
    if False:
        i = 10
        return i + 15
    '\n    Most of the "checks" in this test case rely on the fact that\n    `run_string_as_driver` will throw an exception if the driver string exits\n    with a non-zero exit code (e.g. when the driver scripts throws an\n    exception). Since all of these drivers start named, detached actors, the\n    most likely failure case would be a collision of named actors if they\'re\n    put in the same namespace.\n\n    This test checks that:\n    * When two drivers don\'t specify a namespace, they are placed in different\n      anonymous namespaces.\n    * When two drivers specify a namespace, they collide.\n    * The namespace name (as provided by the runtime context) is correct.\n    '
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=4, ray_client_server_port=50055)
    cluster.wait_for_nodes(1)
    template = '\nimport ray\nray.client("localhost:50055").namespace({namespace}).connect()\n\n@ray.remote\nclass Foo:\n    def ping(self):\n        return "pong"\n\na = Foo.options(lifetime="detached", name="abc").remote()\nray.get(a.ping.remote())\nprint("Current namespace:", ray.get_runtime_context().namespace)\n    '
    anon_driver = template.format(namespace='None')
    run_string_as_driver(anon_driver)
    run_string_as_driver(anon_driver)
    run_in_namespace = template.format(namespace="'namespace'")
    script_output = run_string_as_driver(run_in_namespace)
    with pytest.raises(subprocess.CalledProcessError):
        run_string_as_driver(run_in_namespace)
    assert 'Current namespace: namespace' in script_output
    subprocess.check_output('ray stop --force', shell=True)

def test_connect_to_cluster(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15
    server = ray_client_server.serve('localhost:50055')
    with ray.client('localhost:50055').connect() as client_context:
        assert client_context.dashboard_url == ray._private.worker.get_dashboard_url()
        python_version = '.'.join([str(x) for x in list(sys.version_info)[:3]])
        assert client_context.python_version == python_version
        assert client_context.ray_version == ray.__version__
        assert client_context.ray_commit == ray.__commit__
        protocol_version = ray.util.client.CURRENT_PROTOCOL_VERSION
        assert client_context.protocol_version == protocol_version
    server.stop(0)
    subprocess.check_output('ray stop --force', shell=True)

@pytest.mark.skipif(sys.platform == 'win32', reason='Flaky on Windows.')
def test_local_clusters():
    if False:
        while True:
            i = 10
    '\n    This tests the various behaviors of connecting to local clusters:\n\n    * Using `ray.client("local").connect() ` should always create a new\n      cluster.\n    * Using `ray.cleint().connectIO` should create a new cluster if it doesn\'t\n      connect to an existing one.\n    * Using `ray.client().connect()` should only connect to a cluster if it\n      was created with `ray start --head`, not from a python program.\n\n    It does tests if two calls are in the same cluster by trying to create an\n    actor with the same name in the same namespace, which will error and cause\n    the script have a non-zero exit, which throws an exception.\n    '
    driver_template = '\nimport ray\ninfo = ray.client({address}).namespace("").connect()\n\n@ray.remote\nclass Foo:\n    def ping(self):\n        return "pong"\n\na = Foo.options(name="abc", lifetime="detached").remote()\nray.get(a.ping.remote())\n\nimport time\nwhile True:\n    time.sleep(30)\n\n'
    blocking_local_script = driver_template.format(address="'local'", blocking=True)
    blocking_noaddr_script = driver_template.format(address='', blocking=True)
    p1 = run_string_as_driver_nonblocking(blocking_local_script)
    p2 = run_string_as_driver_nonblocking(blocking_local_script)
    p3 = run_string_as_driver_nonblocking(blocking_noaddr_script)
    p4 = run_string_as_driver_nonblocking(blocking_noaddr_script)
    wait_for_condition(lambda : len(ray._private.services.find_gcs_addresses()) == 4, retry_interval_ms=1000)
    p1.kill()
    p2.kill()
    p3.kill()
    p4.kill()
    subprocess.check_output('ray stop --force', shell=True)
    subprocess.check_output('ray start --head', shell=True)
    run_string_as_driver('\nimport ray\nray.client().connect()\nassert len(ray._private.services.find_gcs_addresses()) == 1\n    ')
    p1 = run_string_as_driver_nonblocking(blocking_local_script)
    wait_for_condition(lambda : len(ray._private.services.find_gcs_addresses()) == 2, retry_interval_ms=1000)
    p1.kill()
    subprocess.check_output('ray stop --force', shell=True)

def test_non_existent_modules():
    if False:
        for i in range(10):
            print('nop')
    exception = None
    try:
        ray.client('badmodule://address')
    except RuntimeError as e:
        exception = e
    assert exception is not None, 'Bad Module did not raise RuntimeException'
    assert 'does not exist' in str(exception)

def test_module_lacks_client_builder():
    if False:
        i = 10
        return i + 15
    mock_importlib = Mock()

    def mock_import_module(module_string):
        if False:
            return 10
        if module_string == 'ray':
            return ray
        else:
            return Mock()
    mock_importlib.import_module = mock_import_module
    with patch('ray.client_builder.importlib', mock_importlib):
        assert isinstance(ray.client(''), ray.ClientBuilder)
        assert isinstance(ray.client('ray://'), ray.ClientBuilder)
        exception = None
        try:
            ray.client('othermodule://')
        except AssertionError as e:
            exception = e
        assert exception is not None, 'Module without ClientBuilder did not raise AssertionError'
        assert 'does not have ClientBuilder' in str(exception)

@pytest.mark.skipif(sys.platform == 'win32', reason='RC Proxy is Flaky on Windows.')
def test_disconnect(call_ray_stop_only, set_enable_auto_connect):
    if False:
        for i in range(10):
            print('nop')
    subprocess.check_output('ray start --head --ray-client-server-port=25555', shell=True)
    with ray.client('localhost:25555').namespace('n1').connect():
        namespace = ray.get_runtime_context().namespace
        assert namespace == 'n1'
        assert ray.util.client.ray.is_connected()
    with pytest.raises(ray.exceptions.RaySystemError):
        ray.put(300)
    with ray.client(None).namespace('n1').connect():
        namespace = ray.get_runtime_context().namespace
        assert namespace == 'n1'
        assert not ray.util.client.ray.is_connected()
    with pytest.raises(ray.exceptions.RaySystemError):
        ray.put(300)
    ctx = ray.client('localhost:25555').namespace('n1').connect()
    namespace = ray.get_runtime_context().namespace
    assert namespace == 'n1'
    assert ray.util.client.ray.is_connected()
    ctx.disconnect()
    ctx.disconnect()
    with pytest.raises(ray.exceptions.RaySystemError):
        ray.put(300)

@pytest.mark.skipif(sys.platform == 'win32', reason='RC Proxy is Flaky on Windows.')
def test_address_resolution(call_ray_stop_only):
    if False:
        return 10
    subprocess.check_output('ray start --head --ray-client-server-port=50055', shell=True)
    with ray.client('localhost:50055').connect():
        assert ray.util.client.ray.is_connected()
    try:
        os.environ['RAY_ADDRESS'] = 'local'
        with ray.client('localhost:50055').connect():
            assert ray.util.client.ray.is_connected()
        with ray.client(None).connect():
            wait_for_condition(lambda : len(ray._private.services.find_gcs_addresses()) == 2, retry_interval_ms=1000)
    finally:
        if os.environ.get('RAY_ADDRESS'):
            del os.environ['RAY_ADDRESS']
        ray.shutdown()

def mock_connect(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Force exit instead of actually attempting to connect\n    '
    raise ConnectionError

def has_client_deprecation_warn(warning: Warning, expected_replacement: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns true if expected_replacement is in the message of the passed\n    warning, and that the warning mentions deprecation.\n    '
    start = 'Starting a connection through `ray.client` will be deprecated'
    message = str(warning.message)
    if start not in message:
        return False
    if expected_replacement not in message:
        return False
    return True

@pytest.mark.skipif(sys.platform == 'win32', reason='pip not supported in Windows runtime envs.')
@pytest.mark.filterwarnings('default:Starting a connection through `ray.client` will be deprecated')
def test_client_deprecation_warn():
    if False:
        print('Hello World!')
    '\n    Tests that calling ray.client directly raises a deprecation warning with\n    a copy pasteable replacement for the client().connect() call converted\n    to ray.init style.\n    '
    with warnings.catch_warnings(record=True) as w:
        ray.client().connect()
        assert any((has_client_deprecation_warn(warning, 'ray.init()') for warning in w))
        ray.shutdown()
    with warnings.catch_warnings(record=True) as w:
        ray.client().namespace('nmspc').env({'pip': ['requests']}).connect()
    expected = 'ray.init(namespace="nmspc", runtime_env=<your_runtime_env>)'
    assert any((has_client_deprecation_warn(warning, expected) for warning in w))
    ray.shutdown()
    server = ray_client_server.serve('localhost:50055')
    with warnings.catch_warnings(record=True) as w:
        with ray.client('localhost:50055').connect():
            pass
    assert any((has_client_deprecation_warn(warning, 'ray.init("ray://localhost:50055")') for warning in w))
    with warnings.catch_warnings(record=True) as w:
        with ray.client('localhost:50055').namespace('nmspc').connect():
            pass
    assert any((has_client_deprecation_warn(warning, 'ray.init("ray://localhost:50055", namespace="nmspc")') for warning in w))
    with warnings.catch_warnings(record=True) as w, patch.dict(os.environ, {'RAY_NAMESPACE': 'aksdj'}):
        with ray.client('localhost:50055').connect():
            pass
    assert any((has_client_deprecation_warn(warning, 'ray.init("ray://localhost:50055")') for warning in w))
    with patch('ray.util.client_connect.connect', mock_connect):
        with warnings.catch_warnings(record=True) as w:
            try:
                ray.client('localhost:50055').env({'pip': ['requests']}).connect()
            except ConnectionError:
                pass
        expected = 'ray.init("ray://localhost:50055", runtime_env=<your_runtime_env>)'
        assert any((has_client_deprecation_warn(warning, expected) for warning in w))
        with warnings.catch_warnings(record=True) as w:
            try:
                ray.client('localhost:50055').namespace('nmspc').env({'pip': ['requests']}).connect()
            except ConnectionError:
                pass
        expected = 'ray.init("ray://localhost:50055", namespace="nmspc", runtime_env=<your_runtime_env>)'
        assert any((has_client_deprecation_warn(warning, expected) for warning in w))
        with warnings.catch_warnings(record=True) as w, patch.dict(os.environ, {'RAY_NAMESPACE': 'abcdef'}):
            try:
                ray.client('localhost:50055').env({'pip': ['requests']}).connect()
            except ConnectionError:
                pass
        expected = 'ray.init("ray://localhost:50055", runtime_env=<your_runtime_env>)'
        assert any((has_client_deprecation_warn(warning, expected) for warning in w))
    server.stop(0)
    subprocess.check_output('ray stop --force', shell=True)

@pytest.mark.parametrize('call_ray_start', ['ray start --head --num-cpus=2 --min-worker-port=0 --max-worker-port=0 --port 0 --ray-client-server-port=50056'], indirect=True)
def test_task_use_prestarted_worker(call_ray_start):
    if False:
        return 10
    ray.init('ray://localhost:50056')
    assert len(list_workers(filters=[('worker_type', '!=', 'DRIVER')])) == 2

    @ray.remote(num_cpus=2)
    def f():
        if False:
            for i in range(10):
                print('nop')
        return 42
    assert ray.get(f.remote()) == 42
    assert len(list_workers(filters=[('worker_type', '!=', 'DRIVER')])) == 2
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))