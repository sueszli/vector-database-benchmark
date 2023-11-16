import os
import subprocess
import sys
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
import pytest
import requests
import ray
import ray._private.state
import ray.actor
from ray import serve
from ray._private.test_utils import SignalActor, wait_for_condition
from ray.cluster_utils import AutoscalingCluster, Cluster
from ray.exceptions import RayActorError
from ray.serve._private.common import ProxyStatus
from ray.serve._private.constants import SERVE_DEFAULT_APP_NAME
from ray.serve._private.logging_utils import get_serve_logs_dir
from ray.serve._private.utils import get_head_node_id
from ray.serve.context import _get_global_client
from ray.serve.schema import ServeInstanceDetails
from ray.tests.conftest import call_ray_stop_only

@pytest.fixture
def shutdown_ray():
    if False:
        print('Hello World!')
    if ray.is_initialized():
        ray.shutdown()
    yield
    if ray.is_initialized():
        ray.shutdown()

@contextmanager
def start_and_shutdown_ray_cli():
    if False:
        while True:
            i = 10
    subprocess.check_output(['ray', 'start', '--head'])
    yield
    subprocess.check_output(['ray', 'stop', '--force'])

@pytest.fixture(scope='function')
def start_and_shutdown_ray_cli_function():
    if False:
        i = 10
        return i + 15
    with start_and_shutdown_ray_cli():
        yield

@pytest.mark.parametrize('ray_instance', [{'LISTEN_FOR_CHANGE_REQUEST_TIMEOUT_S_LOWER_BOUND': '0.1', 'LISTEN_FOR_CHANGE_REQUEST_TIMEOUT_S_UPPER_BOUND': '0.2'}], indirect=True)
def test_long_poll_timeout_with_max_concurrent_queries(ray_instance):
    if False:
        while True:
            i = 10
    'Test that max_concurrent_queries is respected when there are long poll timeouts.\n\n    Previously, when a long poll update occurred (e.g., a timeout or new replicas\n    added), ongoing requests would no longer be counted against\n    `max_concurrent_queries`.\n\n    Issue: https://github.com/ray-project/ray/issues/32652\n    '

    @ray.remote(num_cpus=0)
    class CounterActor:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self._count = 0

        def inc(self):
            if False:
                while True:
                    i = 10
            self._count += 1

        def get(self):
            if False:
                while True:
                    i = 10
            return self._count
    signal_actor = SignalActor.remote()
    counter_actor = CounterActor.remote()

    @serve.deployment(max_concurrent_queries=1)
    async def f():
        await counter_actor.inc.remote()
        await signal_actor.wait.remote()
        return 'hello'
    serve.run(f.bind())

    @ray.remote
    def do_req():
        if False:
            while True:
                i = 10
        return requests.get('http://localhost:8000').text
    first_ref = do_req.remote()

    def check_request_started(num_expected_requests: int) -> bool:
        if False:
            return 10
        with pytest.raises(TimeoutError):
            ray.get(first_ref, timeout=0.1)
        assert ray.get(counter_actor.get.remote()) == num_expected_requests
        return True
    wait_for_condition(check_request_started, timeout=5, num_expected_requests=1)
    new_refs = [do_req.remote() for _ in range(10)]
    (ready, _) = ray.wait(new_refs, timeout=1)
    assert len(ready) == 0
    assert ray.get(counter_actor.get.remote()) == 1
    ray.get(signal_actor.send.remote())
    assert ray.get(first_ref) == 'hello'
    assert ray.get(new_refs) == ['hello'] * 10
    assert ray.get(counter_actor.get.remote()) == 11
    serve.shutdown()

@pytest.mark.parametrize('ray_instance', [], indirect=True)
def test_replica_health_metric(ray_instance):
    if False:
        for i in range(10):
            print('nop')
    'Test replica health metrics'

    @serve.deployment(num_replicas=2)
    def f():
        if False:
            return 10
        return 'hello'
    serve.run(f.bind())

    def count_live_replica_metrics():
        if False:
            while True:
                i = 10
        resp = requests.get('http://127.0.0.1:9999').text
        resp = resp.split('\n')
        count = 0
        for metrics in resp:
            if '# HELP' in metrics or '# TYPE' in metrics:
                continue
            if 'serve_deployment_replica_healthy' in metrics:
                if '1.0' in metrics:
                    count += 1
        return count
    wait_for_condition(lambda : count_live_replica_metrics() == 2, timeout=120, retry_interval_ms=500)
    serve.run(f.options(num_replicas=10).bind())
    wait_for_condition(lambda : count_live_replica_metrics() == 10, timeout=120, retry_interval_ms=500)
    serve.delete(SERVE_DEFAULT_APP_NAME)
    wait_for_condition(lambda : count_live_replica_metrics() == 0, timeout=120, retry_interval_ms=500)
    serve.shutdown()

def test_shutdown_remote(start_and_shutdown_ray_cli_function):
    if False:
        print('Hello World!')
    'Check that serve.shutdown() works on a remote Ray cluster.'
    deploy_serve_script = 'import ray\nfrom ray import serve\n\nray.init(address="auto", namespace="x")\nserve.start()\n\n@serve.deployment\ndef f(*args):\n   return "got f"\n\nserve.run(f.bind())\n'
    shutdown_serve_script = 'import ray\nfrom ray import serve\n\nray.init(address="auto", namespace="x")\nserve.shutdown()\n'
    deploy_file = NamedTemporaryFile(mode='w+', delete=False, suffix='.py')
    shutdown_file = NamedTemporaryFile(mode='w+', delete=False, suffix='.py')
    try:
        deploy_file.write(deploy_serve_script)
        deploy_file.close()
        shutdown_file.write(shutdown_serve_script)
        shutdown_file.close()
        for _ in range(2):
            subprocess.check_output(['python', deploy_file.name])
            assert requests.get('http://localhost:8000/f').text == 'got f'
            subprocess.check_output(['python', shutdown_file.name])
            with pytest.raises(requests.exceptions.ConnectionError):
                requests.get('http://localhost:8000/f')
    finally:
        os.unlink(deploy_file.name)
        os.unlink(shutdown_file.name)

def test_handle_early_detect_failure(shutdown_ray):
    if False:
        print('Hello World!')
    'Check that handle can be notified about replicas failure.\n\n    It should detect replica raises ActorError and take them out of the replicas set.\n    '

    @serve.deployment(num_replicas=2, max_concurrent_queries=1)
    def f(do_crash: bool=False):
        if False:
            i = 10
            return i + 15
        if do_crash:
            os._exit(1)
        return os.getpid()
    handle = serve.run(f.bind())
    pids = ray.get([handle.remote()._to_object_ref_sync() for _ in range(2)])
    assert len(set(pids)) == 2
    client = _get_global_client()
    ray.kill(client._controller, no_restart=True)
    with pytest.raises(RayActorError):
        handle.remote(do_crash=True).result()
    pids = ray.get([handle.remote()._to_object_ref_sync() for _ in range(10)])
    assert len(set(pids)) == 1
    serve.shutdown()

def test_autoscaler_shutdown_node_http_everynode(monkeypatch, shutdown_ray, call_ray_stop_only):
    if False:
        return 10
    monkeypatch.setenv('RAY_SERVE_PROXY_MIN_DRAINING_PERIOD_S', '1')
    cluster = AutoscalingCluster(head_resources={'CPU': 4}, worker_node_types={'cpu_node': {'resources': {'CPU': 4, 'IS_WORKER': 100}, 'node_config': {}, 'max_workers': 1}}, idle_timeout_minutes=0.05)
    cluster.start()
    ray.init(address='auto')
    serve.start(http_options={'location': 'EveryNode'})

    @serve.deployment(num_replicas=2, ray_actor_options={'num_cpus': 3})
    class A:

        def __call__(self, *args):
            if False:
                for i in range(10):
                    print('nop')
            return 'hi'
    serve.run(A.bind(), name='app_f')
    wait_for_condition(lambda : len(ray._private.state.actors()) == 5)
    assert len(ray.nodes()) == 2
    serve.delete('app_f')
    wait_for_condition(lambda : len(list(filter(lambda a: a['State'] == 'ALIVE', ray._private.state.actors().values()))) == 2)
    client = _get_global_client()
    serve_details = ServeInstanceDetails(**ray.get(client._controller.get_serve_instance_details.remote()))
    assert len(serve_details.proxies) == 1
    assert serve_details.proxies[get_head_node_id()].status == ProxyStatus.HEALTHY
    wait_for_condition(lambda : len(list(filter(lambda n: n['Alive'], ray.nodes()))) == 1)
    serve.shutdown()

def test_drain_and_undrain_http_proxy_actors(monkeypatch, shutdown_ray, call_ray_stop_only):
    if False:
        i = 10
        return i + 15
    'Test the state transtion of the proxy actor between\n    HEALTHY, DRAINING and DRAINED\n    '
    monkeypatch.setenv('RAY_SERVE_PROXY_MIN_DRAINING_PERIOD_S', '10')
    cluster = Cluster()
    head_node = cluster.add_node(num_cpus=0)
    cluster.add_node(num_cpus=1)
    cluster.add_node(num_cpus=1)
    cluster.wait_for_nodes()
    ray.init(address=head_node.address)
    serve.start(http_options={'location': 'EveryNode'})

    @serve.deployment
    class HelloModel:

        def __call__(self):
            if False:
                i = 10
                return i + 15
            return 'hello'
    serve.run(HelloModel.options(num_replicas=2).bind())
    wait_for_condition(lambda : len(ray._private.state.actors()) == 6)
    assert len(ray.nodes()) == 3
    client = _get_global_client()
    serve_details = ServeInstanceDetails(**ray.get(client._controller.get_serve_instance_details.remote()))
    proxy_actor_ids = {proxy.actor_id for (_, proxy) in serve_details.proxies.items()}
    assert len(proxy_actor_ids) == 3
    serve.run(HelloModel.options(num_replicas=1).bind())

    def check_proxy_status(proxy_status_to_count):
        if False:
            for i in range(10):
                print('nop')
        serve_details = ServeInstanceDetails(**ray.get(client._controller.get_serve_instance_details.remote()))
        proxy_status_list = [proxy.status for (_, proxy) in serve_details.proxies.items()]
        print('all proxies!!!', [proxy for (_, proxy) in serve_details.proxies.items()])
        current_status = {status: proxy_status_list.count(status) for status in proxy_status_list}
        return (current_status == proxy_status_to_count, current_status)
    wait_for_condition(condition_predictor=check_proxy_status, proxy_status_to_count={ProxyStatus.HEALTHY: 2, ProxyStatus.DRAINING: 1})
    serve.run(HelloModel.options(num_replicas=2).bind())
    wait_for_condition(condition_predictor=check_proxy_status, proxy_status_to_count={ProxyStatus.HEALTHY: 3})
    serve_details = ServeInstanceDetails(**ray.get(client._controller.get_serve_instance_details.remote()))
    {proxy.actor_id for (_, proxy) in serve_details.proxies.items()} == proxy_actor_ids
    serve.run(HelloModel.options(num_replicas=1).bind())
    wait_for_condition(condition_predictor=check_proxy_status, timeout=40, proxy_status_to_count={ProxyStatus.HEALTHY: 2})
    serve.shutdown()

def test_healthz_and_routes_on_head_and_worker_nodes(shutdown_ray, call_ray_stop_only):
    if False:
        i = 10
        return i + 15
    'Test `/-/healthz` and `/-/routes` return the correct responses for head and\n    worker nodes.\n\n    When there are replicas on all nodes, `/-/routes` and `/-/routes` on all nodes\n    should return 200. When there are no replicas on any nodes, `/-/routes` and\n    `/-/routes` on the head node should continue to return 200. `/-/routes` and\n    `/-/routes` on the worker node should start to return 503\n    '
    os.environ['TEST_WORKER_NODE_HTTP_PORT'] = '8001'
    cluster = Cluster()
    cluster.add_node(num_cpus=0)
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)
    serve.start(http_options={'location': 'EveryNode'})

    @serve.deployment(num_replicas=2)
    class HelloModel:

        def __call__(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'hello'
    model = HelloModel.bind()
    serve.run(target=model)

    def check_replicas_on_worker_nodes():
        if False:
            for i in range(10):
                print('nop')
        _actors = ray._private.state.actors().values()
        replica_nodes = [a['Address']['NodeID'] for a in _actors if a['ActorClassName'].startswith('ServeReplica')]
        return len(set(replica_nodes)) == 1
    wait_for_condition(check_replicas_on_worker_nodes)
    wait_for_condition(lambda : len(ray._private.state.actors()) == 5)
    assert len(ray.nodes()) == 2

    def check_request(url: str, expected_code: int, expected_text: str):
        if False:
            print('Hello World!')
        req = requests.get(url)
        return req.status_code == expected_code and req.text == expected_text
    wait_for_condition(condition_predictor=check_request, url='http://127.0.0.1:8000/-/healthz', expected_code=200, expected_text='success')
    assert requests.get('http://127.0.0.1:8000/-/routes').status_code == 200
    assert requests.get('http://127.0.0.1:8000/-/routes').text == '{"/":"default"}'
    wait_for_condition(condition_predictor=check_request, url='http://127.0.0.1:8001/-/healthz', expected_code=200, expected_text='success')
    assert requests.get('http://127.0.0.1:8001/-/routes').status_code == 200
    assert requests.get('http://127.0.0.1:8001/-/routes').text == '{"/":"default"}'
    serve.delete(name=SERVE_DEFAULT_APP_NAME)

    def _check():
        if False:
            for i in range(10):
                print('nop')
        _actors = ray._private.state.actors().values()
        return len(list(filter(lambda a: a['State'] == 'ALIVE', _actors))) == 3
    wait_for_condition(_check)
    wait_for_condition(condition_predictor=check_request, url='http://127.0.0.1:8000/-/healthz', expected_code=200, expected_text='success')
    assert requests.get('http://127.0.0.1:8000/-/routes').status_code == 200
    assert requests.get('http://127.0.0.1:8000/-/routes').text == '{}'
    wait_for_condition(condition_predictor=check_request, url='http://127.0.0.1:8001/-/healthz', expected_code=503, expected_text='This node is being drained.')
    assert requests.get('http://127.0.0.1:8001/-/routes').status_code == 503
    assert requests.get('http://127.0.0.1:8001/-/routes').text == 'This node is being drained.'
    serve.shutdown()

@pytest.mark.parametrize('wait_for_controller_shutdown', (True, False))
def test_controller_shutdown_gracefully(shutdown_ray, call_ray_stop_only, wait_for_controller_shutdown):
    if False:
        return 10
    'Test controller shutdown gracefully when calling `graceful_shutdown()`.\n\n    Called `graceful_shutdown()` on the controller, so it will start shutdown and\n    eventually all actors will be in DEAD state. Test both cases whether to wait for\n    the controller shutdown or not should both resolve graceful shutdown.\n    '
    cluster = Cluster()
    cluster.add_node()
    cluster.add_node()
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)

    @serve.deployment(num_replicas=2)
    class HelloModel:

        def __call__(self):
            if False:
                i = 10
                return i + 15
            return 'hello'
    model = HelloModel.bind()
    serve.run(target=model)
    wait_for_condition(lambda : len(ray._private.state.actors()) == 5)
    assert len(ray.nodes()) == 2
    client = _get_global_client()
    if wait_for_controller_shutdown:
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(client._controller.graceful_shutdown.remote(True))
    else:
        ray.get(client._controller.graceful_shutdown.remote(False))
    wait_for_condition(lambda : all([actor['State'] == 'DEAD' for actor in ray._private.state.actors().values()]))
    serve.shutdown()

def test_client_shutdown_gracefully_when_timeout(shutdown_ray, call_ray_stop_only, caplog):
    if False:
        i = 10
        return i + 15
    'Test client shutdown gracefully when timeout.\n\n    When the controller is taking longer than the timeout to shutdown, the client will\n    log timeout message and exit the process. The controller will continue to shutdown\n    everything gracefully.\n    '
    cluster = Cluster()
    cluster.add_node()
    cluster.add_node()
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)

    @serve.deployment(num_replicas=2)
    class HelloModel:

        def __call__(self):
            if False:
                while True:
                    i = 10
            return 'hello'
    model = HelloModel.bind()
    serve.run(target=model)
    wait_for_condition(lambda : len(ray._private.state.actors()) == 5)
    assert len(ray.nodes()) == 2
    timeout_s = 0.0
    client = _get_global_client()
    client.shutdown(timeout_s=timeout_s)
    assert f'Controller failed to shut down within {timeout_s}s. Check controller logs for more details.' in caplog.text
    wait_for_condition(lambda : all([actor['State'] == 'DEAD' for actor in ray._private.state.actors().values()]))
    serve.shutdown()

def test_serve_shut_down_without_duplicated_logs(shutdown_ray, call_ray_stop_only):
    if False:
        return 10
    'Test Serve shut down without duplicated logs.\n\n    When Serve shutdown is called and executing the shutdown process, the controller\n    log should not be spamming controller shutdown and deleting app messages.\n    '
    cluster = Cluster()
    cluster.add_node()
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)

    @serve.deployment
    class HelloModel:

        def __call__(self):
            if False:
                return 10
            return 'hello'
    model = HelloModel.bind()
    serve.run(target=model)
    serve.shutdown()
    wait_for_condition(lambda : all([actor['State'] == 'DEAD' for actor in ray._private.state.actors().values()]))
    all_serve_logs = ''
    for filename in os.listdir(get_serve_logs_dir()):
        with open(os.path.join(get_serve_logs_dir(), filename), 'r') as f:
            all_serve_logs += f.read()
    assert all_serve_logs.count('Controller shutdown started') == 1
    assert all_serve_logs.count("Deleting application 'default'") == 1
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', '-s', __file__]))