import functools
import sys
import threading
import time
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Dict
import pytest
import requests
import ray
from ray import serve
from ray._private.pydantic_compat import ValidationError
from ray._private.test_utils import SignalActor, wait_for_condition
from ray.serve._private.common import ApplicationStatus
from ray.serve.drivers import DAGDriver

@pytest.mark.parametrize('prefixes', [[None, '/f', None], ['/f', None, '/f']])
def test_deploy_nullify_route_prefix(serve_instance, prefixes):
    if False:
        print('Hello World!')

    @serve.deployment
    def f(*args):
        if False:
            print('Hello World!')
        return 'got me'
    for prefix in prefixes:
        dag = DAGDriver.options(route_prefix=prefix).bind(f.bind())
        handle = serve.run(dag)
        assert requests.get('http://localhost:8000/f').status_code == 200
        assert requests.get('http://localhost:8000/f').text == '"got me"'
        assert handle.predict.remote().result() == 'got me'

@pytest.mark.timeout(10, method='thread')
def test_deploy_empty_bundle(serve_instance):
    if False:
        print('Hello World!')

    @serve.deployment(ray_actor_options={'num_cpus': 0})
    class D:

        def hello(self, _):
            if False:
                print('Hello World!')
            return 'hello'
    serve.run(D.bind())

def test_deployment_error_handling(serve_instance):
    if False:
        while True:
            i = 10

    @serve.deployment
    def f():
        if False:
            print('Hello World!')
        pass
    with pytest.raises(ValidationError, match='1 validation error for RayActorOptionsSchema.*'):
        serve.run(f.options(ray_actor_options={'runtime_env': {'working_dir': '.'}}).bind())

def test_json_serialization_user_config(serve_instance):
    if False:
        i = 10
        return i + 15
    'See https://github.com/ray-project/ray/issues/25345.\n\n    See https://github.com/ray-project/ray/pull/26235 for additional context\n    about this test.\n    '

    @serve.deployment(name='simple-deployment')
    class SimpleDeployment:
        value: str
        nested_value: str

        def reconfigure(self, config: Dict) -> None:
            if False:
                print('Hello World!')
            self.value = config['value']
            self.nested_value = config['nested']['value']

        def get_value(self) -> None:
            if False:
                return 10
            return self.value

        def get_nested_value(self) -> None:
            if False:
                print('Hello World!')
            return self.nested_value
    app = SimpleDeployment.options(user_config={'value': 'Success!', 'nested': {'value': 'Success!'}}).bind()
    handle = serve.run(app)
    assert handle.get_value.remote().result() == 'Success!'
    assert handle.get_nested_value.remote().result() == 'Success!'
    handle = serve.run(SimpleDeployment.options(user_config={'value': 'Failure!', 'another-value': 'Failure!', 'nested': {'value': 'Success!'}}).bind())
    assert handle.get_value.remote().result() == 'Failure!'
    assert handle.get_nested_value.remote().result() == 'Success!'

def test_http_proxy_request_cancellation(serve_instance):
    if False:
        for i in range(10):
            print('nop')
    s = SignalActor.remote()

    @serve.deployment(max_concurrent_queries=1)
    class A:

        def __init__(self) -> None:
            if False:
                print('Hello World!')
            self.counter = 0

        async def __call__(self):
            self.counter += 1
            ret_val = self.counter
            await s.wait.remote()
            return ret_val
    serve.run(A.bind())
    url = 'http://127.0.0.1:8000/A'
    with ThreadPoolExecutor() as pool:
        first_blocking_fut = pool.submit(functools.partial(requests.get, url, timeout=100))
        time.sleep(1)
        assert not first_blocking_fut.done()
        rest_blocking_futs = [pool.submit(functools.partial(requests.get, url, timeout=0.5)) for _ in range(3)]
        time.sleep(1)
        assert all((f.done() for f in rest_blocking_futs))
        ray.get(s.send.remote())
        assert first_blocking_fut.result().text == '1'
    assert requests.get(url).text == '2'

def test_nonserializable_deployment(serve_instance):
    if False:
        print('Hello World!')
    lock = threading.Lock()

    class D:

        def hello(self, _):
            if False:
                while True:
                    i = 10
            return lock
    with pytest.raises(TypeError, match='Could not serialize the deployment[\\s\\S]*was found to be non-serializable.*'):
        serve.deployment(D)

    @serve.deployment
    class E:

        def __init__(self, arg):
            if False:
                i = 10
                return i + 15
            self.arg = arg
    with pytest.raises(TypeError, match='pickle'):
        serve.run(E.bind(lock))
    with pytest.raises(TypeError, match='pickle'):
        serve.run(E.bind(arg=lock))

def test_deploy_application_unhealthy(serve_instance):
    if False:
        for i in range(10):
            print('nop')
    'Test deploying an application that becomes unhealthy.'

    @ray.remote
    class Event:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.is_set = False

        def set(self):
            if False:
                print('Hello World!')
            self.is_set = True

        def is_set(self):
            if False:
                i = 10
                return i + 15
            return self.is_set
    event = Event.remote()

    @serve.deployment(health_check_period_s=1, health_check_timeout_s=3)
    class Model:

        def __call__(self):
            if False:
                print('Hello World!')
            return 'hello world'

        def check_health(self):
            if False:
                while True:
                    i = 10
            if ray.get(event.is_set.remote()):
                raise RuntimeError('Intentionally failing.')
    handle = serve.run(Model.bind(), name='app')
    assert handle.remote().result() == 'hello world'
    assert serve.status().applications['app'].status == ApplicationStatus.RUNNING
    event.set.remote()
    wait_for_condition(lambda : serve.status().applications['app'].status == ApplicationStatus.UNHEALTHY)
    for _ in range(10):
        assert serve.status().applications['app'].status == ApplicationStatus.UNHEALTHY
        time.sleep(0.1)

@pytest.mark.skipif(sys.platform == 'win32', reason='Runtime env support experimental on windows')
def test_deploy_bad_pip_package_deployment(serve_instance):
    if False:
        i = 10
        return i + 15
    'Test deploying with a bad runtime env at deployment level.'

    @serve.deployment(ray_actor_options={'runtime_env': {'pip': ['does_not_exist']}})
    class Model:

        def __call__(self):
            if False:
                print('Hello World!')
            return 'hello world'
    serve.run(Model.bind(), _blocking=False)

    def check_fail():
        if False:
            while True:
                i = 10
        app_status = serve.status().applications['default']
        assert app_status.status == ApplicationStatus.DEPLOY_FAILED
        deployment_message = app_status.deployments['Model'].message
        assert 'No matching distribution found for does_not_exist' in deployment_message
        return True
    wait_for_condition(check_fail, timeout=15)

def test_deploy_same_deployment_name_different_app(serve_instance):
    if False:
        print('Hello World!')

    @serve.deployment
    class Model:

        def __init__(self, name):
            if False:
                for i in range(10):
                    print('nop')
            self.name = name

        def __call__(self):
            if False:
                i = 10
                return i + 15
            return f'hello {self.name}'
    serve.run(Model.bind('alice'), name='app1', route_prefix='/app1')
    serve.run(Model.bind('bob'), name='app2', route_prefix='/app2')
    assert requests.get('http://localhost:8000/app1').text == 'hello alice'
    assert requests.get('http://localhost:8000/app2').text == 'hello bob'
    routes = requests.get('http://localhost:8000/-/routes').json()
    assert routes['/app1'] == 'app1'
    assert routes['/app2'] == 'app2'
    app1_status = serve.status().applications['app1']
    app2_status = serve.status().applications['app2']
    assert app1_status.status == 'RUNNING'
    assert app1_status.deployments['Model'].status == 'HEALTHY'
    assert app2_status.status == 'RUNNING'
    assert app2_status.deployments['Model'].status == 'HEALTHY'
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', '-s', __file__]))