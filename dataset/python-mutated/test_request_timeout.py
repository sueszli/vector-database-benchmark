import asyncio
import os
import sys
import time
from typing import Generator, Set
import pytest
import requests
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse
import ray
from ray import serve
from ray._private.test_utils import SignalActor, wait_for_condition
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient
from ray.serve._private.common import ApplicationStatus
from ray.serve.schema import ServeInstanceDetails
from ray.serve.tests.common.utils import send_signal_on_cancellation
from ray.util.state import list_tasks

@ray.remote
def do_request():
    if False:
        return 10
    return requests.get('http://localhost:8000')

@pytest.fixture
def shutdown_serve():
    if False:
        return 10
    yield
    serve.shutdown()

@pytest.mark.parametrize('ray_instance', [{'RAY_SERVE_REQUEST_PROCESSING_TIMEOUT_S': '5'}], indirect=True)
def test_normal_operation(ray_instance, shutdown_serve):
    if False:
        return 10
    "\n    Verify that a moderate timeout doesn't affect normal operation.\n    "

    @serve.deployment(num_replicas=2)
    def f(*args):
        if False:
            print('Hello World!')
        return 'Success!'
    serve.run(f.bind())
    assert all((response.text == 'Success!' for response in ray.get([do_request.remote() for _ in range(10)])))

@pytest.mark.parametrize('ray_instance', [{'RAY_SERVE_REQUEST_PROCESSING_TIMEOUT_S': '0.1'}], indirect=True)
def test_request_hangs_in_execution(ray_instance, shutdown_serve):
    if False:
        i = 10
        return i + 15
    '\n    Verify that requests are timed out if they take longer than the timeout to execute.\n    '

    @ray.remote
    class PidTracker:

        def __init__(self):
            if False:
                print('Hello World!')
            self.pids = set()

        def add_pid(self, pid: int) -> None:
            if False:
                i = 10
                return i + 15
            self.pids.add(pid)

        def get_pids(self) -> Set[int]:
            if False:
                return 10
            return self.pids
    pid_tracker = PidTracker.remote()
    signal_actor = SignalActor.remote()

    @serve.deployment(num_replicas=2, graceful_shutdown_timeout_s=0)
    class HangsOnFirstRequest:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self._saw_first_request = False

        async def __call__(self):
            ray.get(pid_tracker.add_pid.remote(os.getpid()))
            if not self._saw_first_request:
                self._saw_first_request = True
                await asyncio.sleep(10)
            return 'Success!'
    serve.run(HangsOnFirstRequest.bind())
    response = requests.get('http://localhost:8000')
    assert response.status_code == 408
    ray.get(signal_actor.send.remote())

@serve.deployment(graceful_shutdown_timeout_s=0)
class HangsOnFirstRequest:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._saw_first_request = False
        self.signal_actor = SignalActor.remote()

    async def __call__(self):
        if not self._saw_first_request:
            self._saw_first_request = True
            await self.signal_actor.wait.remote()
        else:
            ray.get(self.signal_actor.send.remote())
        return 'Success!'
hangs_on_first_request_app = HangsOnFirstRequest.bind()

def test_with_rest_api(ray_start_stop):
    if False:
        for i in range(10):
            print('nop')
    'Verify the REST API can configure the request timeout.'
    config = {'proxy_location': 'EveryNode', 'http_options': {'request_timeout_s': 1}, 'applications': [{'name': 'app', 'route_prefix': '/', 'import_path': 'ray.serve.tests.test_request_timeout:hangs_on_first_request_app'}]}
    ServeSubmissionClient('http://localhost:8265').deploy_applications(config)

    def application_running():
        if False:
            return 10
        response = requests.get('http://localhost:52365/api/serve/applications/', timeout=15)
        assert response.status_code == 200
        serve_details = ServeInstanceDetails(**response.json())
        return serve_details.applications['app'].status == ApplicationStatus.RUNNING
    wait_for_condition(application_running, timeout=15)
    print('Application has started running. Testing requests...')
    response = requests.get('http://localhost:8000')
    assert response.status_code == 408
    response = requests.get('http://localhost:8000')
    assert response.status_code == 200
    print('Requests succeeded! Deleting application.')
    ServeSubmissionClient('http://localhost:8265').delete_applications()

@pytest.mark.parametrize('ray_instance', [{'RAY_SERVE_REQUEST_PROCESSING_TIMEOUT_S': '0.5'}], indirect=True)
def test_request_hangs_in_assignment(ray_instance, shutdown_serve):
    if False:
        return 10
    '\n    Verify that requests are timed out if they take longer than the timeout while\n    pending assignment (queued in the handle).\n    '
    signal_actor = SignalActor.remote()

    @serve.deployment(graceful_shutdown_timeout_s=0, max_concurrent_queries=1)
    class HangsOnFirstRequest:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self._saw_first_request = False

        async def __call__(self):
            await signal_actor.wait.remote()
            return 'Success!'
    serve.run(HangsOnFirstRequest.bind())
    response_ref1 = do_request.remote()
    response_ref2 = do_request.remote()
    assert ray.get(response_ref1).status_code == 408
    assert ray.get(response_ref2).status_code == 408
    ray.get(signal_actor.send.remote())
    assert ray.get(do_request.remote()).status_code == 200

@pytest.mark.parametrize('ray_instance', [{'RAY_SERVE_REQUEST_PROCESSING_TIMEOUT_S': '0.1'}], indirect=True)
def test_streaming_request_already_sent_and_timed_out(ray_instance, shutdown_serve):
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify that streaming requests are timed out even if some chunks have already\n    been sent.\n    '

    @serve.deployment(graceful_shutdown_timeout_s=0, max_concurrent_queries=1)
    class SleepForNSeconds:

        def __init__(self, sleep_s: int):
            if False:
                while True:
                    i = 10
            self.sleep_s = sleep_s

        def generate_numbers(self) -> Generator[str, None, None]:
            if False:
                while True:
                    i = 10
            for i in range(2):
                yield f'generated {i}'
                time.sleep(self.sleep_s)

        def __call__(self, request: Request) -> StreamingResponse:
            if False:
                print('Hello World!')
            gen = self.generate_numbers()
            return StreamingResponse(gen, status_code=200, media_type='text/plain')
    serve.run(SleepForNSeconds.bind(0.11))
    r = requests.get('http://localhost:8000', stream=True)
    iterator = r.iter_content(chunk_size=None, decode_unicode=True)
    assert iterator.__next__() == 'generated 0'
    assert r.status_code == 200
    with pytest.raises(requests.exceptions.ChunkedEncodingError) as request_error:
        iterator.__next__()
    assert 'Connection broken' in str(request_error.value)

@pytest.mark.parametrize('ray_instance', [{'RAY_SERVE_REQUEST_PROCESSING_TIMEOUT_S': '0.5'}], indirect=True)
def test_request_timeout_does_not_leak_tasks(ray_instance, shutdown_serve):
    if False:
        return 10
    'Verify that the ASGI-related tasks exit when a request is timed out.\n\n    See https://github.com/ray-project/ray/issues/38368 for details.\n    '

    @serve.deployment
    class Hang:

        async def __call__(self):
            await asyncio.sleep(1000000)
    serve.run(Hang.bind())

    def get_num_running_tasks():
        if False:
            return 10
        return len(list_tasks(address=ray_instance['gcs_address'], filters=[('NAME', '!=', 'ServeController.listen_for_change'), ('TYPE', '=', 'ACTOR_TASK'), ('STATE', '=', 'RUNNING')]))
    wait_for_condition(lambda : get_num_running_tasks() == 0)
    results = ray.get([do_request.remote() for _ in range(10)])
    assert all((r.status_code == 408 for r in results))
    wait_for_condition(lambda : get_num_running_tasks() == 0)

@pytest.mark.parametrize('ray_instance', [{'RAY_SERVE_REQUEST_PROCESSING_TIMEOUT_S': '0.5'}], indirect=True)
@pytest.mark.parametrize('use_fastapi', [False, True])
def test_cancel_on_http_timeout_during_execution(ray_instance, shutdown_serve, use_fastapi: bool):
    if False:
        for i in range(10):
            print('nop')
    'Test the request timing out while the handler is executing.'
    inner_signal_actor = SignalActor.remote()
    outer_signal_actor = SignalActor.remote()

    @serve.deployment
    async def inner():
        await send_signal_on_cancellation(inner_signal_actor)
    if use_fastapi:
        app = FastAPI()

        @serve.deployment
        @serve.ingress(app)
        class Ingress:

            def __init__(self, handle):
                if False:
                    print('Hello World!')
                self._handle = handle.options(use_new_handle_api=True)

            @app.get('/')
            async def wait_for_cancellation(self):
                await self._handle.remote()._to_object_ref()
                await send_signal_on_cancellation(outer_signal_actor)
    else:

        @serve.deployment
        class Ingress:

            def __init__(self, handle):
                if False:
                    i = 10
                    return i + 15
                self._handle = handle.options(use_new_handle_api=True)

            async def __call__(self, request: Request):
                await self._handle.remote()._to_object_ref()
                await send_signal_on_cancellation(outer_signal_actor)
    serve.run(Ingress.bind(inner.bind()))
    assert requests.get('http://localhost:8000').status_code == 408
    ray.get(inner_signal_actor.wait.remote())
    ray.get(outer_signal_actor.wait.remote())

@pytest.mark.parametrize('ray_instance', [{'RAY_SERVE_REQUEST_PROCESSING_TIMEOUT_S': '0.5'}], indirect=True)
def test_cancel_on_http_timeout_during_assignment(ray_instance, shutdown_serve):
    if False:
        i = 10
        return i + 15
    'Test the client disconnecting while the proxy is assigning the request.'
    signal_actor = SignalActor.remote()

    @serve.deployment(max_concurrent_queries=1)
    class Ingress:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self._num_requests = 0

        async def __call__(self, *args):
            self._num_requests += 1
            await signal_actor.wait.remote()
            return self._num_requests
    h = serve.run(Ingress.bind()).options(use_new_handle_api=True)
    initial_response = h.remote()
    wait_for_condition(lambda : ray.get(signal_actor.cur_num_waiters.remote()) == 1)
    assert requests.get('http://localhost:8000').status_code == 408
    ray.get(signal_actor.send.remote())
    assert initial_response.result() == 1
    for i in range(2, 12):
        assert h.remote().result() == i
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', '-s', __file__]))