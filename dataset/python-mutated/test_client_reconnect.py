from concurrent import futures
import contextlib
import os
import threading
import sys
import grpc
from mock import Mock
import numpy as np
import time
import random
import pytest
from typing import Any, Callable, Optional
from unittest.mock import patch
import ray
from ray._private.utils import get_or_create_event_loop
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.tests.conftest import call_ray_start_context
from ray.util.client.common import CLIENT_SERVER_MAX_THREADS, GRPC_OPTIONS

@pytest.fixture(scope='module')
def call_ray_start_shared(request):
    if False:
        return 10
    request = Mock()
    request.param = 'ray start --head --min-worker-port=0 --max-worker-port=0 --port 0 --ray-client-server-port=50051'
    with call_ray_start_context(request) as address:
        yield address
Hook = Callable[[Any], None]

class MiddlemanDataServicer(ray_client_pb2_grpc.RayletDataStreamerServicer):
    """
    Forwards all requests to the real data servicer. Useful for injecting
    errors between a client and server pair.
    """

    def __init__(self, on_response: Optional[Hook]=None, on_request: Optional[Hook]=None):
        if False:
            return 10
        '\n        Args:\n            on_response: Optional hook to inject errors before sending back a\n                response\n        '
        self.stub = None
        self.on_response = on_response
        self.on_request = on_request

    def set_channel(self, channel: grpc.Channel) -> None:
        if False:
            print('Hello World!')
        self.stub = ray_client_pb2_grpc.RayletDataStreamerStub(channel)

    def _requests(self, request_iterator):
        if False:
            i = 10
            return i + 15
        for req in request_iterator:
            if self.on_request:
                self.on_request(req)
            yield req

    def Datapath(self, request_iterator, context):
        if False:
            i = 10
            return i + 15
        try:
            for response in self.stub.Datapath(self._requests(request_iterator), metadata=context.invocation_metadata()):
                if self.on_response:
                    self.on_response(response)
                yield response
        except grpc.RpcError as e:
            context.set_code(e.code())
            context.set_details(e.details())

class MiddlemanLogServicer(ray_client_pb2_grpc.RayletLogStreamerServicer):
    """
    Forwards all requests to the real log servicer. Useful for injecting
    errors between a client and server pair.
    """

    def __init__(self, on_response: Optional[Hook]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            on_response: Optional hook to inject errors before sending back a\n                response\n        '
        self.stub = None
        self.on_response = on_response

    def set_channel(self, channel: grpc.Channel) -> None:
        if False:
            while True:
                i = 10
        self.stub = ray_client_pb2_grpc.RayletLogStreamerStub(channel)

    def Logstream(self, request_iterator, context):
        if False:
            return 10
        try:
            for response in self.stub.Logstream(request_iterator, metadata=context.invocation_metadata()):
                if self.on_response:
                    self.on_response(response)
                yield response
        except grpc.RpcError as e:
            context.set_code(e.code())
            context.set_details(e.details())

class MiddlemanRayletServicer(ray_client_pb2_grpc.RayletDriverServicer):
    """
    Forwards all requests to the raylet driver servicer. Useful for injecting
    errors between a client and server pair.
    """

    def __init__(self, on_request: Optional[Hook]=None, on_response: Optional[Hook]=None):
        if False:
            return 10
        '\n        Args:\n            on_request: Optional hook to inject errors before forwarding a\n                request\n            on_response: Optional hook to inject errors before sending back a\n                response\n        '
        self.stub = None
        self.on_request = on_request
        self.on_response = on_response

    def set_channel(self, channel: grpc.Channel) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.stub = ray_client_pb2_grpc.RayletDriverStub(channel)

    def _call_inner_function(self, request: Any, context, method: str) -> Optional[ray_client_pb2_grpc.RayletDriverStub]:
        if False:
            while True:
                i = 10
        if self.on_request:
            self.on_request(request)
        try:
            response = getattr(self.stub, method)(request, metadata=context.invocation_metadata())
        except grpc.RpcError as e:
            context.set_code(e.code())
            context.set_details(e.details())
            raise
        if self.on_response and method != 'GetObject':
            self.on_response(response)
        return response

    def Init(self, request, context=None) -> ray_client_pb2.InitResponse:
        if False:
            while True:
                i = 10
        return self._call_inner_function(request, context, 'Init')

    def KVPut(self, request, context=None) -> ray_client_pb2.KVPutResponse:
        if False:
            i = 10
            return i + 15
        return self._call_inner_function(request, context, 'KVPut')

    def KVGet(self, request, context=None) -> ray_client_pb2.KVGetResponse:
        if False:
            for i in range(10):
                print('nop')
        return self._call_inner_function(request, context, 'KVGet')

    def KVDel(self, request, context=None) -> ray_client_pb2.KVDelResponse:
        if False:
            return 10
        return self._call_inner_function(request, context, 'KVDel')

    def KVList(self, request, context=None) -> ray_client_pb2.KVListResponse:
        if False:
            for i in range(10):
                print('nop')
        return self._call_inner_function(request, context, 'KVList')

    def KVExists(self, request, context=None) -> ray_client_pb2.KVExistsResponse:
        if False:
            for i in range(10):
                print('nop')
        return self._call_inner_function(request, context, 'KVExists')

    def ListNamedActors(self, request, context=None) -> ray_client_pb2.ClientListNamedActorsResponse:
        if False:
            for i in range(10):
                print('nop')
        return self._call_inner_function(request, context, 'ListNamedActors')

    def ClusterInfo(self, request, context=None) -> ray_client_pb2.ClusterInfoResponse:
        if False:
            i = 10
            return i + 15
        try:
            return self.stub.ClusterInfo(request, metadata=context.invocation_metadata())
        except grpc.RpcError as e:
            context.set_code(e.code())
            context.set_details(e.details())
            raise

    def Terminate(self, req, context=None):
        if False:
            for i in range(10):
                print('nop')
        return self._call_inner_function(req, context, 'Terminate')

    def GetObject(self, request, context=None):
        if False:
            return 10
        for response in self._call_inner_function(request, context, 'GetObject'):
            if self.on_response:
                self.on_response(response)
            yield response

    def PutObject(self, request: ray_client_pb2.PutRequest, context=None) -> ray_client_pb2.PutResponse:
        if False:
            return 10
        return self._call_inner_function(request, context, 'PutObject')

    def WaitObject(self, request: ray_client_pb2.WaitRequest, context=None) -> ray_client_pb2.WaitResponse:
        if False:
            for i in range(10):
                print('nop')
        return self._call_inner_function(request, context, 'WaitObject')

    def Schedule(self, task: ray_client_pb2.ClientTask, context=None) -> ray_client_pb2.ClientTaskTicket:
        if False:
            for i in range(10):
                print('nop')
        return self._call_inner_function(task, context, 'Schedule')

class MiddlemanServer:
    """
    Helper class that wraps the RPC server that middlemans the connection
    between the client and the real ray server. Useful for injecting
    errors between a client and server pair.
    """

    def __init__(self, listen_addr: str, real_addr, on_log_response: Optional[Hook]=None, on_data_request: Optional[Hook]=None, on_data_response: Optional[Hook]=None, on_task_request: Optional[Hook]=None, on_task_response: Optional[Hook]=None):
        if False:
            print('Hello World!')
        '\n        Args:\n            listen_addr: The address the middleman server will listen on\n            real_addr: The address of the real ray server\n            on_log_response: Optional hook to inject errors before sending back\n                a log response\n            on_data_response: Optional hook to inject errors before sending\n                back a data response\n            on_task_request: Optional hook to inject errors before forwarding\n                a raylet driver request\n            on_task_response: Optional hook to inject errors before sending\n                back a raylet driver response\n        '
        self.listen_addr = listen_addr
        self.real_addr = real_addr
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=CLIENT_SERVER_MAX_THREADS), options=GRPC_OPTIONS)
        self.task_servicer = MiddlemanRayletServicer(on_response=on_task_response, on_request=on_task_request)
        self.data_servicer = MiddlemanDataServicer(on_response=on_data_response, on_request=on_data_request)
        self.logs_servicer = MiddlemanLogServicer(on_response=on_log_response)
        ray_client_pb2_grpc.add_RayletDriverServicer_to_server(self.task_servicer, self.server)
        ray_client_pb2_grpc.add_RayletDataStreamerServicer_to_server(self.data_servicer, self.server)
        ray_client_pb2_grpc.add_RayletLogStreamerServicer_to_server(self.logs_servicer, self.server)
        self.server.add_insecure_port(self.listen_addr)
        self.channel = None
        self.reset_channel()

    def reset_channel(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Manually close and reopen the channel to the real ray server. This\n        simulates a disconnection between the client and the server.\n        '
        if self.channel:
            self.channel.close()
        self.channel = grpc.insecure_channel(self.real_addr, options=GRPC_OPTIONS)
        grpc.channel_ready_future(self.channel)
        self.task_servicer.set_channel(self.channel)
        self.data_servicer.set_channel(self.channel)
        self.logs_servicer.set_channel(self.channel)

    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        self.server.start()

    def stop(self, grace: int) -> None:
        if False:
            while True:
                i = 10
        self.server.stop(grace)

@contextlib.contextmanager
def start_middleman_server(on_log_response=None, on_data_request=None, on_data_response=None, on_task_request=None, on_task_response=None):
    if False:
        print('Hello World!')
    '\n    Helper context that starts a middleman server listening on port 10011,\n    and a ray client server on port 50051.\n    '
    ray._inside_client_test = True
    middleman = None
    try:
        middleman = MiddlemanServer(listen_addr='localhost:10011', real_addr='localhost:50051', on_log_response=on_log_response, on_data_request=on_data_request, on_data_response=on_data_response, on_task_request=on_task_request, on_task_response=on_task_response)
        middleman.start()
        ray.init('ray://localhost:10011')
        yield middleman
    finally:
        ray._inside_client_test = False
        ray.util.disconnect()
        if middleman:
            middleman.stop(0)

def test_disconnect_during_get(call_ray_start_shared):
    if False:
        while True:
            i = 10
    '\n    Disconnect the proxy and the client in the middle of a long running get\n    '

    @ray.remote
    def slow_result():
        if False:
            print('Hello World!')
        time.sleep(20)
        return 12345

    def disconnect(middleman):
        if False:
            i = 10
            return i + 15
        time.sleep(3)
        middleman.reset_channel()
    with start_middleman_server() as middleman:
        disconnect_thread = threading.Thread(target=disconnect, args=(middleman,))
        disconnect_thread.start()
        result = ray.get(slow_result.remote())
        assert result == 12345
        disconnect_thread.join()

def test_disconnects_during_large_get(call_ray_start_shared):
    if False:
        return 10
    '\n    Disconnect repeatedly during a large (multi-chunk) get.\n    '
    i = 0
    started = False

    def fail_every_three(_):
        if False:
            print('Hello World!')
        nonlocal i, started
        if not started:
            return
        i += 1
        if i % 3 == 0:
            raise RuntimeError

    @ray.remote
    def large_result():
        if False:
            for i in range(10):
                print('nop')
        return np.random.random((1024, 1024, 6))
    with start_middleman_server(on_task_response=fail_every_three):
        started = True
        result = ray.get(large_result.remote())
        assert result.shape == (1024, 1024, 6)

def test_disconnects_during_large_async_get(call_ray_start_shared):
    if False:
        print('Hello World!')
    '\n    Disconnect repeatedly during a large (multi-chunk) async get.\n    '
    i = 0
    started = False

    def fail_every_three(_):
        if False:
            i = 10
            return i + 15
        nonlocal i, started
        if not started:
            return
        i += 1
        if i % 3 == 0:
            raise RuntimeError

    @ray.remote
    def large_result():
        if False:
            print('Hello World!')
        return np.random.random((1024, 1024, 6))
    with start_middleman_server(on_data_response=fail_every_three):
        started = True

        async def get_large_result():
            return await large_result.remote()
        result = get_or_create_event_loop().run_until_complete(get_large_result())
        assert result.shape == (1024, 1024, 6)

def test_disconnect_during_large_put(call_ray_start_shared):
    if False:
        print('Hello World!')
    '\n    Disconnect during a large (multi-chunk) put.\n    '
    i = 0
    started = False

    def fail_halfway(_):
        if False:
            print('Hello World!')
        nonlocal i, started
        if not started:
            return
        i += 1
        if i == 8:
            raise RuntimeError
    with start_middleman_server(on_data_request=fail_halfway):
        started = True
        objref = ray.put(np.random.random((1024, 1024, 6)))
        assert i > 8
        result = ray.get(objref)
        assert result.shape == (1024, 1024, 6)

def test_disconnect_during_large_schedule(call_ray_start_shared):
    if False:
        i = 10
        return i + 15
    '\n    Disconnect during a remote call with a large (multi-chunk) argument.\n    '
    i = 0
    started = False

    def fail_halfway(_):
        if False:
            i = 10
            return i + 15
        nonlocal i, started
        if not started:
            return
        i += 1
        if i == 8:
            raise RuntimeError

    @ray.remote
    def f(a):
        if False:
            for i in range(10):
                print('nop')
        return a.shape
    with start_middleman_server(on_data_request=fail_halfway):
        started = True
        a = np.random.random((1024, 1024, 6))
        result = ray.get(f.remote(a))
        assert i > 8
        assert result == (1024, 1024, 6)

def test_valid_actor_state(call_ray_start_shared):
    if False:
        for i in range(10):
            print('nop')
    '\n    Repeatedly inject errors in the middle of mutating actor calls. Check\n    at the end that the final state of the actor is consistent with what\n    we would expect had the disconnects not occurred.\n    '

    @ray.remote
    class IncrActor:

        def __init__(self):
            if False:
                print('Hello World!')
            self.val = 0

        def incr(self):
            if False:
                print('Hello World!')
            self.val += 1
            return self.val
    i = 0
    started = False

    def fail_every_seven(_):
        if False:
            while True:
                i = 10
        nonlocal i, started
        i += 1
        if i % 7 == 0 and started:
            raise RuntimeError
    with start_middleman_server(on_data_response=fail_every_seven, on_task_request=fail_every_seven, on_task_response=fail_every_seven):
        started = True
        actor = IncrActor.remote()
        for _ in range(100):
            ref = actor.incr.remote()
        assert ray.get(ref) == 100

def test_valid_actor_state_2(call_ray_start_shared):
    if False:
        print('Hello World!')
    "\n    Do a full disconnect (cancel channel) every 11 requests. Failure\n    happens:\n      - before request sent: request never reaches server\n      - before response received: response never reaches server\n      - while get's are being processed\n    "

    @ray.remote
    class IncrActor:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.val = 0

        def incr(self):
            if False:
                while True:
                    i = 10
            self.val += 1
            return self.val
    i = 0
    with start_middleman_server() as middleman:

        def fail_every_eleven(_):
            if False:
                while True:
                    i = 10
            nonlocal i
            i += 1
            if i % 11 == 0:
                middleman.reset_channel()
        middleman.data_servicer.on_response = fail_every_eleven
        middleman.task_servicer.on_request = fail_every_eleven
        middleman.task_servicer.on_response = fail_every_eleven
        actor = IncrActor.remote()
        for _ in range(100):
            ref = actor.incr.remote()
        assert ray.get(ref) == 100

def test_noisy_puts(call_ray_start_shared):
    if False:
        return 10
    '\n    Randomly kills the data channel with 10% chance when receiving response\n    (requests made it to server, responses dropped) and checks that final\n    result is still consistent\n    '
    random.seed(12345)
    with start_middleman_server() as middleman:

        def fail_randomly(response: ray_client_pb2.DataResponse):
            if False:
                print('Hello World!')
            if random.random() < 0.1:
                raise RuntimeError
        middleman.data_servicer.on_response = fail_randomly
        refs = [ray.put(i * 123) for i in range(500)]
        results = ray.get(refs)
        for (i, result) in enumerate(results):
            assert result == i * 123

def test_client_reconnect_grace_period(call_ray_start_shared):
    if False:
        return 10
    '\n    Tests that the client gives up attempting to reconnect the channel\n    after the grace period expires.\n    '
    with patch.dict(os.environ, {'RAY_CLIENT_RECONNECT_GRACE_PERIOD': '5'}), start_middleman_server() as middleman:
        assert ray.get(ray.put(42)) == 42
        middleman.channel.close()
        start_time = time.time()
        with pytest.raises(ConnectionError):
            ray.get(ray.put(42))
        assert time.time() - start_time < 20
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))