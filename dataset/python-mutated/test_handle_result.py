import json
import sys
import pytest
from mock import ANY
from nameko.exceptions import RemoteError
from nameko.extensions import DependencyProvider
from nameko.rpc import Rpc
from nameko.standalone.rpc import ServiceRpcProxy
from nameko.testing.services import entrypoint_waiter
worker_result_called = []

@pytest.fixture(autouse=True)
def reset():
    if False:
        i = 10
        return i + 15
    yield
    del worker_result_called[:]

class ResultCollector(DependencyProvider):
    """ DependencyProvider that collects worker results
    """

    def worker_result(self, worker_ctx, res, exc_info):
        if False:
            for i in range(10):
                print('nop')
        worker_result_called.append((res, exc_info))

class CustomRpc(Rpc):
    """ Rpc subclass that verifies `result` can be serialized to json,
    and changes the `result` and `exc_info` accordingly.
    """

    def handle_result(self, message, worker_ctx, result, exc_info):
        if False:
            for i in range(10):
                print('nop')
        try:
            json.dumps(result)
        except Exception:
            result = 'something went wrong'
            exc_info = sys.exc_info()
        return super(CustomRpc, self).handle_result(message, worker_ctx, result, exc_info)
custom_rpc = CustomRpc.decorator

class ExampleService(object):
    name = 'exampleservice'
    collector = ResultCollector()

    @custom_rpc
    def echo(self, arg):
        if False:
            while True:
                i = 10
        return arg

    @custom_rpc
    def unserializable(self):
        if False:
            i = 10
            return i + 15
        return object()

@pytest.fixture
def rpc_proxy(rabbit_config):
    if False:
        for i in range(10):
            print('nop')
    with ServiceRpcProxy('exampleservice', rabbit_config) as proxy:
        yield proxy

def test_handle_result(container_factory, rabbit_manager, rabbit_config, rpc_proxy):
    if False:
        return 10
    ' Verify that `handle_result` can modify the return values of the worker,\n    such that other dependencies see the updated values.\n    '
    container = container_factory(ExampleService, rabbit_config)
    container.start()
    assert rpc_proxy.echo('hello') == 'hello'
    with entrypoint_waiter(container, 'unserializable'):
        with pytest.raises(RemoteError) as exc:
            rpc_proxy.unserializable()
        assert 'is not JSON serializable' in str(exc.value)
    assert worker_result_called == [('hello', None), ('something went wrong', (TypeError, ANY, ANY))]