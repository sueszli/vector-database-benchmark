from typing import Callable
import grpc
import pytest
from google.protobuf.any_pb2 import Any as AnyProto
from ray.serve._private.grpc_util import DummyServicer, create_serve_grpc_server, gRPCServer

def fake_service_handler_factory(service_method: str, stream: bool) -> Callable:
    if False:
        for i in range(10):
            print('nop')

    def foo() -> bytes:
        if False:
            while True:
                i = 10
        return f"{('stream' if stream else 'unary')} call from {service_method}".encode()
    return foo

def test_dummy_servicer_can_take_any_methods():
    if False:
        print('Hello World!')
    "Test an instance of DummyServicer can be called with any method name without\n    error.\n\n    When dummy_servicer is called with any custom defined methods, it won't raise error.\n    "
    dummy_servicer = DummyServicer()
    dummy_servicer.foo
    dummy_servicer.bar
    dummy_servicer.baz
    dummy_servicer.my_method
    dummy_servicer.Predict

def test_create_serve_grpc_server():
    if False:
        for i in range(10):
            print('nop')
    'Test `create_serve_grpc_server()` creates the correct server.\n\n    The server created by `create_serve_grpc_server()` should be an instance of\n    Serve defined `gRPCServer`. Also, the handler factory passed with the function\n    should be used to initialize the `gRPCServer`.\n    '
    grpc_server = create_serve_grpc_server(service_handler_factory=fake_service_handler_factory)
    assert isinstance(grpc_server, gRPCServer)
    assert grpc_server.service_handler_factory == fake_service_handler_factory

def test_grpc_server():
    if False:
        for i in range(10):
            print('nop')
    'Test `gRPCServer` did the correct overrides.\n\n    When a add_servicer_to_server function is called on an instance of `gRPCServer`,\n    it correctly overrides `response_serializer` to None, and `unary_unary` and\n    `unary_stream` to be generated from the factory function.\n    '
    service_name = 'ray.serve.ServeAPIService'
    method_name = 'ServeRoutes'

    def add_test_servicer_to_server(servicer, server):
        if False:
            i = 10
            return i + 15
        rpc_method_handlers = {method_name: grpc.unary_unary_rpc_method_handler(servicer.ServeRoutes, request_deserializer=AnyProto.FromString, response_serializer=AnyProto.SerializeToString)}
        generic_handler = grpc.method_handlers_generic_handler(service_name, rpc_method_handlers)
        server.add_generic_rpc_handlers((generic_handler,))
    grpc_server = gRPCServer(thread_pool=None, generic_handlers=(), interceptors=(), options=(), maximum_concurrent_rpcs=None, compression=None, service_handler_factory=fake_service_handler_factory)
    dummy_servicer = DummyServicer()
    assert grpc_server.generic_rpc_handlers == []
    add_test_servicer_to_server(dummy_servicer, grpc_server)
    assert len(grpc_server.generic_rpc_handlers) == 1
    rpc_handler = grpc_server.generic_rpc_handlers[0][0]
    assert rpc_handler.service_name() == service_name
    service_method = f'/{service_name}/{method_name}'
    method_handlers = rpc_handler._method_handlers.get(service_method)
    assert method_handlers.response_serializer is None
    assert method_handlers.unary_unary() == f'unary call from {service_method}'.encode()
    assert method_handlers.unary_stream() == f'stream call from {service_method}'.encode()
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', '-s', __file__]))