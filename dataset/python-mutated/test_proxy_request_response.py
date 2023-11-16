import pickle
from unittest.mock import MagicMock
import pytest
from ray.serve._private.common import StreamingHTTPRequest, gRPCRequest
from ray.serve._private.proxy_request_response import ASGIProxyRequest, ProxyRequest, gRPCProxyRequest
from ray.serve.generated import serve_pb2
from ray.serve.tests.common.utils import FakeGrpcContext

class TestASGIProxyRequest:

    def create_asgi_proxy_request(self, scope: dict) -> ASGIProxyRequest:
        if False:
            i = 10
            return i + 15
        receive = MagicMock()
        send = MagicMock()
        return ASGIProxyRequest(scope=scope, receive=receive, send=send)

    def test_request_type(self):
        if False:
            print('Hello World!')
        'Test calling request_type on an instance of ASGIProxyRequest.\n\n        When the request_type is not passed into the scope, it returns empty string.\n        When the request_type is passed into the scope, it returns the correct value.\n        '
        proxy_request = self.create_asgi_proxy_request(scope={})
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.request_type == ''
        request_type = 'fake-request_type'
        proxy_request = self.create_asgi_proxy_request(scope={'type': request_type})
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.request_type == request_type

    def test_client(self):
        if False:
            return 10
        'Test calling client on an instance of ASGIProxyRequest.\n\n        When the client is not passed into the scope, it returns empty string.\n        When the request_type is passed into the scope, it returns the correct value.\n        '
        proxy_request = self.create_asgi_proxy_request(scope={})
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.client == ''
        client = 'fake-client'
        proxy_request = self.create_asgi_proxy_request(scope={'client': client})
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.client == client

    def test_method(self):
        if False:
            return 10
        'Test calling method on an instance of ASGIProxyRequest.\n\n        When the method is not passed into the scope, it returns "WEBSOCKET". When\n        the method is passed into the scope, it returns the correct value.\n        '
        proxy_request = self.create_asgi_proxy_request(scope={})
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.method == 'WEBSOCKET'
        method = 'fake-method'
        proxy_request = self.create_asgi_proxy_request(scope={'method': method})
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.method == method.upper()

    def test_root_path(self):
        if False:
            for i in range(10):
                print('nop')
        'Test calling root_path on an instance of ASGIProxyRequest.\n\n        When the root_path is not passed into the scope, it returns empty string.\n        When calling set_root_path, it correctly sets the root_path. When the\n        root_path is passed into the scope, it returns the correct value.\n        '
        proxy_request = self.create_asgi_proxy_request(scope={})
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.root_path == ''
        root_path = 'fake-root_path'
        proxy_request.set_root_path(root_path)
        assert proxy_request.root_path == root_path
        proxy_request = self.create_asgi_proxy_request(scope={'root_path': root_path})
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.root_path == root_path

    def test_path(self):
        if False:
            for i in range(10):
                print('nop')
        'Test calling path on an instance of ASGIProxyRequest.\n\n        When the path is not passed into the scope, it returns empty string.\n        When calling set_path, it correctly sets the path. When the\n        path is passed into the scope, it returns the correct value.\n        '
        proxy_request = self.create_asgi_proxy_request(scope={})
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.path == ''
        path = 'fake-path'
        proxy_request.set_path(path)
        assert proxy_request.path == path
        proxy_request = self.create_asgi_proxy_request(scope={'path': path})
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.path == path

    def test_headers(self):
        if False:
            print('Hello World!')
        'Test calling headers on an instance of ASGIProxyRequest.\n\n        When the headers are not passed into the scope, it returns empty list.\n        When the headers are passed into the scope, it returns the correct value.\n        '
        proxy_request = self.create_asgi_proxy_request(scope={})
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.headers == []
        headers = [(b'fake-header-key', b'fake-header-value')]
        proxy_request = self.create_asgi_proxy_request(scope={'headers': headers})
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.headers == headers

    def test_request_object(self):
        if False:
            return 10
        'Test calling request_object on an instance of ASGIProxyRequest.\n\n        When the request_object is called, it returns a StreamingHTTPRequest object\n        with the correct pickled_asgi_scope and http_proxy_handle.\n        '
        proxy_handle = MagicMock()
        headers = [(b'fake-header-key', b'fake-header-value')]
        scope = {'headers': headers}
        proxy_request = self.create_asgi_proxy_request(scope=scope)
        request_object = proxy_request.request_object(proxy_handle=proxy_handle)
        assert isinstance(request_object, StreamingHTTPRequest)
        assert pickle.loads(request_object.pickled_asgi_scope) == scope
        assert request_object.http_proxy_handle == proxy_handle

    def test_is_route_request(self):
        if False:
            for i in range(10):
                print('nop')
        'Test calling is_route_request on an instance of ASGIProxyRequest.\n\n        When the is_route_request is called with `/-/routes`, it returns true.\n        When the is_route_request is called with other path, it returns false.\n        '
        scope = {'path': '/-/routes'}
        proxy_request = self.create_asgi_proxy_request(scope=scope)
        assert proxy_request.is_route_request is True
        scope = {'path': '/foo'}
        proxy_request = self.create_asgi_proxy_request(scope=scope)
        assert proxy_request.is_route_request is False

    def test_is_health_request(self):
        if False:
            print('Hello World!')
        'Test calling is_health_request on an instance of ASGIProxyRequest.\n\n        When the is_health_request is called with `/-/healthz`, it returns true.\n        When the is_health_request is called with other path, it returns false.\n        '
        scope = {'path': '/-/healthz'}
        proxy_request = self.create_asgi_proxy_request(scope=scope)
        assert proxy_request.is_health_request is True
        scope = {'path': '/foo'}
        proxy_request = self.create_asgi_proxy_request(scope=scope)
        assert proxy_request.is_health_request is False

class TestgRPCProxyRequest:

    def test_calling_list_applications_method(self):
        if False:
            i = 10
            return i + 15
        'Test initialize gRPCProxyRequest with list applications service method.\n\n        When the gRPCProxyRequest is initialized with list application service method,\n        calling is_route_request should return true and calling is_health_request\n        should return false.\n        '
        context = FakeGrpcContext()
        request_proto = serve_pb2.ListApplicationsRequest()
        service_method = '/ray.serve.RayServeAPIService/ListApplications'
        proxy_request = gRPCProxyRequest(request_proto=request_proto, context=context, service_method=service_method, stream=False)
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.is_route_request is True
        assert proxy_request.is_health_request is False

    def test_calling_healthz_method(self):
        if False:
            print('Hello World!')
        'Test initialize gRPCProxyRequest with healthz service method.\n\n        When the gRPCProxyRequest is initialized with healthz service method, calling\n        is_route_request should return false and calling is_health_request\n        should return true.\n        '
        context = FakeGrpcContext()
        request_proto = serve_pb2.HealthzRequest()
        service_method = '/ray.serve.RayServeAPIService/Healthz'
        proxy_request = gRPCProxyRequest(request_proto=request_proto, context=context, service_method=service_method, stream=False)
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.is_route_request is False
        assert proxy_request.is_health_request is True

    def test_calling_user_defined_method(self):
        if False:
            print('Hello World!')
        'Test initialize gRPCProxyRequest with user defined service method.\n\n        When the gRPCProxyRequest is initialized with user defined service method,\n        all attributes should be setup accordingly. Calling both is_route_request\n        and is_health_request should return false. `send_request_id()` should\n        also work accordingly to be able to send the into back to the client.\n        `request_object()` generates a gRPCRequest object with the correct attributes.\n        '
        request_proto = serve_pb2.UserDefinedMessage(name='foo', num=30, foo='bar')
        application = 'fake-application'
        request_id = 'fake-request_id'
        multiplexed_model_id = 'fake-multiplexed_model_id'
        metadata = (('foo', 'bar'), ('application', application), ('request_id', request_id), ('multiplexed_model_id', multiplexed_model_id))
        context = MagicMock()
        context.invocation_metadata.return_value = metadata
        method_name = 'Method1'
        service_method = f'/custom.defined.Service/{method_name}'
        proxy_request = gRPCProxyRequest(request_proto=request_proto, context=context, service_method=service_method, stream=MagicMock())
        assert isinstance(proxy_request, ProxyRequest)
        assert proxy_request.route_path == application
        assert pickle.loads(proxy_request.request) == request_proto
        assert proxy_request.method_name == method_name
        assert proxy_request.app_name == application
        assert proxy_request.request_id == request_id
        assert proxy_request.multiplexed_model_id == multiplexed_model_id
        assert proxy_request.is_route_request is False
        assert proxy_request.is_health_request is False
        proxy_request.send_request_id(request_id=request_id)
        context.set_trailing_metadata.assert_called_with([('request_id', request_id)])
        proxy_handle = MagicMock()
        request_object = proxy_request.request_object(proxy_handle=proxy_handle)
        assert isinstance(request_object, gRPCRequest)
        assert pickle.loads(request_object.grpc_user_request) == request_proto
        assert request_object.grpc_proxy_handle == proxy_handle
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', '-s', __file__]))