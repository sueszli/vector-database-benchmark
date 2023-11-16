import pytest
from ray.serve._private.common import RequestProtocol
from ray.serve._private.router import RequestMetadata

def test_request_metadata():
    if False:
        i = 10
        return i + 15
    'Test logic on RequestMetadata.\n\n    Ensure the default values are set correctly and both is_http_request and\n    is_grpc_request returns the correct value. When the _request_protocol is set to\n    HTTP, is_http_request should return True and is_grpc_request should return False.\n    When the _request_protocol is set to gRPC, is_http_request should return False and\n    is_grpc_request should return True.\n    '
    request_id = 'request-id'
    endpoint = 'endpoint'
    request_metadata = RequestMetadata(request_id=request_id, endpoint=endpoint)
    assert request_metadata.request_id == request_id
    assert request_metadata.endpoint == endpoint
    assert request_metadata.call_method == '__call__'
    assert request_metadata.route == ''
    assert request_metadata.app_name == ''
    assert request_metadata.multiplexed_model_id == ''
    assert request_metadata.is_streaming is False
    assert request_metadata._request_protocol == RequestProtocol.UNDEFINED
    assert request_metadata.is_http_request is False
    assert request_metadata.is_grpc_request is False
    request_metadata._request_protocol = RequestProtocol.HTTP
    assert request_metadata.is_http_request is True
    assert request_metadata.is_grpc_request is False
    request_metadata._request_protocol = RequestProtocol.GRPC
    assert request_metadata.is_http_request is False
    assert request_metadata.is_grpc_request is True
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', '-s', __file__]))