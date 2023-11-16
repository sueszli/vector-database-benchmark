import pytest
from azure.core.exceptions import ServiceRequestError
from azure.core.rest import HttpRequest
from azure.core.polling import LROPoller
from azure.core.polling.base_polling import LROBasePolling

@pytest.fixture
def deserialization_callback():
    if False:
        i = 10
        return i + 15

    def _callback(response):
        if False:
            for i in range(10):
                print('nop')
        return response.http_response.json()
    return _callback

@pytest.fixture
def lro_poller(client, deserialization_callback):
    if False:
        print('Hello World!')

    def _callback(request, **kwargs):
        if False:
            i = 10
            return i + 15
        initial_response = client.send_request(request=request, _return_pipeline_response=True)
        return LROPoller(client._client, initial_response, deserialization_callback, LROBasePolling(0, **kwargs))
    return _callback

def test_post_with_location_and_operation_location_headers(lro_poller):
    if False:
        i = 10
        return i + 15
    poller = lro_poller(HttpRequest('POST', '/polling/post/location-and-operation-location'))
    result = poller.result()
    assert result == {'location_result': True}

def test_post_with_location_and_operation_location_headers_no_body(lro_poller):
    if False:
        while True:
            i = 10
    poller = lro_poller(HttpRequest('POST', '/polling/post/location-and-operation-location-no-body'))
    result = poller.result()
    assert result is None

def test_post_resource_location(lro_poller):
    if False:
        i = 10
        return i + 15
    poller = lro_poller(HttpRequest('POST', '/polling/post/resource-location'))
    result = poller.result()
    assert result == {'location_result': True}

def test_put_no_polling(lro_poller):
    if False:
        print('Hello World!')
    result = lro_poller(HttpRequest('PUT', '/polling/no-polling')).result()
    assert result['properties']['provisioningState'] == 'Succeeded'

def test_put_location(lro_poller):
    if False:
        i = 10
        return i + 15
    result = lro_poller(HttpRequest('PUT', '/polling/location')).result()
    assert result['location_result']

def test_put_initial_response_body_invalid(lro_poller):
    if False:
        while True:
            i = 10
    result = lro_poller(HttpRequest('PUT', '/polling/initial-body-invalid')).result()
    assert result['location_result']

def test_put_operation_location_polling_fail(lro_poller):
    if False:
        print('Hello World!')
    with pytest.raises(ServiceRequestError):
        lro_poller(HttpRequest('PUT', '/polling/bad-operation-location'), retry_total=0).result()

def test_put_location_polling_fail(lro_poller):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ServiceRequestError):
        lro_poller(HttpRequest('PUT', '/polling/bad-location'), retry_total=0).result()

def test_patch_location(lro_poller):
    if False:
        while True:
            i = 10
    result = lro_poller(HttpRequest('PATCH', '/polling/location')).result()
    assert result['location_result']

def test_patch_operation_location_polling_fail(lro_poller):
    if False:
        while True:
            i = 10
    with pytest.raises(ServiceRequestError):
        lro_poller(HttpRequest('PUT', '/polling/bad-operation-location'), retry_total=0).result()

def test_patch_location_polling_fail(lro_poller):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ServiceRequestError):
        lro_poller(HttpRequest('PUT', '/polling/bad-location'), retry_total=0).result()

def test_delete_operation_location(lro_poller):
    if False:
        return 10
    result = lro_poller(HttpRequest('DELETE', '/polling/operation-location')).result()
    assert result['status'] == 'Succeeded'

def test_request_id(lro_poller):
    if False:
        return 10
    result = lro_poller(HttpRequest('POST', '/polling/request-id'), request_id='123456789').result()

def test_continuation_token(client, lro_poller, deserialization_callback):
    if False:
        print('Hello World!')
    poller = lro_poller(HttpRequest('POST', '/polling/post/location-and-operation-location'))
    token = poller.continuation_token()
    new_poller = LROPoller.from_continuation_token(continuation_token=token, polling_method=LROBasePolling(0), client=client._client, deserialization_callback=deserialization_callback)
    result = new_poller.result()
    assert result == {'location_result': True}