import base64
import json
import pickle
import re
from utils import HTTP_REQUESTS
from azure.core.pipeline._tools import is_rest
import types
from unittest import mock
import pytest
from requests import Request, Response
from azure.core.polling import async_poller, AsyncLROPoller
from azure.core.exceptions import DecodeError, HttpResponseError
from azure.core import AsyncPipelineClient
from azure.core.pipeline import PipelineResponse, AsyncPipeline, PipelineContext
from azure.core.pipeline.transport import AsyncioRequestsTransportResponse, AsyncHttpTransport
from azure.core.polling.base_polling import LROBasePolling
from azure.core.polling.async_base_polling import AsyncLROBasePolling
from utils import ASYNCIO_REQUESTS_TRANSPORT_RESPONSES, request_and_responses_product, create_transport_response
from rest_client_async import AsyncMockRestClient

class SimpleResource:
    """An implementation of Python 3 SimpleNamespace.
    Used to deserialize resource objects from response bodies where
    no particular object type has been specified.
    """

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        self.__dict__.update(kwargs)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        keys = sorted(self.__dict__)
        items = ('{}={!r}'.format(k, self.__dict__[k]) for k in keys)
        return '{}({})'.format(type(self).__name__, ', '.join(items))

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.__dict__ == other.__dict__

class BadEndpointError(Exception):
    pass
TEST_NAME = 'foo'
RESPONSE_BODY = {'properties': {'provisioningState': 'InProgress'}}
ASYNC_BODY = json.dumps({'status': 'Succeeded'})
ASYNC_URL = 'http://dummyurlFromAzureAsyncOPHeader_Return200'
LOCATION_BODY = json.dumps({'name': TEST_NAME})
LOCATION_URL = 'http://dummyurlurlFromLocationHeader_Return200'
RESOURCE_BODY = json.dumps({'name': TEST_NAME})
RESOURCE_URL = 'http://subscriptions/sub1/resourcegroups/g1/resourcetype1/resource1'
ERROR = 'http://dummyurl_ReturnError'
POLLING_STATUS = 200
CLIENT = AsyncPipelineClient('http://example.org')
CLIENT.http_request_type = None
CLIENT.http_response_type = None

async def mock_run(client_self, request, **kwargs):
    return TestBasePolling.mock_update(client_self.http_request_type, client_self.http_response_type, request.url)
CLIENT._pipeline.run = types.MethodType(mock_run, CLIENT)

@pytest.fixture
def client():
    if False:
        i = 10
        return i + 15
    return AsyncPipelineClient('https://baseurl')

@pytest.fixture
def async_pipeline_client_builder():
    if False:
        return 10
    'Build a client that use the "send" callback as final transport layer\n\n    send will receive "request" and kwargs as any transport layer\n    '

    def create_client(send_cb):
        if False:
            return 10

        class TestHttpTransport(AsyncHttpTransport):

            async def open(self):
                pass

            async def close(self):
                pass

            async def __aexit__(self, *args, **kwargs):
                pass

            async def send(self, request, **kwargs):
                return await send_cb(request, **kwargs)
        return AsyncPipelineClient('http://example.org/', pipeline=AsyncPipeline(transport=TestHttpTransport()))
    return create_client

@pytest.fixture
def deserialization_cb():
    if False:
        print('Hello World!')

    def cb(pipeline_response):
        if False:
            return 10
        return json.loads(pipeline_response.http_response.text())
    return cb

@pytest.fixture
def polling_response():
    if False:
        for i in range(10):
            print('nop')
    polling = AsyncLROBasePolling()
    headers = {}
    response = Response()
    response.headers = headers
    response.status_code = 200
    polling._pipeline_response = PipelineResponse(None, AsyncioRequestsTransportResponse(None, response), PipelineContext(None))
    polling._initial_response = polling._pipeline_response
    return (polling, headers)

def test_base_polling_continuation_token(client, polling_response):
    if False:
        return 10
    (polling, _) = polling_response
    continuation_token = polling.get_continuation_token()
    assert isinstance(continuation_token, str)
    polling_args = AsyncLROBasePolling.from_continuation_token(continuation_token, deserialization_callback='deserialization_callback', client=client)
    new_polling = AsyncLROBasePolling()
    new_polling.initialize(*polling_args)

@pytest.mark.asyncio
@pytest.mark.parametrize('http_request,http_response', request_and_responses_product(ASYNCIO_REQUESTS_TRANSPORT_RESPONSES))
async def test_post(async_pipeline_client_builder, deserialization_cb, http_request, http_response):
    initial_response = TestBasePolling.mock_send(http_request, http_response, 'POST', 202, {'location': 'http://example.org/location', 'operation-location': 'http://example.org/async_monitor'}, '')

    async def send(request, **kwargs):
        assert request.method == 'GET'
        if request.url == 'http://example.org/location':
            return TestBasePolling.mock_send(http_request, http_response, 'GET', 200, body={'location_result': True}).http_response
        elif request.url == 'http://example.org/async_monitor':
            return TestBasePolling.mock_send(http_request, http_response, 'GET', 200, body={'status': 'Succeeded'}).http_response
        else:
            pytest.fail('No other query allowed')
    client = async_pipeline_client_builder(send)
    poll = async_poller(client, initial_response, deserialization_cb, AsyncLROBasePolling(0))
    result = await poll
    assert result['location_result'] == True

    async def send(request, **kwargs):
        assert request.method == 'GET'
        if request.url == 'http://example.org/location':
            return TestBasePolling.mock_send(http_request, http_response, 'GET', 200, body=None).http_response
        elif request.url == 'http://example.org/async_monitor':
            return TestBasePolling.mock_send(http_request, http_response, 'GET', 200, body={'status': 'Succeeded'}).http_response
        else:
            pytest.fail('No other query allowed')
    client = async_pipeline_client_builder(send)
    poll = async_poller(client, initial_response, deserialization_cb, AsyncLROBasePolling(0))
    result = await poll
    assert result is None

@pytest.mark.asyncio
@pytest.mark.parametrize('http_request,http_response', request_and_responses_product(ASYNCIO_REQUESTS_TRANSPORT_RESPONSES))
async def test_post_resource_location(async_pipeline_client_builder, deserialization_cb, http_request, http_response):
    initial_response = TestBasePolling.mock_send(http_request, http_response, 'POST', 202, {'operation-location': 'http://example.org/async_monitor'}, '')

    async def send(request, **kwargs):
        assert request.method == 'GET'
        if request.url == 'http://example.org/resource_location':
            return TestBasePolling.mock_send(http_request, http_response, 'GET', 200, body={'location_result': True}).http_response
        elif request.url == 'http://example.org/async_monitor':
            return TestBasePolling.mock_send(http_request, http_response, 'GET', 200, body={'status': 'Succeeded', 'resourceLocation': 'http://example.org/resource_location'}).http_response
        else:
            pytest.fail('No other query allowed')
    client = async_pipeline_client_builder(send)
    poll = async_poller(client, initial_response, deserialization_cb, AsyncLROBasePolling(0))
    result = await poll
    assert result['location_result'] == True

@pytest.mark.asyncio
@pytest.mark.parametrize('http_request,http_response', request_and_responses_product(ASYNCIO_REQUESTS_TRANSPORT_RESPONSES))
async def test_post_direct_success(async_pipeline_client_builder, deserialization_cb, http_request, http_response):
    initial_response = TestBasePolling.mock_send(http_request, http_response, 'POST', 202, {'operation-location': 'http://example.org/async_monitor'}, {'status': 'succeeded'})

    async def send(request, **kwargs):
        pytest.fail('No requests allowed')
    client = async_pipeline_client_builder(send)
    poll = async_poller(client, initial_response, deserialization_cb, AsyncLROBasePolling(0))
    result = await poll
    assert result['status'] == 'succeeded'

@pytest.mark.asyncio
@pytest.mark.parametrize('http_request,http_response', request_and_responses_product(ASYNCIO_REQUESTS_TRANSPORT_RESPONSES))
async def test_post_fail(async_pipeline_client_builder, deserialization_cb, http_request, http_response):
    initial_response = TestBasePolling.mock_send(http_request, http_response, 'POST', 500, {'status': 'failed'})

    async def send(request, **kwargs):
        pytest.fail('No requests allowed')
    client = async_pipeline_client_builder(send)
    with pytest.raises(HttpResponseError):
        poll = async_poller(client, initial_response, deserialization_cb, AsyncLROBasePolling(0))
        result = await poll

class TestBasePolling(object):
    convert = re.compile('([a-z0-9])([A-Z])')

    @staticmethod
    def mock_send(http_request, http_response, method, status, headers=None, body=RESPONSE_BODY):
        if False:
            while True:
                i = 10
        if headers is None:
            headers = {}
        response = Response()
        response._content_consumed = True
        response._content = json.dumps(body).encode('ascii') if body is not None else b''
        response.request = Request()
        response.request.method = method
        response.request.url = RESOURCE_URL
        response.request.headers = {'x-ms-client-request-id': '67f4dd4e-6262-45e1-8bed-5c45cf23b6d9'}
        response.status_code = status
        response.headers = headers
        response.headers.update({'content-type': 'application/json; charset=utf8'})
        response.reason = 'OK'
        if is_rest(http_request):
            request = http_request(response.request.method, response.request.url, headers=response.request.headers, content=body)
        else:
            request = CLIENT._request(response.request.method, response.request.url, None, response.request.headers, body, None, None)
        response = create_transport_response(http_response, request, response)
        if is_rest(http_response):
            response.body()
        return PipelineResponse(request, response, None)

    @staticmethod
    def mock_update(http_request, http_response, url, headers=None):
        if False:
            for i in range(10):
                print('nop')
        response = Response()
        response._content_consumed = True
        response.request = mock.create_autospec(Request)
        response.request.method = 'GET'
        response.headers = headers or {}
        response.headers.update({'content-type': 'application/json; charset=utf8'})
        response.reason = 'OK'
        if url == ASYNC_URL:
            response.request.url = url
            response.status_code = POLLING_STATUS
            response._content = ASYNC_BODY.encode('ascii')
            response.randomFieldFromPollAsyncOpHeader = None
        elif url == LOCATION_URL:
            response.request.url = url
            response.status_code = POLLING_STATUS
            response._content = LOCATION_BODY.encode('ascii')
            response.randomFieldFromPollLocationHeader = None
        elif url == ERROR:
            raise BadEndpointError('boom')
        elif url == RESOURCE_URL:
            response.request.url = url
            response.status_code = POLLING_STATUS
            response._content = RESOURCE_BODY.encode('ascii')
        else:
            raise Exception('URL does not match')
        request = http_request(response.request.method, response.request.url)
        response = create_transport_response(http_response, request, response)
        if is_rest(http_response):
            response.body()
        return PipelineResponse(request, response, None)

    @staticmethod
    def mock_outputs(pipeline_response):
        if False:
            print('Hello World!')
        response = pipeline_response.http_response
        try:
            body = json.loads(response.text())
        except ValueError:
            raise DecodeError('Impossible to deserialize')
        body = {TestBasePolling.convert.sub('\\1_\\2', k).lower(): v for (k, v) in body.items()}
        properties = body.setdefault('properties', {})
        if 'name' in body:
            properties['name'] = body['name']
        if properties:
            properties = {TestBasePolling.convert.sub('\\1_\\2', k).lower(): v for (k, v) in properties.items()}
            del body['properties']
            body.update(properties)
            resource = SimpleResource(**body)
        else:
            raise DecodeError('Impossible to deserialize')
            resource = SimpleResource(**body)
        return resource

    @staticmethod
    def mock_deserialization_no_body(pipeline_response):
        if False:
            print('Hello World!')
        "Use this mock when you don't expect a return (last body irrelevant)"
        return None

@pytest.mark.asyncio
@pytest.mark.parametrize('http_request,http_response', request_and_responses_product(ASYNCIO_REQUESTS_TRANSPORT_RESPONSES))
async def test_long_running_put(http_request, http_response):
    CLIENT.http_request_type = http_request
    CLIENT.http_response_type = http_response
    response = TestBasePolling.mock_send(http_request, http_response, 'PUT', 1000, {})
    with pytest.raises(HttpResponseError):
        await async_poller(CLIENT, response, TestBasePolling.mock_outputs, AsyncLROBasePolling(0))
    response_body = {'properties': {'provisioningState': 'Succeeded'}, 'name': TEST_NAME}
    response = TestBasePolling.mock_send(http_request, http_response, 'PUT', 201, {}, response_body)

    def no_update_allowed(url, headers=None):
        if False:
            while True:
                i = 10
        raise ValueError('Should not try to update')
    polling_method = AsyncLROBasePolling(0)
    poll = await async_poller(CLIENT, response, TestBasePolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert not hasattr(polling_method._pipeline_response, 'randomFieldFromPollAsyncOpHeader')
    response = TestBasePolling.mock_send(http_request, http_response, 'PUT', 201, {'operation-location': ASYNC_URL})
    polling_method = AsyncLROBasePolling(0)
    poll = await async_poller(CLIENT, response, TestBasePolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert not hasattr(polling_method._pipeline_response, 'randomFieldFromPollAsyncOpHeader')
    response = TestBasePolling.mock_send(http_request, http_response, 'PUT', 201, {'location': LOCATION_URL})
    polling_method = AsyncLROBasePolling(0)
    poll = await async_poller(CLIENT, response, TestBasePolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollLocationHeader is None
    response_body = {}
    response = TestBasePolling.mock_send(http_request, http_response, 'PUT', 201, {'location': LOCATION_URL}, response_body)
    polling_method = AsyncLROBasePolling(0)
    poll = await async_poller(CLIENT, response, TestBasePolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollLocationHeader is None
    response = TestBasePolling.mock_send(http_request, http_response, 'PUT', 201, {'operation-location': ERROR})
    with pytest.raises(BadEndpointError):
        poll = await async_poller(CLIENT, response, TestBasePolling.mock_outputs, AsyncLROBasePolling(0))
    response = TestBasePolling.mock_send(http_request, http_response, 'PUT', 201, {'location': ERROR})
    with pytest.raises(BadEndpointError):
        poll = await async_poller(CLIENT, response, TestBasePolling.mock_outputs, AsyncLROBasePolling(0))

@pytest.mark.asyncio
@pytest.mark.parametrize('http_request,http_response', request_and_responses_product(ASYNCIO_REQUESTS_TRANSPORT_RESPONSES))
async def test_long_running_patch(http_request, http_response):
    CLIENT.http_request_type = http_request
    CLIENT.http_response_type = http_response
    response = TestBasePolling.mock_send(http_request, http_response, 'PATCH', 202, {'location': LOCATION_URL}, body={'properties': {'provisioningState': 'Succeeded'}})
    polling_method = AsyncLROBasePolling(0)
    poll = await async_poller(CLIENT, response, TestBasePolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollLocationHeader is None
    response = TestBasePolling.mock_send(http_request, http_response, 'PATCH', 202, {'operation-location': ASYNC_URL}, body={'properties': {'provisioningState': 'Succeeded'}})
    polling_method = AsyncLROBasePolling(0)
    poll = await async_poller(CLIENT, response, TestBasePolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert not hasattr(polling_method._pipeline_response, 'randomFieldFromPollAsyncOpHeader')
    response = TestBasePolling.mock_send(http_request, http_response, 'PATCH', 200, {'location': LOCATION_URL}, body={'properties': {'provisioningState': 'Succeeded'}})
    polling_method = AsyncLROBasePolling(0)
    poll = await async_poller(CLIENT, response, TestBasePolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollLocationHeader is None
    response = TestBasePolling.mock_send(http_request, http_response, 'PATCH', 200, {'operation-location': ASYNC_URL}, body={'properties': {'provisioningState': 'Succeeded'}})
    polling_method = AsyncLROBasePolling(0)
    poll = await async_poller(CLIENT, response, TestBasePolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert not hasattr(polling_method._pipeline_response, 'randomFieldFromPollAsyncOpHeader')
    response = TestBasePolling.mock_send(http_request, http_response, 'PATCH', 202, {'operation-location': ERROR})
    with pytest.raises(BadEndpointError):
        poll = await async_poller(CLIENT, response, TestBasePolling.mock_outputs, AsyncLROBasePolling(0))
    response = TestBasePolling.mock_send(http_request, http_response, 'PATCH', 202, {'location': ERROR})
    with pytest.raises(BadEndpointError):
        poll = await async_poller(CLIENT, response, TestBasePolling.mock_outputs, AsyncLROBasePolling(0))

@pytest.mark.asyncio
@pytest.mark.parametrize('http_request,http_response', request_and_responses_product(ASYNCIO_REQUESTS_TRANSPORT_RESPONSES))
async def test_long_running_delete(http_request, http_response):
    CLIENT.http_request_type = http_request
    CLIENT.http_response_type = http_response
    response = TestBasePolling.mock_send(http_request, http_response, 'DELETE', 202, {'operation-location': ASYNC_URL}, body='')
    polling_method = AsyncLROBasePolling(0)
    poll = await async_poller(CLIENT, response, TestBasePolling.mock_deserialization_no_body, polling_method)
    assert poll is None
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollAsyncOpHeader is None

@pytest.mark.asyncio
@pytest.mark.parametrize('http_request,http_response', request_and_responses_product(ASYNCIO_REQUESTS_TRANSPORT_RESPONSES))
async def test_long_running_post(http_request, http_response):
    CLIENT.http_request_type = http_request
    CLIENT.http_response_type = http_response
    response = TestBasePolling.mock_send(http_request, http_response, 'POST', 201, {'operation-location': ASYNC_URL}, body={'properties': {'provisioningState': 'Succeeded'}})
    polling_method = AsyncLROBasePolling(0)
    poll = await async_poller(CLIENT, response, TestBasePolling.mock_deserialization_no_body, polling_method)
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollAsyncOpHeader is None
    response = TestBasePolling.mock_send(http_request, http_response, 'POST', 202, {'operation-location': ASYNC_URL}, body={'properties': {'provisioningState': 'Succeeded'}})
    polling_method = AsyncLROBasePolling(0)
    poll = await async_poller(CLIENT, response, TestBasePolling.mock_deserialization_no_body, polling_method)
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollAsyncOpHeader is None
    response = TestBasePolling.mock_send(http_request, http_response, 'POST', 202, {'location': LOCATION_URL}, body={'properties': {'provisioningState': 'Succeeded'}})
    polling_method = AsyncLROBasePolling(0)
    poll = await async_poller(CLIENT, response, TestBasePolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollLocationHeader is None
    response = TestBasePolling.mock_send(http_request, http_response, 'POST', 202, {'operation-location': ERROR})
    with pytest.raises(BadEndpointError):
        await async_poller(CLIENT, response, TestBasePolling.mock_outputs, AsyncLROBasePolling(0))
    response = TestBasePolling.mock_send(http_request, http_response, 'POST', 202, {'location': ERROR})
    with pytest.raises(BadEndpointError):
        await async_poller(CLIENT, response, TestBasePolling.mock_outputs, AsyncLROBasePolling(0))

@pytest.mark.asyncio
@pytest.mark.parametrize('http_request,http_response', request_and_responses_product(ASYNCIO_REQUESTS_TRANSPORT_RESPONSES))
async def test_long_running_negative(http_request, http_response):
    global LOCATION_BODY
    global POLLING_STATUS
    CLIENT.http_request_type = http_request
    CLIENT.http_response_type = http_response
    LOCATION_BODY = '{'
    response = TestBasePolling.mock_send(http_request, http_response, 'POST', 202, {'location': LOCATION_URL})
    poll = async_poller(CLIENT, response, TestBasePolling.mock_outputs, AsyncLROBasePolling(0))
    with pytest.raises(DecodeError):
        await poll
    LOCATION_BODY = '{\'"}'
    response = TestBasePolling.mock_send(http_request, http_response, 'POST', 202, {'location': LOCATION_URL})
    poll = async_poller(CLIENT, response, TestBasePolling.mock_outputs, AsyncLROBasePolling(0))
    with pytest.raises(DecodeError):
        await poll
    LOCATION_BODY = '{'
    POLLING_STATUS = 203
    response = TestBasePolling.mock_send(http_request, http_response, 'POST', 202, {'location': LOCATION_URL})
    poll = async_poller(CLIENT, response, TestBasePolling.mock_outputs, AsyncLROBasePolling(0))
    with pytest.raises(HttpResponseError) as error:
        await poll
    assert error.value.continuation_token == base64.b64encode(pickle.dumps(response)).decode('ascii')
    LOCATION_BODY = json.dumps({'name': TEST_NAME})
    POLLING_STATUS = 200

@pytest.mark.asyncio
@pytest.mark.parametrize('http_request,http_response', request_and_responses_product(ASYNCIO_REQUESTS_TRANSPORT_RESPONSES))
async def test_post_final_state_via(async_pipeline_client_builder, deserialization_cb, http_request, http_response):
    CLIENT.http_request_type = http_request
    CLIENT.http_response_type = http_response
    initial_response = TestBasePolling.mock_send(http_request, http_response, 'POST', 202, {'location': 'http://example.org/location', 'operation-location': 'http://example.org/async_monitor'}, '')

    async def send(request, **kwargs):
        assert request.method == 'GET'
        if request.url == 'http://example.org/location':
            return TestBasePolling.mock_send(http_request, http_response, 'GET', 200, body={'location_result': True}).http_response
        elif request.url == 'http://example.org/async_monitor':
            return TestBasePolling.mock_send(http_request, http_response, 'GET', 200, body={'status': 'Succeeded'}).http_response
        else:
            pytest.fail('No other query allowed')
    client = async_pipeline_client_builder(send)
    poll = async_poller(client, initial_response, deserialization_cb, AsyncLROBasePolling(0, lro_options={'final-state-via': 'location'}))
    result = await poll
    assert result['location_result'] == True
    poll = async_poller(client, initial_response, deserialization_cb, AsyncLROBasePolling(0, lro_options={'final-state-via': 'operation-location'}))
    result = await poll
    assert result['status'] == 'Succeeded'
    poll = async_poller(client, initial_response, deserialization_cb, AsyncLROBasePolling(0))
    result = await poll
    assert result['location_result'] == True

    async def send(request, **kwargs):
        assert request.method == 'GET'
        if request.url == 'http://example.org/location':
            return TestBasePolling.mock_send(http_request, http_response, 'GET', 200, body=None).http_response
        elif request.url == 'http://example.org/async_monitor':
            return TestBasePolling.mock_send(http_request, http_response, 'GET', 200, body={'status': 'Succeeded'}).http_response
        else:
            pytest.fail('No other query allowed')
    client = async_pipeline_client_builder(send)
    poll = async_poller(client, initial_response, deserialization_cb, AsyncLROBasePolling(0, lro_options={'final-state-via': 'location'}))
    result = await poll
    assert result is None

@pytest.mark.asyncio
@pytest.mark.parametrize('http_request', HTTP_REQUESTS)
async def test_final_get_via_location(port, http_request, deserialization_cb):
    client = AsyncMockRestClient(port)
    request = http_request('PUT', 'http://localhost:{}/polling/polling-with-options'.format(port))
    request.set_json_body({'hello': 'world!'})
    initial_response = await client._client._pipeline.run(request)
    poller = AsyncLROPoller(client._client, initial_response, deserialization_cb, AsyncLROBasePolling(0, lro_options={'final-state-via': 'location'}))
    result = await poller.result()
    assert result == {'returnedFrom': 'locationHeaderUrl'}
'Reproduce the bad design of azure-mgmt-core 1.0.0-1.4.0'

class ARMPolling(LROBasePolling):
    pass

class AsyncARMPolling(ARMPolling, AsyncLROBasePolling):
    pass

@pytest.mark.asyncio
async def test_async_polling_inheritance(async_pipeline_client_builder, deserialization_cb):
    rest_http = request_and_responses_product(ASYNCIO_REQUESTS_TRANSPORT_RESPONSES)[1]

    async def send(request, **kwargs):
        assert request.method == 'GET'
        if request.url == 'http://example.org/location':
            return TestBasePolling.mock_send(rest_http[0], rest_http[1], 'GET', 200, body={'success': True}).http_response
        elif request.url == 'http://example.org/async_monitor':
            return TestBasePolling.mock_send(rest_http[0], rest_http[1], 'GET', 200, body={'status': 'Succeeded'}).http_response
        else:
            pytest.fail('No other query allowed')
    client = async_pipeline_client_builder(send)
    initial_response = TestBasePolling.mock_send(rest_http[0], rest_http[1], 'POST', 200, {'location': 'http://example.org/location', 'operation-location': 'http://example.org/async_monitor'}, '')
    polling = AsyncARMPolling()
    polling.initialize(client=client, initial_response=initial_response, deserialization_callback=deserialization_cb)
    await polling.run()
    resource = polling.resource()
    assert resource['success']