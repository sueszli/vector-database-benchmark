import base64
import json
import pickle
import re
import types
from unittest import mock
import pytest
from requests import Request, Response
from azure.core.polling import async_poller
from azure.core.exceptions import DecodeError, HttpResponseError
from azure.core import AsyncPipelineClient
from azure.core.pipeline import PipelineResponse, AsyncPipeline
from azure.core.pipeline.transport import AsyncioRequestsTransportResponse, AsyncHttpTransport
from azure.core.polling.base_polling import LongRunningOperation, BadStatus, LocationPolling
from azure.mgmt.core.polling.async_arm_polling import AsyncARMPolling

class SimpleResource:
    """An implementation of Python 3 SimpleNamespace.
    Used to deserialize resource objects from response bodies where
    no particular object type has been specified.
    """

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.__dict__.update(kwargs)

    def __repr__(self):
        if False:
            return 10
        keys = sorted(self.__dict__)
        items = ('{}={!r}'.format(k, self.__dict__[k]) for k in keys)
        return '{}({})'.format(type(self).__name__, ', '.join(items))

    def __eq__(self, other):
        if False:
            return 10
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

async def mock_run(client_self, request, **kwargs):
    return TestArmPolling.mock_update(request.url)
CLIENT._pipeline.run = types.MethodType(mock_run, CLIENT)

@pytest.fixture
def async_pipeline_client_builder():
    if False:
        i = 10
        return i + 15
    'Build a client that use the "send" callback as final transport layer\n\n    send will receive "request" and kwargs as any transport layer\n    '

    def create_client(send_cb):
        if False:
            print('Hello World!')

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
        while True:
            i = 10

    def cb(pipeline_response):
        if False:
            return 10
        return json.loads(pipeline_response.http_response.text())
    return cb

@pytest.mark.asyncio
async def test_post(async_pipeline_client_builder, deserialization_cb):
    initial_response = TestArmPolling.mock_send('POST', 202, {'location': 'http://example.org/location', 'azure-asyncoperation': 'http://example.org/async_monitor'}, '')

    async def send(request, **kwargs):
        assert request.method == 'GET'
        if request.url == 'http://example.org/location':
            return TestArmPolling.mock_send('GET', 200, body={'location_result': True}).http_response
        elif request.url == 'http://example.org/async_monitor':
            return TestArmPolling.mock_send('GET', 200, body={'status': 'Succeeded'}).http_response
        else:
            pytest.fail('No other query allowed')
    client = async_pipeline_client_builder(send)
    poll = async_poller(client, initial_response, deserialization_cb, AsyncARMPolling(0, lro_options={'final-state-via': 'location'}))
    result = await poll
    assert result['location_result'] == True
    poll = async_poller(client, initial_response, deserialization_cb, AsyncARMPolling(0, lro_options={'final-state-via': 'azure-async-operation'}))
    result = await poll
    assert result['status'] == 'Succeeded'
    poll = async_poller(client, initial_response, deserialization_cb, AsyncARMPolling(0))
    result = await poll
    assert result['location_result'] == True

    async def send(request, **kwargs):
        assert request.method == 'GET'
        if request.url == 'http://example.org/location':
            return TestArmPolling.mock_send('GET', 200, body=None).http_response
        elif request.url == 'http://example.org/async_monitor':
            return TestArmPolling.mock_send('GET', 200, body={'status': 'Succeeded'}).http_response
        else:
            pytest.fail('No other query allowed')
    client = async_pipeline_client_builder(send)
    poll = async_poller(client, initial_response, deserialization_cb, AsyncARMPolling(0, lro_options={'final-state-via': 'location'}))
    result = await poll
    assert result is None

class TestArmPolling(object):
    convert = re.compile('([a-z0-9])([A-Z])')

    @staticmethod
    def mock_send(method, status, headers=None, body=RESPONSE_BODY):
        if False:
            for i in range(10):
                print('nop')
        if headers is None:
            headers = {}
        response = Response()
        response._content_consumed = True
        response._content = json.dumps(body).encode('ascii') if body is not None else None
        response.request = Request()
        response.request.method = method
        response.request.url = RESOURCE_URL
        response.request.headers = {'x-ms-client-request-id': '67f4dd4e-6262-45e1-8bed-5c45cf23b6d9'}
        response.status_code = status
        response.headers = headers
        response.headers.update({'content-type': 'application/json; charset=utf8'})
        response.reason = 'OK'
        request = CLIENT._request(response.request.method, response.request.url, None, response.request.headers, body, None, None)
        return PipelineResponse(request, AsyncioRequestsTransportResponse(request, response), None)

    @staticmethod
    def mock_update(url, headers=None):
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
        request = CLIENT._request(response.request.method, response.request.url, None, {}, None, None, None)
        return PipelineResponse(request, AsyncioRequestsTransportResponse(request, response), None)

    @staticmethod
    def mock_outputs(pipeline_response):
        if False:
            while True:
                i = 10
        response = pipeline_response.http_response
        try:
            body = json.loads(response.text())
        except ValueError:
            raise DecodeError('Impossible to deserialize')
        body = {TestArmPolling.convert.sub('\\1_\\2', k).lower(): v for (k, v) in body.items()}
        properties = body.setdefault('properties', {})
        if 'name' in body:
            properties['name'] = body['name']
        if properties:
            properties = {TestArmPolling.convert.sub('\\1_\\2', k).lower(): v for (k, v) in properties.items()}
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
            i = 10
            return i + 15
        "Use this mock when you don't expect a return (last body irrelevant)"
        return None

@pytest.mark.asyncio
async def test_long_running_put():
    response = TestArmPolling.mock_send('PUT', 1000, {})
    with pytest.raises(HttpResponseError):
        await async_poller(CLIENT, response, TestArmPolling.mock_outputs, AsyncARMPolling(0))
    response_body = {'properties': {'provisioningState': 'Succeeded'}, 'name': TEST_NAME}
    response = TestArmPolling.mock_send('PUT', 201, {}, response_body)

    def no_update_allowed(url, headers=None):
        if False:
            return 10
        raise ValueError('Should not try to update')
    polling_method = AsyncARMPolling(0)
    poll = await async_poller(CLIENT, response, TestArmPolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert not hasattr(polling_method._pipeline_response, 'randomFieldFromPollAsyncOpHeader')
    response = TestArmPolling.mock_send('PUT', 201, {'azure-asyncoperation': ASYNC_URL})
    polling_method = AsyncARMPolling(0)
    poll = await async_poller(CLIENT, response, TestArmPolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert not hasattr(polling_method._pipeline_response, 'randomFieldFromPollAsyncOpHeader')
    response = TestArmPolling.mock_send('PUT', 201, {'location': LOCATION_URL})
    polling_method = AsyncARMPolling(0)
    poll = await async_poller(CLIENT, response, TestArmPolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollLocationHeader is None
    response_body = {}
    response = TestArmPolling.mock_send('PUT', 201, {'location': LOCATION_URL}, response_body)
    polling_method = AsyncARMPolling(0)
    poll = await async_poller(CLIENT, response, TestArmPolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollLocationHeader is None
    response = TestArmPolling.mock_send('PUT', 201, {'azure-asyncoperation': ERROR})
    with pytest.raises(BadEndpointError):
        poll = await async_poller(CLIENT, response, TestArmPolling.mock_outputs, AsyncARMPolling(0))
    response = TestArmPolling.mock_send('PUT', 201, {'location': ERROR})
    with pytest.raises(BadEndpointError):
        poll = await async_poller(CLIENT, response, TestArmPolling.mock_outputs, AsyncARMPolling(0))

@pytest.mark.asyncio
async def test_long_running_patch():
    response = TestArmPolling.mock_send('PATCH', 202, {'location': LOCATION_URL}, body={'properties': {'provisioningState': 'Succeeded'}})
    polling_method = AsyncARMPolling(0)
    poll = await async_poller(CLIENT, response, TestArmPolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollLocationHeader is None
    response = TestArmPolling.mock_send('PATCH', 202, {'azure-asyncoperation': ASYNC_URL}, body={'properties': {'provisioningState': 'Succeeded'}})
    polling_method = AsyncARMPolling(0)
    poll = await async_poller(CLIENT, response, TestArmPolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert not hasattr(polling_method._pipeline_response, 'randomFieldFromPollAsyncOpHeader')
    response = TestArmPolling.mock_send('PATCH', 200, {'location': LOCATION_URL}, body={'properties': {'provisioningState': 'Succeeded'}})
    polling_method = AsyncARMPolling(0)
    poll = await async_poller(CLIENT, response, TestArmPolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollLocationHeader is None
    response = TestArmPolling.mock_send('PATCH', 200, {'azure-asyncoperation': ASYNC_URL}, body={'properties': {'provisioningState': 'Succeeded'}})
    polling_method = AsyncARMPolling(0)
    poll = await async_poller(CLIENT, response, TestArmPolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert not hasattr(polling_method._pipeline_response, 'randomFieldFromPollAsyncOpHeader')
    response = TestArmPolling.mock_send('PATCH', 202, {'azure-asyncoperation': ERROR})
    with pytest.raises(BadEndpointError):
        poll = await async_poller(CLIENT, response, TestArmPolling.mock_outputs, AsyncARMPolling(0))
    response = TestArmPolling.mock_send('PATCH', 202, {'location': ERROR})
    with pytest.raises(BadEndpointError):
        poll = await async_poller(CLIENT, response, TestArmPolling.mock_outputs, AsyncARMPolling(0))

@pytest.mark.asyncio
async def test_long_running_delete():
    response = TestArmPolling.mock_send('DELETE', 202, {'azure-asyncoperation': ASYNC_URL}, body='')
    polling_method = AsyncARMPolling(0)
    poll = await async_poller(CLIENT, response, TestArmPolling.mock_deserialization_no_body, polling_method)
    assert poll is None
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollAsyncOpHeader is None

@pytest.mark.asyncio
async def test_long_running_post():
    response = TestArmPolling.mock_send('POST', 201, {'azure-asyncoperation': ASYNC_URL}, body={'properties': {'provisioningState': 'Succeeded'}})
    polling_method = AsyncARMPolling(0)
    poll = await async_poller(CLIENT, response, TestArmPolling.mock_deserialization_no_body, polling_method)
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollAsyncOpHeader is None
    response = TestArmPolling.mock_send('POST', 202, {'azure-asyncoperation': ASYNC_URL}, body={'properties': {'provisioningState': 'Succeeded'}})
    polling_method = AsyncARMPolling(0)
    poll = await async_poller(CLIENT, response, TestArmPolling.mock_deserialization_no_body, polling_method)
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollAsyncOpHeader is None
    response = TestArmPolling.mock_send('POST', 202, {'location': LOCATION_URL}, body={'properties': {'provisioningState': 'Succeeded'}})
    polling_method = AsyncARMPolling(0)
    poll = await async_poller(CLIENT, response, TestArmPolling.mock_outputs, polling_method)
    assert poll.name == TEST_NAME
    assert polling_method._pipeline_response.http_response.internal_response.randomFieldFromPollLocationHeader is None
    response = TestArmPolling.mock_send('POST', 202, {'azure-asyncoperation': ERROR})
    with pytest.raises(BadEndpointError):
        await async_poller(CLIENT, response, TestArmPolling.mock_outputs, AsyncARMPolling(0))
    response = TestArmPolling.mock_send('POST', 202, {'location': ERROR})
    with pytest.raises(BadEndpointError):
        await async_poller(CLIENT, response, TestArmPolling.mock_outputs, AsyncARMPolling(0))

@pytest.mark.asyncio
async def test_long_running_negative():
    global LOCATION_BODY
    global POLLING_STATUS
    LOCATION_BODY = '{'
    response = TestArmPolling.mock_send('POST', 202, {'location': LOCATION_URL})
    poll = async_poller(CLIENT, response, TestArmPolling.mock_outputs, AsyncARMPolling(0))
    with pytest.raises(DecodeError):
        await poll
    LOCATION_BODY = '{\'"}'
    response = TestArmPolling.mock_send('POST', 202, {'location': LOCATION_URL})
    poll = async_poller(CLIENT, response, TestArmPolling.mock_outputs, AsyncARMPolling(0))
    with pytest.raises(DecodeError):
        await poll
    LOCATION_BODY = '{'
    POLLING_STATUS = 203
    response = TestArmPolling.mock_send('POST', 202, {'location': LOCATION_URL})
    poll = async_poller(CLIENT, response, TestArmPolling.mock_outputs, AsyncARMPolling(0))
    with pytest.raises(HttpResponseError) as error:
        await poll
    assert error.value.continuation_token == base64.b64encode(pickle.dumps(response)).decode('ascii')
    LOCATION_BODY = json.dumps({'name': TEST_NAME})
    POLLING_STATUS = 200

def test_polling_with_path_format_arguments():
    if False:
        print('Hello World!')
    method = AsyncARMPolling(timeout=0, path_format_arguments={'host': 'host:3000', 'accountName': 'local'})
    client = AsyncPipelineClient(base_url='http://{accountName}{host}')
    method._operation = LocationPolling()
    method._operation._location_url = '/results/1'
    method._client = client
    assert 'http://localhost:3000/results/1' == method._client.format_url(method._operation.get_polling_url(), **method._path_format_arguments)