import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
from collections.abc import Iterable
import json
import math
from google.api_core import gapic_v1, grpc_helpers, grpc_helpers_async, path_template
from google.api_core import client_options
from google.api_core import exceptions as core_exceptions
import google.auth
from google.auth import credentials as ga_credentials
from google.auth.exceptions import MutualTLSChannelError
from google.cloud.location import locations_pb2
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
from google.oauth2 import service_account
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.rpc import status_pb2
from google.type import expr_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.tasks_v2.services.cloud_tasks import CloudTasksAsyncClient, CloudTasksClient, pagers, transports
from google.cloud.tasks_v2.types import cloudtasks
from google.cloud.tasks_v2.types import queue
from google.cloud.tasks_v2.types import queue as gct_queue
from google.cloud.tasks_v2.types import target
from google.cloud.tasks_v2.types import task
from google.cloud.tasks_v2.types import task as gct_task

def client_cert_source_callback():
    if False:
        for i in range(10):
            print('nop')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        print('Hello World!')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert CloudTasksClient._get_default_mtls_endpoint(None) is None
    assert CloudTasksClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CloudTasksClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CloudTasksClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CloudTasksClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CloudTasksClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CloudTasksClient, 'grpc'), (CloudTasksAsyncClient, 'grpc_asyncio'), (CloudTasksClient, 'rest')])
def test_cloud_tasks_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('cloudtasks.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudtasks.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CloudTasksGrpcTransport, 'grpc'), (transports.CloudTasksGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.CloudTasksRestTransport, 'rest')])
def test_cloud_tasks_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(CloudTasksClient, 'grpc'), (CloudTasksAsyncClient, 'grpc_asyncio'), (CloudTasksClient, 'rest')])
def test_cloud_tasks_client_from_service_account_file(client_class, transport_name):
    if False:
        while True:
            i = 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_file') as factory:
        factory.return_value = creds
        client = client_class.from_service_account_file('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        client = client_class.from_service_account_json('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('cloudtasks.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudtasks.googleapis.com')

def test_cloud_tasks_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = CloudTasksClient.get_transport_class()
    available_transports = [transports.CloudTasksGrpcTransport, transports.CloudTasksRestTransport]
    assert transport in available_transports
    transport = CloudTasksClient.get_transport_class('grpc')
    assert transport == transports.CloudTasksGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudTasksClient, transports.CloudTasksGrpcTransport, 'grpc'), (CloudTasksAsyncClient, transports.CloudTasksGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudTasksClient, transports.CloudTasksRestTransport, 'rest')])
@mock.patch.object(CloudTasksClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudTasksClient))
@mock.patch.object(CloudTasksAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudTasksAsyncClient))
def test_cloud_tasks_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(CloudTasksClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CloudTasksClient, 'get_transport_class') as gtc:
        client = client_class(transport=transport_name)
        gtc.assert_called()
    options = client_options.ClientOptions(api_endpoint='squid.clam.whelk')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(transport=transport_name, client_options=options)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)
    with mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'never'}):
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(transport=transport_name)
            patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)
    with mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'always'}):
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(transport=transport_name)
            patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_MTLS_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)
    with mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'Unsupported'}):
        with pytest.raises(MutualTLSChannelError):
            client = client_class(transport=transport_name)
    with mock.patch.dict(os.environ, {'GOOGLE_API_USE_CLIENT_CERTIFICATE': 'Unsupported'}):
        with pytest.raises(ValueError):
            client = client_class(transport=transport_name)
    options = client_options.ClientOptions(quota_project_id='octopus')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id='octopus', client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)
    options = client_options.ClientOptions(api_audience='https://language.googleapis.com')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience='https://language.googleapis.com')

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CloudTasksClient, transports.CloudTasksGrpcTransport, 'grpc', 'true'), (CloudTasksAsyncClient, transports.CloudTasksGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CloudTasksClient, transports.CloudTasksGrpcTransport, 'grpc', 'false'), (CloudTasksAsyncClient, transports.CloudTasksGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (CloudTasksClient, transports.CloudTasksRestTransport, 'rest', 'true'), (CloudTasksClient, transports.CloudTasksRestTransport, 'rest', 'false')])
@mock.patch.object(CloudTasksClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudTasksClient))
@mock.patch.object(CloudTasksAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudTasksAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_cloud_tasks_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
    if False:
        while True:
            i = 10
    with mock.patch.dict(os.environ, {'GOOGLE_API_USE_CLIENT_CERTIFICATE': use_client_cert_env}):
        options = client_options.ClientOptions(client_cert_source=client_cert_source_callback)
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options, transport=transport_name)
            if use_client_cert_env == 'false':
                expected_client_cert_source = None
                expected_host = client.DEFAULT_ENDPOINT
            else:
                expected_client_cert_source = client_cert_source_callback
                expected_host = client.DEFAULT_MTLS_ENDPOINT
            patched.assert_called_once_with(credentials=None, credentials_file=None, host=expected_host, scopes=None, client_cert_source_for_mtls=expected_client_cert_source, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)
    with mock.patch.dict(os.environ, {'GOOGLE_API_USE_CLIENT_CERTIFICATE': use_client_cert_env}):
        with mock.patch.object(transport_class, '__init__') as patched:
            with mock.patch('google.auth.transport.mtls.has_default_client_cert_source', return_value=True):
                with mock.patch('google.auth.transport.mtls.default_client_cert_source', return_value=client_cert_source_callback):
                    if use_client_cert_env == 'false':
                        expected_host = client.DEFAULT_ENDPOINT
                        expected_client_cert_source = None
                    else:
                        expected_host = client.DEFAULT_MTLS_ENDPOINT
                        expected_client_cert_source = client_cert_source_callback
                    patched.return_value = None
                    client = client_class(transport=transport_name)
                    patched.assert_called_once_with(credentials=None, credentials_file=None, host=expected_host, scopes=None, client_cert_source_for_mtls=expected_client_cert_source, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)
    with mock.patch.dict(os.environ, {'GOOGLE_API_USE_CLIENT_CERTIFICATE': use_client_cert_env}):
        with mock.patch.object(transport_class, '__init__') as patched:
            with mock.patch('google.auth.transport.mtls.has_default_client_cert_source', return_value=False):
                patched.return_value = None
                client = client_class(transport=transport_name)
                patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class', [CloudTasksClient, CloudTasksAsyncClient])
@mock.patch.object(CloudTasksClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudTasksClient))
@mock.patch.object(CloudTasksAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudTasksAsyncClient))
def test_cloud_tasks_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        while True:
            i = 10
    mock_client_cert_source = mock.Mock()
    with mock.patch.dict(os.environ, {'GOOGLE_API_USE_CLIENT_CERTIFICATE': 'true'}):
        mock_api_endpoint = 'foo'
        options = client_options.ClientOptions(client_cert_source=mock_client_cert_source, api_endpoint=mock_api_endpoint)
        (api_endpoint, cert_source) = client_class.get_mtls_endpoint_and_cert_source(options)
        assert api_endpoint == mock_api_endpoint
        assert cert_source == mock_client_cert_source
    with mock.patch.dict(os.environ, {'GOOGLE_API_USE_CLIENT_CERTIFICATE': 'false'}):
        mock_client_cert_source = mock.Mock()
        mock_api_endpoint = 'foo'
        options = client_options.ClientOptions(client_cert_source=mock_client_cert_source, api_endpoint=mock_api_endpoint)
        (api_endpoint, cert_source) = client_class.get_mtls_endpoint_and_cert_source(options)
        assert api_endpoint == mock_api_endpoint
        assert cert_source is None
    with mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'never'}):
        (api_endpoint, cert_source) = client_class.get_mtls_endpoint_and_cert_source()
        assert api_endpoint == client_class.DEFAULT_ENDPOINT
        assert cert_source is None
    with mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'always'}):
        (api_endpoint, cert_source) = client_class.get_mtls_endpoint_and_cert_source()
        assert api_endpoint == client_class.DEFAULT_MTLS_ENDPOINT
        assert cert_source is None
    with mock.patch.dict(os.environ, {'GOOGLE_API_USE_CLIENT_CERTIFICATE': 'true'}):
        with mock.patch('google.auth.transport.mtls.has_default_client_cert_source', return_value=False):
            (api_endpoint, cert_source) = client_class.get_mtls_endpoint_and_cert_source()
            assert api_endpoint == client_class.DEFAULT_ENDPOINT
            assert cert_source is None
    with mock.patch.dict(os.environ, {'GOOGLE_API_USE_CLIENT_CERTIFICATE': 'true'}):
        with mock.patch('google.auth.transport.mtls.has_default_client_cert_source', return_value=True):
            with mock.patch('google.auth.transport.mtls.default_client_cert_source', return_value=mock_client_cert_source):
                (api_endpoint, cert_source) = client_class.get_mtls_endpoint_and_cert_source()
                assert api_endpoint == client_class.DEFAULT_MTLS_ENDPOINT
                assert cert_source == mock_client_cert_source

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudTasksClient, transports.CloudTasksGrpcTransport, 'grpc'), (CloudTasksAsyncClient, transports.CloudTasksGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudTasksClient, transports.CloudTasksRestTransport, 'rest')])
def test_cloud_tasks_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudTasksClient, transports.CloudTasksGrpcTransport, 'grpc', grpc_helpers), (CloudTasksAsyncClient, transports.CloudTasksGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (CloudTasksClient, transports.CloudTasksRestTransport, 'rest', None)])
def test_cloud_tasks_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_cloud_tasks_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.tasks_v2.services.cloud_tasks.transports.CloudTasksGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CloudTasksClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudTasksClient, transports.CloudTasksGrpcTransport, 'grpc', grpc_helpers), (CloudTasksAsyncClient, transports.CloudTasksGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_cloud_tasks_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel') as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        file_creds = ga_credentials.AnonymousCredentials()
        load_creds.return_value = (file_creds, None)
        adc.return_value = (creds, None)
        client = client_class(client_options=options, transport=transport_name)
        create_channel.assert_called_with('cloudtasks.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='cloudtasks.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [cloudtasks.ListQueuesRequest, dict])
def test_list_queues(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_queues), '__call__') as call:
        call.return_value = cloudtasks.ListQueuesResponse(next_page_token='next_page_token_value')
        response = client.list_queues(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.ListQueuesRequest()
    assert isinstance(response, pagers.ListQueuesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_queues_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_queues), '__call__') as call:
        client.list_queues()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.ListQueuesRequest()

@pytest.mark.asyncio
async def test_list_queues_async(transport: str='grpc_asyncio', request_type=cloudtasks.ListQueuesRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_queues), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloudtasks.ListQueuesResponse(next_page_token='next_page_token_value'))
        response = await client.list_queues(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.ListQueuesRequest()
    assert isinstance(response, pagers.ListQueuesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_queues_async_from_dict():
    await test_list_queues_async(request_type=dict)

def test_list_queues_field_headers():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.ListQueuesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_queues), '__call__') as call:
        call.return_value = cloudtasks.ListQueuesResponse()
        client.list_queues(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_queues_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.ListQueuesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_queues), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloudtasks.ListQueuesResponse())
        await client.list_queues(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_queues_flattened():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_queues), '__call__') as call:
        call.return_value = cloudtasks.ListQueuesResponse()
        client.list_queues(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_queues_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_queues(cloudtasks.ListQueuesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_queues_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_queues), '__call__') as call:
        call.return_value = cloudtasks.ListQueuesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloudtasks.ListQueuesResponse())
        response = await client.list_queues(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_queues_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_queues(cloudtasks.ListQueuesRequest(), parent='parent_value')

def test_list_queues_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_queues), '__call__') as call:
        call.side_effect = (cloudtasks.ListQueuesResponse(queues=[queue.Queue(), queue.Queue(), queue.Queue()], next_page_token='abc'), cloudtasks.ListQueuesResponse(queues=[], next_page_token='def'), cloudtasks.ListQueuesResponse(queues=[queue.Queue()], next_page_token='ghi'), cloudtasks.ListQueuesResponse(queues=[queue.Queue(), queue.Queue()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_queues(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, queue.Queue) for i in results))

def test_list_queues_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_queues), '__call__') as call:
        call.side_effect = (cloudtasks.ListQueuesResponse(queues=[queue.Queue(), queue.Queue(), queue.Queue()], next_page_token='abc'), cloudtasks.ListQueuesResponse(queues=[], next_page_token='def'), cloudtasks.ListQueuesResponse(queues=[queue.Queue()], next_page_token='ghi'), cloudtasks.ListQueuesResponse(queues=[queue.Queue(), queue.Queue()]), RuntimeError)
        pages = list(client.list_queues(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_queues_async_pager():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_queues), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloudtasks.ListQueuesResponse(queues=[queue.Queue(), queue.Queue(), queue.Queue()], next_page_token='abc'), cloudtasks.ListQueuesResponse(queues=[], next_page_token='def'), cloudtasks.ListQueuesResponse(queues=[queue.Queue()], next_page_token='ghi'), cloudtasks.ListQueuesResponse(queues=[queue.Queue(), queue.Queue()]), RuntimeError)
        async_pager = await client.list_queues(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, queue.Queue) for i in responses))

@pytest.mark.asyncio
async def test_list_queues_async_pages():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_queues), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloudtasks.ListQueuesResponse(queues=[queue.Queue(), queue.Queue(), queue.Queue()], next_page_token='abc'), cloudtasks.ListQueuesResponse(queues=[], next_page_token='def'), cloudtasks.ListQueuesResponse(queues=[queue.Queue()], next_page_token='ghi'), cloudtasks.ListQueuesResponse(queues=[queue.Queue(), queue.Queue()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_queues(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloudtasks.GetQueueRequest, dict])
def test_get_queue(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_queue), '__call__') as call:
        call.return_value = queue.Queue(name='name_value', state=queue.Queue.State.RUNNING)
        response = client.get_queue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.GetQueueRequest()
    assert isinstance(response, queue.Queue)
    assert response.name == 'name_value'
    assert response.state == queue.Queue.State.RUNNING

def test_get_queue_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_queue), '__call__') as call:
        client.get_queue()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.GetQueueRequest()

@pytest.mark.asyncio
async def test_get_queue_async(transport: str='grpc_asyncio', request_type=cloudtasks.GetQueueRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_queue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(queue.Queue(name='name_value', state=queue.Queue.State.RUNNING))
        response = await client.get_queue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.GetQueueRequest()
    assert isinstance(response, queue.Queue)
    assert response.name == 'name_value'
    assert response.state == queue.Queue.State.RUNNING

@pytest.mark.asyncio
async def test_get_queue_async_from_dict():
    await test_get_queue_async(request_type=dict)

def test_get_queue_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.GetQueueRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_queue), '__call__') as call:
        call.return_value = queue.Queue()
        client.get_queue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_queue_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.GetQueueRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_queue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(queue.Queue())
        await client.get_queue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_queue_flattened():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_queue), '__call__') as call:
        call.return_value = queue.Queue()
        client.get_queue(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_queue_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_queue(cloudtasks.GetQueueRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_queue_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_queue), '__call__') as call:
        call.return_value = queue.Queue()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(queue.Queue())
        response = await client.get_queue(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_queue_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_queue(cloudtasks.GetQueueRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloudtasks.CreateQueueRequest, dict])
def test_create_queue(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_queue), '__call__') as call:
        call.return_value = gct_queue.Queue(name='name_value', state=gct_queue.Queue.State.RUNNING)
        response = client.create_queue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.CreateQueueRequest()
    assert isinstance(response, gct_queue.Queue)
    assert response.name == 'name_value'
    assert response.state == gct_queue.Queue.State.RUNNING

def test_create_queue_empty_call():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_queue), '__call__') as call:
        client.create_queue()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.CreateQueueRequest()

@pytest.mark.asyncio
async def test_create_queue_async(transport: str='grpc_asyncio', request_type=cloudtasks.CreateQueueRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_queue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_queue.Queue(name='name_value', state=gct_queue.Queue.State.RUNNING))
        response = await client.create_queue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.CreateQueueRequest()
    assert isinstance(response, gct_queue.Queue)
    assert response.name == 'name_value'
    assert response.state == gct_queue.Queue.State.RUNNING

@pytest.mark.asyncio
async def test_create_queue_async_from_dict():
    await test_create_queue_async(request_type=dict)

def test_create_queue_field_headers():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.CreateQueueRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_queue), '__call__') as call:
        call.return_value = gct_queue.Queue()
        client.create_queue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_queue_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.CreateQueueRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_queue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_queue.Queue())
        await client.create_queue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_queue_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_queue), '__call__') as call:
        call.return_value = gct_queue.Queue()
        client.create_queue(parent='parent_value', queue=gct_queue.Queue(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].queue
        mock_val = gct_queue.Queue(name='name_value')
        assert arg == mock_val

def test_create_queue_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_queue(cloudtasks.CreateQueueRequest(), parent='parent_value', queue=gct_queue.Queue(name='name_value'))

@pytest.mark.asyncio
async def test_create_queue_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_queue), '__call__') as call:
        call.return_value = gct_queue.Queue()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_queue.Queue())
        response = await client.create_queue(parent='parent_value', queue=gct_queue.Queue(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].queue
        mock_val = gct_queue.Queue(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_queue_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_queue(cloudtasks.CreateQueueRequest(), parent='parent_value', queue=gct_queue.Queue(name='name_value'))

@pytest.mark.parametrize('request_type', [cloudtasks.UpdateQueueRequest, dict])
def test_update_queue(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_queue), '__call__') as call:
        call.return_value = gct_queue.Queue(name='name_value', state=gct_queue.Queue.State.RUNNING)
        response = client.update_queue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.UpdateQueueRequest()
    assert isinstance(response, gct_queue.Queue)
    assert response.name == 'name_value'
    assert response.state == gct_queue.Queue.State.RUNNING

def test_update_queue_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_queue), '__call__') as call:
        client.update_queue()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.UpdateQueueRequest()

@pytest.mark.asyncio
async def test_update_queue_async(transport: str='grpc_asyncio', request_type=cloudtasks.UpdateQueueRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_queue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_queue.Queue(name='name_value', state=gct_queue.Queue.State.RUNNING))
        response = await client.update_queue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.UpdateQueueRequest()
    assert isinstance(response, gct_queue.Queue)
    assert response.name == 'name_value'
    assert response.state == gct_queue.Queue.State.RUNNING

@pytest.mark.asyncio
async def test_update_queue_async_from_dict():
    await test_update_queue_async(request_type=dict)

def test_update_queue_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.UpdateQueueRequest()
    request.queue.name = 'name_value'
    with mock.patch.object(type(client.transport.update_queue), '__call__') as call:
        call.return_value = gct_queue.Queue()
        client.update_queue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'queue.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_queue_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.UpdateQueueRequest()
    request.queue.name = 'name_value'
    with mock.patch.object(type(client.transport.update_queue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_queue.Queue())
        await client.update_queue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'queue.name=name_value') in kw['metadata']

def test_update_queue_flattened():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_queue), '__call__') as call:
        call.return_value = gct_queue.Queue()
        client.update_queue(queue=gct_queue.Queue(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].queue
        mock_val = gct_queue.Queue(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_queue_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_queue(cloudtasks.UpdateQueueRequest(), queue=gct_queue.Queue(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_queue_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_queue), '__call__') as call:
        call.return_value = gct_queue.Queue()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_queue.Queue())
        response = await client.update_queue(queue=gct_queue.Queue(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].queue
        mock_val = gct_queue.Queue(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_queue_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_queue(cloudtasks.UpdateQueueRequest(), queue=gct_queue.Queue(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [cloudtasks.DeleteQueueRequest, dict])
def test_delete_queue(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_queue), '__call__') as call:
        call.return_value = None
        response = client.delete_queue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.DeleteQueueRequest()
    assert response is None

def test_delete_queue_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_queue), '__call__') as call:
        client.delete_queue()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.DeleteQueueRequest()

@pytest.mark.asyncio
async def test_delete_queue_async(transport: str='grpc_asyncio', request_type=cloudtasks.DeleteQueueRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_queue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_queue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.DeleteQueueRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_queue_async_from_dict():
    await test_delete_queue_async(request_type=dict)

def test_delete_queue_field_headers():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.DeleteQueueRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_queue), '__call__') as call:
        call.return_value = None
        client.delete_queue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_queue_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.DeleteQueueRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_queue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_queue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_queue_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_queue), '__call__') as call:
        call.return_value = None
        client.delete_queue(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_queue_flattened_error():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_queue(cloudtasks.DeleteQueueRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_queue_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_queue), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_queue(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_queue_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_queue(cloudtasks.DeleteQueueRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloudtasks.PurgeQueueRequest, dict])
def test_purge_queue(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.purge_queue), '__call__') as call:
        call.return_value = queue.Queue(name='name_value', state=queue.Queue.State.RUNNING)
        response = client.purge_queue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.PurgeQueueRequest()
    assert isinstance(response, queue.Queue)
    assert response.name == 'name_value'
    assert response.state == queue.Queue.State.RUNNING

def test_purge_queue_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.purge_queue), '__call__') as call:
        client.purge_queue()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.PurgeQueueRequest()

@pytest.mark.asyncio
async def test_purge_queue_async(transport: str='grpc_asyncio', request_type=cloudtasks.PurgeQueueRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.purge_queue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(queue.Queue(name='name_value', state=queue.Queue.State.RUNNING))
        response = await client.purge_queue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.PurgeQueueRequest()
    assert isinstance(response, queue.Queue)
    assert response.name == 'name_value'
    assert response.state == queue.Queue.State.RUNNING

@pytest.mark.asyncio
async def test_purge_queue_async_from_dict():
    await test_purge_queue_async(request_type=dict)

def test_purge_queue_field_headers():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.PurgeQueueRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.purge_queue), '__call__') as call:
        call.return_value = queue.Queue()
        client.purge_queue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_purge_queue_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.PurgeQueueRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.purge_queue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(queue.Queue())
        await client.purge_queue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_purge_queue_flattened():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.purge_queue), '__call__') as call:
        call.return_value = queue.Queue()
        client.purge_queue(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_purge_queue_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.purge_queue(cloudtasks.PurgeQueueRequest(), name='name_value')

@pytest.mark.asyncio
async def test_purge_queue_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.purge_queue), '__call__') as call:
        call.return_value = queue.Queue()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(queue.Queue())
        response = await client.purge_queue(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_purge_queue_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.purge_queue(cloudtasks.PurgeQueueRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloudtasks.PauseQueueRequest, dict])
def test_pause_queue(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.pause_queue), '__call__') as call:
        call.return_value = queue.Queue(name='name_value', state=queue.Queue.State.RUNNING)
        response = client.pause_queue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.PauseQueueRequest()
    assert isinstance(response, queue.Queue)
    assert response.name == 'name_value'
    assert response.state == queue.Queue.State.RUNNING

def test_pause_queue_empty_call():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.pause_queue), '__call__') as call:
        client.pause_queue()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.PauseQueueRequest()

@pytest.mark.asyncio
async def test_pause_queue_async(transport: str='grpc_asyncio', request_type=cloudtasks.PauseQueueRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.pause_queue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(queue.Queue(name='name_value', state=queue.Queue.State.RUNNING))
        response = await client.pause_queue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.PauseQueueRequest()
    assert isinstance(response, queue.Queue)
    assert response.name == 'name_value'
    assert response.state == queue.Queue.State.RUNNING

@pytest.mark.asyncio
async def test_pause_queue_async_from_dict():
    await test_pause_queue_async(request_type=dict)

def test_pause_queue_field_headers():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.PauseQueueRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.pause_queue), '__call__') as call:
        call.return_value = queue.Queue()
        client.pause_queue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_pause_queue_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.PauseQueueRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.pause_queue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(queue.Queue())
        await client.pause_queue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_pause_queue_flattened():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.pause_queue), '__call__') as call:
        call.return_value = queue.Queue()
        client.pause_queue(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_pause_queue_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.pause_queue(cloudtasks.PauseQueueRequest(), name='name_value')

@pytest.mark.asyncio
async def test_pause_queue_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.pause_queue), '__call__') as call:
        call.return_value = queue.Queue()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(queue.Queue())
        response = await client.pause_queue(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_pause_queue_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.pause_queue(cloudtasks.PauseQueueRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloudtasks.ResumeQueueRequest, dict])
def test_resume_queue(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resume_queue), '__call__') as call:
        call.return_value = queue.Queue(name='name_value', state=queue.Queue.State.RUNNING)
        response = client.resume_queue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.ResumeQueueRequest()
    assert isinstance(response, queue.Queue)
    assert response.name == 'name_value'
    assert response.state == queue.Queue.State.RUNNING

def test_resume_queue_empty_call():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.resume_queue), '__call__') as call:
        client.resume_queue()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.ResumeQueueRequest()

@pytest.mark.asyncio
async def test_resume_queue_async(transport: str='grpc_asyncio', request_type=cloudtasks.ResumeQueueRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resume_queue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(queue.Queue(name='name_value', state=queue.Queue.State.RUNNING))
        response = await client.resume_queue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.ResumeQueueRequest()
    assert isinstance(response, queue.Queue)
    assert response.name == 'name_value'
    assert response.state == queue.Queue.State.RUNNING

@pytest.mark.asyncio
async def test_resume_queue_async_from_dict():
    await test_resume_queue_async(request_type=dict)

def test_resume_queue_field_headers():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.ResumeQueueRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.resume_queue), '__call__') as call:
        call.return_value = queue.Queue()
        client.resume_queue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_resume_queue_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.ResumeQueueRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.resume_queue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(queue.Queue())
        await client.resume_queue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_resume_queue_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.resume_queue), '__call__') as call:
        call.return_value = queue.Queue()
        client.resume_queue(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_resume_queue_flattened_error():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.resume_queue(cloudtasks.ResumeQueueRequest(), name='name_value')

@pytest.mark.asyncio
async def test_resume_queue_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.resume_queue), '__call__') as call:
        call.return_value = queue.Queue()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(queue.Queue())
        response = await client.resume_queue(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_resume_queue_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.resume_queue(cloudtasks.ResumeQueueRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_empty_call():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        client.get_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()

@pytest.mark.asyncio
async def test_get_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.GetIamPolicyRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_get_iam_policy_async_from_dict():
    await test_get_iam_policy_async(request_type=dict)

def test_get_iam_policy_field_headers():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_iam_policy_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_get_iam_policy_from_dict_foreign():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_get_iam_policy_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(resource='resource_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

def test_get_iam_policy_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

@pytest.mark.asyncio
async def test_get_iam_policy_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(resource='resource_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_iam_policy_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        client.set_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()

@pytest.mark.asyncio
async def test_set_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.SetIamPolicyRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.set_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_set_iam_policy_async_from_dict():
    await test_set_iam_policy_async(request_type=dict)

def test_set_iam_policy_field_headers():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_iam_policy_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.set_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_set_iam_policy_from_dict_foreign():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774), 'update_mask': field_mask_pb2.FieldMask(paths=['paths_value'])})
        call.assert_called()

def test_set_iam_policy_flattened():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(resource='resource_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

def test_set_iam_policy_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

@pytest.mark.asyncio
async def test_set_iam_policy_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(resource='resource_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_set_iam_policy_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response = client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        client.test_iam_permissions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()

@pytest.mark.asyncio
async def test_test_iam_permissions_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value']))
        response = await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

@pytest.mark.asyncio
async def test_test_iam_permissions_async_from_dict():
    await test_test_iam_permissions_async(request_type=dict)

def test_test_iam_permissions_field_headers():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_test_iam_permissions_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_test_iam_permissions_from_dict_foreign():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_test_iam_permissions_flattened():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(resource='resource_value', permissions=['permissions_value'])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val
        arg = args[0].permissions
        mock_val = ['permissions_value']
        assert arg == mock_val

def test_test_iam_permissions_flattened_error():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

@pytest.mark.asyncio
async def test_test_iam_permissions_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(resource='resource_value', permissions=['permissions_value'])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val
        arg = args[0].permissions
        mock_val = ['permissions_value']
        assert arg == mock_val

@pytest.mark.asyncio
async def test_test_iam_permissions_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

@pytest.mark.parametrize('request_type', [cloudtasks.ListTasksRequest, dict])
def test_list_tasks(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.return_value = cloudtasks.ListTasksResponse(next_page_token='next_page_token_value')
        response = client.list_tasks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.ListTasksRequest()
    assert isinstance(response, pagers.ListTasksPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tasks_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        client.list_tasks()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.ListTasksRequest()

@pytest.mark.asyncio
async def test_list_tasks_async(transport: str='grpc_asyncio', request_type=cloudtasks.ListTasksRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloudtasks.ListTasksResponse(next_page_token='next_page_token_value'))
        response = await client.list_tasks(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.ListTasksRequest()
    assert isinstance(response, pagers.ListTasksAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_tasks_async_from_dict():
    await test_list_tasks_async(request_type=dict)

def test_list_tasks_field_headers():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.ListTasksRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.return_value = cloudtasks.ListTasksResponse()
        client.list_tasks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_tasks_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.ListTasksRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloudtasks.ListTasksResponse())
        await client.list_tasks(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_tasks_flattened():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.return_value = cloudtasks.ListTasksResponse()
        client.list_tasks(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_tasks_flattened_error():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_tasks(cloudtasks.ListTasksRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_tasks_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.return_value = cloudtasks.ListTasksResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloudtasks.ListTasksResponse())
        response = await client.list_tasks(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_tasks_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_tasks(cloudtasks.ListTasksRequest(), parent='parent_value')

def test_list_tasks_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.side_effect = (cloudtasks.ListTasksResponse(tasks=[task.Task(), task.Task(), task.Task()], next_page_token='abc'), cloudtasks.ListTasksResponse(tasks=[], next_page_token='def'), cloudtasks.ListTasksResponse(tasks=[task.Task()], next_page_token='ghi'), cloudtasks.ListTasksResponse(tasks=[task.Task(), task.Task()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_tasks(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, task.Task) for i in results))

def test_list_tasks_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tasks), '__call__') as call:
        call.side_effect = (cloudtasks.ListTasksResponse(tasks=[task.Task(), task.Task(), task.Task()], next_page_token='abc'), cloudtasks.ListTasksResponse(tasks=[], next_page_token='def'), cloudtasks.ListTasksResponse(tasks=[task.Task()], next_page_token='ghi'), cloudtasks.ListTasksResponse(tasks=[task.Task(), task.Task()]), RuntimeError)
        pages = list(client.list_tasks(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_tasks_async_pager():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tasks), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloudtasks.ListTasksResponse(tasks=[task.Task(), task.Task(), task.Task()], next_page_token='abc'), cloudtasks.ListTasksResponse(tasks=[], next_page_token='def'), cloudtasks.ListTasksResponse(tasks=[task.Task()], next_page_token='ghi'), cloudtasks.ListTasksResponse(tasks=[task.Task(), task.Task()]), RuntimeError)
        async_pager = await client.list_tasks(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, task.Task) for i in responses))

@pytest.mark.asyncio
async def test_list_tasks_async_pages():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tasks), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloudtasks.ListTasksResponse(tasks=[task.Task(), task.Task(), task.Task()], next_page_token='abc'), cloudtasks.ListTasksResponse(tasks=[], next_page_token='def'), cloudtasks.ListTasksResponse(tasks=[task.Task()], next_page_token='ghi'), cloudtasks.ListTasksResponse(tasks=[task.Task(), task.Task()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_tasks(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloudtasks.GetTaskRequest, dict])
def test_get_task(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_task), '__call__') as call:
        call.return_value = task.Task(name='name_value', dispatch_count=1496, response_count=1527, view=task.Task.View.BASIC)
        response = client.get_task(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.GetTaskRequest()
    assert isinstance(response, task.Task)
    assert response.name == 'name_value'
    assert response.dispatch_count == 1496
    assert response.response_count == 1527
    assert response.view == task.Task.View.BASIC

def test_get_task_empty_call():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_task), '__call__') as call:
        client.get_task()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.GetTaskRequest()

@pytest.mark.asyncio
async def test_get_task_async(transport: str='grpc_asyncio', request_type=cloudtasks.GetTaskRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_task), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(task.Task(name='name_value', dispatch_count=1496, response_count=1527, view=task.Task.View.BASIC))
        response = await client.get_task(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.GetTaskRequest()
    assert isinstance(response, task.Task)
    assert response.name == 'name_value'
    assert response.dispatch_count == 1496
    assert response.response_count == 1527
    assert response.view == task.Task.View.BASIC

@pytest.mark.asyncio
async def test_get_task_async_from_dict():
    await test_get_task_async(request_type=dict)

def test_get_task_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.GetTaskRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_task), '__call__') as call:
        call.return_value = task.Task()
        client.get_task(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_task_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.GetTaskRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_task), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(task.Task())
        await client.get_task(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_task_flattened():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_task), '__call__') as call:
        call.return_value = task.Task()
        client.get_task(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_task_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_task(cloudtasks.GetTaskRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_task_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_task), '__call__') as call:
        call.return_value = task.Task()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(task.Task())
        response = await client.get_task(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_task_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_task(cloudtasks.GetTaskRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloudtasks.CreateTaskRequest, dict])
def test_create_task(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_task), '__call__') as call:
        call.return_value = gct_task.Task(name='name_value', dispatch_count=1496, response_count=1527, view=gct_task.Task.View.BASIC)
        response = client.create_task(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.CreateTaskRequest()
    assert isinstance(response, gct_task.Task)
    assert response.name == 'name_value'
    assert response.dispatch_count == 1496
    assert response.response_count == 1527
    assert response.view == gct_task.Task.View.BASIC

def test_create_task_empty_call():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_task), '__call__') as call:
        client.create_task()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.CreateTaskRequest()

@pytest.mark.asyncio
async def test_create_task_async(transport: str='grpc_asyncio', request_type=cloudtasks.CreateTaskRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_task), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_task.Task(name='name_value', dispatch_count=1496, response_count=1527, view=gct_task.Task.View.BASIC))
        response = await client.create_task(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.CreateTaskRequest()
    assert isinstance(response, gct_task.Task)
    assert response.name == 'name_value'
    assert response.dispatch_count == 1496
    assert response.response_count == 1527
    assert response.view == gct_task.Task.View.BASIC

@pytest.mark.asyncio
async def test_create_task_async_from_dict():
    await test_create_task_async(request_type=dict)

def test_create_task_field_headers():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.CreateTaskRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_task), '__call__') as call:
        call.return_value = gct_task.Task()
        client.create_task(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_task_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.CreateTaskRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_task), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_task.Task())
        await client.create_task(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_task_flattened():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_task), '__call__') as call:
        call.return_value = gct_task.Task()
        client.create_task(parent='parent_value', task=gct_task.Task(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].task
        mock_val = gct_task.Task(name='name_value')
        assert arg == mock_val

def test_create_task_flattened_error():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_task(cloudtasks.CreateTaskRequest(), parent='parent_value', task=gct_task.Task(name='name_value'))

@pytest.mark.asyncio
async def test_create_task_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_task), '__call__') as call:
        call.return_value = gct_task.Task()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_task.Task())
        response = await client.create_task(parent='parent_value', task=gct_task.Task(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].task
        mock_val = gct_task.Task(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_task_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_task(cloudtasks.CreateTaskRequest(), parent='parent_value', task=gct_task.Task(name='name_value'))

@pytest.mark.parametrize('request_type', [cloudtasks.DeleteTaskRequest, dict])
def test_delete_task(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_task), '__call__') as call:
        call.return_value = None
        response = client.delete_task(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.DeleteTaskRequest()
    assert response is None

def test_delete_task_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_task), '__call__') as call:
        client.delete_task()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.DeleteTaskRequest()

@pytest.mark.asyncio
async def test_delete_task_async(transport: str='grpc_asyncio', request_type=cloudtasks.DeleteTaskRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_task), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_task(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.DeleteTaskRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_task_async_from_dict():
    await test_delete_task_async(request_type=dict)

def test_delete_task_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.DeleteTaskRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_task), '__call__') as call:
        call.return_value = None
        client.delete_task(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_task_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.DeleteTaskRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_task), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_task(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_task_flattened():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_task), '__call__') as call:
        call.return_value = None
        client.delete_task(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_task_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_task(cloudtasks.DeleteTaskRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_task_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_task), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_task(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_task_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_task(cloudtasks.DeleteTaskRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloudtasks.RunTaskRequest, dict])
def test_run_task(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.run_task), '__call__') as call:
        call.return_value = task.Task(name='name_value', dispatch_count=1496, response_count=1527, view=task.Task.View.BASIC)
        response = client.run_task(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.RunTaskRequest()
    assert isinstance(response, task.Task)
    assert response.name == 'name_value'
    assert response.dispatch_count == 1496
    assert response.response_count == 1527
    assert response.view == task.Task.View.BASIC

def test_run_task_empty_call():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.run_task), '__call__') as call:
        client.run_task()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.RunTaskRequest()

@pytest.mark.asyncio
async def test_run_task_async(transport: str='grpc_asyncio', request_type=cloudtasks.RunTaskRequest):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.run_task), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(task.Task(name='name_value', dispatch_count=1496, response_count=1527, view=task.Task.View.BASIC))
        response = await client.run_task(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudtasks.RunTaskRequest()
    assert isinstance(response, task.Task)
    assert response.name == 'name_value'
    assert response.dispatch_count == 1496
    assert response.response_count == 1527
    assert response.view == task.Task.View.BASIC

@pytest.mark.asyncio
async def test_run_task_async_from_dict():
    await test_run_task_async(request_type=dict)

def test_run_task_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.RunTaskRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.run_task), '__call__') as call:
        call.return_value = task.Task()
        client.run_task(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_run_task_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudtasks.RunTaskRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.run_task), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(task.Task())
        await client.run_task(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_run_task_flattened():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.run_task), '__call__') as call:
        call.return_value = task.Task()
        client.run_task(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_run_task_flattened_error():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.run_task(cloudtasks.RunTaskRequest(), name='name_value')

@pytest.mark.asyncio
async def test_run_task_flattened_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.run_task), '__call__') as call:
        call.return_value = task.Task()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(task.Task())
        response = await client.run_task(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_run_task_flattened_error_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.run_task(cloudtasks.RunTaskRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloudtasks.ListQueuesRequest, dict])
def test_list_queues_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloudtasks.ListQueuesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloudtasks.ListQueuesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_queues(request)
    assert isinstance(response, pagers.ListQueuesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_queues_rest_required_fields(request_type=cloudtasks.ListQueuesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_queues._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_queues._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloudtasks.ListQueuesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloudtasks.ListQueuesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_queues(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_queues_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_queues._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_queues_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'post_list_queues') as post, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_list_queues') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudtasks.ListQueuesRequest.pb(cloudtasks.ListQueuesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloudtasks.ListQueuesResponse.to_json(cloudtasks.ListQueuesResponse())
        request = cloudtasks.ListQueuesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloudtasks.ListQueuesResponse()
        client.list_queues(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_queues_rest_bad_request(transport: str='rest', request_type=cloudtasks.ListQueuesRequest):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_queues(request)

def test_list_queues_rest_flattened():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloudtasks.ListQueuesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloudtasks.ListQueuesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_queues(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/queues' % client.transport._host, args[1])

def test_list_queues_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_queues(cloudtasks.ListQueuesRequest(), parent='parent_value')

def test_list_queues_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (cloudtasks.ListQueuesResponse(queues=[queue.Queue(), queue.Queue(), queue.Queue()], next_page_token='abc'), cloudtasks.ListQueuesResponse(queues=[], next_page_token='def'), cloudtasks.ListQueuesResponse(queues=[queue.Queue()], next_page_token='ghi'), cloudtasks.ListQueuesResponse(queues=[queue.Queue(), queue.Queue()]))
        response = response + response
        response = tuple((cloudtasks.ListQueuesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_queues(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, queue.Queue) for i in results))
        pages = list(client.list_queues(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloudtasks.GetQueueRequest, dict])
def test_get_queue_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = queue.Queue(name='name_value', state=queue.Queue.State.RUNNING)
        response_value = Response()
        response_value.status_code = 200
        return_value = queue.Queue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_queue(request)
    assert isinstance(response, queue.Queue)
    assert response.name == 'name_value'
    assert response.state == queue.Queue.State.RUNNING

def test_get_queue_rest_required_fields(request_type=cloudtasks.GetQueueRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_queue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_queue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = queue.Queue()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = queue.Queue.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_queue(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_queue_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_queue._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_queue_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'post_get_queue') as post, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_get_queue') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudtasks.GetQueueRequest.pb(cloudtasks.GetQueueRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = queue.Queue.to_json(queue.Queue())
        request = cloudtasks.GetQueueRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = queue.Queue()
        client.get_queue(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_queue_rest_bad_request(transport: str='rest', request_type=cloudtasks.GetQueueRequest):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_queue(request)

def test_get_queue_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = queue.Queue()
        sample_request = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = queue.Queue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_queue(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/queues/*}' % client.transport._host, args[1])

def test_get_queue_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_queue(cloudtasks.GetQueueRequest(), name='name_value')

def test_get_queue_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudtasks.CreateQueueRequest, dict])
def test_create_queue_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['queue'] = {'name': 'name_value', 'app_engine_routing_override': {'service': 'service_value', 'version': 'version_value', 'instance': 'instance_value', 'host': 'host_value'}, 'rate_limits': {'max_dispatches_per_second': 0.26380000000000003, 'max_burst_size': 1519, 'max_concurrent_dispatches': 2671}, 'retry_config': {'max_attempts': 1303, 'max_retry_duration': {'seconds': 751, 'nanos': 543}, 'min_backoff': {}, 'max_backoff': {}, 'max_doublings': 1388}, 'state': 1, 'purge_time': {'seconds': 751, 'nanos': 543}, 'stackdriver_logging_config': {'sampling_ratio': 0.1497}}
    test_field = cloudtasks.CreateQueueRequest.meta.fields['queue']

    def get_message_fields(field):
        if False:
            i = 10
            return i + 15
        message_fields = []
        if hasattr(field, 'message') and field.message:
            is_field_type_proto_plus_type = not hasattr(field.message, 'DESCRIPTOR')
            if is_field_type_proto_plus_type:
                message_fields = field.message.meta.fields.values()
            else:
                message_fields = field.message.DESCRIPTOR.fields
        return message_fields
    runtime_nested_fields = [(field.name, nested_field.name) for field in get_message_fields(test_field) for nested_field in get_message_fields(field)]
    subfields_not_in_runtime = []
    for (field, value) in request_init['queue'].items():
        result = None
        is_repeated = False
        if isinstance(value, list) and len(value):
            is_repeated = True
            result = value[0]
        if isinstance(value, dict):
            result = value
        if result and hasattr(result, 'keys'):
            for subfield in result.keys():
                if (field, subfield) not in runtime_nested_fields:
                    subfields_not_in_runtime.append({'field': field, 'subfield': subfield, 'is_repeated': is_repeated})
    for subfield_to_delete in subfields_not_in_runtime:
        field = subfield_to_delete.get('field')
        field_repeated = subfield_to_delete.get('is_repeated')
        subfield = subfield_to_delete.get('subfield')
        if subfield:
            if field_repeated:
                for i in range(0, len(request_init['queue'][field])):
                    del request_init['queue'][field][i][subfield]
            else:
                del request_init['queue'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gct_queue.Queue(name='name_value', state=gct_queue.Queue.State.RUNNING)
        response_value = Response()
        response_value.status_code = 200
        return_value = gct_queue.Queue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_queue(request)
    assert isinstance(response, gct_queue.Queue)
    assert response.name == 'name_value'
    assert response.state == gct_queue.Queue.State.RUNNING

def test_create_queue_rest_required_fields(request_type=cloudtasks.CreateQueueRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_queue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_queue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gct_queue.Queue()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gct_queue.Queue.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_queue(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_queue_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_queue._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'queue'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_queue_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'post_create_queue') as post, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_create_queue') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudtasks.CreateQueueRequest.pb(cloudtasks.CreateQueueRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gct_queue.Queue.to_json(gct_queue.Queue())
        request = cloudtasks.CreateQueueRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gct_queue.Queue()
        client.create_queue(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_queue_rest_bad_request(transport: str='rest', request_type=cloudtasks.CreateQueueRequest):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_queue(request)

def test_create_queue_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gct_queue.Queue()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', queue=gct_queue.Queue(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gct_queue.Queue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_queue(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/queues' % client.transport._host, args[1])

def test_create_queue_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_queue(cloudtasks.CreateQueueRequest(), parent='parent_value', queue=gct_queue.Queue(name='name_value'))

def test_create_queue_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudtasks.UpdateQueueRequest, dict])
def test_update_queue_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'queue': {'name': 'projects/sample1/locations/sample2/queues/sample3'}}
    request_init['queue'] = {'name': 'projects/sample1/locations/sample2/queues/sample3', 'app_engine_routing_override': {'service': 'service_value', 'version': 'version_value', 'instance': 'instance_value', 'host': 'host_value'}, 'rate_limits': {'max_dispatches_per_second': 0.26380000000000003, 'max_burst_size': 1519, 'max_concurrent_dispatches': 2671}, 'retry_config': {'max_attempts': 1303, 'max_retry_duration': {'seconds': 751, 'nanos': 543}, 'min_backoff': {}, 'max_backoff': {}, 'max_doublings': 1388}, 'state': 1, 'purge_time': {'seconds': 751, 'nanos': 543}, 'stackdriver_logging_config': {'sampling_ratio': 0.1497}}
    test_field = cloudtasks.UpdateQueueRequest.meta.fields['queue']

    def get_message_fields(field):
        if False:
            print('Hello World!')
        message_fields = []
        if hasattr(field, 'message') and field.message:
            is_field_type_proto_plus_type = not hasattr(field.message, 'DESCRIPTOR')
            if is_field_type_proto_plus_type:
                message_fields = field.message.meta.fields.values()
            else:
                message_fields = field.message.DESCRIPTOR.fields
        return message_fields
    runtime_nested_fields = [(field.name, nested_field.name) for field in get_message_fields(test_field) for nested_field in get_message_fields(field)]
    subfields_not_in_runtime = []
    for (field, value) in request_init['queue'].items():
        result = None
        is_repeated = False
        if isinstance(value, list) and len(value):
            is_repeated = True
            result = value[0]
        if isinstance(value, dict):
            result = value
        if result and hasattr(result, 'keys'):
            for subfield in result.keys():
                if (field, subfield) not in runtime_nested_fields:
                    subfields_not_in_runtime.append({'field': field, 'subfield': subfield, 'is_repeated': is_repeated})
    for subfield_to_delete in subfields_not_in_runtime:
        field = subfield_to_delete.get('field')
        field_repeated = subfield_to_delete.get('is_repeated')
        subfield = subfield_to_delete.get('subfield')
        if subfield:
            if field_repeated:
                for i in range(0, len(request_init['queue'][field])):
                    del request_init['queue'][field][i][subfield]
            else:
                del request_init['queue'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gct_queue.Queue(name='name_value', state=gct_queue.Queue.State.RUNNING)
        response_value = Response()
        response_value.status_code = 200
        return_value = gct_queue.Queue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_queue(request)
    assert isinstance(response, gct_queue.Queue)
    assert response.name == 'name_value'
    assert response.state == gct_queue.Queue.State.RUNNING

def test_update_queue_rest_required_fields(request_type=cloudtasks.UpdateQueueRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_queue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_queue._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gct_queue.Queue()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gct_queue.Queue.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_queue(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_queue_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_queue._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('queue',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_queue_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'post_update_queue') as post, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_update_queue') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudtasks.UpdateQueueRequest.pb(cloudtasks.UpdateQueueRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gct_queue.Queue.to_json(gct_queue.Queue())
        request = cloudtasks.UpdateQueueRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gct_queue.Queue()
        client.update_queue(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_queue_rest_bad_request(transport: str='rest', request_type=cloudtasks.UpdateQueueRequest):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'queue': {'name': 'projects/sample1/locations/sample2/queues/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_queue(request)

def test_update_queue_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gct_queue.Queue()
        sample_request = {'queue': {'name': 'projects/sample1/locations/sample2/queues/sample3'}}
        mock_args = dict(queue=gct_queue.Queue(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gct_queue.Queue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_queue(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{queue.name=projects/*/locations/*/queues/*}' % client.transport._host, args[1])

def test_update_queue_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_queue(cloudtasks.UpdateQueueRequest(), queue=gct_queue.Queue(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_queue_rest_error():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudtasks.DeleteQueueRequest, dict])
def test_delete_queue_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_queue(request)
    assert response is None

def test_delete_queue_rest_required_fields(request_type=cloudtasks.DeleteQueueRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_queue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_queue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = None
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = ''
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_queue(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_queue_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_queue._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_queue_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_delete_queue') as pre:
        pre.assert_not_called()
        pb_message = cloudtasks.DeleteQueueRequest.pb(cloudtasks.DeleteQueueRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = cloudtasks.DeleteQueueRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_queue(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_queue_rest_bad_request(transport: str='rest', request_type=cloudtasks.DeleteQueueRequest):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_queue(request)

def test_delete_queue_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_queue(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/queues/*}' % client.transport._host, args[1])

def test_delete_queue_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_queue(cloudtasks.DeleteQueueRequest(), name='name_value')

def test_delete_queue_rest_error():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudtasks.PurgeQueueRequest, dict])
def test_purge_queue_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = queue.Queue(name='name_value', state=queue.Queue.State.RUNNING)
        response_value = Response()
        response_value.status_code = 200
        return_value = queue.Queue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.purge_queue(request)
    assert isinstance(response, queue.Queue)
    assert response.name == 'name_value'
    assert response.state == queue.Queue.State.RUNNING

def test_purge_queue_rest_required_fields(request_type=cloudtasks.PurgeQueueRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).purge_queue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).purge_queue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = queue.Queue()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = queue.Queue.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.purge_queue(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_purge_queue_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.purge_queue._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_purge_queue_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'post_purge_queue') as post, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_purge_queue') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudtasks.PurgeQueueRequest.pb(cloudtasks.PurgeQueueRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = queue.Queue.to_json(queue.Queue())
        request = cloudtasks.PurgeQueueRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = queue.Queue()
        client.purge_queue(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_purge_queue_rest_bad_request(transport: str='rest', request_type=cloudtasks.PurgeQueueRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.purge_queue(request)

def test_purge_queue_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = queue.Queue()
        sample_request = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = queue.Queue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.purge_queue(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/queues/*}:purge' % client.transport._host, args[1])

def test_purge_queue_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.purge_queue(cloudtasks.PurgeQueueRequest(), name='name_value')

def test_purge_queue_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudtasks.PauseQueueRequest, dict])
def test_pause_queue_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = queue.Queue(name='name_value', state=queue.Queue.State.RUNNING)
        response_value = Response()
        response_value.status_code = 200
        return_value = queue.Queue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.pause_queue(request)
    assert isinstance(response, queue.Queue)
    assert response.name == 'name_value'
    assert response.state == queue.Queue.State.RUNNING

def test_pause_queue_rest_required_fields(request_type=cloudtasks.PauseQueueRequest):
    if False:
        return 10
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).pause_queue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).pause_queue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = queue.Queue()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = queue.Queue.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.pause_queue(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_pause_queue_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.pause_queue._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_pause_queue_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'post_pause_queue') as post, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_pause_queue') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudtasks.PauseQueueRequest.pb(cloudtasks.PauseQueueRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = queue.Queue.to_json(queue.Queue())
        request = cloudtasks.PauseQueueRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = queue.Queue()
        client.pause_queue(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_pause_queue_rest_bad_request(transport: str='rest', request_type=cloudtasks.PauseQueueRequest):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.pause_queue(request)

def test_pause_queue_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = queue.Queue()
        sample_request = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = queue.Queue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.pause_queue(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/queues/*}:pause' % client.transport._host, args[1])

def test_pause_queue_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.pause_queue(cloudtasks.PauseQueueRequest(), name='name_value')

def test_pause_queue_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudtasks.ResumeQueueRequest, dict])
def test_resume_queue_rest(request_type):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = queue.Queue(name='name_value', state=queue.Queue.State.RUNNING)
        response_value = Response()
        response_value.status_code = 200
        return_value = queue.Queue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.resume_queue(request)
    assert isinstance(response, queue.Queue)
    assert response.name == 'name_value'
    assert response.state == queue.Queue.State.RUNNING

def test_resume_queue_rest_required_fields(request_type=cloudtasks.ResumeQueueRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resume_queue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resume_queue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = queue.Queue()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = queue.Queue.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.resume_queue(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_resume_queue_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.resume_queue._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_resume_queue_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'post_resume_queue') as post, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_resume_queue') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudtasks.ResumeQueueRequest.pb(cloudtasks.ResumeQueueRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = queue.Queue.to_json(queue.Queue())
        request = cloudtasks.ResumeQueueRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = queue.Queue()
        client.resume_queue(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_resume_queue_rest_bad_request(transport: str='rest', request_type=cloudtasks.ResumeQueueRequest):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.resume_queue(request)

def test_resume_queue_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = queue.Queue()
        sample_request = {'name': 'projects/sample1/locations/sample2/queues/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = queue.Queue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.resume_queue(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/queues/*}:resume' % client.transport._host, args[1])

def test_resume_queue_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.resume_queue(cloudtasks.ResumeQueueRequest(), name='name_value')

def test_resume_queue_rest_error():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_rest_required_fields(request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_iam_policy_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_iam_policy_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'post_get_iam_policy') as post, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_get_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.GetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.GetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.get_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

def test_get_iam_policy_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        sample_request = {'resource': 'projects/sample1/locations/sample2/queues/sample3'}
        mock_args = dict(resource='resource_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_iam_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{resource=projects/*/locations/*/queues/*}:getIamPolicy' % client.transport._host, args[1])

def test_get_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

def test_get_iam_policy_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_rest_required_fields(request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.set_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_iam_policy_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'policy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_iam_policy_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'post_set_iam_policy') as post, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_set_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.SetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.SetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.set_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

def test_set_iam_policy_rest_flattened():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        sample_request = {'resource': 'projects/sample1/locations/sample2/queues/sample3'}
        mock_args = dict(resource='resource_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_iam_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{resource=projects/*/locations/*/queues/*}:setIamPolicy' % client.transport._host, args[1])

def test_set_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

def test_set_iam_policy_rest_error():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.test_iam_permissions(request)
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_rest_required_fields(request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['resource'] = ''
    request_init['permissions'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    jsonified_request['permissions'] = 'permissions_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    assert 'permissions' in jsonified_request
    assert jsonified_request['permissions'] == 'permissions_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = iam_policy_pb2.TestIamPermissionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.test_iam_permissions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_test_iam_permissions_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.test_iam_permissions._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'permissions'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_test_iam_permissions_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'post_test_iam_permissions') as post, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_test_iam_permissions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.TestIamPermissionsRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(iam_policy_pb2.TestIamPermissionsResponse())
        request = iam_policy_pb2.TestIamPermissionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_test_iam_permissions_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

def test_test_iam_permissions_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse()
        sample_request = {'resource': 'projects/sample1/locations/sample2/queues/sample3'}
        mock_args = dict(resource='resource_value', permissions=['permissions_value'])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.test_iam_permissions(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{resource=projects/*/locations/*/queues/*}:testIamPermissions' % client.transport._host, args[1])

def test_test_iam_permissions_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

def test_test_iam_permissions_rest_error():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudtasks.ListTasksRequest, dict])
def test_list_tasks_rest(request_type):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloudtasks.ListTasksResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloudtasks.ListTasksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_tasks(request)
    assert isinstance(response, pagers.ListTasksPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tasks_rest_required_fields(request_type=cloudtasks.ListTasksRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tasks._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tasks._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token', 'response_view'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloudtasks.ListTasksResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloudtasks.ListTasksResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_tasks(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_tasks_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_tasks._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken', 'responseView')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_tasks_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'post_list_tasks') as post, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_list_tasks') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudtasks.ListTasksRequest.pb(cloudtasks.ListTasksRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloudtasks.ListTasksResponse.to_json(cloudtasks.ListTasksResponse())
        request = cloudtasks.ListTasksRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloudtasks.ListTasksResponse()
        client.list_tasks(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_tasks_rest_bad_request(transport: str='rest', request_type=cloudtasks.ListTasksRequest):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_tasks(request)

def test_list_tasks_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloudtasks.ListTasksResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/queues/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloudtasks.ListTasksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_tasks(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*/queues/*}/tasks' % client.transport._host, args[1])

def test_list_tasks_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_tasks(cloudtasks.ListTasksRequest(), parent='parent_value')

def test_list_tasks_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (cloudtasks.ListTasksResponse(tasks=[task.Task(), task.Task(), task.Task()], next_page_token='abc'), cloudtasks.ListTasksResponse(tasks=[], next_page_token='def'), cloudtasks.ListTasksResponse(tasks=[task.Task()], next_page_token='ghi'), cloudtasks.ListTasksResponse(tasks=[task.Task(), task.Task()]))
        response = response + response
        response = tuple((cloudtasks.ListTasksResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/queues/sample3'}
        pager = client.list_tasks(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, task.Task) for i in results))
        pages = list(client.list_tasks(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloudtasks.GetTaskRequest, dict])
def test_get_task_rest(request_type):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3/tasks/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = task.Task(name='name_value', dispatch_count=1496, response_count=1527, view=task.Task.View.BASIC)
        response_value = Response()
        response_value.status_code = 200
        return_value = task.Task.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_task(request)
    assert isinstance(response, task.Task)
    assert response.name == 'name_value'
    assert response.dispatch_count == 1496
    assert response.response_count == 1527
    assert response.view == task.Task.View.BASIC

def test_get_task_rest_required_fields(request_type=cloudtasks.GetTaskRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_task._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_task._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('response_view',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = task.Task()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = task.Task.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_task(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_task_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_task._get_unset_required_fields({})
    assert set(unset_fields) == set(('responseView',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_task_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'post_get_task') as post, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_get_task') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudtasks.GetTaskRequest.pb(cloudtasks.GetTaskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = task.Task.to_json(task.Task())
        request = cloudtasks.GetTaskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = task.Task()
        client.get_task(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_task_rest_bad_request(transport: str='rest', request_type=cloudtasks.GetTaskRequest):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3/tasks/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_task(request)

def test_get_task_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = task.Task()
        sample_request = {'name': 'projects/sample1/locations/sample2/queues/sample3/tasks/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = task.Task.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_task(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/queues/*/tasks/*}' % client.transport._host, args[1])

def test_get_task_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_task(cloudtasks.GetTaskRequest(), name='name_value')

def test_get_task_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudtasks.CreateTaskRequest, dict])
def test_create_task_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gct_task.Task(name='name_value', dispatch_count=1496, response_count=1527, view=gct_task.Task.View.BASIC)
        response_value = Response()
        response_value.status_code = 200
        return_value = gct_task.Task.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_task(request)
    assert isinstance(response, gct_task.Task)
    assert response.name == 'name_value'
    assert response.dispatch_count == 1496
    assert response.response_count == 1527
    assert response.view == gct_task.Task.View.BASIC

def test_create_task_rest_required_fields(request_type=cloudtasks.CreateTaskRequest):
    if False:
        return 10
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_task._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_task._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gct_task.Task()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gct_task.Task.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_task(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_task_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_task._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'task'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_task_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'post_create_task') as post, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_create_task') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudtasks.CreateTaskRequest.pb(cloudtasks.CreateTaskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gct_task.Task.to_json(gct_task.Task())
        request = cloudtasks.CreateTaskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gct_task.Task()
        client.create_task(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_task_rest_bad_request(transport: str='rest', request_type=cloudtasks.CreateTaskRequest):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/queues/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_task(request)

def test_create_task_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gct_task.Task()
        sample_request = {'parent': 'projects/sample1/locations/sample2/queues/sample3'}
        mock_args = dict(parent='parent_value', task=gct_task.Task(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gct_task.Task.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_task(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*/queues/*}/tasks' % client.transport._host, args[1])

def test_create_task_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_task(cloudtasks.CreateTaskRequest(), parent='parent_value', task=gct_task.Task(name='name_value'))

def test_create_task_rest_error():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudtasks.DeleteTaskRequest, dict])
def test_delete_task_rest(request_type):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3/tasks/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_task(request)
    assert response is None

def test_delete_task_rest_required_fields(request_type=cloudtasks.DeleteTaskRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_task._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_task._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = None
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = ''
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_task(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_task_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_task._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_task_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_delete_task') as pre:
        pre.assert_not_called()
        pb_message = cloudtasks.DeleteTaskRequest.pb(cloudtasks.DeleteTaskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = cloudtasks.DeleteTaskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_task(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_task_rest_bad_request(transport: str='rest', request_type=cloudtasks.DeleteTaskRequest):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3/tasks/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_task(request)

def test_delete_task_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/queues/sample3/tasks/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_task(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/queues/*/tasks/*}' % client.transport._host, args[1])

def test_delete_task_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_task(cloudtasks.DeleteTaskRequest(), name='name_value')

def test_delete_task_rest_error():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudtasks.RunTaskRequest, dict])
def test_run_task_rest(request_type):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3/tasks/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = task.Task(name='name_value', dispatch_count=1496, response_count=1527, view=task.Task.View.BASIC)
        response_value = Response()
        response_value.status_code = 200
        return_value = task.Task.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.run_task(request)
    assert isinstance(response, task.Task)
    assert response.name == 'name_value'
    assert response.dispatch_count == 1496
    assert response.response_count == 1527
    assert response.view == task.Task.View.BASIC

def test_run_task_rest_required_fields(request_type=cloudtasks.RunTaskRequest):
    if False:
        return 10
    transport_class = transports.CloudTasksRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).run_task._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).run_task._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = task.Task()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = task.Task.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.run_task(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_run_task_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.run_task._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_run_task_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudTasksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudTasksRestInterceptor())
    client = CloudTasksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudTasksRestInterceptor, 'post_run_task') as post, mock.patch.object(transports.CloudTasksRestInterceptor, 'pre_run_task') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudtasks.RunTaskRequest.pb(cloudtasks.RunTaskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = task.Task.to_json(task.Task())
        request = cloudtasks.RunTaskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = task.Task()
        client.run_task(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_run_task_rest_bad_request(transport: str='rest', request_type=cloudtasks.RunTaskRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/queues/sample3/tasks/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.run_task(request)

def test_run_task_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = task.Task()
        sample_request = {'name': 'projects/sample1/locations/sample2/queues/sample3/tasks/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = task.Task.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.run_task(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/queues/*/tasks/*}:run' % client.transport._host, args[1])

def test_run_task_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.run_task(cloudtasks.RunTaskRequest(), name='name_value')

def test_run_task_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudTasksGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CloudTasksGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudTasksClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CloudTasksGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudTasksClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudTasksClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CloudTasksGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudTasksClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.CloudTasksGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CloudTasksClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.CloudTasksGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CloudTasksGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CloudTasksGrpcTransport, transports.CloudTasksGrpcAsyncIOTransport, transports.CloudTasksRestTransport])
def test_transport_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        i = 10
        return i + 15
    transport = CloudTasksClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CloudTasksGrpcTransport)

def test_cloud_tasks_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CloudTasksTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_cloud_tasks_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.tasks_v2.services.cloud_tasks.transports.CloudTasksTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CloudTasksTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_queues', 'get_queue', 'create_queue', 'update_queue', 'delete_queue', 'purge_queue', 'pause_queue', 'resume_queue', 'get_iam_policy', 'set_iam_policy', 'test_iam_permissions', 'list_tasks', 'get_task', 'create_task', 'delete_task', 'run_task', 'get_location', 'list_locations')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_cloud_tasks_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.tasks_v2.services.cloud_tasks.transports.CloudTasksTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudTasksTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_cloud_tasks_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.tasks_v2.services.cloud_tasks.transports.CloudTasksTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudTasksTransport()
        adc.assert_called_once()

def test_cloud_tasks_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CloudTasksClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CloudTasksGrpcTransport, transports.CloudTasksGrpcAsyncIOTransport])
def test_cloud_tasks_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CloudTasksGrpcTransport, transports.CloudTasksGrpcAsyncIOTransport, transports.CloudTasksRestTransport])
def test_cloud_tasks_transport_auth_gdch_credentials(transport_class):
    if False:
        i = 10
        return i + 15
    host = 'https://language.com'
    api_audience_tests = [None, 'https://language2.com']
    api_audience_expect = [host, 'https://language2.com']
    for (t, e) in zip(api_audience_tests, api_audience_expect):
        with mock.patch.object(google.auth, 'default', autospec=True) as adc:
            gdch_mock = mock.MagicMock()
            type(gdch_mock).with_gdch_audience = mock.PropertyMock(return_value=gdch_mock)
            adc.return_value = (gdch_mock, None)
            transport_class(host=host, api_audience=t)
            gdch_mock.with_gdch_audience.assert_called_once_with(e)

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CloudTasksGrpcTransport, grpc_helpers), (transports.CloudTasksGrpcAsyncIOTransport, grpc_helpers_async)])
def test_cloud_tasks_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudtasks.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='cloudtasks.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CloudTasksGrpcTransport, transports.CloudTasksGrpcAsyncIOTransport])
def test_cloud_tasks_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch.object(transport_class, 'create_channel') as mock_create_channel:
        mock_ssl_channel_creds = mock.Mock()
        transport_class(host='squid.clam.whelk', credentials=cred, ssl_channel_credentials=mock_ssl_channel_creds)
        mock_create_channel.assert_called_once_with('squid.clam.whelk:443', credentials=cred, credentials_file=None, scopes=None, ssl_credentials=mock_ssl_channel_creds, quota_project_id=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])
    with mock.patch.object(transport_class, 'create_channel', return_value=mock.Mock()):
        with mock.patch('grpc.ssl_channel_credentials') as mock_ssl_cred:
            transport_class(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
            (expected_cert, expected_key) = client_cert_source_callback()
            mock_ssl_cred.assert_called_once_with(certificate_chain=expected_cert, private_key=expected_key)

def test_cloud_tasks_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.CloudTasksRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_tasks_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudtasks.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudtasks.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudtasks.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_tasks_host_with_port(transport_name):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudtasks.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudtasks.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudtasks.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_cloud_tasks_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = CloudTasksClient(credentials=creds1, transport=transport_name)
    client2 = CloudTasksClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_queues._session
    session2 = client2.transport.list_queues._session
    assert session1 != session2
    session1 = client1.transport.get_queue._session
    session2 = client2.transport.get_queue._session
    assert session1 != session2
    session1 = client1.transport.create_queue._session
    session2 = client2.transport.create_queue._session
    assert session1 != session2
    session1 = client1.transport.update_queue._session
    session2 = client2.transport.update_queue._session
    assert session1 != session2
    session1 = client1.transport.delete_queue._session
    session2 = client2.transport.delete_queue._session
    assert session1 != session2
    session1 = client1.transport.purge_queue._session
    session2 = client2.transport.purge_queue._session
    assert session1 != session2
    session1 = client1.transport.pause_queue._session
    session2 = client2.transport.pause_queue._session
    assert session1 != session2
    session1 = client1.transport.resume_queue._session
    session2 = client2.transport.resume_queue._session
    assert session1 != session2
    session1 = client1.transport.get_iam_policy._session
    session2 = client2.transport.get_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.set_iam_policy._session
    session2 = client2.transport.set_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.test_iam_permissions._session
    session2 = client2.transport.test_iam_permissions._session
    assert session1 != session2
    session1 = client1.transport.list_tasks._session
    session2 = client2.transport.list_tasks._session
    assert session1 != session2
    session1 = client1.transport.get_task._session
    session2 = client2.transport.get_task._session
    assert session1 != session2
    session1 = client1.transport.create_task._session
    session2 = client2.transport.create_task._session
    assert session1 != session2
    session1 = client1.transport.delete_task._session
    session2 = client2.transport.delete_task._session
    assert session1 != session2
    session1 = client1.transport.run_task._session
    session2 = client2.transport.run_task._session
    assert session1 != session2

def test_cloud_tasks_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudTasksGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_cloud_tasks_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudTasksGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CloudTasksGrpcTransport, transports.CloudTasksGrpcAsyncIOTransport])
def test_cloud_tasks_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        print('Hello World!')
    with mock.patch('grpc.ssl_channel_credentials', autospec=True) as grpc_ssl_channel_cred:
        with mock.patch.object(transport_class, 'create_channel') as grpc_create_channel:
            mock_ssl_cred = mock.Mock()
            grpc_ssl_channel_cred.return_value = mock_ssl_cred
            mock_grpc_channel = mock.Mock()
            grpc_create_channel.return_value = mock_grpc_channel
            cred = ga_credentials.AnonymousCredentials()
            with pytest.warns(DeprecationWarning):
                with mock.patch.object(google.auth, 'default') as adc:
                    adc.return_value = (cred, None)
                    transport = transport_class(host='squid.clam.whelk', api_mtls_endpoint='mtls.squid.clam.whelk', client_cert_source=client_cert_source_callback)
                    adc.assert_called_once()
            grpc_ssl_channel_cred.assert_called_once_with(certificate_chain=b'cert bytes', private_key=b'key bytes')
            grpc_create_channel.assert_called_once_with('mtls.squid.clam.whelk:443', credentials=cred, credentials_file=None, scopes=None, ssl_credentials=mock_ssl_cred, quota_project_id=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])
            assert transport.grpc_channel == mock_grpc_channel
            assert transport._ssl_channel_credentials == mock_ssl_cred

@pytest.mark.parametrize('transport_class', [transports.CloudTasksGrpcTransport, transports.CloudTasksGrpcAsyncIOTransport])
def test_cloud_tasks_transport_channel_mtls_with_adc(transport_class):
    if False:
        while True:
            i = 10
    mock_ssl_cred = mock.Mock()
    with mock.patch.multiple('google.auth.transport.grpc.SslCredentials', __init__=mock.Mock(return_value=None), ssl_credentials=mock.PropertyMock(return_value=mock_ssl_cred)):
        with mock.patch.object(transport_class, 'create_channel') as grpc_create_channel:
            mock_grpc_channel = mock.Mock()
            grpc_create_channel.return_value = mock_grpc_channel
            mock_cred = mock.Mock()
            with pytest.warns(DeprecationWarning):
                transport = transport_class(host='squid.clam.whelk', credentials=mock_cred, api_mtls_endpoint='mtls.squid.clam.whelk', client_cert_source=None)
            grpc_create_channel.assert_called_once_with('mtls.squid.clam.whelk:443', credentials=mock_cred, credentials_file=None, scopes=None, ssl_credentials=mock_ssl_cred, quota_project_id=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])
            assert transport.grpc_channel == mock_grpc_channel

def test_queue_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    queue = 'whelk'
    expected = 'projects/{project}/locations/{location}/queues/{queue}'.format(project=project, location=location, queue=queue)
    actual = CloudTasksClient.queue_path(project, location, queue)
    assert expected == actual

def test_parse_queue_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus', 'location': 'oyster', 'queue': 'nudibranch'}
    path = CloudTasksClient.queue_path(**expected)
    actual = CloudTasksClient.parse_queue_path(path)
    assert expected == actual

def test_task_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    queue = 'winkle'
    task = 'nautilus'
    expected = 'projects/{project}/locations/{location}/queues/{queue}/tasks/{task}'.format(project=project, location=location, queue=queue, task=task)
    actual = CloudTasksClient.task_path(project, location, queue, task)
    assert expected == actual

def test_parse_task_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'scallop', 'location': 'abalone', 'queue': 'squid', 'task': 'clam'}
    path = CloudTasksClient.task_path(**expected)
    actual = CloudTasksClient.parse_task_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CloudTasksClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'octopus'}
    path = CloudTasksClient.common_billing_account_path(**expected)
    actual = CloudTasksClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CloudTasksClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'nudibranch'}
    path = CloudTasksClient.common_folder_path(**expected)
    actual = CloudTasksClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CloudTasksClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'mussel'}
    path = CloudTasksClient.common_organization_path(**expected)
    actual = CloudTasksClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = CloudTasksClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nautilus'}
    path = CloudTasksClient.common_project_path(**expected)
    actual = CloudTasksClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CloudTasksClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'squid', 'location': 'clam'}
    path = CloudTasksClient.common_location_path(**expected)
    actual = CloudTasksClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CloudTasksTransport, '_prep_wrapped_messages') as prep:
        client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CloudTasksTransport, '_prep_wrapped_messages') as prep:
        transport_class = CloudTasksClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_location(request)

@pytest.mark.parametrize('request_type', [locations_pb2.GetLocationRequest, dict])
def test_get_location_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = locations_pb2.Location()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_location(request)
    assert isinstance(response, locations_pb2.Location)

def test_list_locations_rest_bad_request(transport: str='rest', request_type=locations_pb2.ListLocationsRequest):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_locations(request)

@pytest.mark.parametrize('request_type', [locations_pb2.ListLocationsRequest, dict])
def test_list_locations_rest(request_type):
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = locations_pb2.ListLocationsResponse()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_locations(request)
    assert isinstance(response, locations_pb2.ListLocationsResponse)

def test_list_locations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = locations_pb2.ListLocationsRequest()
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, locations_pb2.ListLocationsResponse)

@pytest.mark.asyncio
async def test_list_locations_async(transport: str='grpc'):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = locations_pb2.ListLocationsRequest()
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, locations_pb2.ListLocationsResponse)

def test_list_locations_field_headers():
    if False:
        return 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = locations_pb2.ListLocationsRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        client.list_locations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

@pytest.mark.asyncio
async def test_list_locations_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = locations_pb2.ListLocationsRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        await client.list_locations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

def test_list_locations_from_dict():
    if False:
        for i in range(10):
            print('nop')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = locations_pb2.GetLocationRequest()
    with mock.patch.object(type(client.transport.get_location), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, locations_pb2.Location)

@pytest.mark.asyncio
async def test_get_location_async(transport: str='grpc_asyncio'):
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = locations_pb2.GetLocationRequest()
    with mock.patch.object(type(client.transport.get_location), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, locations_pb2.Location)

def test_get_location_field_headers():
    if False:
        while True:
            i = 10
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    request = locations_pb2.GetLocationRequest()
    request.name = 'locations/abc'
    with mock.patch.object(type(client.transport.get_location), '__call__') as call:
        call.return_value = locations_pb2.Location()
        client.get_location(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations/abc') in kw['metadata']

@pytest.mark.asyncio
async def test_get_location_field_headers_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = locations_pb2.GetLocationRequest()
    request.name = 'locations/abc'
    with mock.patch.object(type(client.transport.get_location), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        await client.get_location(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations/abc') in kw['metadata']

def test_get_location_from_dict():
    if False:
        print('Hello World!')
    client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = CloudTasksAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        while True:
            i = 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = CloudTasksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CloudTasksClient, transports.CloudTasksGrpcTransport), (CloudTasksAsyncClient, transports.CloudTasksGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)