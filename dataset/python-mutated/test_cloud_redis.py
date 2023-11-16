import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
from collections.abc import Iterable
import json
import math
from google.api_core import future, gapic_v1, grpc_helpers, grpc_helpers_async, operation, operations_v1, path_template
from google.api_core import client_options
from google.api_core import exceptions as core_exceptions
from google.api_core import operation_async
import google.auth
from google.auth import credentials as ga_credentials
from google.auth.exceptions import MutualTLSChannelError
from google.cloud.location import locations_pb2
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import duration_pb2
from google.protobuf import empty_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.type import dayofweek_pb2
from google.type import timeofday_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.redis_v1.services.cloud_redis import CloudRedisAsyncClient, CloudRedisClient, pagers, transports
from google.cloud.redis_v1.types import cloud_redis

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        for i in range(10):
            print('nop')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert CloudRedisClient._get_default_mtls_endpoint(None) is None
    assert CloudRedisClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CloudRedisClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CloudRedisClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CloudRedisClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CloudRedisClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CloudRedisClient, 'grpc'), (CloudRedisAsyncClient, 'grpc_asyncio'), (CloudRedisClient, 'rest')])
def test_cloud_redis_client_from_service_account_info(client_class, transport_name):
    if False:
        i = 10
        return i + 15
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('redis.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://redis.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CloudRedisGrpcTransport, 'grpc'), (transports.CloudRedisGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.CloudRedisRestTransport, 'rest')])
def test_cloud_redis_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(CloudRedisClient, 'grpc'), (CloudRedisAsyncClient, 'grpc_asyncio'), (CloudRedisClient, 'rest')])
def test_cloud_redis_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('redis.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://redis.googleapis.com')

def test_cloud_redis_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = CloudRedisClient.get_transport_class()
    available_transports = [transports.CloudRedisGrpcTransport, transports.CloudRedisRestTransport]
    assert transport in available_transports
    transport = CloudRedisClient.get_transport_class('grpc')
    assert transport == transports.CloudRedisGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudRedisClient, transports.CloudRedisGrpcTransport, 'grpc'), (CloudRedisAsyncClient, transports.CloudRedisGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudRedisClient, transports.CloudRedisRestTransport, 'rest')])
@mock.patch.object(CloudRedisClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudRedisClient))
@mock.patch.object(CloudRedisAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudRedisAsyncClient))
def test_cloud_redis_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(CloudRedisClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CloudRedisClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CloudRedisClient, transports.CloudRedisGrpcTransport, 'grpc', 'true'), (CloudRedisAsyncClient, transports.CloudRedisGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CloudRedisClient, transports.CloudRedisGrpcTransport, 'grpc', 'false'), (CloudRedisAsyncClient, transports.CloudRedisGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (CloudRedisClient, transports.CloudRedisRestTransport, 'rest', 'true'), (CloudRedisClient, transports.CloudRedisRestTransport, 'rest', 'false')])
@mock.patch.object(CloudRedisClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudRedisClient))
@mock.patch.object(CloudRedisAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudRedisAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_cloud_redis_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
    if False:
        print('Hello World!')
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

@pytest.mark.parametrize('client_class', [CloudRedisClient, CloudRedisAsyncClient])
@mock.patch.object(CloudRedisClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudRedisClient))
@mock.patch.object(CloudRedisAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudRedisAsyncClient))
def test_cloud_redis_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudRedisClient, transports.CloudRedisGrpcTransport, 'grpc'), (CloudRedisAsyncClient, transports.CloudRedisGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudRedisClient, transports.CloudRedisRestTransport, 'rest')])
def test_cloud_redis_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudRedisClient, transports.CloudRedisGrpcTransport, 'grpc', grpc_helpers), (CloudRedisAsyncClient, transports.CloudRedisGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (CloudRedisClient, transports.CloudRedisRestTransport, 'rest', None)])
def test_cloud_redis_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_cloud_redis_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.redis_v1.services.cloud_redis.transports.CloudRedisGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CloudRedisClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudRedisClient, transports.CloudRedisGrpcTransport, 'grpc', grpc_helpers), (CloudRedisAsyncClient, transports.CloudRedisGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_cloud_redis_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
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
        create_channel.assert_called_with('redis.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='redis.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [cloud_redis.ListInstancesRequest, dict])
def test_list_instances(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = cloud_redis.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_instances(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.ListInstancesRequest()
    assert isinstance(response, pagers.ListInstancesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_instances_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        client.list_instances()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.ListInstancesRequest()

@pytest.mark.asyncio
async def test_list_instances_async(transport: str='grpc_asyncio', request_type=cloud_redis.ListInstancesRequest):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_redis.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_instances(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.ListInstancesRequest()
    assert isinstance(response, pagers.ListInstancesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_instances_async_from_dict():
    await test_list_instances_async(request_type=dict)

def test_list_instances_field_headers():
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.ListInstancesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = cloud_redis.ListInstancesResponse()
        client.list_instances(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_instances_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.ListInstancesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_redis.ListInstancesResponse())
        await client.list_instances(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_instances_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = cloud_redis.ListInstancesResponse()
        client.list_instances(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_instances_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_instances(cloud_redis.ListInstancesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_instances_flattened_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = cloud_redis.ListInstancesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_redis.ListInstancesResponse())
        response = await client.list_instances(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_instances_flattened_error_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_instances(cloud_redis.ListInstancesRequest(), parent='parent_value')

def test_list_instances_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.side_effect = (cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance(), cloud_redis.Instance(), cloud_redis.Instance()], next_page_token='abc'), cloud_redis.ListInstancesResponse(instances=[], next_page_token='def'), cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance()], next_page_token='ghi'), cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance(), cloud_redis.Instance()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_instances(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_redis.Instance) for i in results))

def test_list_instances_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.side_effect = (cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance(), cloud_redis.Instance(), cloud_redis.Instance()], next_page_token='abc'), cloud_redis.ListInstancesResponse(instances=[], next_page_token='def'), cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance()], next_page_token='ghi'), cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance(), cloud_redis.Instance()]), RuntimeError)
        pages = list(client.list_instances(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_instances_async_pager():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_instances), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance(), cloud_redis.Instance(), cloud_redis.Instance()], next_page_token='abc'), cloud_redis.ListInstancesResponse(instances=[], next_page_token='def'), cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance()], next_page_token='ghi'), cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance(), cloud_redis.Instance()]), RuntimeError)
        async_pager = await client.list_instances(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, cloud_redis.Instance) for i in responses))

@pytest.mark.asyncio
async def test_list_instances_async_pages():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_instances), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance(), cloud_redis.Instance(), cloud_redis.Instance()], next_page_token='abc'), cloud_redis.ListInstancesResponse(instances=[], next_page_token='def'), cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance()], next_page_token='ghi'), cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance(), cloud_redis.Instance()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_instances(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_redis.GetInstanceRequest, dict])
def test_get_instance(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = cloud_redis.Instance(name='name_value', display_name='display_name_value', location_id='location_id_value', alternative_location_id='alternative_location_id_value', redis_version='redis_version_value', reserved_ip_range='reserved_ip_range_value', secondary_ip_range='secondary_ip_range_value', host='host_value', port=453, current_location_id='current_location_id_value', state=cloud_redis.Instance.State.CREATING, status_message='status_message_value', tier=cloud_redis.Instance.Tier.BASIC, memory_size_gb=1499, authorized_network='authorized_network_value', persistence_iam_identity='persistence_iam_identity_value', connect_mode=cloud_redis.Instance.ConnectMode.DIRECT_PEERING, auth_enabled=True, transit_encryption_mode=cloud_redis.Instance.TransitEncryptionMode.SERVER_AUTHENTICATION, replica_count=1384, read_endpoint='read_endpoint_value', read_endpoint_port=1920, read_replicas_mode=cloud_redis.Instance.ReadReplicasMode.READ_REPLICAS_DISABLED, customer_managed_key='customer_managed_key_value', suspension_reasons=[cloud_redis.Instance.SuspensionReason.CUSTOMER_MANAGED_KEY_ISSUE], maintenance_version='maintenance_version_value', available_maintenance_versions=['available_maintenance_versions_value'])
        response = client.get_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.GetInstanceRequest()
    assert isinstance(response, cloud_redis.Instance)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.location_id == 'location_id_value'
    assert response.alternative_location_id == 'alternative_location_id_value'
    assert response.redis_version == 'redis_version_value'
    assert response.reserved_ip_range == 'reserved_ip_range_value'
    assert response.secondary_ip_range == 'secondary_ip_range_value'
    assert response.host == 'host_value'
    assert response.port == 453
    assert response.current_location_id == 'current_location_id_value'
    assert response.state == cloud_redis.Instance.State.CREATING
    assert response.status_message == 'status_message_value'
    assert response.tier == cloud_redis.Instance.Tier.BASIC
    assert response.memory_size_gb == 1499
    assert response.authorized_network == 'authorized_network_value'
    assert response.persistence_iam_identity == 'persistence_iam_identity_value'
    assert response.connect_mode == cloud_redis.Instance.ConnectMode.DIRECT_PEERING
    assert response.auth_enabled is True
    assert response.transit_encryption_mode == cloud_redis.Instance.TransitEncryptionMode.SERVER_AUTHENTICATION
    assert response.replica_count == 1384
    assert response.read_endpoint == 'read_endpoint_value'
    assert response.read_endpoint_port == 1920
    assert response.read_replicas_mode == cloud_redis.Instance.ReadReplicasMode.READ_REPLICAS_DISABLED
    assert response.customer_managed_key == 'customer_managed_key_value'
    assert response.suspension_reasons == [cloud_redis.Instance.SuspensionReason.CUSTOMER_MANAGED_KEY_ISSUE]
    assert response.maintenance_version == 'maintenance_version_value'
    assert response.available_maintenance_versions == ['available_maintenance_versions_value']

def test_get_instance_empty_call():
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        client.get_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.GetInstanceRequest()

@pytest.mark.asyncio
async def test_get_instance_async(transport: str='grpc_asyncio', request_type=cloud_redis.GetInstanceRequest):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_redis.Instance(name='name_value', display_name='display_name_value', location_id='location_id_value', alternative_location_id='alternative_location_id_value', redis_version='redis_version_value', reserved_ip_range='reserved_ip_range_value', secondary_ip_range='secondary_ip_range_value', host='host_value', port=453, current_location_id='current_location_id_value', state=cloud_redis.Instance.State.CREATING, status_message='status_message_value', tier=cloud_redis.Instance.Tier.BASIC, memory_size_gb=1499, authorized_network='authorized_network_value', persistence_iam_identity='persistence_iam_identity_value', connect_mode=cloud_redis.Instance.ConnectMode.DIRECT_PEERING, auth_enabled=True, transit_encryption_mode=cloud_redis.Instance.TransitEncryptionMode.SERVER_AUTHENTICATION, replica_count=1384, read_endpoint='read_endpoint_value', read_endpoint_port=1920, read_replicas_mode=cloud_redis.Instance.ReadReplicasMode.READ_REPLICAS_DISABLED, customer_managed_key='customer_managed_key_value', suspension_reasons=[cloud_redis.Instance.SuspensionReason.CUSTOMER_MANAGED_KEY_ISSUE], maintenance_version='maintenance_version_value', available_maintenance_versions=['available_maintenance_versions_value']))
        response = await client.get_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.GetInstanceRequest()
    assert isinstance(response, cloud_redis.Instance)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.location_id == 'location_id_value'
    assert response.alternative_location_id == 'alternative_location_id_value'
    assert response.redis_version == 'redis_version_value'
    assert response.reserved_ip_range == 'reserved_ip_range_value'
    assert response.secondary_ip_range == 'secondary_ip_range_value'
    assert response.host == 'host_value'
    assert response.port == 453
    assert response.current_location_id == 'current_location_id_value'
    assert response.state == cloud_redis.Instance.State.CREATING
    assert response.status_message == 'status_message_value'
    assert response.tier == cloud_redis.Instance.Tier.BASIC
    assert response.memory_size_gb == 1499
    assert response.authorized_network == 'authorized_network_value'
    assert response.persistence_iam_identity == 'persistence_iam_identity_value'
    assert response.connect_mode == cloud_redis.Instance.ConnectMode.DIRECT_PEERING
    assert response.auth_enabled is True
    assert response.transit_encryption_mode == cloud_redis.Instance.TransitEncryptionMode.SERVER_AUTHENTICATION
    assert response.replica_count == 1384
    assert response.read_endpoint == 'read_endpoint_value'
    assert response.read_endpoint_port == 1920
    assert response.read_replicas_mode == cloud_redis.Instance.ReadReplicasMode.READ_REPLICAS_DISABLED
    assert response.customer_managed_key == 'customer_managed_key_value'
    assert response.suspension_reasons == [cloud_redis.Instance.SuspensionReason.CUSTOMER_MANAGED_KEY_ISSUE]
    assert response.maintenance_version == 'maintenance_version_value'
    assert response.available_maintenance_versions == ['available_maintenance_versions_value']

@pytest.mark.asyncio
async def test_get_instance_async_from_dict():
    await test_get_instance_async(request_type=dict)

def test_get_instance_field_headers():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.GetInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = cloud_redis.Instance()
        client.get_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_instance_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.GetInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_redis.Instance())
        await client.get_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_instance_flattened():
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = cloud_redis.Instance()
        client.get_instance(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_instance_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_instance(cloud_redis.GetInstanceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_instance_flattened_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = cloud_redis.Instance()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_redis.Instance())
        response = await client.get_instance(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_instance_flattened_error_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_instance(cloud_redis.GetInstanceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_redis.GetInstanceAuthStringRequest, dict])
def test_get_instance_auth_string(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_instance_auth_string), '__call__') as call:
        call.return_value = cloud_redis.InstanceAuthString(auth_string='auth_string_value')
        response = client.get_instance_auth_string(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.GetInstanceAuthStringRequest()
    assert isinstance(response, cloud_redis.InstanceAuthString)
    assert response.auth_string == 'auth_string_value'

def test_get_instance_auth_string_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_instance_auth_string), '__call__') as call:
        client.get_instance_auth_string()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.GetInstanceAuthStringRequest()

@pytest.mark.asyncio
async def test_get_instance_auth_string_async(transport: str='grpc_asyncio', request_type=cloud_redis.GetInstanceAuthStringRequest):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_instance_auth_string), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_redis.InstanceAuthString(auth_string='auth_string_value'))
        response = await client.get_instance_auth_string(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.GetInstanceAuthStringRequest()
    assert isinstance(response, cloud_redis.InstanceAuthString)
    assert response.auth_string == 'auth_string_value'

@pytest.mark.asyncio
async def test_get_instance_auth_string_async_from_dict():
    await test_get_instance_auth_string_async(request_type=dict)

def test_get_instance_auth_string_field_headers():
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.GetInstanceAuthStringRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_instance_auth_string), '__call__') as call:
        call.return_value = cloud_redis.InstanceAuthString()
        client.get_instance_auth_string(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_instance_auth_string_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.GetInstanceAuthStringRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_instance_auth_string), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_redis.InstanceAuthString())
        await client.get_instance_auth_string(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_instance_auth_string_flattened():
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_instance_auth_string), '__call__') as call:
        call.return_value = cloud_redis.InstanceAuthString()
        client.get_instance_auth_string(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_instance_auth_string_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_instance_auth_string(cloud_redis.GetInstanceAuthStringRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_instance_auth_string_flattened_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_instance_auth_string), '__call__') as call:
        call.return_value = cloud_redis.InstanceAuthString()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_redis.InstanceAuthString())
        response = await client.get_instance_auth_string(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_instance_auth_string_flattened_error_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_instance_auth_string(cloud_redis.GetInstanceAuthStringRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_redis.CreateInstanceRequest, dict])
def test_create_instance(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.CreateInstanceRequest()
    assert isinstance(response, future.Future)

def test_create_instance_empty_call():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        client.create_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.CreateInstanceRequest()

@pytest.mark.asyncio
async def test_create_instance_async(transport: str='grpc_asyncio', request_type=cloud_redis.CreateInstanceRequest):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.CreateInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_instance_async_from_dict():
    await test_create_instance_async(request_type=dict)

def test_create_instance_field_headers():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.CreateInstanceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_instance_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.CreateInstanceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_instance_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_instance(parent='parent_value', instance_id='instance_id_value', instance=cloud_redis.Instance(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].instance_id
        mock_val = 'instance_id_value'
        assert arg == mock_val
        arg = args[0].instance
        mock_val = cloud_redis.Instance(name='name_value')
        assert arg == mock_val

def test_create_instance_flattened_error():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_instance(cloud_redis.CreateInstanceRequest(), parent='parent_value', instance_id='instance_id_value', instance=cloud_redis.Instance(name='name_value'))

@pytest.mark.asyncio
async def test_create_instance_flattened_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_instance(parent='parent_value', instance_id='instance_id_value', instance=cloud_redis.Instance(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].instance_id
        mock_val = 'instance_id_value'
        assert arg == mock_val
        arg = args[0].instance
        mock_val = cloud_redis.Instance(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_instance_flattened_error_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_instance(cloud_redis.CreateInstanceRequest(), parent='parent_value', instance_id='instance_id_value', instance=cloud_redis.Instance(name='name_value'))

@pytest.mark.parametrize('request_type', [cloud_redis.UpdateInstanceRequest, dict])
def test_update_instance(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.UpdateInstanceRequest()
    assert isinstance(response, future.Future)

def test_update_instance_empty_call():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        client.update_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.UpdateInstanceRequest()

@pytest.mark.asyncio
async def test_update_instance_async(transport: str='grpc_asyncio', request_type=cloud_redis.UpdateInstanceRequest):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.UpdateInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_instance_async_from_dict():
    await test_update_instance_async(request_type=dict)

def test_update_instance_field_headers():
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.UpdateInstanceRequest()
    request.instance.name = 'name_value'
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'instance.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_instance_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.UpdateInstanceRequest()
    request.instance.name = 'name_value'
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'instance.name=name_value') in kw['metadata']

def test_update_instance_flattened():
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_instance(update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), instance=cloud_redis.Instance(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val
        arg = args[0].instance
        mock_val = cloud_redis.Instance(name='name_value')
        assert arg == mock_val

def test_update_instance_flattened_error():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_instance(cloud_redis.UpdateInstanceRequest(), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), instance=cloud_redis.Instance(name='name_value'))

@pytest.mark.asyncio
async def test_update_instance_flattened_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_instance(update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), instance=cloud_redis.Instance(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val
        arg = args[0].instance
        mock_val = cloud_redis.Instance(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_instance_flattened_error_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_instance(cloud_redis.UpdateInstanceRequest(), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), instance=cloud_redis.Instance(name='name_value'))

@pytest.mark.parametrize('request_type', [cloud_redis.UpgradeInstanceRequest, dict])
def test_upgrade_instance(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.upgrade_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.upgrade_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.UpgradeInstanceRequest()
    assert isinstance(response, future.Future)

def test_upgrade_instance_empty_call():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.upgrade_instance), '__call__') as call:
        client.upgrade_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.UpgradeInstanceRequest()

@pytest.mark.asyncio
async def test_upgrade_instance_async(transport: str='grpc_asyncio', request_type=cloud_redis.UpgradeInstanceRequest):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.upgrade_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.upgrade_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.UpgradeInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_upgrade_instance_async_from_dict():
    await test_upgrade_instance_async(request_type=dict)

def test_upgrade_instance_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.UpgradeInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.upgrade_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.upgrade_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_upgrade_instance_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.UpgradeInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.upgrade_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.upgrade_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_upgrade_instance_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.upgrade_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.upgrade_instance(name='name_value', redis_version='redis_version_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].redis_version
        mock_val = 'redis_version_value'
        assert arg == mock_val

def test_upgrade_instance_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.upgrade_instance(cloud_redis.UpgradeInstanceRequest(), name='name_value', redis_version='redis_version_value')

@pytest.mark.asyncio
async def test_upgrade_instance_flattened_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.upgrade_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.upgrade_instance(name='name_value', redis_version='redis_version_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].redis_version
        mock_val = 'redis_version_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_upgrade_instance_flattened_error_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.upgrade_instance(cloud_redis.UpgradeInstanceRequest(), name='name_value', redis_version='redis_version_value')

@pytest.mark.parametrize('request_type', [cloud_redis.ImportInstanceRequest, dict])
def test_import_instance(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.import_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.ImportInstanceRequest()
    assert isinstance(response, future.Future)

def test_import_instance_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.import_instance), '__call__') as call:
        client.import_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.ImportInstanceRequest()

@pytest.mark.asyncio
async def test_import_instance_async(transport: str='grpc_asyncio', request_type=cloud_redis.ImportInstanceRequest):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.import_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.ImportInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_import_instance_async_from_dict():
    await test_import_instance_async(request_type=dict)

def test_import_instance_field_headers():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.ImportInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.import_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.import_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_import_instance_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.ImportInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.import_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.import_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_import_instance_flattened():
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.import_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.import_instance(name='name_value', input_config=cloud_redis.InputConfig(gcs_source=cloud_redis.GcsSource(uri='uri_value')))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].input_config
        mock_val = cloud_redis.InputConfig(gcs_source=cloud_redis.GcsSource(uri='uri_value'))
        assert arg == mock_val

def test_import_instance_flattened_error():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.import_instance(cloud_redis.ImportInstanceRequest(), name='name_value', input_config=cloud_redis.InputConfig(gcs_source=cloud_redis.GcsSource(uri='uri_value')))

@pytest.mark.asyncio
async def test_import_instance_flattened_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.import_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.import_instance(name='name_value', input_config=cloud_redis.InputConfig(gcs_source=cloud_redis.GcsSource(uri='uri_value')))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].input_config
        mock_val = cloud_redis.InputConfig(gcs_source=cloud_redis.GcsSource(uri='uri_value'))
        assert arg == mock_val

@pytest.mark.asyncio
async def test_import_instance_flattened_error_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.import_instance(cloud_redis.ImportInstanceRequest(), name='name_value', input_config=cloud_redis.InputConfig(gcs_source=cloud_redis.GcsSource(uri='uri_value')))

@pytest.mark.parametrize('request_type', [cloud_redis.ExportInstanceRequest, dict])
def test_export_instance(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.export_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.export_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.ExportInstanceRequest()
    assert isinstance(response, future.Future)

def test_export_instance_empty_call():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.export_instance), '__call__') as call:
        client.export_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.ExportInstanceRequest()

@pytest.mark.asyncio
async def test_export_instance_async(transport: str='grpc_asyncio', request_type=cloud_redis.ExportInstanceRequest):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.export_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.export_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.ExportInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_export_instance_async_from_dict():
    await test_export_instance_async(request_type=dict)

def test_export_instance_field_headers():
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.ExportInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.export_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.export_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_export_instance_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.ExportInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.export_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.export_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_export_instance_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.export_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.export_instance(name='name_value', output_config=cloud_redis.OutputConfig(gcs_destination=cloud_redis.GcsDestination(uri='uri_value')))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].output_config
        mock_val = cloud_redis.OutputConfig(gcs_destination=cloud_redis.GcsDestination(uri='uri_value'))
        assert arg == mock_val

def test_export_instance_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.export_instance(cloud_redis.ExportInstanceRequest(), name='name_value', output_config=cloud_redis.OutputConfig(gcs_destination=cloud_redis.GcsDestination(uri='uri_value')))

@pytest.mark.asyncio
async def test_export_instance_flattened_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.export_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.export_instance(name='name_value', output_config=cloud_redis.OutputConfig(gcs_destination=cloud_redis.GcsDestination(uri='uri_value')))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].output_config
        mock_val = cloud_redis.OutputConfig(gcs_destination=cloud_redis.GcsDestination(uri='uri_value'))
        assert arg == mock_val

@pytest.mark.asyncio
async def test_export_instance_flattened_error_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.export_instance(cloud_redis.ExportInstanceRequest(), name='name_value', output_config=cloud_redis.OutputConfig(gcs_destination=cloud_redis.GcsDestination(uri='uri_value')))

@pytest.mark.parametrize('request_type', [cloud_redis.FailoverInstanceRequest, dict])
def test_failover_instance(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.failover_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.failover_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.FailoverInstanceRequest()
    assert isinstance(response, future.Future)

def test_failover_instance_empty_call():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.failover_instance), '__call__') as call:
        client.failover_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.FailoverInstanceRequest()

@pytest.mark.asyncio
async def test_failover_instance_async(transport: str='grpc_asyncio', request_type=cloud_redis.FailoverInstanceRequest):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.failover_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.failover_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.FailoverInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_failover_instance_async_from_dict():
    await test_failover_instance_async(request_type=dict)

def test_failover_instance_field_headers():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.FailoverInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.failover_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.failover_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_failover_instance_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.FailoverInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.failover_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.failover_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_failover_instance_flattened():
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.failover_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.failover_instance(name='name_value', data_protection_mode=cloud_redis.FailoverInstanceRequest.DataProtectionMode.LIMITED_DATA_LOSS)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].data_protection_mode
        mock_val = cloud_redis.FailoverInstanceRequest.DataProtectionMode.LIMITED_DATA_LOSS
        assert arg == mock_val

def test_failover_instance_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.failover_instance(cloud_redis.FailoverInstanceRequest(), name='name_value', data_protection_mode=cloud_redis.FailoverInstanceRequest.DataProtectionMode.LIMITED_DATA_LOSS)

@pytest.mark.asyncio
async def test_failover_instance_flattened_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.failover_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.failover_instance(name='name_value', data_protection_mode=cloud_redis.FailoverInstanceRequest.DataProtectionMode.LIMITED_DATA_LOSS)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].data_protection_mode
        mock_val = cloud_redis.FailoverInstanceRequest.DataProtectionMode.LIMITED_DATA_LOSS
        assert arg == mock_val

@pytest.mark.asyncio
async def test_failover_instance_flattened_error_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.failover_instance(cloud_redis.FailoverInstanceRequest(), name='name_value', data_protection_mode=cloud_redis.FailoverInstanceRequest.DataProtectionMode.LIMITED_DATA_LOSS)

@pytest.mark.parametrize('request_type', [cloud_redis.DeleteInstanceRequest, dict])
def test_delete_instance(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.DeleteInstanceRequest()
    assert isinstance(response, future.Future)

def test_delete_instance_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        client.delete_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.DeleteInstanceRequest()

@pytest.mark.asyncio
async def test_delete_instance_async(transport: str='grpc_asyncio', request_type=cloud_redis.DeleteInstanceRequest):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.DeleteInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_instance_async_from_dict():
    await test_delete_instance_async(request_type=dict)

def test_delete_instance_field_headers():
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.DeleteInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_instance_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.DeleteInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_instance_flattened():
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_instance(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_instance_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_instance(cloud_redis.DeleteInstanceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_instance_flattened_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_instance(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_instance_flattened_error_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_instance(cloud_redis.DeleteInstanceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_redis.RescheduleMaintenanceRequest, dict])
def test_reschedule_maintenance(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reschedule_maintenance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.reschedule_maintenance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.RescheduleMaintenanceRequest()
    assert isinstance(response, future.Future)

def test_reschedule_maintenance_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.reschedule_maintenance), '__call__') as call:
        client.reschedule_maintenance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.RescheduleMaintenanceRequest()

@pytest.mark.asyncio
async def test_reschedule_maintenance_async(transport: str='grpc_asyncio', request_type=cloud_redis.RescheduleMaintenanceRequest):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reschedule_maintenance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reschedule_maintenance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_redis.RescheduleMaintenanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_reschedule_maintenance_async_from_dict():
    await test_reschedule_maintenance_async(request_type=dict)

def test_reschedule_maintenance_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.RescheduleMaintenanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.reschedule_maintenance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reschedule_maintenance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_reschedule_maintenance_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_redis.RescheduleMaintenanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.reschedule_maintenance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.reschedule_maintenance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_reschedule_maintenance_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reschedule_maintenance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reschedule_maintenance(name='name_value', reschedule_type=cloud_redis.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE, schedule_time=timestamp_pb2.Timestamp(seconds=751))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].reschedule_type
        mock_val = cloud_redis.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE
        assert arg == mock_val
        assert TimestampRule().to_proto(args[0].schedule_time) == timestamp_pb2.Timestamp(seconds=751)

def test_reschedule_maintenance_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.reschedule_maintenance(cloud_redis.RescheduleMaintenanceRequest(), name='name_value', reschedule_type=cloud_redis.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE, schedule_time=timestamp_pb2.Timestamp(seconds=751))

@pytest.mark.asyncio
async def test_reschedule_maintenance_flattened_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reschedule_maintenance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reschedule_maintenance(name='name_value', reschedule_type=cloud_redis.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE, schedule_time=timestamp_pb2.Timestamp(seconds=751))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].reschedule_type
        mock_val = cloud_redis.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE
        assert arg == mock_val
        assert TimestampRule().to_proto(args[0].schedule_time) == timestamp_pb2.Timestamp(seconds=751)

@pytest.mark.asyncio
async def test_reschedule_maintenance_flattened_error_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.reschedule_maintenance(cloud_redis.RescheduleMaintenanceRequest(), name='name_value', reschedule_type=cloud_redis.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE, schedule_time=timestamp_pb2.Timestamp(seconds=751))

@pytest.mark.parametrize('request_type', [cloud_redis.ListInstancesRequest, dict])
def test_list_instances_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_redis.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_redis.ListInstancesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_instances(request)
    assert isinstance(response, pagers.ListInstancesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_instances_rest_required_fields(request_type=cloud_redis.ListInstancesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudRedisRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_redis.ListInstancesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_redis.ListInstancesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_instances(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_instances_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_instances_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudRedisRestInterceptor())
    client = CloudRedisClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudRedisRestInterceptor, 'post_list_instances') as post, mock.patch.object(transports.CloudRedisRestInterceptor, 'pre_list_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_redis.ListInstancesRequest.pb(cloud_redis.ListInstancesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_redis.ListInstancesResponse.to_json(cloud_redis.ListInstancesResponse())
        request = cloud_redis.ListInstancesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_redis.ListInstancesResponse()
        client.list_instances(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_instances_rest_bad_request(transport: str='rest', request_type=cloud_redis.ListInstancesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_instances(request)

def test_list_instances_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_redis.ListInstancesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_redis.ListInstancesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_instances(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/instances' % client.transport._host, args[1])

def test_list_instances_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_instances(cloud_redis.ListInstancesRequest(), parent='parent_value')

def test_list_instances_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance(), cloud_redis.Instance(), cloud_redis.Instance()], next_page_token='abc'), cloud_redis.ListInstancesResponse(instances=[], next_page_token='def'), cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance()], next_page_token='ghi'), cloud_redis.ListInstancesResponse(instances=[cloud_redis.Instance(), cloud_redis.Instance()]))
        response = response + response
        response = tuple((cloud_redis.ListInstancesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_instances(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_redis.Instance) for i in results))
        pages = list(client.list_instances(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_redis.GetInstanceRequest, dict])
def test_get_instance_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_redis.Instance(name='name_value', display_name='display_name_value', location_id='location_id_value', alternative_location_id='alternative_location_id_value', redis_version='redis_version_value', reserved_ip_range='reserved_ip_range_value', secondary_ip_range='secondary_ip_range_value', host='host_value', port=453, current_location_id='current_location_id_value', state=cloud_redis.Instance.State.CREATING, status_message='status_message_value', tier=cloud_redis.Instance.Tier.BASIC, memory_size_gb=1499, authorized_network='authorized_network_value', persistence_iam_identity='persistence_iam_identity_value', connect_mode=cloud_redis.Instance.ConnectMode.DIRECT_PEERING, auth_enabled=True, transit_encryption_mode=cloud_redis.Instance.TransitEncryptionMode.SERVER_AUTHENTICATION, replica_count=1384, read_endpoint='read_endpoint_value', read_endpoint_port=1920, read_replicas_mode=cloud_redis.Instance.ReadReplicasMode.READ_REPLICAS_DISABLED, customer_managed_key='customer_managed_key_value', suspension_reasons=[cloud_redis.Instance.SuspensionReason.CUSTOMER_MANAGED_KEY_ISSUE], maintenance_version='maintenance_version_value', available_maintenance_versions=['available_maintenance_versions_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_redis.Instance.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_instance(request)
    assert isinstance(response, cloud_redis.Instance)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.location_id == 'location_id_value'
    assert response.alternative_location_id == 'alternative_location_id_value'
    assert response.redis_version == 'redis_version_value'
    assert response.reserved_ip_range == 'reserved_ip_range_value'
    assert response.secondary_ip_range == 'secondary_ip_range_value'
    assert response.host == 'host_value'
    assert response.port == 453
    assert response.current_location_id == 'current_location_id_value'
    assert response.state == cloud_redis.Instance.State.CREATING
    assert response.status_message == 'status_message_value'
    assert response.tier == cloud_redis.Instance.Tier.BASIC
    assert response.memory_size_gb == 1499
    assert response.authorized_network == 'authorized_network_value'
    assert response.persistence_iam_identity == 'persistence_iam_identity_value'
    assert response.connect_mode == cloud_redis.Instance.ConnectMode.DIRECT_PEERING
    assert response.auth_enabled is True
    assert response.transit_encryption_mode == cloud_redis.Instance.TransitEncryptionMode.SERVER_AUTHENTICATION
    assert response.replica_count == 1384
    assert response.read_endpoint == 'read_endpoint_value'
    assert response.read_endpoint_port == 1920
    assert response.read_replicas_mode == cloud_redis.Instance.ReadReplicasMode.READ_REPLICAS_DISABLED
    assert response.customer_managed_key == 'customer_managed_key_value'
    assert response.suspension_reasons == [cloud_redis.Instance.SuspensionReason.CUSTOMER_MANAGED_KEY_ISSUE]
    assert response.maintenance_version == 'maintenance_version_value'
    assert response.available_maintenance_versions == ['available_maintenance_versions_value']

def test_get_instance_rest_required_fields(request_type=cloud_redis.GetInstanceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudRedisRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_redis.Instance()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_redis.Instance.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_instance_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_instance_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudRedisRestInterceptor())
    client = CloudRedisClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudRedisRestInterceptor, 'post_get_instance') as post, mock.patch.object(transports.CloudRedisRestInterceptor, 'pre_get_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_redis.GetInstanceRequest.pb(cloud_redis.GetInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_redis.Instance.to_json(cloud_redis.Instance())
        request = cloud_redis.GetInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_redis.Instance()
        client.get_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_instance_rest_bad_request(transport: str='rest', request_type=cloud_redis.GetInstanceRequest):
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_instance(request)

def test_get_instance_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_redis.Instance()
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_redis.Instance.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/instances/*}' % client.transport._host, args[1])

def test_get_instance_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_instance(cloud_redis.GetInstanceRequest(), name='name_value')

def test_get_instance_rest_error():
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_redis.GetInstanceAuthStringRequest, dict])
def test_get_instance_auth_string_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_redis.InstanceAuthString(auth_string='auth_string_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_redis.InstanceAuthString.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_instance_auth_string(request)
    assert isinstance(response, cloud_redis.InstanceAuthString)
    assert response.auth_string == 'auth_string_value'

def test_get_instance_auth_string_rest_required_fields(request_type=cloud_redis.GetInstanceAuthStringRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudRedisRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_instance_auth_string._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_instance_auth_string._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_redis.InstanceAuthString()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_redis.InstanceAuthString.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_instance_auth_string(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_instance_auth_string_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_instance_auth_string._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_instance_auth_string_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudRedisRestInterceptor())
    client = CloudRedisClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudRedisRestInterceptor, 'post_get_instance_auth_string') as post, mock.patch.object(transports.CloudRedisRestInterceptor, 'pre_get_instance_auth_string') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_redis.GetInstanceAuthStringRequest.pb(cloud_redis.GetInstanceAuthStringRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_redis.InstanceAuthString.to_json(cloud_redis.InstanceAuthString())
        request = cloud_redis.GetInstanceAuthStringRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_redis.InstanceAuthString()
        client.get_instance_auth_string(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_instance_auth_string_rest_bad_request(transport: str='rest', request_type=cloud_redis.GetInstanceAuthStringRequest):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_instance_auth_string(request)

def test_get_instance_auth_string_rest_flattened():
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_redis.InstanceAuthString()
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_redis.InstanceAuthString.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_instance_auth_string(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/instances/*}/authString' % client.transport._host, args[1])

def test_get_instance_auth_string_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_instance_auth_string(cloud_redis.GetInstanceAuthStringRequest(), name='name_value')

def test_get_instance_auth_string_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_redis.CreateInstanceRequest, dict])
def test_create_instance_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['instance'] = {'name': 'name_value', 'display_name': 'display_name_value', 'labels': {}, 'location_id': 'location_id_value', 'alternative_location_id': 'alternative_location_id_value', 'redis_version': 'redis_version_value', 'reserved_ip_range': 'reserved_ip_range_value', 'secondary_ip_range': 'secondary_ip_range_value', 'host': 'host_value', 'port': 453, 'current_location_id': 'current_location_id_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'state': 1, 'status_message': 'status_message_value', 'redis_configs': {}, 'tier': 1, 'memory_size_gb': 1499, 'authorized_network': 'authorized_network_value', 'persistence_iam_identity': 'persistence_iam_identity_value', 'connect_mode': 1, 'auth_enabled': True, 'server_ca_certs': [{'serial_number': 'serial_number_value', 'cert': 'cert_value', 'create_time': {}, 'expire_time': {}, 'sha1_fingerprint': 'sha1_fingerprint_value'}], 'transit_encryption_mode': 1, 'maintenance_policy': {'create_time': {}, 'update_time': {}, 'description': 'description_value', 'weekly_maintenance_window': [{'day': 1, 'start_time': {'hours': 561, 'minutes': 773, 'seconds': 751, 'nanos': 543}, 'duration': {'seconds': 751, 'nanos': 543}}]}, 'maintenance_schedule': {'start_time': {}, 'end_time': {}, 'can_reschedule': True, 'schedule_deadline_time': {}}, 'replica_count': 1384, 'nodes': [{'id': 'id_value', 'zone': 'zone_value'}], 'read_endpoint': 'read_endpoint_value', 'read_endpoint_port': 1920, 'read_replicas_mode': 1, 'customer_managed_key': 'customer_managed_key_value', 'persistence_config': {'persistence_mode': 1, 'rdb_snapshot_period': 3, 'rdb_next_snapshot_time': {}, 'rdb_snapshot_start_time': {}}, 'suspension_reasons': [1], 'maintenance_version': 'maintenance_version_value', 'available_maintenance_versions': ['available_maintenance_versions_value1', 'available_maintenance_versions_value2']}
    test_field = cloud_redis.CreateInstanceRequest.meta.fields['instance']

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
    for (field, value) in request_init['instance'].items():
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
                for i in range(0, len(request_init['instance'][field])):
                    del request_init['instance'][field][i][subfield]
            else:
                del request_init['instance'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_instance(request)
    assert response.operation.name == 'operations/spam'

def test_create_instance_rest_required_fields(request_type=cloud_redis.CreateInstanceRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudRedisRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['instance_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'instanceId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'instanceId' in jsonified_request
    assert jsonified_request['instanceId'] == request_init['instance_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['instanceId'] = 'instance_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_instance._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('instance_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'instanceId' in jsonified_request
    assert jsonified_request['instanceId'] == 'instance_id_value'
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_instance(request)
            expected_params = [('instanceId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_instance_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(('instanceId',)) & set(('parent', 'instanceId', 'instance'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_instance_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudRedisRestInterceptor())
    client = CloudRedisClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudRedisRestInterceptor, 'post_create_instance') as post, mock.patch.object(transports.CloudRedisRestInterceptor, 'pre_create_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_redis.CreateInstanceRequest.pb(cloud_redis.CreateInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_redis.CreateInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_instance_rest_bad_request(transport: str='rest', request_type=cloud_redis.CreateInstanceRequest):
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_instance(request)

def test_create_instance_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', instance_id='instance_id_value', instance=cloud_redis.Instance(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/instances' % client.transport._host, args[1])

def test_create_instance_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_instance(cloud_redis.CreateInstanceRequest(), parent='parent_value', instance_id='instance_id_value', instance=cloud_redis.Instance(name='name_value'))

def test_create_instance_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_redis.UpdateInstanceRequest, dict])
def test_update_instance_rest(request_type):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'instance': {'name': 'projects/sample1/locations/sample2/instances/sample3'}}
    request_init['instance'] = {'name': 'projects/sample1/locations/sample2/instances/sample3', 'display_name': 'display_name_value', 'labels': {}, 'location_id': 'location_id_value', 'alternative_location_id': 'alternative_location_id_value', 'redis_version': 'redis_version_value', 'reserved_ip_range': 'reserved_ip_range_value', 'secondary_ip_range': 'secondary_ip_range_value', 'host': 'host_value', 'port': 453, 'current_location_id': 'current_location_id_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'state': 1, 'status_message': 'status_message_value', 'redis_configs': {}, 'tier': 1, 'memory_size_gb': 1499, 'authorized_network': 'authorized_network_value', 'persistence_iam_identity': 'persistence_iam_identity_value', 'connect_mode': 1, 'auth_enabled': True, 'server_ca_certs': [{'serial_number': 'serial_number_value', 'cert': 'cert_value', 'create_time': {}, 'expire_time': {}, 'sha1_fingerprint': 'sha1_fingerprint_value'}], 'transit_encryption_mode': 1, 'maintenance_policy': {'create_time': {}, 'update_time': {}, 'description': 'description_value', 'weekly_maintenance_window': [{'day': 1, 'start_time': {'hours': 561, 'minutes': 773, 'seconds': 751, 'nanos': 543}, 'duration': {'seconds': 751, 'nanos': 543}}]}, 'maintenance_schedule': {'start_time': {}, 'end_time': {}, 'can_reschedule': True, 'schedule_deadline_time': {}}, 'replica_count': 1384, 'nodes': [{'id': 'id_value', 'zone': 'zone_value'}], 'read_endpoint': 'read_endpoint_value', 'read_endpoint_port': 1920, 'read_replicas_mode': 1, 'customer_managed_key': 'customer_managed_key_value', 'persistence_config': {'persistence_mode': 1, 'rdb_snapshot_period': 3, 'rdb_next_snapshot_time': {}, 'rdb_snapshot_start_time': {}}, 'suspension_reasons': [1], 'maintenance_version': 'maintenance_version_value', 'available_maintenance_versions': ['available_maintenance_versions_value1', 'available_maintenance_versions_value2']}
    test_field = cloud_redis.UpdateInstanceRequest.meta.fields['instance']

    def get_message_fields(field):
        if False:
            for i in range(10):
                print('nop')
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
    for (field, value) in request_init['instance'].items():
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
                for i in range(0, len(request_init['instance'][field])):
                    del request_init['instance'][field][i][subfield]
            else:
                del request_init['instance'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_instance(request)
    assert response.operation.name == 'operations/spam'

def test_update_instance_rest_required_fields(request_type=cloud_redis.UpdateInstanceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudRedisRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_instance._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_instance_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('updateMask', 'instance'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_instance_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudRedisRestInterceptor())
    client = CloudRedisClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudRedisRestInterceptor, 'post_update_instance') as post, mock.patch.object(transports.CloudRedisRestInterceptor, 'pre_update_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_redis.UpdateInstanceRequest.pb(cloud_redis.UpdateInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_redis.UpdateInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_instance_rest_bad_request(transport: str='rest', request_type=cloud_redis.UpdateInstanceRequest):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'instance': {'name': 'projects/sample1/locations/sample2/instances/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_instance(request)

def test_update_instance_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'instance': {'name': 'projects/sample1/locations/sample2/instances/sample3'}}
        mock_args = dict(update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), instance=cloud_redis.Instance(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{instance.name=projects/*/locations/*/instances/*}' % client.transport._host, args[1])

def test_update_instance_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_instance(cloud_redis.UpdateInstanceRequest(), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), instance=cloud_redis.Instance(name='name_value'))

def test_update_instance_rest_error():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_redis.UpgradeInstanceRequest, dict])
def test_upgrade_instance_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.upgrade_instance(request)
    assert response.operation.name == 'operations/spam'

def test_upgrade_instance_rest_required_fields(request_type=cloud_redis.UpgradeInstanceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudRedisRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['redis_version'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).upgrade_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['redisVersion'] = 'redis_version_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).upgrade_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'redisVersion' in jsonified_request
    assert jsonified_request['redisVersion'] == 'redis_version_value'
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.upgrade_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_upgrade_instance_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.upgrade_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'redisVersion'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_upgrade_instance_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudRedisRestInterceptor())
    client = CloudRedisClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudRedisRestInterceptor, 'post_upgrade_instance') as post, mock.patch.object(transports.CloudRedisRestInterceptor, 'pre_upgrade_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_redis.UpgradeInstanceRequest.pb(cloud_redis.UpgradeInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_redis.UpgradeInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.upgrade_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_upgrade_instance_rest_bad_request(transport: str='rest', request_type=cloud_redis.UpgradeInstanceRequest):
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.upgrade_instance(request)

def test_upgrade_instance_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value', redis_version='redis_version_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.upgrade_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/instances/*}:upgrade' % client.transport._host, args[1])

def test_upgrade_instance_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.upgrade_instance(cloud_redis.UpgradeInstanceRequest(), name='name_value', redis_version='redis_version_value')

def test_upgrade_instance_rest_error():
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_redis.ImportInstanceRequest, dict])
def test_import_instance_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.import_instance(request)
    assert response.operation.name == 'operations/spam'

def test_import_instance_rest_required_fields(request_type=cloud_redis.ImportInstanceRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudRedisRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).import_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).import_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.import_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_import_instance_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.import_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'inputConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_import_instance_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudRedisRestInterceptor())
    client = CloudRedisClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudRedisRestInterceptor, 'post_import_instance') as post, mock.patch.object(transports.CloudRedisRestInterceptor, 'pre_import_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_redis.ImportInstanceRequest.pb(cloud_redis.ImportInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_redis.ImportInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.import_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_import_instance_rest_bad_request(transport: str='rest', request_type=cloud_redis.ImportInstanceRequest):
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.import_instance(request)

def test_import_instance_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value', input_config=cloud_redis.InputConfig(gcs_source=cloud_redis.GcsSource(uri='uri_value')))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.import_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/instances/*}:import' % client.transport._host, args[1])

def test_import_instance_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.import_instance(cloud_redis.ImportInstanceRequest(), name='name_value', input_config=cloud_redis.InputConfig(gcs_source=cloud_redis.GcsSource(uri='uri_value')))

def test_import_instance_rest_error():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_redis.ExportInstanceRequest, dict])
def test_export_instance_rest(request_type):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.export_instance(request)
    assert response.operation.name == 'operations/spam'

def test_export_instance_rest_required_fields(request_type=cloud_redis.ExportInstanceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudRedisRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).export_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).export_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.export_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_export_instance_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.export_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'outputConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_export_instance_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudRedisRestInterceptor())
    client = CloudRedisClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudRedisRestInterceptor, 'post_export_instance') as post, mock.patch.object(transports.CloudRedisRestInterceptor, 'pre_export_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_redis.ExportInstanceRequest.pb(cloud_redis.ExportInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_redis.ExportInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.export_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_export_instance_rest_bad_request(transport: str='rest', request_type=cloud_redis.ExportInstanceRequest):
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.export_instance(request)

def test_export_instance_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value', output_config=cloud_redis.OutputConfig(gcs_destination=cloud_redis.GcsDestination(uri='uri_value')))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.export_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/instances/*}:export' % client.transport._host, args[1])

def test_export_instance_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.export_instance(cloud_redis.ExportInstanceRequest(), name='name_value', output_config=cloud_redis.OutputConfig(gcs_destination=cloud_redis.GcsDestination(uri='uri_value')))

def test_export_instance_rest_error():
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_redis.FailoverInstanceRequest, dict])
def test_failover_instance_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.failover_instance(request)
    assert response.operation.name == 'operations/spam'

def test_failover_instance_rest_required_fields(request_type=cloud_redis.FailoverInstanceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudRedisRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).failover_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).failover_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.failover_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_failover_instance_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.failover_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_failover_instance_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudRedisRestInterceptor())
    client = CloudRedisClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudRedisRestInterceptor, 'post_failover_instance') as post, mock.patch.object(transports.CloudRedisRestInterceptor, 'pre_failover_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_redis.FailoverInstanceRequest.pb(cloud_redis.FailoverInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_redis.FailoverInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.failover_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_failover_instance_rest_bad_request(transport: str='rest', request_type=cloud_redis.FailoverInstanceRequest):
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.failover_instance(request)

def test_failover_instance_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value', data_protection_mode=cloud_redis.FailoverInstanceRequest.DataProtectionMode.LIMITED_DATA_LOSS)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.failover_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/instances/*}:failover' % client.transport._host, args[1])

def test_failover_instance_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.failover_instance(cloud_redis.FailoverInstanceRequest(), name='name_value', data_protection_mode=cloud_redis.FailoverInstanceRequest.DataProtectionMode.LIMITED_DATA_LOSS)

def test_failover_instance_rest_error():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_redis.DeleteInstanceRequest, dict])
def test_delete_instance_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_instance(request)
    assert response.operation.name == 'operations/spam'

def test_delete_instance_rest_required_fields(request_type=cloud_redis.DeleteInstanceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudRedisRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_instance_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_instance_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudRedisRestInterceptor())
    client = CloudRedisClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudRedisRestInterceptor, 'post_delete_instance') as post, mock.patch.object(transports.CloudRedisRestInterceptor, 'pre_delete_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_redis.DeleteInstanceRequest.pb(cloud_redis.DeleteInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_redis.DeleteInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_instance_rest_bad_request(transport: str='rest', request_type=cloud_redis.DeleteInstanceRequest):
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_instance(request)

def test_delete_instance_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/instances/*}' % client.transport._host, args[1])

def test_delete_instance_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_instance(cloud_redis.DeleteInstanceRequest(), name='name_value')

def test_delete_instance_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_redis.RescheduleMaintenanceRequest, dict])
def test_reschedule_maintenance_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.reschedule_maintenance(request)
    assert response.operation.name == 'operations/spam'

def test_reschedule_maintenance_rest_required_fields(request_type=cloud_redis.RescheduleMaintenanceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudRedisRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).reschedule_maintenance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).reschedule_maintenance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.reschedule_maintenance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_reschedule_maintenance_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.reschedule_maintenance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'rescheduleType'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_reschedule_maintenance_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudRedisRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudRedisRestInterceptor())
    client = CloudRedisClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudRedisRestInterceptor, 'post_reschedule_maintenance') as post, mock.patch.object(transports.CloudRedisRestInterceptor, 'pre_reschedule_maintenance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_redis.RescheduleMaintenanceRequest.pb(cloud_redis.RescheduleMaintenanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_redis.RescheduleMaintenanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.reschedule_maintenance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_reschedule_maintenance_rest_bad_request(transport: str='rest', request_type=cloud_redis.RescheduleMaintenanceRequest):
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.reschedule_maintenance(request)

def test_reschedule_maintenance_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value', reschedule_type=cloud_redis.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE, schedule_time=timestamp_pb2.Timestamp(seconds=751))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.reschedule_maintenance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/instances/*}:rescheduleMaintenance' % client.transport._host, args[1])

def test_reschedule_maintenance_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.reschedule_maintenance(cloud_redis.RescheduleMaintenanceRequest(), name='name_value', reschedule_type=cloud_redis.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE, schedule_time=timestamp_pb2.Timestamp(seconds=751))

def test_reschedule_maintenance_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.CloudRedisGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CloudRedisGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudRedisClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CloudRedisGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudRedisClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudRedisClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CloudRedisGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudRedisClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudRedisGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CloudRedisClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.CloudRedisGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CloudRedisGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CloudRedisGrpcTransport, transports.CloudRedisGrpcAsyncIOTransport, transports.CloudRedisRestTransport])
def test_transport_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        for i in range(10):
            print('nop')
    transport = CloudRedisClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CloudRedisGrpcTransport)

def test_cloud_redis_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CloudRedisTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_cloud_redis_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.redis_v1.services.cloud_redis.transports.CloudRedisTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CloudRedisTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_instances', 'get_instance', 'get_instance_auth_string', 'create_instance', 'update_instance', 'upgrade_instance', 'import_instance', 'export_instance', 'failover_instance', 'delete_instance', 'reschedule_maintenance', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    with pytest.raises(NotImplementedError):
        transport.operations_client
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_cloud_redis_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.redis_v1.services.cloud_redis.transports.CloudRedisTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudRedisTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_cloud_redis_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.redis_v1.services.cloud_redis.transports.CloudRedisTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudRedisTransport()
        adc.assert_called_once()

def test_cloud_redis_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CloudRedisClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CloudRedisGrpcTransport, transports.CloudRedisGrpcAsyncIOTransport])
def test_cloud_redis_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CloudRedisGrpcTransport, transports.CloudRedisGrpcAsyncIOTransport, transports.CloudRedisRestTransport])
def test_cloud_redis_transport_auth_gdch_credentials(transport_class):
    if False:
        return 10
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CloudRedisGrpcTransport, grpc_helpers), (transports.CloudRedisGrpcAsyncIOTransport, grpc_helpers_async)])
def test_cloud_redis_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('redis.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='redis.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CloudRedisGrpcTransport, transports.CloudRedisGrpcAsyncIOTransport])
def test_cloud_redis_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        while True:
            i = 10
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

def test_cloud_redis_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.CloudRedisRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_cloud_redis_rest_lro_client():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_redis_host_no_port(transport_name):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='redis.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('redis.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://redis.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_redis_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='redis.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('redis.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://redis.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_cloud_redis_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = CloudRedisClient(credentials=creds1, transport=transport_name)
    client2 = CloudRedisClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_instances._session
    session2 = client2.transport.list_instances._session
    assert session1 != session2
    session1 = client1.transport.get_instance._session
    session2 = client2.transport.get_instance._session
    assert session1 != session2
    session1 = client1.transport.get_instance_auth_string._session
    session2 = client2.transport.get_instance_auth_string._session
    assert session1 != session2
    session1 = client1.transport.create_instance._session
    session2 = client2.transport.create_instance._session
    assert session1 != session2
    session1 = client1.transport.update_instance._session
    session2 = client2.transport.update_instance._session
    assert session1 != session2
    session1 = client1.transport.upgrade_instance._session
    session2 = client2.transport.upgrade_instance._session
    assert session1 != session2
    session1 = client1.transport.import_instance._session
    session2 = client2.transport.import_instance._session
    assert session1 != session2
    session1 = client1.transport.export_instance._session
    session2 = client2.transport.export_instance._session
    assert session1 != session2
    session1 = client1.transport.failover_instance._session
    session2 = client2.transport.failover_instance._session
    assert session1 != session2
    session1 = client1.transport.delete_instance._session
    session2 = client2.transport.delete_instance._session
    assert session1 != session2
    session1 = client1.transport.reschedule_maintenance._session
    session2 = client2.transport.reschedule_maintenance._session
    assert session1 != session2

def test_cloud_redis_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudRedisGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_cloud_redis_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudRedisGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CloudRedisGrpcTransport, transports.CloudRedisGrpcAsyncIOTransport])
def test_cloud_redis_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        for i in range(10):
            print('nop')
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

@pytest.mark.parametrize('transport_class', [transports.CloudRedisGrpcTransport, transports.CloudRedisGrpcAsyncIOTransport])
def test_cloud_redis_transport_channel_mtls_with_adc(transport_class):
    if False:
        return 10
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

def test_cloud_redis_grpc_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_cloud_redis_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_instance_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    instance = 'whelk'
    expected = 'projects/{project}/locations/{location}/instances/{instance}'.format(project=project, location=location, instance=instance)
    actual = CloudRedisClient.instance_path(project, location, instance)
    assert expected == actual

def test_parse_instance_path():
    if False:
        return 10
    expected = {'project': 'octopus', 'location': 'oyster', 'instance': 'nudibranch'}
    path = CloudRedisClient.instance_path(**expected)
    actual = CloudRedisClient.parse_instance_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CloudRedisClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'mussel'}
    path = CloudRedisClient.common_billing_account_path(**expected)
    actual = CloudRedisClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CloudRedisClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'nautilus'}
    path = CloudRedisClient.common_folder_path(**expected)
    actual = CloudRedisClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CloudRedisClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'abalone'}
    path = CloudRedisClient.common_organization_path(**expected)
    actual = CloudRedisClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = CloudRedisClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'clam'}
    path = CloudRedisClient.common_project_path(**expected)
    actual = CloudRedisClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CloudRedisClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = CloudRedisClient.common_location_path(**expected)
    actual = CloudRedisClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CloudRedisTransport, '_prep_wrapped_messages') as prep:
        client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CloudRedisTransport, '_prep_wrapped_messages') as prep:
        transport_class = CloudRedisClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_cancel_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.CancelOperationRequest):
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/operations/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.cancel_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.CancelOperationRequest, dict])
def test_cancel_operation_rest(request_type):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/operations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = '{}'
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.cancel_operation(request)
    assert response is None

def test_delete_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.DeleteOperationRequest):
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/operations/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.DeleteOperationRequest, dict])
def test_delete_operation_rest(request_type):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/operations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = '{}'
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_operation(request)
    assert response is None

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/operations/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.GetOperationRequest, dict])
def test_get_operation_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/operations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_operation(request)
    assert isinstance(response, operations_pb2.Operation)

def test_list_operations_rest_bad_request(transport: str='rest', request_type=operations_pb2.ListOperationsRequest):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_operations(request)

@pytest.mark.parametrize('request_type', [operations_pb2.ListOperationsRequest, dict])
def test_list_operations_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.ListOperationsResponse()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_operations(request)
    assert isinstance(response, operations_pb2.ListOperationsResponse)

def test_delete_operation(transport: str='grpc'):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = operations_pb2.DeleteOperationRequest()
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert response is None

@pytest.mark.asyncio
async def test_delete_operation_async(transport: str='grpc'):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = operations_pb2.DeleteOperationRequest()
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert response is None

def test_delete_operation_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = operations_pb2.DeleteOperationRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_operation_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = operations_pb2.DeleteOperationRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

def test_delete_operation_from_dict():
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = operations_pb2.CancelOperationRequest()
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert response is None

@pytest.mark.asyncio
async def test_cancel_operation_async(transport: str='grpc'):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = operations_pb2.CancelOperationRequest()
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert response is None

def test_cancel_operation_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = operations_pb2.CancelOperationRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        client.cancel_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

@pytest.mark.asyncio
async def test_cancel_operation_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = operations_pb2.CancelOperationRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.cancel_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

def test_cancel_operation_from_dict():
    if False:
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = operations_pb2.GetOperationRequest()
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, operations_pb2.Operation)

@pytest.mark.asyncio
async def test_get_operation_async(transport: str='grpc'):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = operations_pb2.GetOperationRequest()
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, operations_pb2.Operation)

def test_get_operation_field_headers():
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = operations_pb2.GetOperationRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        client.get_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

@pytest.mark.asyncio
async def test_get_operation_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = operations_pb2.GetOperationRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        await client.get_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

def test_get_operation_from_dict():
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = operations_pb2.ListOperationsRequest()
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, operations_pb2.ListOperationsResponse)

@pytest.mark.asyncio
async def test_list_operations_async(transport: str='grpc'):
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = operations_pb2.ListOperationsRequest()
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, operations_pb2.ListOperationsResponse)

def test_list_operations_field_headers():
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    request = operations_pb2.ListOperationsRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        client.list_operations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

@pytest.mark.asyncio
async def test_list_operations_field_headers_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = operations_pb2.ListOperationsRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        await client.list_operations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

def test_list_operations_from_dict():
    if False:
        return 10
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = CloudRedisAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = CloudRedisClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CloudRedisClient, transports.CloudRedisGrpcTransport), (CloudRedisAsyncClient, transports.CloudRedisGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)