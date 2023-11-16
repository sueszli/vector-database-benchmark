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
from google.cloud.memcache_v1beta2.services.cloud_memcache import CloudMemcacheAsyncClient, CloudMemcacheClient, pagers, transports
from google.cloud.memcache_v1beta2.types import cloud_memcache

def client_cert_source_callback():
    if False:
        return 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        for i in range(10):
            print('nop')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        while True:
            i = 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert CloudMemcacheClient._get_default_mtls_endpoint(None) is None
    assert CloudMemcacheClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CloudMemcacheClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CloudMemcacheClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CloudMemcacheClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CloudMemcacheClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CloudMemcacheClient, 'grpc'), (CloudMemcacheAsyncClient, 'grpc_asyncio'), (CloudMemcacheClient, 'rest')])
def test_cloud_memcache_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('memcache.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://memcache.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CloudMemcacheGrpcTransport, 'grpc'), (transports.CloudMemcacheGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.CloudMemcacheRestTransport, 'rest')])
def test_cloud_memcache_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(CloudMemcacheClient, 'grpc'), (CloudMemcacheAsyncClient, 'grpc_asyncio'), (CloudMemcacheClient, 'rest')])
def test_cloud_memcache_client_from_service_account_file(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_file') as factory:
        factory.return_value = creds
        client = client_class.from_service_account_file('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        client = client_class.from_service_account_json('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('memcache.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://memcache.googleapis.com')

def test_cloud_memcache_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = CloudMemcacheClient.get_transport_class()
    available_transports = [transports.CloudMemcacheGrpcTransport, transports.CloudMemcacheRestTransport]
    assert transport in available_transports
    transport = CloudMemcacheClient.get_transport_class('grpc')
    assert transport == transports.CloudMemcacheGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudMemcacheClient, transports.CloudMemcacheGrpcTransport, 'grpc'), (CloudMemcacheAsyncClient, transports.CloudMemcacheGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudMemcacheClient, transports.CloudMemcacheRestTransport, 'rest')])
@mock.patch.object(CloudMemcacheClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudMemcacheClient))
@mock.patch.object(CloudMemcacheAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudMemcacheAsyncClient))
def test_cloud_memcache_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(CloudMemcacheClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CloudMemcacheClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CloudMemcacheClient, transports.CloudMemcacheGrpcTransport, 'grpc', 'true'), (CloudMemcacheAsyncClient, transports.CloudMemcacheGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CloudMemcacheClient, transports.CloudMemcacheGrpcTransport, 'grpc', 'false'), (CloudMemcacheAsyncClient, transports.CloudMemcacheGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (CloudMemcacheClient, transports.CloudMemcacheRestTransport, 'rest', 'true'), (CloudMemcacheClient, transports.CloudMemcacheRestTransport, 'rest', 'false')])
@mock.patch.object(CloudMemcacheClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudMemcacheClient))
@mock.patch.object(CloudMemcacheAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudMemcacheAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_cloud_memcache_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [CloudMemcacheClient, CloudMemcacheAsyncClient])
@mock.patch.object(CloudMemcacheClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudMemcacheClient))
@mock.patch.object(CloudMemcacheAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudMemcacheAsyncClient))
def test_cloud_memcache_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudMemcacheClient, transports.CloudMemcacheGrpcTransport, 'grpc'), (CloudMemcacheAsyncClient, transports.CloudMemcacheGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudMemcacheClient, transports.CloudMemcacheRestTransport, 'rest')])
def test_cloud_memcache_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudMemcacheClient, transports.CloudMemcacheGrpcTransport, 'grpc', grpc_helpers), (CloudMemcacheAsyncClient, transports.CloudMemcacheGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (CloudMemcacheClient, transports.CloudMemcacheRestTransport, 'rest', None)])
def test_cloud_memcache_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_cloud_memcache_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.memcache_v1beta2.services.cloud_memcache.transports.CloudMemcacheGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CloudMemcacheClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudMemcacheClient, transports.CloudMemcacheGrpcTransport, 'grpc', grpc_helpers), (CloudMemcacheAsyncClient, transports.CloudMemcacheGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_cloud_memcache_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
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
        create_channel.assert_called_with('memcache.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='memcache.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [cloud_memcache.ListInstancesRequest, dict])
def test_list_instances(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = cloud_memcache.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_instances(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.ListInstancesRequest()
    assert isinstance(response, pagers.ListInstancesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_instances_empty_call():
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        client.list_instances()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.ListInstancesRequest()

@pytest.mark.asyncio
async def test_list_instances_async(transport: str='grpc_asyncio', request_type=cloud_memcache.ListInstancesRequest):
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_memcache.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_instances(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.ListInstancesRequest()
    assert isinstance(response, pagers.ListInstancesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_instances_async_from_dict():
    await test_list_instances_async(request_type=dict)

def test_list_instances_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.ListInstancesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = cloud_memcache.ListInstancesResponse()
        client.list_instances(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_instances_field_headers_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.ListInstancesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_memcache.ListInstancesResponse())
        await client.list_instances(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_instances_flattened():
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = cloud_memcache.ListInstancesResponse()
        client.list_instances(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_instances_flattened_error():
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_instances(cloud_memcache.ListInstancesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_instances_flattened_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = cloud_memcache.ListInstancesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_memcache.ListInstancesResponse())
        response = await client.list_instances(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_instances_flattened_error_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_instances(cloud_memcache.ListInstancesRequest(), parent='parent_value')

def test_list_instances_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.side_effect = (cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance(), cloud_memcache.Instance(), cloud_memcache.Instance()], next_page_token='abc'), cloud_memcache.ListInstancesResponse(resources=[], next_page_token='def'), cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance()], next_page_token='ghi'), cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance(), cloud_memcache.Instance()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_instances(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_memcache.Instance) for i in results))

def test_list_instances_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.side_effect = (cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance(), cloud_memcache.Instance(), cloud_memcache.Instance()], next_page_token='abc'), cloud_memcache.ListInstancesResponse(resources=[], next_page_token='def'), cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance()], next_page_token='ghi'), cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance(), cloud_memcache.Instance()]), RuntimeError)
        pages = list(client.list_instances(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_instances_async_pager():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_instances), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance(), cloud_memcache.Instance(), cloud_memcache.Instance()], next_page_token='abc'), cloud_memcache.ListInstancesResponse(resources=[], next_page_token='def'), cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance()], next_page_token='ghi'), cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance(), cloud_memcache.Instance()]), RuntimeError)
        async_pager = await client.list_instances(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, cloud_memcache.Instance) for i in responses))

@pytest.mark.asyncio
async def test_list_instances_async_pages():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_instances), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance(), cloud_memcache.Instance(), cloud_memcache.Instance()], next_page_token='abc'), cloud_memcache.ListInstancesResponse(resources=[], next_page_token='def'), cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance()], next_page_token='ghi'), cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance(), cloud_memcache.Instance()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_instances(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_memcache.GetInstanceRequest, dict])
def test_get_instance(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = cloud_memcache.Instance(name='name_value', display_name='display_name_value', authorized_network='authorized_network_value', zones=['zones_value'], node_count=1070, memcache_version=cloud_memcache.MemcacheVersion.MEMCACHE_1_5, state=cloud_memcache.Instance.State.CREATING, memcache_full_version='memcache_full_version_value', discovery_endpoint='discovery_endpoint_value', update_available=True)
        response = client.get_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.GetInstanceRequest()
    assert isinstance(response, cloud_memcache.Instance)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.authorized_network == 'authorized_network_value'
    assert response.zones == ['zones_value']
    assert response.node_count == 1070
    assert response.memcache_version == cloud_memcache.MemcacheVersion.MEMCACHE_1_5
    assert response.state == cloud_memcache.Instance.State.CREATING
    assert response.memcache_full_version == 'memcache_full_version_value'
    assert response.discovery_endpoint == 'discovery_endpoint_value'
    assert response.update_available is True

def test_get_instance_empty_call():
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        client.get_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.GetInstanceRequest()

@pytest.mark.asyncio
async def test_get_instance_async(transport: str='grpc_asyncio', request_type=cloud_memcache.GetInstanceRequest):
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_memcache.Instance(name='name_value', display_name='display_name_value', authorized_network='authorized_network_value', zones=['zones_value'], node_count=1070, memcache_version=cloud_memcache.MemcacheVersion.MEMCACHE_1_5, state=cloud_memcache.Instance.State.CREATING, memcache_full_version='memcache_full_version_value', discovery_endpoint='discovery_endpoint_value', update_available=True))
        response = await client.get_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.GetInstanceRequest()
    assert isinstance(response, cloud_memcache.Instance)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.authorized_network == 'authorized_network_value'
    assert response.zones == ['zones_value']
    assert response.node_count == 1070
    assert response.memcache_version == cloud_memcache.MemcacheVersion.MEMCACHE_1_5
    assert response.state == cloud_memcache.Instance.State.CREATING
    assert response.memcache_full_version == 'memcache_full_version_value'
    assert response.discovery_endpoint == 'discovery_endpoint_value'
    assert response.update_available is True

@pytest.mark.asyncio
async def test_get_instance_async_from_dict():
    await test_get_instance_async(request_type=dict)

def test_get_instance_field_headers():
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.GetInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = cloud_memcache.Instance()
        client.get_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_instance_field_headers_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.GetInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_memcache.Instance())
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
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = cloud_memcache.Instance()
        client.get_instance(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_instance_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_instance(cloud_memcache.GetInstanceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_instance_flattened_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = cloud_memcache.Instance()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_memcache.Instance())
        response = await client.get_instance(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_instance_flattened_error_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_instance(cloud_memcache.GetInstanceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_memcache.CreateInstanceRequest, dict])
def test_create_instance(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.CreateInstanceRequest()
    assert isinstance(response, future.Future)

def test_create_instance_empty_call():
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        client.create_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.CreateInstanceRequest()

@pytest.mark.asyncio
async def test_create_instance_async(transport: str='grpc_asyncio', request_type=cloud_memcache.CreateInstanceRequest):
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.CreateInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_instance_async_from_dict():
    await test_create_instance_async(request_type=dict)

def test_create_instance_field_headers():
    if False:
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.CreateInstanceRequest()
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.CreateInstanceRequest()
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
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_instance(parent='parent_value', instance_id='instance_id_value', resource=cloud_memcache.Instance(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].instance_id
        mock_val = 'instance_id_value'
        assert arg == mock_val
        arg = args[0].resource
        mock_val = cloud_memcache.Instance(name='name_value')
        assert arg == mock_val

def test_create_instance_flattened_error():
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_instance(cloud_memcache.CreateInstanceRequest(), parent='parent_value', instance_id='instance_id_value', resource=cloud_memcache.Instance(name='name_value'))

@pytest.mark.asyncio
async def test_create_instance_flattened_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_instance(parent='parent_value', instance_id='instance_id_value', resource=cloud_memcache.Instance(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].instance_id
        mock_val = 'instance_id_value'
        assert arg == mock_val
        arg = args[0].resource
        mock_val = cloud_memcache.Instance(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_instance_flattened_error_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_instance(cloud_memcache.CreateInstanceRequest(), parent='parent_value', instance_id='instance_id_value', resource=cloud_memcache.Instance(name='name_value'))

@pytest.mark.parametrize('request_type', [cloud_memcache.UpdateInstanceRequest, dict])
def test_update_instance(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.UpdateInstanceRequest()
    assert isinstance(response, future.Future)

def test_update_instance_empty_call():
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        client.update_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.UpdateInstanceRequest()

@pytest.mark.asyncio
async def test_update_instance_async(transport: str='grpc_asyncio', request_type=cloud_memcache.UpdateInstanceRequest):
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.UpdateInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_instance_async_from_dict():
    await test_update_instance_async(request_type=dict)

def test_update_instance_field_headers():
    if False:
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.UpdateInstanceRequest()
    request.resource.name = 'name_value'
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_instance_field_headers_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.UpdateInstanceRequest()
    request.resource.name = 'name_value'
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource.name=name_value') in kw['metadata']

def test_update_instance_flattened():
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_instance(update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), resource=cloud_memcache.Instance(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val
        arg = args[0].resource
        mock_val = cloud_memcache.Instance(name='name_value')
        assert arg == mock_val

def test_update_instance_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_instance(cloud_memcache.UpdateInstanceRequest(), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), resource=cloud_memcache.Instance(name='name_value'))

@pytest.mark.asyncio
async def test_update_instance_flattened_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_instance(update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), resource=cloud_memcache.Instance(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val
        arg = args[0].resource
        mock_val = cloud_memcache.Instance(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_instance_flattened_error_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_instance(cloud_memcache.UpdateInstanceRequest(), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), resource=cloud_memcache.Instance(name='name_value'))

@pytest.mark.parametrize('request_type', [cloud_memcache.UpdateParametersRequest, dict])
def test_update_parameters(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_parameters), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_parameters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.UpdateParametersRequest()
    assert isinstance(response, future.Future)

def test_update_parameters_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_parameters), '__call__') as call:
        client.update_parameters()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.UpdateParametersRequest()

@pytest.mark.asyncio
async def test_update_parameters_async(transport: str='grpc_asyncio', request_type=cloud_memcache.UpdateParametersRequest):
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_parameters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_parameters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.UpdateParametersRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_parameters_async_from_dict():
    await test_update_parameters_async(request_type=dict)

def test_update_parameters_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.UpdateParametersRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_parameters), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_parameters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_parameters_field_headers_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.UpdateParametersRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_parameters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_parameters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_parameters_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_parameters), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_parameters(name='name_value', update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), parameters=cloud_memcache.MemcacheParameters(id='id_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val
        arg = args[0].parameters
        mock_val = cloud_memcache.MemcacheParameters(id='id_value')
        assert arg == mock_val

def test_update_parameters_flattened_error():
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_parameters(cloud_memcache.UpdateParametersRequest(), name='name_value', update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), parameters=cloud_memcache.MemcacheParameters(id='id_value'))

@pytest.mark.asyncio
async def test_update_parameters_flattened_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_parameters), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_parameters(name='name_value', update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), parameters=cloud_memcache.MemcacheParameters(id='id_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val
        arg = args[0].parameters
        mock_val = cloud_memcache.MemcacheParameters(id='id_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_parameters_flattened_error_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_parameters(cloud_memcache.UpdateParametersRequest(), name='name_value', update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), parameters=cloud_memcache.MemcacheParameters(id='id_value'))

@pytest.mark.parametrize('request_type', [cloud_memcache.DeleteInstanceRequest, dict])
def test_delete_instance(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.DeleteInstanceRequest()
    assert isinstance(response, future.Future)

def test_delete_instance_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        client.delete_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.DeleteInstanceRequest()

@pytest.mark.asyncio
async def test_delete_instance_async(transport: str='grpc_asyncio', request_type=cloud_memcache.DeleteInstanceRequest):
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.DeleteInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_instance_async_from_dict():
    await test_delete_instance_async(request_type=dict)

def test_delete_instance_field_headers():
    if False:
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.DeleteInstanceRequest()
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.DeleteInstanceRequest()
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
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_instance(cloud_memcache.DeleteInstanceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_instance_flattened_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_instance(cloud_memcache.DeleteInstanceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_memcache.ApplyParametersRequest, dict])
def test_apply_parameters(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.apply_parameters), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.apply_parameters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.ApplyParametersRequest()
    assert isinstance(response, future.Future)

def test_apply_parameters_empty_call():
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.apply_parameters), '__call__') as call:
        client.apply_parameters()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.ApplyParametersRequest()

@pytest.mark.asyncio
async def test_apply_parameters_async(transport: str='grpc_asyncio', request_type=cloud_memcache.ApplyParametersRequest):
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.apply_parameters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.apply_parameters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.ApplyParametersRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_apply_parameters_async_from_dict():
    await test_apply_parameters_async(request_type=dict)

def test_apply_parameters_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.ApplyParametersRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.apply_parameters), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.apply_parameters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_apply_parameters_field_headers_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.ApplyParametersRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.apply_parameters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.apply_parameters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_apply_parameters_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.apply_parameters), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.apply_parameters(name='name_value', node_ids=['node_ids_value'], apply_all=True)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].node_ids
        mock_val = ['node_ids_value']
        assert arg == mock_val
        arg = args[0].apply_all
        mock_val = True
        assert arg == mock_val

def test_apply_parameters_flattened_error():
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.apply_parameters(cloud_memcache.ApplyParametersRequest(), name='name_value', node_ids=['node_ids_value'], apply_all=True)

@pytest.mark.asyncio
async def test_apply_parameters_flattened_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.apply_parameters), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.apply_parameters(name='name_value', node_ids=['node_ids_value'], apply_all=True)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].node_ids
        mock_val = ['node_ids_value']
        assert arg == mock_val
        arg = args[0].apply_all
        mock_val = True
        assert arg == mock_val

@pytest.mark.asyncio
async def test_apply_parameters_flattened_error_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.apply_parameters(cloud_memcache.ApplyParametersRequest(), name='name_value', node_ids=['node_ids_value'], apply_all=True)

@pytest.mark.parametrize('request_type', [cloud_memcache.ApplySoftwareUpdateRequest, dict])
def test_apply_software_update(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.apply_software_update), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.apply_software_update(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.ApplySoftwareUpdateRequest()
    assert isinstance(response, future.Future)

def test_apply_software_update_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.apply_software_update), '__call__') as call:
        client.apply_software_update()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.ApplySoftwareUpdateRequest()

@pytest.mark.asyncio
async def test_apply_software_update_async(transport: str='grpc_asyncio', request_type=cloud_memcache.ApplySoftwareUpdateRequest):
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.apply_software_update), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.apply_software_update(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.ApplySoftwareUpdateRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_apply_software_update_async_from_dict():
    await test_apply_software_update_async(request_type=dict)

def test_apply_software_update_field_headers():
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.ApplySoftwareUpdateRequest()
    request.instance = 'instance_value'
    with mock.patch.object(type(client.transport.apply_software_update), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.apply_software_update(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'instance=instance_value') in kw['metadata']

@pytest.mark.asyncio
async def test_apply_software_update_field_headers_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.ApplySoftwareUpdateRequest()
    request.instance = 'instance_value'
    with mock.patch.object(type(client.transport.apply_software_update), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.apply_software_update(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'instance=instance_value') in kw['metadata']

def test_apply_software_update_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.apply_software_update), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.apply_software_update(instance='instance_value', node_ids=['node_ids_value'], apply_all=True)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].instance
        mock_val = 'instance_value'
        assert arg == mock_val
        arg = args[0].node_ids
        mock_val = ['node_ids_value']
        assert arg == mock_val
        arg = args[0].apply_all
        mock_val = True
        assert arg == mock_val

def test_apply_software_update_flattened_error():
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.apply_software_update(cloud_memcache.ApplySoftwareUpdateRequest(), instance='instance_value', node_ids=['node_ids_value'], apply_all=True)

@pytest.mark.asyncio
async def test_apply_software_update_flattened_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.apply_software_update), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.apply_software_update(instance='instance_value', node_ids=['node_ids_value'], apply_all=True)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].instance
        mock_val = 'instance_value'
        assert arg == mock_val
        arg = args[0].node_ids
        mock_val = ['node_ids_value']
        assert arg == mock_val
        arg = args[0].apply_all
        mock_val = True
        assert arg == mock_val

@pytest.mark.asyncio
async def test_apply_software_update_flattened_error_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.apply_software_update(cloud_memcache.ApplySoftwareUpdateRequest(), instance='instance_value', node_ids=['node_ids_value'], apply_all=True)

@pytest.mark.parametrize('request_type', [cloud_memcache.RescheduleMaintenanceRequest, dict])
def test_reschedule_maintenance(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reschedule_maintenance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.reschedule_maintenance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.RescheduleMaintenanceRequest()
    assert isinstance(response, future.Future)

def test_reschedule_maintenance_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.reschedule_maintenance), '__call__') as call:
        client.reschedule_maintenance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.RescheduleMaintenanceRequest()

@pytest.mark.asyncio
async def test_reschedule_maintenance_async(transport: str='grpc_asyncio', request_type=cloud_memcache.RescheduleMaintenanceRequest):
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reschedule_maintenance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reschedule_maintenance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_memcache.RescheduleMaintenanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_reschedule_maintenance_async_from_dict():
    await test_reschedule_maintenance_async(request_type=dict)

def test_reschedule_maintenance_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.RescheduleMaintenanceRequest()
    request.instance = 'instance_value'
    with mock.patch.object(type(client.transport.reschedule_maintenance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reschedule_maintenance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'instance=instance_value') in kw['metadata']

@pytest.mark.asyncio
async def test_reschedule_maintenance_field_headers_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_memcache.RescheduleMaintenanceRequest()
    request.instance = 'instance_value'
    with mock.patch.object(type(client.transport.reschedule_maintenance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.reschedule_maintenance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'instance=instance_value') in kw['metadata']

def test_reschedule_maintenance_flattened():
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reschedule_maintenance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reschedule_maintenance(instance='instance_value', reschedule_type=cloud_memcache.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE, schedule_time=timestamp_pb2.Timestamp(seconds=751))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].instance
        mock_val = 'instance_value'
        assert arg == mock_val
        arg = args[0].reschedule_type
        mock_val = cloud_memcache.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE
        assert arg == mock_val
        assert TimestampRule().to_proto(args[0].schedule_time) == timestamp_pb2.Timestamp(seconds=751)

def test_reschedule_maintenance_flattened_error():
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.reschedule_maintenance(cloud_memcache.RescheduleMaintenanceRequest(), instance='instance_value', reschedule_type=cloud_memcache.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE, schedule_time=timestamp_pb2.Timestamp(seconds=751))

@pytest.mark.asyncio
async def test_reschedule_maintenance_flattened_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reschedule_maintenance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reschedule_maintenance(instance='instance_value', reschedule_type=cloud_memcache.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE, schedule_time=timestamp_pb2.Timestamp(seconds=751))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].instance
        mock_val = 'instance_value'
        assert arg == mock_val
        arg = args[0].reschedule_type
        mock_val = cloud_memcache.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE
        assert arg == mock_val
        assert TimestampRule().to_proto(args[0].schedule_time) == timestamp_pb2.Timestamp(seconds=751)

@pytest.mark.asyncio
async def test_reschedule_maintenance_flattened_error_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.reschedule_maintenance(cloud_memcache.RescheduleMaintenanceRequest(), instance='instance_value', reschedule_type=cloud_memcache.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE, schedule_time=timestamp_pb2.Timestamp(seconds=751))

@pytest.mark.parametrize('request_type', [cloud_memcache.ListInstancesRequest, dict])
def test_list_instances_rest(request_type):
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_memcache.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_memcache.ListInstancesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_instances(request)
    assert isinstance(response, pagers.ListInstancesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_instances_rest_required_fields(request_type=cloud_memcache.ListInstancesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudMemcacheRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_memcache.ListInstancesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_memcache.ListInstancesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_instances(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_instances_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_instances_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudMemcacheRestInterceptor())
    client = CloudMemcacheClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudMemcacheRestInterceptor, 'post_list_instances') as post, mock.patch.object(transports.CloudMemcacheRestInterceptor, 'pre_list_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_memcache.ListInstancesRequest.pb(cloud_memcache.ListInstancesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_memcache.ListInstancesResponse.to_json(cloud_memcache.ListInstancesResponse())
        request = cloud_memcache.ListInstancesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_memcache.ListInstancesResponse()
        client.list_instances(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_instances_rest_bad_request(transport: str='rest', request_type=cloud_memcache.ListInstancesRequest):
    if False:
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_memcache.ListInstancesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_memcache.ListInstancesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_instances(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/{parent=projects/*/locations/*}/instances' % client.transport._host, args[1])

def test_list_instances_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_instances(cloud_memcache.ListInstancesRequest(), parent='parent_value')

def test_list_instances_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance(), cloud_memcache.Instance(), cloud_memcache.Instance()], next_page_token='abc'), cloud_memcache.ListInstancesResponse(resources=[], next_page_token='def'), cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance()], next_page_token='ghi'), cloud_memcache.ListInstancesResponse(resources=[cloud_memcache.Instance(), cloud_memcache.Instance()]))
        response = response + response
        response = tuple((cloud_memcache.ListInstancesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_instances(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_memcache.Instance) for i in results))
        pages = list(client.list_instances(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_memcache.GetInstanceRequest, dict])
def test_get_instance_rest(request_type):
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_memcache.Instance(name='name_value', display_name='display_name_value', authorized_network='authorized_network_value', zones=['zones_value'], node_count=1070, memcache_version=cloud_memcache.MemcacheVersion.MEMCACHE_1_5, state=cloud_memcache.Instance.State.CREATING, memcache_full_version='memcache_full_version_value', discovery_endpoint='discovery_endpoint_value', update_available=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_memcache.Instance.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_instance(request)
    assert isinstance(response, cloud_memcache.Instance)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.authorized_network == 'authorized_network_value'
    assert response.zones == ['zones_value']
    assert response.node_count == 1070
    assert response.memcache_version == cloud_memcache.MemcacheVersion.MEMCACHE_1_5
    assert response.state == cloud_memcache.Instance.State.CREATING
    assert response.memcache_full_version == 'memcache_full_version_value'
    assert response.discovery_endpoint == 'discovery_endpoint_value'
    assert response.update_available is True

def test_get_instance_rest_required_fields(request_type=cloud_memcache.GetInstanceRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudMemcacheRestTransport
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
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_memcache.Instance()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_memcache.Instance.pb(return_value)
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
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_instance_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudMemcacheRestInterceptor())
    client = CloudMemcacheClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudMemcacheRestInterceptor, 'post_get_instance') as post, mock.patch.object(transports.CloudMemcacheRestInterceptor, 'pre_get_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_memcache.GetInstanceRequest.pb(cloud_memcache.GetInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_memcache.Instance.to_json(cloud_memcache.Instance())
        request = cloud_memcache.GetInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_memcache.Instance()
        client.get_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_instance_rest_bad_request(transport: str='rest', request_type=cloud_memcache.GetInstanceRequest):
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_memcache.Instance()
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_memcache.Instance.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/{name=projects/*/locations/*/instances/*}' % client.transport._host, args[1])

def test_get_instance_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_instance(cloud_memcache.GetInstanceRequest(), name='name_value')

def test_get_instance_rest_error():
    if False:
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_memcache.CreateInstanceRequest, dict])
def test_create_instance_rest(request_type):
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['resource'] = {'name': 'name_value', 'display_name': 'display_name_value', 'labels': {}, 'authorized_network': 'authorized_network_value', 'zones': ['zones_value1', 'zones_value2'], 'node_count': 1070, 'node_config': {'cpu_count': 976, 'memory_size_mb': 1505}, 'memcache_version': 1, 'parameters': {'id': 'id_value', 'params': {}}, 'memcache_nodes': [{'node_id': 'node_id_value', 'zone': 'zone_value', 'state': 1, 'host': 'host_value', 'port': 453, 'parameters': {}, 'update_available': True}], 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'state': 1, 'memcache_full_version': 'memcache_full_version_value', 'instance_messages': [{'code': 1, 'message': 'message_value'}], 'discovery_endpoint': 'discovery_endpoint_value', 'update_available': True, 'maintenance_policy': {'create_time': {}, 'update_time': {}, 'description': 'description_value', 'weekly_maintenance_window': [{'day': 1, 'start_time': {'hours': 561, 'minutes': 773, 'seconds': 751, 'nanos': 543}, 'duration': {'seconds': 751, 'nanos': 543}}]}, 'maintenance_schedule': {'start_time': {}, 'end_time': {}, 'schedule_deadline_time': {}}}
    test_field = cloud_memcache.CreateInstanceRequest.meta.fields['resource']

    def get_message_fields(field):
        if False:
            while True:
                i = 10
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
    for (field, value) in request_init['resource'].items():
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
                for i in range(0, len(request_init['resource'][field])):
                    del request_init['resource'][field][i][subfield]
            else:
                del request_init['resource'][field][subfield]
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

def test_create_instance_rest_required_fields(request_type=cloud_memcache.CreateInstanceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudMemcacheRestTransport
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
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(('instanceId',)) & set(('parent', 'instanceId', 'resource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_instance_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudMemcacheRestInterceptor())
    client = CloudMemcacheClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudMemcacheRestInterceptor, 'post_create_instance') as post, mock.patch.object(transports.CloudMemcacheRestInterceptor, 'pre_create_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_memcache.CreateInstanceRequest.pb(cloud_memcache.CreateInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_memcache.CreateInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_instance_rest_bad_request(transport: str='rest', request_type=cloud_memcache.CreateInstanceRequest):
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', instance_id='instance_id_value', resource=cloud_memcache.Instance(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/{parent=projects/*/locations/*}/instances' % client.transport._host, args[1])

def test_create_instance_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_instance(cloud_memcache.CreateInstanceRequest(), parent='parent_value', instance_id='instance_id_value', resource=cloud_memcache.Instance(name='name_value'))

def test_create_instance_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_memcache.UpdateInstanceRequest, dict])
def test_update_instance_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': {'name': 'projects/sample1/locations/sample2/instances/sample3'}}
    request_init['resource'] = {'name': 'projects/sample1/locations/sample2/instances/sample3', 'display_name': 'display_name_value', 'labels': {}, 'authorized_network': 'authorized_network_value', 'zones': ['zones_value1', 'zones_value2'], 'node_count': 1070, 'node_config': {'cpu_count': 976, 'memory_size_mb': 1505}, 'memcache_version': 1, 'parameters': {'id': 'id_value', 'params': {}}, 'memcache_nodes': [{'node_id': 'node_id_value', 'zone': 'zone_value', 'state': 1, 'host': 'host_value', 'port': 453, 'parameters': {}, 'update_available': True}], 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'state': 1, 'memcache_full_version': 'memcache_full_version_value', 'instance_messages': [{'code': 1, 'message': 'message_value'}], 'discovery_endpoint': 'discovery_endpoint_value', 'update_available': True, 'maintenance_policy': {'create_time': {}, 'update_time': {}, 'description': 'description_value', 'weekly_maintenance_window': [{'day': 1, 'start_time': {'hours': 561, 'minutes': 773, 'seconds': 751, 'nanos': 543}, 'duration': {'seconds': 751, 'nanos': 543}}]}, 'maintenance_schedule': {'start_time': {}, 'end_time': {}, 'schedule_deadline_time': {}}}
    test_field = cloud_memcache.UpdateInstanceRequest.meta.fields['resource']

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
    for (field, value) in request_init['resource'].items():
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
                for i in range(0, len(request_init['resource'][field])):
                    del request_init['resource'][field][i][subfield]
            else:
                del request_init['resource'][field][subfield]
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

def test_update_instance_rest_required_fields(request_type=cloud_memcache.UpdateInstanceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudMemcacheRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_instance._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('updateMask', 'resource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_instance_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudMemcacheRestInterceptor())
    client = CloudMemcacheClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudMemcacheRestInterceptor, 'post_update_instance') as post, mock.patch.object(transports.CloudMemcacheRestInterceptor, 'pre_update_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_memcache.UpdateInstanceRequest.pb(cloud_memcache.UpdateInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_memcache.UpdateInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_instance_rest_bad_request(transport: str='rest', request_type=cloud_memcache.UpdateInstanceRequest):
    if False:
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': {'name': 'projects/sample1/locations/sample2/instances/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_instance(request)

def test_update_instance_rest_flattened():
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'resource': {'name': 'projects/sample1/locations/sample2/instances/sample3'}}
        mock_args = dict(update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), resource=cloud_memcache.Instance(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/{resource.name=projects/*/locations/*/instances/*}' % client.transport._host, args[1])

def test_update_instance_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_instance(cloud_memcache.UpdateInstanceRequest(), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), resource=cloud_memcache.Instance(name='name_value'))

def test_update_instance_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_memcache.UpdateParametersRequest, dict])
def test_update_parameters_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_parameters(request)
    assert response.operation.name == 'operations/spam'

def test_update_parameters_rest_required_fields(request_type=cloud_memcache.UpdateParametersRequest):
    if False:
        return 10
    transport_class = transports.CloudMemcacheRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_parameters._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_parameters._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_parameters(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_parameters_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_parameters._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_parameters_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudMemcacheRestInterceptor())
    client = CloudMemcacheClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudMemcacheRestInterceptor, 'post_update_parameters') as post, mock.patch.object(transports.CloudMemcacheRestInterceptor, 'pre_update_parameters') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_memcache.UpdateParametersRequest.pb(cloud_memcache.UpdateParametersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_memcache.UpdateParametersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_parameters(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_parameters_rest_bad_request(transport: str='rest', request_type=cloud_memcache.UpdateParametersRequest):
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_parameters(request)

def test_update_parameters_rest_flattened():
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value', update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), parameters=cloud_memcache.MemcacheParameters(id='id_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_parameters(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/{name=projects/*/locations/*/instances/*}:updateParameters' % client.transport._host, args[1])

def test_update_parameters_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_parameters(cloud_memcache.UpdateParametersRequest(), name='name_value', update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), parameters=cloud_memcache.MemcacheParameters(id='id_value'))

def test_update_parameters_rest_error():
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_memcache.DeleteInstanceRequest, dict])
def test_delete_instance_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_delete_instance_rest_required_fields(request_type=cloud_memcache.DeleteInstanceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudMemcacheRestTransport
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
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_instance_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudMemcacheRestInterceptor())
    client = CloudMemcacheClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudMemcacheRestInterceptor, 'post_delete_instance') as post, mock.patch.object(transports.CloudMemcacheRestInterceptor, 'pre_delete_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_memcache.DeleteInstanceRequest.pb(cloud_memcache.DeleteInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_memcache.DeleteInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_instance_rest_bad_request(transport: str='rest', request_type=cloud_memcache.DeleteInstanceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        assert path_template.validate('%s/v1beta2/{name=projects/*/locations/*/instances/*}' % client.transport._host, args[1])

def test_delete_instance_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_instance(cloud_memcache.DeleteInstanceRequest(), name='name_value')

def test_delete_instance_rest_error():
    if False:
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_memcache.ApplyParametersRequest, dict])
def test_apply_parameters_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.apply_parameters(request)
    assert response.operation.name == 'operations/spam'

def test_apply_parameters_rest_required_fields(request_type=cloud_memcache.ApplyParametersRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudMemcacheRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).apply_parameters._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).apply_parameters._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.apply_parameters(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_apply_parameters_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.apply_parameters._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_apply_parameters_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudMemcacheRestInterceptor())
    client = CloudMemcacheClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudMemcacheRestInterceptor, 'post_apply_parameters') as post, mock.patch.object(transports.CloudMemcacheRestInterceptor, 'pre_apply_parameters') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_memcache.ApplyParametersRequest.pb(cloud_memcache.ApplyParametersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_memcache.ApplyParametersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.apply_parameters(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_apply_parameters_rest_bad_request(transport: str='rest', request_type=cloud_memcache.ApplyParametersRequest):
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.apply_parameters(request)

def test_apply_parameters_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(name='name_value', node_ids=['node_ids_value'], apply_all=True)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.apply_parameters(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/{name=projects/*/locations/*/instances/*}:applyParameters' % client.transport._host, args[1])

def test_apply_parameters_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.apply_parameters(cloud_memcache.ApplyParametersRequest(), name='name_value', node_ids=['node_ids_value'], apply_all=True)

def test_apply_parameters_rest_error():
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_memcache.ApplySoftwareUpdateRequest, dict])
def test_apply_software_update_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'instance': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.apply_software_update(request)
    assert response.operation.name == 'operations/spam'

def test_apply_software_update_rest_required_fields(request_type=cloud_memcache.ApplySoftwareUpdateRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudMemcacheRestTransport
    request_init = {}
    request_init['instance'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).apply_software_update._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instance'] = 'instance_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).apply_software_update._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'instance' in jsonified_request
    assert jsonified_request['instance'] == 'instance_value'
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.apply_software_update(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_apply_software_update_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.apply_software_update._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('instance',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_apply_software_update_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudMemcacheRestInterceptor())
    client = CloudMemcacheClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudMemcacheRestInterceptor, 'post_apply_software_update') as post, mock.patch.object(transports.CloudMemcacheRestInterceptor, 'pre_apply_software_update') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_memcache.ApplySoftwareUpdateRequest.pb(cloud_memcache.ApplySoftwareUpdateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_memcache.ApplySoftwareUpdateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.apply_software_update(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_apply_software_update_rest_bad_request(transport: str='rest', request_type=cloud_memcache.ApplySoftwareUpdateRequest):
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'instance': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.apply_software_update(request)

def test_apply_software_update_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'instance': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(instance='instance_value', node_ids=['node_ids_value'], apply_all=True)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.apply_software_update(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/{instance=projects/*/locations/*/instances/*}:applySoftwareUpdate' % client.transport._host, args[1])

def test_apply_software_update_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.apply_software_update(cloud_memcache.ApplySoftwareUpdateRequest(), instance='instance_value', node_ids=['node_ids_value'], apply_all=True)

def test_apply_software_update_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_memcache.RescheduleMaintenanceRequest, dict])
def test_reschedule_maintenance_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'instance': 'projects/sample1/locations/sample2/instances/sample3'}
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

def test_reschedule_maintenance_rest_required_fields(request_type=cloud_memcache.RescheduleMaintenanceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudMemcacheRestTransport
    request_init = {}
    request_init['instance'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).reschedule_maintenance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instance'] = 'instance_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).reschedule_maintenance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'instance' in jsonified_request
    assert jsonified_request['instance'] == 'instance_value'
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.reschedule_maintenance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('instance', 'rescheduleType'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_reschedule_maintenance_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudMemcacheRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudMemcacheRestInterceptor())
    client = CloudMemcacheClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudMemcacheRestInterceptor, 'post_reschedule_maintenance') as post, mock.patch.object(transports.CloudMemcacheRestInterceptor, 'pre_reschedule_maintenance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_memcache.RescheduleMaintenanceRequest.pb(cloud_memcache.RescheduleMaintenanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_memcache.RescheduleMaintenanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.reschedule_maintenance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_reschedule_maintenance_rest_bad_request(transport: str='rest', request_type=cloud_memcache.RescheduleMaintenanceRequest):
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'instance': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.reschedule_maintenance(request)

def test_reschedule_maintenance_rest_flattened():
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'instance': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(instance='instance_value', reschedule_type=cloud_memcache.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE, schedule_time=timestamp_pb2.Timestamp(seconds=751))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.reschedule_maintenance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/{instance=projects/*/locations/*/instances/*}:rescheduleMaintenance' % client.transport._host, args[1])

def test_reschedule_maintenance_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.reschedule_maintenance(cloud_memcache.RescheduleMaintenanceRequest(), instance='instance_value', reschedule_type=cloud_memcache.RescheduleMaintenanceRequest.RescheduleType.IMMEDIATE, schedule_time=timestamp_pb2.Timestamp(seconds=751))

def test_reschedule_maintenance_rest_error():
    if False:
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudMemcacheGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CloudMemcacheGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudMemcacheClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CloudMemcacheGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudMemcacheClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudMemcacheClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CloudMemcacheGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudMemcacheClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.CloudMemcacheGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CloudMemcacheClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudMemcacheGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CloudMemcacheGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CloudMemcacheGrpcTransport, transports.CloudMemcacheGrpcAsyncIOTransport, transports.CloudMemcacheRestTransport])
def test_transport_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        while True:
            i = 10
    transport = CloudMemcacheClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CloudMemcacheGrpcTransport)

def test_cloud_memcache_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CloudMemcacheTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_cloud_memcache_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.memcache_v1beta2.services.cloud_memcache.transports.CloudMemcacheTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CloudMemcacheTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_instances', 'get_instance', 'create_instance', 'update_instance', 'update_parameters', 'delete_instance', 'apply_parameters', 'apply_software_update', 'reschedule_maintenance', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_cloud_memcache_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.memcache_v1beta2.services.cloud_memcache.transports.CloudMemcacheTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudMemcacheTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_cloud_memcache_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.memcache_v1beta2.services.cloud_memcache.transports.CloudMemcacheTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudMemcacheTransport()
        adc.assert_called_once()

def test_cloud_memcache_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CloudMemcacheClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CloudMemcacheGrpcTransport, transports.CloudMemcacheGrpcAsyncIOTransport])
def test_cloud_memcache_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CloudMemcacheGrpcTransport, transports.CloudMemcacheGrpcAsyncIOTransport, transports.CloudMemcacheRestTransport])
def test_cloud_memcache_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CloudMemcacheGrpcTransport, grpc_helpers), (transports.CloudMemcacheGrpcAsyncIOTransport, grpc_helpers_async)])
def test_cloud_memcache_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('memcache.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='memcache.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CloudMemcacheGrpcTransport, transports.CloudMemcacheGrpcAsyncIOTransport])
def test_cloud_memcache_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        print('Hello World!')
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

def test_cloud_memcache_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.CloudMemcacheRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_cloud_memcache_rest_lro_client():
    if False:
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_memcache_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='memcache.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('memcache.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://memcache.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_memcache_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='memcache.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('memcache.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://memcache.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_cloud_memcache_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = CloudMemcacheClient(credentials=creds1, transport=transport_name)
    client2 = CloudMemcacheClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_instances._session
    session2 = client2.transport.list_instances._session
    assert session1 != session2
    session1 = client1.transport.get_instance._session
    session2 = client2.transport.get_instance._session
    assert session1 != session2
    session1 = client1.transport.create_instance._session
    session2 = client2.transport.create_instance._session
    assert session1 != session2
    session1 = client1.transport.update_instance._session
    session2 = client2.transport.update_instance._session
    assert session1 != session2
    session1 = client1.transport.update_parameters._session
    session2 = client2.transport.update_parameters._session
    assert session1 != session2
    session1 = client1.transport.delete_instance._session
    session2 = client2.transport.delete_instance._session
    assert session1 != session2
    session1 = client1.transport.apply_parameters._session
    session2 = client2.transport.apply_parameters._session
    assert session1 != session2
    session1 = client1.transport.apply_software_update._session
    session2 = client2.transport.apply_software_update._session
    assert session1 != session2
    session1 = client1.transport.reschedule_maintenance._session
    session2 = client2.transport.reschedule_maintenance._session
    assert session1 != session2

def test_cloud_memcache_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudMemcacheGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_cloud_memcache_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudMemcacheGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CloudMemcacheGrpcTransport, transports.CloudMemcacheGrpcAsyncIOTransport])
def test_cloud_memcache_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        return 10
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

@pytest.mark.parametrize('transport_class', [transports.CloudMemcacheGrpcTransport, transports.CloudMemcacheGrpcAsyncIOTransport])
def test_cloud_memcache_transport_channel_mtls_with_adc(transport_class):
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

def test_cloud_memcache_grpc_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_cloud_memcache_grpc_lro_async_client():
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_instance_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    instance = 'whelk'
    expected = 'projects/{project}/locations/{location}/instances/{instance}'.format(project=project, location=location, instance=instance)
    actual = CloudMemcacheClient.instance_path(project, location, instance)
    assert expected == actual

def test_parse_instance_path():
    if False:
        return 10
    expected = {'project': 'octopus', 'location': 'oyster', 'instance': 'nudibranch'}
    path = CloudMemcacheClient.instance_path(**expected)
    actual = CloudMemcacheClient.parse_instance_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CloudMemcacheClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'mussel'}
    path = CloudMemcacheClient.common_billing_account_path(**expected)
    actual = CloudMemcacheClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CloudMemcacheClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'nautilus'}
    path = CloudMemcacheClient.common_folder_path(**expected)
    actual = CloudMemcacheClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CloudMemcacheClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'abalone'}
    path = CloudMemcacheClient.common_organization_path(**expected)
    actual = CloudMemcacheClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = CloudMemcacheClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'clam'}
    path = CloudMemcacheClient.common_project_path(**expected)
    actual = CloudMemcacheClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CloudMemcacheClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = CloudMemcacheClient.common_location_path(**expected)
    actual = CloudMemcacheClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CloudMemcacheTransport, '_prep_wrapped_messages') as prep:
        client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CloudMemcacheTransport, '_prep_wrapped_messages') as prep:
        transport_class = CloudMemcacheClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = CloudMemcacheAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        for i in range(10):
            print('nop')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = CloudMemcacheClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CloudMemcacheClient, transports.CloudMemcacheGrpcTransport), (CloudMemcacheAsyncClient, transports.CloudMemcacheGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)