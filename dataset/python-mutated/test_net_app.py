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
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.netapp_v1.services.net_app import NetAppAsyncClient, NetAppClient, pagers, transports
from google.cloud.netapp_v1.types import active_directory as gcn_active_directory
from google.cloud.netapp_v1.types import active_directory
from google.cloud.netapp_v1.types import cloud_netapp_service, common, kms
from google.cloud.netapp_v1.types import replication
from google.cloud.netapp_v1.types import replication as gcn_replication
from google.cloud.netapp_v1.types import snapshot
from google.cloud.netapp_v1.types import snapshot as gcn_snapshot
from google.cloud.netapp_v1.types import storage_pool
from google.cloud.netapp_v1.types import storage_pool as gcn_storage_pool
from google.cloud.netapp_v1.types import volume
from google.cloud.netapp_v1.types import volume as gcn_volume

def client_cert_source_callback():
    if False:
        return 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        while True:
            i = 10
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert NetAppClient._get_default_mtls_endpoint(None) is None
    assert NetAppClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert NetAppClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert NetAppClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert NetAppClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert NetAppClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(NetAppClient, 'grpc'), (NetAppAsyncClient, 'grpc_asyncio'), (NetAppClient, 'rest')])
def test_net_app_client_from_service_account_info(client_class, transport_name):
    if False:
        while True:
            i = 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('netapp.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://netapp.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.NetAppGrpcTransport, 'grpc'), (transports.NetAppGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.NetAppRestTransport, 'rest')])
def test_net_app_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(NetAppClient, 'grpc'), (NetAppAsyncClient, 'grpc_asyncio'), (NetAppClient, 'rest')])
def test_net_app_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('netapp.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://netapp.googleapis.com')

def test_net_app_client_get_transport_class():
    if False:
        return 10
    transport = NetAppClient.get_transport_class()
    available_transports = [transports.NetAppGrpcTransport, transports.NetAppRestTransport]
    assert transport in available_transports
    transport = NetAppClient.get_transport_class('grpc')
    assert transport == transports.NetAppGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(NetAppClient, transports.NetAppGrpcTransport, 'grpc'), (NetAppAsyncClient, transports.NetAppGrpcAsyncIOTransport, 'grpc_asyncio'), (NetAppClient, transports.NetAppRestTransport, 'rest')])
@mock.patch.object(NetAppClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetAppClient))
@mock.patch.object(NetAppAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetAppAsyncClient))
def test_net_app_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(NetAppClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(NetAppClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(NetAppClient, transports.NetAppGrpcTransport, 'grpc', 'true'), (NetAppAsyncClient, transports.NetAppGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (NetAppClient, transports.NetAppGrpcTransport, 'grpc', 'false'), (NetAppAsyncClient, transports.NetAppGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (NetAppClient, transports.NetAppRestTransport, 'rest', 'true'), (NetAppClient, transports.NetAppRestTransport, 'rest', 'false')])
@mock.patch.object(NetAppClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetAppClient))
@mock.patch.object(NetAppAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetAppAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_net_app_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [NetAppClient, NetAppAsyncClient])
@mock.patch.object(NetAppClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetAppClient))
@mock.patch.object(NetAppAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetAppAsyncClient))
def test_net_app_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        i = 10
        return i + 15
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(NetAppClient, transports.NetAppGrpcTransport, 'grpc'), (NetAppAsyncClient, transports.NetAppGrpcAsyncIOTransport, 'grpc_asyncio'), (NetAppClient, transports.NetAppRestTransport, 'rest')])
def test_net_app_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(NetAppClient, transports.NetAppGrpcTransport, 'grpc', grpc_helpers), (NetAppAsyncClient, transports.NetAppGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (NetAppClient, transports.NetAppRestTransport, 'rest', None)])
def test_net_app_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_net_app_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.netapp_v1.services.net_app.transports.NetAppGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = NetAppClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(NetAppClient, transports.NetAppGrpcTransport, 'grpc', grpc_helpers), (NetAppAsyncClient, transports.NetAppGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_net_app_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
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
        create_channel.assert_called_with('netapp.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='netapp.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [storage_pool.ListStoragePoolsRequest, dict])
def test_list_storage_pools(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_storage_pools), '__call__') as call:
        call.return_value = storage_pool.ListStoragePoolsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_storage_pools(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == storage_pool.ListStoragePoolsRequest()
    assert isinstance(response, pagers.ListStoragePoolsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_storage_pools_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_storage_pools), '__call__') as call:
        client.list_storage_pools()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == storage_pool.ListStoragePoolsRequest()

@pytest.mark.asyncio
async def test_list_storage_pools_async(transport: str='grpc_asyncio', request_type=storage_pool.ListStoragePoolsRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_storage_pools), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(storage_pool.ListStoragePoolsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_storage_pools(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == storage_pool.ListStoragePoolsRequest()
    assert isinstance(response, pagers.ListStoragePoolsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_storage_pools_async_from_dict():
    await test_list_storage_pools_async(request_type=dict)

def test_list_storage_pools_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = storage_pool.ListStoragePoolsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_storage_pools), '__call__') as call:
        call.return_value = storage_pool.ListStoragePoolsResponse()
        client.list_storage_pools(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_storage_pools_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = storage_pool.ListStoragePoolsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_storage_pools), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(storage_pool.ListStoragePoolsResponse())
        await client.list_storage_pools(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_storage_pools_flattened():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_storage_pools), '__call__') as call:
        call.return_value = storage_pool.ListStoragePoolsResponse()
        client.list_storage_pools(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_storage_pools_flattened_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_storage_pools(storage_pool.ListStoragePoolsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_storage_pools_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_storage_pools), '__call__') as call:
        call.return_value = storage_pool.ListStoragePoolsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(storage_pool.ListStoragePoolsResponse())
        response = await client.list_storage_pools(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_storage_pools_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_storage_pools(storage_pool.ListStoragePoolsRequest(), parent='parent_value')

def test_list_storage_pools_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_storage_pools), '__call__') as call:
        call.side_effect = (storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool(), storage_pool.StoragePool(), storage_pool.StoragePool()], next_page_token='abc'), storage_pool.ListStoragePoolsResponse(storage_pools=[], next_page_token='def'), storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool()], next_page_token='ghi'), storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool(), storage_pool.StoragePool()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_storage_pools(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, storage_pool.StoragePool) for i in results))

def test_list_storage_pools_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_storage_pools), '__call__') as call:
        call.side_effect = (storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool(), storage_pool.StoragePool(), storage_pool.StoragePool()], next_page_token='abc'), storage_pool.ListStoragePoolsResponse(storage_pools=[], next_page_token='def'), storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool()], next_page_token='ghi'), storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool(), storage_pool.StoragePool()]), RuntimeError)
        pages = list(client.list_storage_pools(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_storage_pools_async_pager():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_storage_pools), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool(), storage_pool.StoragePool(), storage_pool.StoragePool()], next_page_token='abc'), storage_pool.ListStoragePoolsResponse(storage_pools=[], next_page_token='def'), storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool()], next_page_token='ghi'), storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool(), storage_pool.StoragePool()]), RuntimeError)
        async_pager = await client.list_storage_pools(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, storage_pool.StoragePool) for i in responses))

@pytest.mark.asyncio
async def test_list_storage_pools_async_pages():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_storage_pools), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool(), storage_pool.StoragePool(), storage_pool.StoragePool()], next_page_token='abc'), storage_pool.ListStoragePoolsResponse(storage_pools=[], next_page_token='def'), storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool()], next_page_token='ghi'), storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool(), storage_pool.StoragePool()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_storage_pools(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [gcn_storage_pool.CreateStoragePoolRequest, dict])
def test_create_storage_pool(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_storage_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_storage_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_storage_pool.CreateStoragePoolRequest()
    assert isinstance(response, future.Future)

def test_create_storage_pool_empty_call():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_storage_pool), '__call__') as call:
        client.create_storage_pool()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_storage_pool.CreateStoragePoolRequest()

@pytest.mark.asyncio
async def test_create_storage_pool_async(transport: str='grpc_asyncio', request_type=gcn_storage_pool.CreateStoragePoolRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_storage_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_storage_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_storage_pool.CreateStoragePoolRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_storage_pool_async_from_dict():
    await test_create_storage_pool_async(request_type=dict)

def test_create_storage_pool_field_headers():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_storage_pool.CreateStoragePoolRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_storage_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_storage_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_storage_pool_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_storage_pool.CreateStoragePoolRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_storage_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_storage_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_storage_pool_flattened():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_storage_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_storage_pool(parent='parent_value', storage_pool=gcn_storage_pool.StoragePool(name='name_value'), storage_pool_id='storage_pool_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].storage_pool
        mock_val = gcn_storage_pool.StoragePool(name='name_value')
        assert arg == mock_val
        arg = args[0].storage_pool_id
        mock_val = 'storage_pool_id_value'
        assert arg == mock_val

def test_create_storage_pool_flattened_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_storage_pool(gcn_storage_pool.CreateStoragePoolRequest(), parent='parent_value', storage_pool=gcn_storage_pool.StoragePool(name='name_value'), storage_pool_id='storage_pool_id_value')

@pytest.mark.asyncio
async def test_create_storage_pool_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_storage_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_storage_pool(parent='parent_value', storage_pool=gcn_storage_pool.StoragePool(name='name_value'), storage_pool_id='storage_pool_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].storage_pool
        mock_val = gcn_storage_pool.StoragePool(name='name_value')
        assert arg == mock_val
        arg = args[0].storage_pool_id
        mock_val = 'storage_pool_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_storage_pool_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_storage_pool(gcn_storage_pool.CreateStoragePoolRequest(), parent='parent_value', storage_pool=gcn_storage_pool.StoragePool(name='name_value'), storage_pool_id='storage_pool_id_value')

@pytest.mark.parametrize('request_type', [storage_pool.GetStoragePoolRequest, dict])
def test_get_storage_pool(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_storage_pool), '__call__') as call:
        call.return_value = storage_pool.StoragePool(name='name_value', service_level=common.ServiceLevel.PREMIUM, capacity_gib=1247, volume_capacity_gib=2006, volume_count=1312, state=storage_pool.StoragePool.State.READY, state_details='state_details_value', description='description_value', network='network_value', active_directory='active_directory_value', kms_config='kms_config_value', ldap_enabled=True, psa_range='psa_range_value', encryption_type=common.EncryptionType.SERVICE_MANAGED, global_access_allowed=True)
        response = client.get_storage_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == storage_pool.GetStoragePoolRequest()
    assert isinstance(response, storage_pool.StoragePool)
    assert response.name == 'name_value'
    assert response.service_level == common.ServiceLevel.PREMIUM
    assert response.capacity_gib == 1247
    assert response.volume_capacity_gib == 2006
    assert response.volume_count == 1312
    assert response.state == storage_pool.StoragePool.State.READY
    assert response.state_details == 'state_details_value'
    assert response.description == 'description_value'
    assert response.network == 'network_value'
    assert response.active_directory == 'active_directory_value'
    assert response.kms_config == 'kms_config_value'
    assert response.ldap_enabled is True
    assert response.psa_range == 'psa_range_value'
    assert response.encryption_type == common.EncryptionType.SERVICE_MANAGED
    assert response.global_access_allowed is True

def test_get_storage_pool_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_storage_pool), '__call__') as call:
        client.get_storage_pool()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == storage_pool.GetStoragePoolRequest()

@pytest.mark.asyncio
async def test_get_storage_pool_async(transport: str='grpc_asyncio', request_type=storage_pool.GetStoragePoolRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_storage_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(storage_pool.StoragePool(name='name_value', service_level=common.ServiceLevel.PREMIUM, capacity_gib=1247, volume_capacity_gib=2006, volume_count=1312, state=storage_pool.StoragePool.State.READY, state_details='state_details_value', description='description_value', network='network_value', active_directory='active_directory_value', kms_config='kms_config_value', ldap_enabled=True, psa_range='psa_range_value', encryption_type=common.EncryptionType.SERVICE_MANAGED, global_access_allowed=True))
        response = await client.get_storage_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == storage_pool.GetStoragePoolRequest()
    assert isinstance(response, storage_pool.StoragePool)
    assert response.name == 'name_value'
    assert response.service_level == common.ServiceLevel.PREMIUM
    assert response.capacity_gib == 1247
    assert response.volume_capacity_gib == 2006
    assert response.volume_count == 1312
    assert response.state == storage_pool.StoragePool.State.READY
    assert response.state_details == 'state_details_value'
    assert response.description == 'description_value'
    assert response.network == 'network_value'
    assert response.active_directory == 'active_directory_value'
    assert response.kms_config == 'kms_config_value'
    assert response.ldap_enabled is True
    assert response.psa_range == 'psa_range_value'
    assert response.encryption_type == common.EncryptionType.SERVICE_MANAGED
    assert response.global_access_allowed is True

@pytest.mark.asyncio
async def test_get_storage_pool_async_from_dict():
    await test_get_storage_pool_async(request_type=dict)

def test_get_storage_pool_field_headers():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = storage_pool.GetStoragePoolRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_storage_pool), '__call__') as call:
        call.return_value = storage_pool.StoragePool()
        client.get_storage_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_storage_pool_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = storage_pool.GetStoragePoolRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_storage_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(storage_pool.StoragePool())
        await client.get_storage_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_storage_pool_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_storage_pool), '__call__') as call:
        call.return_value = storage_pool.StoragePool()
        client.get_storage_pool(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_storage_pool_flattened_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_storage_pool(storage_pool.GetStoragePoolRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_storage_pool_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_storage_pool), '__call__') as call:
        call.return_value = storage_pool.StoragePool()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(storage_pool.StoragePool())
        response = await client.get_storage_pool(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_storage_pool_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_storage_pool(storage_pool.GetStoragePoolRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_storage_pool.UpdateStoragePoolRequest, dict])
def test_update_storage_pool(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_storage_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_storage_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_storage_pool.UpdateStoragePoolRequest()
    assert isinstance(response, future.Future)

def test_update_storage_pool_empty_call():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_storage_pool), '__call__') as call:
        client.update_storage_pool()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_storage_pool.UpdateStoragePoolRequest()

@pytest.mark.asyncio
async def test_update_storage_pool_async(transport: str='grpc_asyncio', request_type=gcn_storage_pool.UpdateStoragePoolRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_storage_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_storage_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_storage_pool.UpdateStoragePoolRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_storage_pool_async_from_dict():
    await test_update_storage_pool_async(request_type=dict)

def test_update_storage_pool_field_headers():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_storage_pool.UpdateStoragePoolRequest()
    request.storage_pool.name = 'name_value'
    with mock.patch.object(type(client.transport.update_storage_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_storage_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'storage_pool.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_storage_pool_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_storage_pool.UpdateStoragePoolRequest()
    request.storage_pool.name = 'name_value'
    with mock.patch.object(type(client.transport.update_storage_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_storage_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'storage_pool.name=name_value') in kw['metadata']

def test_update_storage_pool_flattened():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_storage_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_storage_pool(storage_pool=gcn_storage_pool.StoragePool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].storage_pool
        mock_val = gcn_storage_pool.StoragePool(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_storage_pool_flattened_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_storage_pool(gcn_storage_pool.UpdateStoragePoolRequest(), storage_pool=gcn_storage_pool.StoragePool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_storage_pool_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_storage_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_storage_pool(storage_pool=gcn_storage_pool.StoragePool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].storage_pool
        mock_val = gcn_storage_pool.StoragePool(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_storage_pool_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_storage_pool(gcn_storage_pool.UpdateStoragePoolRequest(), storage_pool=gcn_storage_pool.StoragePool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [storage_pool.DeleteStoragePoolRequest, dict])
def test_delete_storage_pool(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_storage_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_storage_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == storage_pool.DeleteStoragePoolRequest()
    assert isinstance(response, future.Future)

def test_delete_storage_pool_empty_call():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_storage_pool), '__call__') as call:
        client.delete_storage_pool()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == storage_pool.DeleteStoragePoolRequest()

@pytest.mark.asyncio
async def test_delete_storage_pool_async(transport: str='grpc_asyncio', request_type=storage_pool.DeleteStoragePoolRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_storage_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_storage_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == storage_pool.DeleteStoragePoolRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_storage_pool_async_from_dict():
    await test_delete_storage_pool_async(request_type=dict)

def test_delete_storage_pool_field_headers():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = storage_pool.DeleteStoragePoolRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_storage_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_storage_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_storage_pool_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = storage_pool.DeleteStoragePoolRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_storage_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_storage_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_storage_pool_flattened():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_storage_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_storage_pool(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_storage_pool_flattened_error():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_storage_pool(storage_pool.DeleteStoragePoolRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_storage_pool_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_storage_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_storage_pool(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_storage_pool_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_storage_pool(storage_pool.DeleteStoragePoolRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [volume.ListVolumesRequest, dict])
def test_list_volumes(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.return_value = volume.ListVolumesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_volumes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.ListVolumesRequest()
    assert isinstance(response, pagers.ListVolumesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_volumes_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        client.list_volumes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.ListVolumesRequest()

@pytest.mark.asyncio
async def test_list_volumes_async(transport: str='grpc_asyncio', request_type=volume.ListVolumesRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.ListVolumesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_volumes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.ListVolumesRequest()
    assert isinstance(response, pagers.ListVolumesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_volumes_async_from_dict():
    await test_list_volumes_async(request_type=dict)

def test_list_volumes_field_headers():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.ListVolumesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.return_value = volume.ListVolumesResponse()
        client.list_volumes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_volumes_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.ListVolumesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.ListVolumesResponse())
        await client.list_volumes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_volumes_flattened():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.return_value = volume.ListVolumesResponse()
        client.list_volumes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_volumes_flattened_error():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_volumes(volume.ListVolumesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_volumes_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.return_value = volume.ListVolumesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.ListVolumesResponse())
        response = await client.list_volumes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_volumes_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_volumes(volume.ListVolumesRequest(), parent='parent_value')

def test_list_volumes_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.side_effect = (volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume(), volume.Volume()], next_page_token='abc'), volume.ListVolumesResponse(volumes=[], next_page_token='def'), volume.ListVolumesResponse(volumes=[volume.Volume()], next_page_token='ghi'), volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_volumes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, volume.Volume) for i in results))

def test_list_volumes_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_volumes), '__call__') as call:
        call.side_effect = (volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume(), volume.Volume()], next_page_token='abc'), volume.ListVolumesResponse(volumes=[], next_page_token='def'), volume.ListVolumesResponse(volumes=[volume.Volume()], next_page_token='ghi'), volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume()]), RuntimeError)
        pages = list(client.list_volumes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_volumes_async_pager():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_volumes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume(), volume.Volume()], next_page_token='abc'), volume.ListVolumesResponse(volumes=[], next_page_token='def'), volume.ListVolumesResponse(volumes=[volume.Volume()], next_page_token='ghi'), volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume()]), RuntimeError)
        async_pager = await client.list_volumes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, volume.Volume) for i in responses))

@pytest.mark.asyncio
async def test_list_volumes_async_pages():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_volumes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume(), volume.Volume()], next_page_token='abc'), volume.ListVolumesResponse(volumes=[], next_page_token='def'), volume.ListVolumesResponse(volumes=[volume.Volume()], next_page_token='ghi'), volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_volumes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [volume.GetVolumeRequest, dict])
def test_get_volume(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_volume), '__call__') as call:
        call.return_value = volume.Volume(name='name_value', state=volume.Volume.State.READY, state_details='state_details_value', share_name='share_name_value', psa_range='psa_range_value', storage_pool='storage_pool_value', network='network_value', service_level=common.ServiceLevel.PREMIUM, capacity_gib=1247, protocols=[volume.Protocols.NFSV3], smb_settings=[volume.SMBSettings.ENCRYPT_DATA], unix_permissions='unix_permissions_value', description='description_value', snap_reserve=0.1293, snapshot_directory=True, used_gib=834, security_style=volume.SecurityStyle.NTFS, kerberos_enabled=True, ldap_enabled=True, active_directory='active_directory_value', kms_config='kms_config_value', encryption_type=common.EncryptionType.SERVICE_MANAGED, has_replication=True, restricted_actions=[volume.RestrictedAction.DELETE])
        response = client.get_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.GetVolumeRequest()
    assert isinstance(response, volume.Volume)
    assert response.name == 'name_value'
    assert response.state == volume.Volume.State.READY
    assert response.state_details == 'state_details_value'
    assert response.share_name == 'share_name_value'
    assert response.psa_range == 'psa_range_value'
    assert response.storage_pool == 'storage_pool_value'
    assert response.network == 'network_value'
    assert response.service_level == common.ServiceLevel.PREMIUM
    assert response.capacity_gib == 1247
    assert response.protocols == [volume.Protocols.NFSV3]
    assert response.smb_settings == [volume.SMBSettings.ENCRYPT_DATA]
    assert response.unix_permissions == 'unix_permissions_value'
    assert response.description == 'description_value'
    assert math.isclose(response.snap_reserve, 0.1293, rel_tol=1e-06)
    assert response.snapshot_directory is True
    assert response.used_gib == 834
    assert response.security_style == volume.SecurityStyle.NTFS
    assert response.kerberos_enabled is True
    assert response.ldap_enabled is True
    assert response.active_directory == 'active_directory_value'
    assert response.kms_config == 'kms_config_value'
    assert response.encryption_type == common.EncryptionType.SERVICE_MANAGED
    assert response.has_replication is True
    assert response.restricted_actions == [volume.RestrictedAction.DELETE]

def test_get_volume_empty_call():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_volume), '__call__') as call:
        client.get_volume()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.GetVolumeRequest()

@pytest.mark.asyncio
async def test_get_volume_async(transport: str='grpc_asyncio', request_type=volume.GetVolumeRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.Volume(name='name_value', state=volume.Volume.State.READY, state_details='state_details_value', share_name='share_name_value', psa_range='psa_range_value', storage_pool='storage_pool_value', network='network_value', service_level=common.ServiceLevel.PREMIUM, capacity_gib=1247, protocols=[volume.Protocols.NFSV3], smb_settings=[volume.SMBSettings.ENCRYPT_DATA], unix_permissions='unix_permissions_value', description='description_value', snap_reserve=0.1293, snapshot_directory=True, used_gib=834, security_style=volume.SecurityStyle.NTFS, kerberos_enabled=True, ldap_enabled=True, active_directory='active_directory_value', kms_config='kms_config_value', encryption_type=common.EncryptionType.SERVICE_MANAGED, has_replication=True, restricted_actions=[volume.RestrictedAction.DELETE]))
        response = await client.get_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.GetVolumeRequest()
    assert isinstance(response, volume.Volume)
    assert response.name == 'name_value'
    assert response.state == volume.Volume.State.READY
    assert response.state_details == 'state_details_value'
    assert response.share_name == 'share_name_value'
    assert response.psa_range == 'psa_range_value'
    assert response.storage_pool == 'storage_pool_value'
    assert response.network == 'network_value'
    assert response.service_level == common.ServiceLevel.PREMIUM
    assert response.capacity_gib == 1247
    assert response.protocols == [volume.Protocols.NFSV3]
    assert response.smb_settings == [volume.SMBSettings.ENCRYPT_DATA]
    assert response.unix_permissions == 'unix_permissions_value'
    assert response.description == 'description_value'
    assert math.isclose(response.snap_reserve, 0.1293, rel_tol=1e-06)
    assert response.snapshot_directory is True
    assert response.used_gib == 834
    assert response.security_style == volume.SecurityStyle.NTFS
    assert response.kerberos_enabled is True
    assert response.ldap_enabled is True
    assert response.active_directory == 'active_directory_value'
    assert response.kms_config == 'kms_config_value'
    assert response.encryption_type == common.EncryptionType.SERVICE_MANAGED
    assert response.has_replication is True
    assert response.restricted_actions == [volume.RestrictedAction.DELETE]

@pytest.mark.asyncio
async def test_get_volume_async_from_dict():
    await test_get_volume_async(request_type=dict)

def test_get_volume_field_headers():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.GetVolumeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_volume), '__call__') as call:
        call.return_value = volume.Volume()
        client.get_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_volume_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.GetVolumeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.Volume())
        await client.get_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_volume_flattened():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_volume), '__call__') as call:
        call.return_value = volume.Volume()
        client.get_volume(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_volume_flattened_error():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_volume(volume.GetVolumeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_volume_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_volume), '__call__') as call:
        call.return_value = volume.Volume()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(volume.Volume())
        response = await client.get_volume(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_volume_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_volume(volume.GetVolumeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_volume.CreateVolumeRequest, dict])
def test_create_volume(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_volume.CreateVolumeRequest()
    assert isinstance(response, future.Future)

def test_create_volume_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_volume), '__call__') as call:
        client.create_volume()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_volume.CreateVolumeRequest()

@pytest.mark.asyncio
async def test_create_volume_async(transport: str='grpc_asyncio', request_type=gcn_volume.CreateVolumeRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_volume.CreateVolumeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_volume_async_from_dict():
    await test_create_volume_async(request_type=dict)

def test_create_volume_field_headers():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_volume.CreateVolumeRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_volume_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_volume.CreateVolumeRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_volume_flattened():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_volume(parent='parent_value', volume=gcn_volume.Volume(name='name_value'), volume_id='volume_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].volume
        mock_val = gcn_volume.Volume(name='name_value')
        assert arg == mock_val
        arg = args[0].volume_id
        mock_val = 'volume_id_value'
        assert arg == mock_val

def test_create_volume_flattened_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_volume(gcn_volume.CreateVolumeRequest(), parent='parent_value', volume=gcn_volume.Volume(name='name_value'), volume_id='volume_id_value')

@pytest.mark.asyncio
async def test_create_volume_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_volume(parent='parent_value', volume=gcn_volume.Volume(name='name_value'), volume_id='volume_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].volume
        mock_val = gcn_volume.Volume(name='name_value')
        assert arg == mock_val
        arg = args[0].volume_id
        mock_val = 'volume_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_volume_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_volume(gcn_volume.CreateVolumeRequest(), parent='parent_value', volume=gcn_volume.Volume(name='name_value'), volume_id='volume_id_value')

@pytest.mark.parametrize('request_type', [gcn_volume.UpdateVolumeRequest, dict])
def test_update_volume(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_volume.UpdateVolumeRequest()
    assert isinstance(response, future.Future)

def test_update_volume_empty_call():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_volume), '__call__') as call:
        client.update_volume()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_volume.UpdateVolumeRequest()

@pytest.mark.asyncio
async def test_update_volume_async(transport: str='grpc_asyncio', request_type=gcn_volume.UpdateVolumeRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_volume.UpdateVolumeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_volume_async_from_dict():
    await test_update_volume_async(request_type=dict)

def test_update_volume_field_headers():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_volume.UpdateVolumeRequest()
    request.volume.name = 'name_value'
    with mock.patch.object(type(client.transport.update_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'volume.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_volume_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_volume.UpdateVolumeRequest()
    request.volume.name = 'name_value'
    with mock.patch.object(type(client.transport.update_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'volume.name=name_value') in kw['metadata']

def test_update_volume_flattened():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_volume(volume=gcn_volume.Volume(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].volume
        mock_val = gcn_volume.Volume(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_volume_flattened_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_volume(gcn_volume.UpdateVolumeRequest(), volume=gcn_volume.Volume(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_volume_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_volume(volume=gcn_volume.Volume(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].volume
        mock_val = gcn_volume.Volume(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_volume_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_volume(gcn_volume.UpdateVolumeRequest(), volume=gcn_volume.Volume(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [volume.DeleteVolumeRequest, dict])
def test_delete_volume(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.DeleteVolumeRequest()
    assert isinstance(response, future.Future)

def test_delete_volume_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_volume), '__call__') as call:
        client.delete_volume()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.DeleteVolumeRequest()

@pytest.mark.asyncio
async def test_delete_volume_async(transport: str='grpc_asyncio', request_type=volume.DeleteVolumeRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.DeleteVolumeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_volume_async_from_dict():
    await test_delete_volume_async(request_type=dict)

def test_delete_volume_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.DeleteVolumeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_volume_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.DeleteVolumeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_volume_flattened():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_volume(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_volume_flattened_error():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_volume(volume.DeleteVolumeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_volume_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_volume(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_volume_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_volume(volume.DeleteVolumeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [volume.RevertVolumeRequest, dict])
def test_revert_volume(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.revert_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.revert_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.RevertVolumeRequest()
    assert isinstance(response, future.Future)

def test_revert_volume_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.revert_volume), '__call__') as call:
        client.revert_volume()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.RevertVolumeRequest()

@pytest.mark.asyncio
async def test_revert_volume_async(transport: str='grpc_asyncio', request_type=volume.RevertVolumeRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.revert_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.revert_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == volume.RevertVolumeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_revert_volume_async_from_dict():
    await test_revert_volume_async(request_type=dict)

def test_revert_volume_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.RevertVolumeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.revert_volume), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.revert_volume(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_revert_volume_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = volume.RevertVolumeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.revert_volume), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.revert_volume(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [snapshot.ListSnapshotsRequest, dict])
def test_list_snapshots(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_snapshots), '__call__') as call:
        call.return_value = snapshot.ListSnapshotsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_snapshots(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snapshot.ListSnapshotsRequest()
    assert isinstance(response, pagers.ListSnapshotsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_snapshots_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_snapshots), '__call__') as call:
        client.list_snapshots()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snapshot.ListSnapshotsRequest()

@pytest.mark.asyncio
async def test_list_snapshots_async(transport: str='grpc_asyncio', request_type=snapshot.ListSnapshotsRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_snapshots), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(snapshot.ListSnapshotsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_snapshots(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snapshot.ListSnapshotsRequest()
    assert isinstance(response, pagers.ListSnapshotsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_snapshots_async_from_dict():
    await test_list_snapshots_async(request_type=dict)

def test_list_snapshots_field_headers():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = snapshot.ListSnapshotsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_snapshots), '__call__') as call:
        call.return_value = snapshot.ListSnapshotsResponse()
        client.list_snapshots(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_snapshots_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = snapshot.ListSnapshotsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_snapshots), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(snapshot.ListSnapshotsResponse())
        await client.list_snapshots(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_snapshots_flattened():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_snapshots), '__call__') as call:
        call.return_value = snapshot.ListSnapshotsResponse()
        client.list_snapshots(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_snapshots_flattened_error():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_snapshots(snapshot.ListSnapshotsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_snapshots_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_snapshots), '__call__') as call:
        call.return_value = snapshot.ListSnapshotsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(snapshot.ListSnapshotsResponse())
        response = await client.list_snapshots(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_snapshots_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_snapshots(snapshot.ListSnapshotsRequest(), parent='parent_value')

def test_list_snapshots_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_snapshots), '__call__') as call:
        call.side_effect = (snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot(), snapshot.Snapshot(), snapshot.Snapshot()], next_page_token='abc'), snapshot.ListSnapshotsResponse(snapshots=[], next_page_token='def'), snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot()], next_page_token='ghi'), snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot(), snapshot.Snapshot()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_snapshots(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, snapshot.Snapshot) for i in results))

def test_list_snapshots_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_snapshots), '__call__') as call:
        call.side_effect = (snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot(), snapshot.Snapshot(), snapshot.Snapshot()], next_page_token='abc'), snapshot.ListSnapshotsResponse(snapshots=[], next_page_token='def'), snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot()], next_page_token='ghi'), snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot(), snapshot.Snapshot()]), RuntimeError)
        pages = list(client.list_snapshots(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_snapshots_async_pager():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_snapshots), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot(), snapshot.Snapshot(), snapshot.Snapshot()], next_page_token='abc'), snapshot.ListSnapshotsResponse(snapshots=[], next_page_token='def'), snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot()], next_page_token='ghi'), snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot(), snapshot.Snapshot()]), RuntimeError)
        async_pager = await client.list_snapshots(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, snapshot.Snapshot) for i in responses))

@pytest.mark.asyncio
async def test_list_snapshots_async_pages():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_snapshots), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot(), snapshot.Snapshot(), snapshot.Snapshot()], next_page_token='abc'), snapshot.ListSnapshotsResponse(snapshots=[], next_page_token='def'), snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot()], next_page_token='ghi'), snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot(), snapshot.Snapshot()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_snapshots(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [snapshot.GetSnapshotRequest, dict])
def test_get_snapshot(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_snapshot), '__call__') as call:
        call.return_value = snapshot.Snapshot(name='name_value', state=snapshot.Snapshot.State.READY, state_details='state_details_value', description='description_value', used_bytes=0.10790000000000001)
        response = client.get_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snapshot.GetSnapshotRequest()
    assert isinstance(response, snapshot.Snapshot)
    assert response.name == 'name_value'
    assert response.state == snapshot.Snapshot.State.READY
    assert response.state_details == 'state_details_value'
    assert response.description == 'description_value'
    assert math.isclose(response.used_bytes, 0.10790000000000001, rel_tol=1e-06)

def test_get_snapshot_empty_call():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_snapshot), '__call__') as call:
        client.get_snapshot()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snapshot.GetSnapshotRequest()

@pytest.mark.asyncio
async def test_get_snapshot_async(transport: str='grpc_asyncio', request_type=snapshot.GetSnapshotRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(snapshot.Snapshot(name='name_value', state=snapshot.Snapshot.State.READY, state_details='state_details_value', description='description_value', used_bytes=0.10790000000000001))
        response = await client.get_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snapshot.GetSnapshotRequest()
    assert isinstance(response, snapshot.Snapshot)
    assert response.name == 'name_value'
    assert response.state == snapshot.Snapshot.State.READY
    assert response.state_details == 'state_details_value'
    assert response.description == 'description_value'
    assert math.isclose(response.used_bytes, 0.10790000000000001, rel_tol=1e-06)

@pytest.mark.asyncio
async def test_get_snapshot_async_from_dict():
    await test_get_snapshot_async(request_type=dict)

def test_get_snapshot_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = snapshot.GetSnapshotRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_snapshot), '__call__') as call:
        call.return_value = snapshot.Snapshot()
        client.get_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_snapshot_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = snapshot.GetSnapshotRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(snapshot.Snapshot())
        await client.get_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_snapshot_flattened():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_snapshot), '__call__') as call:
        call.return_value = snapshot.Snapshot()
        client.get_snapshot(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_snapshot_flattened_error():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_snapshot(snapshot.GetSnapshotRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_snapshot_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_snapshot), '__call__') as call:
        call.return_value = snapshot.Snapshot()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(snapshot.Snapshot())
        response = await client.get_snapshot(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_snapshot_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_snapshot(snapshot.GetSnapshotRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_snapshot.CreateSnapshotRequest, dict])
def test_create_snapshot(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_snapshot.CreateSnapshotRequest()
    assert isinstance(response, future.Future)

def test_create_snapshot_empty_call():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_snapshot), '__call__') as call:
        client.create_snapshot()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_snapshot.CreateSnapshotRequest()

@pytest.mark.asyncio
async def test_create_snapshot_async(transport: str='grpc_asyncio', request_type=gcn_snapshot.CreateSnapshotRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_snapshot.CreateSnapshotRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_snapshot_async_from_dict():
    await test_create_snapshot_async(request_type=dict)

def test_create_snapshot_field_headers():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_snapshot.CreateSnapshotRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_snapshot_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_snapshot.CreateSnapshotRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_snapshot_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_snapshot(parent='parent_value', snapshot=gcn_snapshot.Snapshot(name='name_value'), snapshot_id='snapshot_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].snapshot
        mock_val = gcn_snapshot.Snapshot(name='name_value')
        assert arg == mock_val
        arg = args[0].snapshot_id
        mock_val = 'snapshot_id_value'
        assert arg == mock_val

def test_create_snapshot_flattened_error():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_snapshot(gcn_snapshot.CreateSnapshotRequest(), parent='parent_value', snapshot=gcn_snapshot.Snapshot(name='name_value'), snapshot_id='snapshot_id_value')

@pytest.mark.asyncio
async def test_create_snapshot_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_snapshot(parent='parent_value', snapshot=gcn_snapshot.Snapshot(name='name_value'), snapshot_id='snapshot_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].snapshot
        mock_val = gcn_snapshot.Snapshot(name='name_value')
        assert arg == mock_val
        arg = args[0].snapshot_id
        mock_val = 'snapshot_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_snapshot_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_snapshot(gcn_snapshot.CreateSnapshotRequest(), parent='parent_value', snapshot=gcn_snapshot.Snapshot(name='name_value'), snapshot_id='snapshot_id_value')

@pytest.mark.parametrize('request_type', [snapshot.DeleteSnapshotRequest, dict])
def test_delete_snapshot(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snapshot.DeleteSnapshotRequest()
    assert isinstance(response, future.Future)

def test_delete_snapshot_empty_call():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_snapshot), '__call__') as call:
        client.delete_snapshot()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snapshot.DeleteSnapshotRequest()

@pytest.mark.asyncio
async def test_delete_snapshot_async(transport: str='grpc_asyncio', request_type=snapshot.DeleteSnapshotRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snapshot.DeleteSnapshotRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_snapshot_async_from_dict():
    await test_delete_snapshot_async(request_type=dict)

def test_delete_snapshot_field_headers():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = snapshot.DeleteSnapshotRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_snapshot_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = snapshot.DeleteSnapshotRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_snapshot_flattened():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_snapshot(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_snapshot_flattened_error():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_snapshot(snapshot.DeleteSnapshotRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_snapshot_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_snapshot(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_snapshot_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_snapshot(snapshot.DeleteSnapshotRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_snapshot.UpdateSnapshotRequest, dict])
def test_update_snapshot(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_snapshot.UpdateSnapshotRequest()
    assert isinstance(response, future.Future)

def test_update_snapshot_empty_call():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_snapshot), '__call__') as call:
        client.update_snapshot()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_snapshot.UpdateSnapshotRequest()

@pytest.mark.asyncio
async def test_update_snapshot_async(transport: str='grpc_asyncio', request_type=gcn_snapshot.UpdateSnapshotRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_snapshot.UpdateSnapshotRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_snapshot_async_from_dict():
    await test_update_snapshot_async(request_type=dict)

def test_update_snapshot_field_headers():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_snapshot.UpdateSnapshotRequest()
    request.snapshot.name = 'name_value'
    with mock.patch.object(type(client.transport.update_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_snapshot(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'snapshot.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_snapshot_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_snapshot.UpdateSnapshotRequest()
    request.snapshot.name = 'name_value'
    with mock.patch.object(type(client.transport.update_snapshot), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_snapshot(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'snapshot.name=name_value') in kw['metadata']

def test_update_snapshot_flattened():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_snapshot(snapshot=gcn_snapshot.Snapshot(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].snapshot
        mock_val = gcn_snapshot.Snapshot(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_snapshot_flattened_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_snapshot(gcn_snapshot.UpdateSnapshotRequest(), snapshot=gcn_snapshot.Snapshot(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_snapshot_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_snapshot), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_snapshot(snapshot=gcn_snapshot.Snapshot(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].snapshot
        mock_val = gcn_snapshot.Snapshot(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_snapshot_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_snapshot(gcn_snapshot.UpdateSnapshotRequest(), snapshot=gcn_snapshot.Snapshot(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [active_directory.ListActiveDirectoriesRequest, dict])
def test_list_active_directories(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_active_directories), '__call__') as call:
        call.return_value = active_directory.ListActiveDirectoriesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_active_directories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == active_directory.ListActiveDirectoriesRequest()
    assert isinstance(response, pagers.ListActiveDirectoriesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_active_directories_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_active_directories), '__call__') as call:
        client.list_active_directories()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == active_directory.ListActiveDirectoriesRequest()

@pytest.mark.asyncio
async def test_list_active_directories_async(transport: str='grpc_asyncio', request_type=active_directory.ListActiveDirectoriesRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_active_directories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(active_directory.ListActiveDirectoriesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_active_directories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == active_directory.ListActiveDirectoriesRequest()
    assert isinstance(response, pagers.ListActiveDirectoriesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_active_directories_async_from_dict():
    await test_list_active_directories_async(request_type=dict)

def test_list_active_directories_field_headers():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = active_directory.ListActiveDirectoriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_active_directories), '__call__') as call:
        call.return_value = active_directory.ListActiveDirectoriesResponse()
        client.list_active_directories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_active_directories_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = active_directory.ListActiveDirectoriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_active_directories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(active_directory.ListActiveDirectoriesResponse())
        await client.list_active_directories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_active_directories_flattened():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_active_directories), '__call__') as call:
        call.return_value = active_directory.ListActiveDirectoriesResponse()
        client.list_active_directories(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_active_directories_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_active_directories(active_directory.ListActiveDirectoriesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_active_directories_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_active_directories), '__call__') as call:
        call.return_value = active_directory.ListActiveDirectoriesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(active_directory.ListActiveDirectoriesResponse())
        response = await client.list_active_directories(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_active_directories_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_active_directories(active_directory.ListActiveDirectoriesRequest(), parent='parent_value')

def test_list_active_directories_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_active_directories), '__call__') as call:
        call.side_effect = (active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory(), active_directory.ActiveDirectory(), active_directory.ActiveDirectory()], next_page_token='abc'), active_directory.ListActiveDirectoriesResponse(active_directories=[], next_page_token='def'), active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory()], next_page_token='ghi'), active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory(), active_directory.ActiveDirectory()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_active_directories(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, active_directory.ActiveDirectory) for i in results))

def test_list_active_directories_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_active_directories), '__call__') as call:
        call.side_effect = (active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory(), active_directory.ActiveDirectory(), active_directory.ActiveDirectory()], next_page_token='abc'), active_directory.ListActiveDirectoriesResponse(active_directories=[], next_page_token='def'), active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory()], next_page_token='ghi'), active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory(), active_directory.ActiveDirectory()]), RuntimeError)
        pages = list(client.list_active_directories(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_active_directories_async_pager():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_active_directories), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory(), active_directory.ActiveDirectory(), active_directory.ActiveDirectory()], next_page_token='abc'), active_directory.ListActiveDirectoriesResponse(active_directories=[], next_page_token='def'), active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory()], next_page_token='ghi'), active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory(), active_directory.ActiveDirectory()]), RuntimeError)
        async_pager = await client.list_active_directories(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, active_directory.ActiveDirectory) for i in responses))

@pytest.mark.asyncio
async def test_list_active_directories_async_pages():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_active_directories), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory(), active_directory.ActiveDirectory(), active_directory.ActiveDirectory()], next_page_token='abc'), active_directory.ListActiveDirectoriesResponse(active_directories=[], next_page_token='def'), active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory()], next_page_token='ghi'), active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory(), active_directory.ActiveDirectory()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_active_directories(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [active_directory.GetActiveDirectoryRequest, dict])
def test_get_active_directory(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_active_directory), '__call__') as call:
        call.return_value = active_directory.ActiveDirectory(name='name_value', state=active_directory.ActiveDirectory.State.CREATING, domain='domain_value', site='site_value', dns='dns_value', net_bios_prefix='net_bios_prefix_value', organizational_unit='organizational_unit_value', aes_encryption=True, username='username_value', password='password_value', backup_operators=['backup_operators_value'], security_operators=['security_operators_value'], kdc_hostname='kdc_hostname_value', kdc_ip='kdc_ip_value', nfs_users_with_ldap=True, description='description_value', ldap_signing=True, encrypt_dc_connections=True, state_details='state_details_value')
        response = client.get_active_directory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == active_directory.GetActiveDirectoryRequest()
    assert isinstance(response, active_directory.ActiveDirectory)
    assert response.name == 'name_value'
    assert response.state == active_directory.ActiveDirectory.State.CREATING
    assert response.domain == 'domain_value'
    assert response.site == 'site_value'
    assert response.dns == 'dns_value'
    assert response.net_bios_prefix == 'net_bios_prefix_value'
    assert response.organizational_unit == 'organizational_unit_value'
    assert response.aes_encryption is True
    assert response.username == 'username_value'
    assert response.password == 'password_value'
    assert response.backup_operators == ['backup_operators_value']
    assert response.security_operators == ['security_operators_value']
    assert response.kdc_hostname == 'kdc_hostname_value'
    assert response.kdc_ip == 'kdc_ip_value'
    assert response.nfs_users_with_ldap is True
    assert response.description == 'description_value'
    assert response.ldap_signing is True
    assert response.encrypt_dc_connections is True
    assert response.state_details == 'state_details_value'

def test_get_active_directory_empty_call():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_active_directory), '__call__') as call:
        client.get_active_directory()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == active_directory.GetActiveDirectoryRequest()

@pytest.mark.asyncio
async def test_get_active_directory_async(transport: str='grpc_asyncio', request_type=active_directory.GetActiveDirectoryRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_active_directory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(active_directory.ActiveDirectory(name='name_value', state=active_directory.ActiveDirectory.State.CREATING, domain='domain_value', site='site_value', dns='dns_value', net_bios_prefix='net_bios_prefix_value', organizational_unit='organizational_unit_value', aes_encryption=True, username='username_value', password='password_value', backup_operators=['backup_operators_value'], security_operators=['security_operators_value'], kdc_hostname='kdc_hostname_value', kdc_ip='kdc_ip_value', nfs_users_with_ldap=True, description='description_value', ldap_signing=True, encrypt_dc_connections=True, state_details='state_details_value'))
        response = await client.get_active_directory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == active_directory.GetActiveDirectoryRequest()
    assert isinstance(response, active_directory.ActiveDirectory)
    assert response.name == 'name_value'
    assert response.state == active_directory.ActiveDirectory.State.CREATING
    assert response.domain == 'domain_value'
    assert response.site == 'site_value'
    assert response.dns == 'dns_value'
    assert response.net_bios_prefix == 'net_bios_prefix_value'
    assert response.organizational_unit == 'organizational_unit_value'
    assert response.aes_encryption is True
    assert response.username == 'username_value'
    assert response.password == 'password_value'
    assert response.backup_operators == ['backup_operators_value']
    assert response.security_operators == ['security_operators_value']
    assert response.kdc_hostname == 'kdc_hostname_value'
    assert response.kdc_ip == 'kdc_ip_value'
    assert response.nfs_users_with_ldap is True
    assert response.description == 'description_value'
    assert response.ldap_signing is True
    assert response.encrypt_dc_connections is True
    assert response.state_details == 'state_details_value'

@pytest.mark.asyncio
async def test_get_active_directory_async_from_dict():
    await test_get_active_directory_async(request_type=dict)

def test_get_active_directory_field_headers():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = active_directory.GetActiveDirectoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_active_directory), '__call__') as call:
        call.return_value = active_directory.ActiveDirectory()
        client.get_active_directory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_active_directory_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = active_directory.GetActiveDirectoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_active_directory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(active_directory.ActiveDirectory())
        await client.get_active_directory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_active_directory_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_active_directory), '__call__') as call:
        call.return_value = active_directory.ActiveDirectory()
        client.get_active_directory(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_active_directory_flattened_error():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_active_directory(active_directory.GetActiveDirectoryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_active_directory_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_active_directory), '__call__') as call:
        call.return_value = active_directory.ActiveDirectory()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(active_directory.ActiveDirectory())
        response = await client.get_active_directory(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_active_directory_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_active_directory(active_directory.GetActiveDirectoryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_active_directory.CreateActiveDirectoryRequest, dict])
def test_create_active_directory(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_active_directory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_active_directory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_active_directory.CreateActiveDirectoryRequest()
    assert isinstance(response, future.Future)

def test_create_active_directory_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_active_directory), '__call__') as call:
        client.create_active_directory()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_active_directory.CreateActiveDirectoryRequest()

@pytest.mark.asyncio
async def test_create_active_directory_async(transport: str='grpc_asyncio', request_type=gcn_active_directory.CreateActiveDirectoryRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_active_directory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_active_directory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_active_directory.CreateActiveDirectoryRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_active_directory_async_from_dict():
    await test_create_active_directory_async(request_type=dict)

def test_create_active_directory_field_headers():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_active_directory.CreateActiveDirectoryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_active_directory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_active_directory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_active_directory_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_active_directory.CreateActiveDirectoryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_active_directory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_active_directory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_active_directory_flattened():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_active_directory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_active_directory(parent='parent_value', active_directory=gcn_active_directory.ActiveDirectory(name='name_value'), active_directory_id='active_directory_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].active_directory
        mock_val = gcn_active_directory.ActiveDirectory(name='name_value')
        assert arg == mock_val
        arg = args[0].active_directory_id
        mock_val = 'active_directory_id_value'
        assert arg == mock_val

def test_create_active_directory_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_active_directory(gcn_active_directory.CreateActiveDirectoryRequest(), parent='parent_value', active_directory=gcn_active_directory.ActiveDirectory(name='name_value'), active_directory_id='active_directory_id_value')

@pytest.mark.asyncio
async def test_create_active_directory_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_active_directory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_active_directory(parent='parent_value', active_directory=gcn_active_directory.ActiveDirectory(name='name_value'), active_directory_id='active_directory_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].active_directory
        mock_val = gcn_active_directory.ActiveDirectory(name='name_value')
        assert arg == mock_val
        arg = args[0].active_directory_id
        mock_val = 'active_directory_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_active_directory_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_active_directory(gcn_active_directory.CreateActiveDirectoryRequest(), parent='parent_value', active_directory=gcn_active_directory.ActiveDirectory(name='name_value'), active_directory_id='active_directory_id_value')

@pytest.mark.parametrize('request_type', [gcn_active_directory.UpdateActiveDirectoryRequest, dict])
def test_update_active_directory(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_active_directory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_active_directory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_active_directory.UpdateActiveDirectoryRequest()
    assert isinstance(response, future.Future)

def test_update_active_directory_empty_call():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_active_directory), '__call__') as call:
        client.update_active_directory()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_active_directory.UpdateActiveDirectoryRequest()

@pytest.mark.asyncio
async def test_update_active_directory_async(transport: str='grpc_asyncio', request_type=gcn_active_directory.UpdateActiveDirectoryRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_active_directory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_active_directory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_active_directory.UpdateActiveDirectoryRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_active_directory_async_from_dict():
    await test_update_active_directory_async(request_type=dict)

def test_update_active_directory_field_headers():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_active_directory.UpdateActiveDirectoryRequest()
    request.active_directory.name = 'name_value'
    with mock.patch.object(type(client.transport.update_active_directory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_active_directory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'active_directory.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_active_directory_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_active_directory.UpdateActiveDirectoryRequest()
    request.active_directory.name = 'name_value'
    with mock.patch.object(type(client.transport.update_active_directory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_active_directory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'active_directory.name=name_value') in kw['metadata']

def test_update_active_directory_flattened():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_active_directory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_active_directory(active_directory=gcn_active_directory.ActiveDirectory(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].active_directory
        mock_val = gcn_active_directory.ActiveDirectory(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_active_directory_flattened_error():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_active_directory(gcn_active_directory.UpdateActiveDirectoryRequest(), active_directory=gcn_active_directory.ActiveDirectory(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_active_directory_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_active_directory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_active_directory(active_directory=gcn_active_directory.ActiveDirectory(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].active_directory
        mock_val = gcn_active_directory.ActiveDirectory(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_active_directory_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_active_directory(gcn_active_directory.UpdateActiveDirectoryRequest(), active_directory=gcn_active_directory.ActiveDirectory(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [active_directory.DeleteActiveDirectoryRequest, dict])
def test_delete_active_directory(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_active_directory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_active_directory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == active_directory.DeleteActiveDirectoryRequest()
    assert isinstance(response, future.Future)

def test_delete_active_directory_empty_call():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_active_directory), '__call__') as call:
        client.delete_active_directory()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == active_directory.DeleteActiveDirectoryRequest()

@pytest.mark.asyncio
async def test_delete_active_directory_async(transport: str='grpc_asyncio', request_type=active_directory.DeleteActiveDirectoryRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_active_directory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_active_directory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == active_directory.DeleteActiveDirectoryRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_active_directory_async_from_dict():
    await test_delete_active_directory_async(request_type=dict)

def test_delete_active_directory_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = active_directory.DeleteActiveDirectoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_active_directory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_active_directory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_active_directory_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = active_directory.DeleteActiveDirectoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_active_directory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_active_directory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_active_directory_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_active_directory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_active_directory(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_active_directory_flattened_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_active_directory(active_directory.DeleteActiveDirectoryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_active_directory_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_active_directory), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_active_directory(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_active_directory_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_active_directory(active_directory.DeleteActiveDirectoryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [kms.ListKmsConfigsRequest, dict])
def test_list_kms_configs(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_kms_configs), '__call__') as call:
        call.return_value = kms.ListKmsConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_kms_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.ListKmsConfigsRequest()
    assert isinstance(response, pagers.ListKmsConfigsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_kms_configs_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_kms_configs), '__call__') as call:
        client.list_kms_configs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.ListKmsConfigsRequest()

@pytest.mark.asyncio
async def test_list_kms_configs_async(transport: str='grpc_asyncio', request_type=kms.ListKmsConfigsRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_kms_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(kms.ListKmsConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_kms_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.ListKmsConfigsRequest()
    assert isinstance(response, pagers.ListKmsConfigsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_kms_configs_async_from_dict():
    await test_list_kms_configs_async(request_type=dict)

def test_list_kms_configs_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = kms.ListKmsConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_kms_configs), '__call__') as call:
        call.return_value = kms.ListKmsConfigsResponse()
        client.list_kms_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_kms_configs_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = kms.ListKmsConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_kms_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(kms.ListKmsConfigsResponse())
        await client.list_kms_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_kms_configs_flattened():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_kms_configs), '__call__') as call:
        call.return_value = kms.ListKmsConfigsResponse()
        client.list_kms_configs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_kms_configs_flattened_error():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_kms_configs(kms.ListKmsConfigsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_kms_configs_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_kms_configs), '__call__') as call:
        call.return_value = kms.ListKmsConfigsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(kms.ListKmsConfigsResponse())
        response = await client.list_kms_configs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_kms_configs_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_kms_configs(kms.ListKmsConfigsRequest(), parent='parent_value')

def test_list_kms_configs_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_kms_configs), '__call__') as call:
        call.side_effect = (kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig(), kms.KmsConfig(), kms.KmsConfig()], next_page_token='abc'), kms.ListKmsConfigsResponse(kms_configs=[], next_page_token='def'), kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig()], next_page_token='ghi'), kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig(), kms.KmsConfig()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_kms_configs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, kms.KmsConfig) for i in results))

def test_list_kms_configs_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_kms_configs), '__call__') as call:
        call.side_effect = (kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig(), kms.KmsConfig(), kms.KmsConfig()], next_page_token='abc'), kms.ListKmsConfigsResponse(kms_configs=[], next_page_token='def'), kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig()], next_page_token='ghi'), kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig(), kms.KmsConfig()]), RuntimeError)
        pages = list(client.list_kms_configs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_kms_configs_async_pager():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_kms_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig(), kms.KmsConfig(), kms.KmsConfig()], next_page_token='abc'), kms.ListKmsConfigsResponse(kms_configs=[], next_page_token='def'), kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig()], next_page_token='ghi'), kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig(), kms.KmsConfig()]), RuntimeError)
        async_pager = await client.list_kms_configs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, kms.KmsConfig) for i in responses))

@pytest.mark.asyncio
async def test_list_kms_configs_async_pages():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_kms_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig(), kms.KmsConfig(), kms.KmsConfig()], next_page_token='abc'), kms.ListKmsConfigsResponse(kms_configs=[], next_page_token='def'), kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig()], next_page_token='ghi'), kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig(), kms.KmsConfig()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_kms_configs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [kms.CreateKmsConfigRequest, dict])
def test_create_kms_config(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_kms_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_kms_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.CreateKmsConfigRequest()
    assert isinstance(response, future.Future)

def test_create_kms_config_empty_call():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_kms_config), '__call__') as call:
        client.create_kms_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.CreateKmsConfigRequest()

@pytest.mark.asyncio
async def test_create_kms_config_async(transport: str='grpc_asyncio', request_type=kms.CreateKmsConfigRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_kms_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_kms_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.CreateKmsConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_kms_config_async_from_dict():
    await test_create_kms_config_async(request_type=dict)

def test_create_kms_config_field_headers():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = kms.CreateKmsConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_kms_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_kms_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_kms_config_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = kms.CreateKmsConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_kms_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_kms_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_kms_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_kms_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_kms_config(parent='parent_value', kms_config=kms.KmsConfig(name='name_value'), kms_config_id='kms_config_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].kms_config
        mock_val = kms.KmsConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].kms_config_id
        mock_val = 'kms_config_id_value'
        assert arg == mock_val

def test_create_kms_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_kms_config(kms.CreateKmsConfigRequest(), parent='parent_value', kms_config=kms.KmsConfig(name='name_value'), kms_config_id='kms_config_id_value')

@pytest.mark.asyncio
async def test_create_kms_config_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_kms_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_kms_config(parent='parent_value', kms_config=kms.KmsConfig(name='name_value'), kms_config_id='kms_config_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].kms_config
        mock_val = kms.KmsConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].kms_config_id
        mock_val = 'kms_config_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_kms_config_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_kms_config(kms.CreateKmsConfigRequest(), parent='parent_value', kms_config=kms.KmsConfig(name='name_value'), kms_config_id='kms_config_id_value')

@pytest.mark.parametrize('request_type', [kms.GetKmsConfigRequest, dict])
def test_get_kms_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_kms_config), '__call__') as call:
        call.return_value = kms.KmsConfig(name='name_value', crypto_key_name='crypto_key_name_value', state=kms.KmsConfig.State.READY, state_details='state_details_value', description='description_value', instructions='instructions_value', service_account='service_account_value')
        response = client.get_kms_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.GetKmsConfigRequest()
    assert isinstance(response, kms.KmsConfig)
    assert response.name == 'name_value'
    assert response.crypto_key_name == 'crypto_key_name_value'
    assert response.state == kms.KmsConfig.State.READY
    assert response.state_details == 'state_details_value'
    assert response.description == 'description_value'
    assert response.instructions == 'instructions_value'
    assert response.service_account == 'service_account_value'

def test_get_kms_config_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_kms_config), '__call__') as call:
        client.get_kms_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.GetKmsConfigRequest()

@pytest.mark.asyncio
async def test_get_kms_config_async(transport: str='grpc_asyncio', request_type=kms.GetKmsConfigRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_kms_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(kms.KmsConfig(name='name_value', crypto_key_name='crypto_key_name_value', state=kms.KmsConfig.State.READY, state_details='state_details_value', description='description_value', instructions='instructions_value', service_account='service_account_value'))
        response = await client.get_kms_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.GetKmsConfigRequest()
    assert isinstance(response, kms.KmsConfig)
    assert response.name == 'name_value'
    assert response.crypto_key_name == 'crypto_key_name_value'
    assert response.state == kms.KmsConfig.State.READY
    assert response.state_details == 'state_details_value'
    assert response.description == 'description_value'
    assert response.instructions == 'instructions_value'
    assert response.service_account == 'service_account_value'

@pytest.mark.asyncio
async def test_get_kms_config_async_from_dict():
    await test_get_kms_config_async(request_type=dict)

def test_get_kms_config_field_headers():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = kms.GetKmsConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_kms_config), '__call__') as call:
        call.return_value = kms.KmsConfig()
        client.get_kms_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_kms_config_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = kms.GetKmsConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_kms_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(kms.KmsConfig())
        await client.get_kms_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_kms_config_flattened():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_kms_config), '__call__') as call:
        call.return_value = kms.KmsConfig()
        client.get_kms_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_kms_config_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_kms_config(kms.GetKmsConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_kms_config_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_kms_config), '__call__') as call:
        call.return_value = kms.KmsConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(kms.KmsConfig())
        response = await client.get_kms_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_kms_config_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_kms_config(kms.GetKmsConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [kms.UpdateKmsConfigRequest, dict])
def test_update_kms_config(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_kms_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_kms_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.UpdateKmsConfigRequest()
    assert isinstance(response, future.Future)

def test_update_kms_config_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_kms_config), '__call__') as call:
        client.update_kms_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.UpdateKmsConfigRequest()

@pytest.mark.asyncio
async def test_update_kms_config_async(transport: str='grpc_asyncio', request_type=kms.UpdateKmsConfigRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_kms_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_kms_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.UpdateKmsConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_kms_config_async_from_dict():
    await test_update_kms_config_async(request_type=dict)

def test_update_kms_config_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = kms.UpdateKmsConfigRequest()
    request.kms_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_kms_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_kms_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'kms_config.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_kms_config_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = kms.UpdateKmsConfigRequest()
    request.kms_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_kms_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_kms_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'kms_config.name=name_value') in kw['metadata']

def test_update_kms_config_flattened():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_kms_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_kms_config(kms_config=kms.KmsConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].kms_config
        mock_val = kms.KmsConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_kms_config_flattened_error():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_kms_config(kms.UpdateKmsConfigRequest(), kms_config=kms.KmsConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_kms_config_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_kms_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_kms_config(kms_config=kms.KmsConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].kms_config
        mock_val = kms.KmsConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_kms_config_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_kms_config(kms.UpdateKmsConfigRequest(), kms_config=kms.KmsConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [kms.EncryptVolumesRequest, dict])
def test_encrypt_volumes(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.encrypt_volumes), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.encrypt_volumes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.EncryptVolumesRequest()
    assert isinstance(response, future.Future)

def test_encrypt_volumes_empty_call():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.encrypt_volumes), '__call__') as call:
        client.encrypt_volumes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.EncryptVolumesRequest()

@pytest.mark.asyncio
async def test_encrypt_volumes_async(transport: str='grpc_asyncio', request_type=kms.EncryptVolumesRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.encrypt_volumes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.encrypt_volumes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.EncryptVolumesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_encrypt_volumes_async_from_dict():
    await test_encrypt_volumes_async(request_type=dict)

def test_encrypt_volumes_field_headers():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = kms.EncryptVolumesRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.encrypt_volumes), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.encrypt_volumes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_encrypt_volumes_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = kms.EncryptVolumesRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.encrypt_volumes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.encrypt_volumes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [kms.VerifyKmsConfigRequest, dict])
def test_verify_kms_config(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.verify_kms_config), '__call__') as call:
        call.return_value = kms.VerifyKmsConfigResponse(healthy=True, health_error='health_error_value', instructions='instructions_value')
        response = client.verify_kms_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.VerifyKmsConfigRequest()
    assert isinstance(response, kms.VerifyKmsConfigResponse)
    assert response.healthy is True
    assert response.health_error == 'health_error_value'
    assert response.instructions == 'instructions_value'

def test_verify_kms_config_empty_call():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.verify_kms_config), '__call__') as call:
        client.verify_kms_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.VerifyKmsConfigRequest()

@pytest.mark.asyncio
async def test_verify_kms_config_async(transport: str='grpc_asyncio', request_type=kms.VerifyKmsConfigRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.verify_kms_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(kms.VerifyKmsConfigResponse(healthy=True, health_error='health_error_value', instructions='instructions_value'))
        response = await client.verify_kms_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.VerifyKmsConfigRequest()
    assert isinstance(response, kms.VerifyKmsConfigResponse)
    assert response.healthy is True
    assert response.health_error == 'health_error_value'
    assert response.instructions == 'instructions_value'

@pytest.mark.asyncio
async def test_verify_kms_config_async_from_dict():
    await test_verify_kms_config_async(request_type=dict)

def test_verify_kms_config_field_headers():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = kms.VerifyKmsConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.verify_kms_config), '__call__') as call:
        call.return_value = kms.VerifyKmsConfigResponse()
        client.verify_kms_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_verify_kms_config_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = kms.VerifyKmsConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.verify_kms_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(kms.VerifyKmsConfigResponse())
        await client.verify_kms_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [kms.DeleteKmsConfigRequest, dict])
def test_delete_kms_config(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_kms_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_kms_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.DeleteKmsConfigRequest()
    assert isinstance(response, future.Future)

def test_delete_kms_config_empty_call():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_kms_config), '__call__') as call:
        client.delete_kms_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.DeleteKmsConfigRequest()

@pytest.mark.asyncio
async def test_delete_kms_config_async(transport: str='grpc_asyncio', request_type=kms.DeleteKmsConfigRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_kms_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_kms_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == kms.DeleteKmsConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_kms_config_async_from_dict():
    await test_delete_kms_config_async(request_type=dict)

def test_delete_kms_config_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = kms.DeleteKmsConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_kms_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_kms_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_kms_config_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = kms.DeleteKmsConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_kms_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_kms_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_kms_config_flattened():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_kms_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_kms_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_kms_config_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_kms_config(kms.DeleteKmsConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_kms_config_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_kms_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_kms_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_kms_config_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_kms_config(kms.DeleteKmsConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [replication.ListReplicationsRequest, dict])
def test_list_replications(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_replications), '__call__') as call:
        call.return_value = replication.ListReplicationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_replications(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.ListReplicationsRequest()
    assert isinstance(response, pagers.ListReplicationsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_replications_empty_call():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_replications), '__call__') as call:
        client.list_replications()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.ListReplicationsRequest()

@pytest.mark.asyncio
async def test_list_replications_async(transport: str='grpc_asyncio', request_type=replication.ListReplicationsRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_replications), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(replication.ListReplicationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_replications(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.ListReplicationsRequest()
    assert isinstance(response, pagers.ListReplicationsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_replications_async_from_dict():
    await test_list_replications_async(request_type=dict)

def test_list_replications_field_headers():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = replication.ListReplicationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_replications), '__call__') as call:
        call.return_value = replication.ListReplicationsResponse()
        client.list_replications(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_replications_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = replication.ListReplicationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_replications), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(replication.ListReplicationsResponse())
        await client.list_replications(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_replications_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_replications), '__call__') as call:
        call.return_value = replication.ListReplicationsResponse()
        client.list_replications(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_replications_flattened_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_replications(replication.ListReplicationsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_replications_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_replications), '__call__') as call:
        call.return_value = replication.ListReplicationsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(replication.ListReplicationsResponse())
        response = await client.list_replications(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_replications_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_replications(replication.ListReplicationsRequest(), parent='parent_value')

def test_list_replications_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_replications), '__call__') as call:
        call.side_effect = (replication.ListReplicationsResponse(replications=[replication.Replication(), replication.Replication(), replication.Replication()], next_page_token='abc'), replication.ListReplicationsResponse(replications=[], next_page_token='def'), replication.ListReplicationsResponse(replications=[replication.Replication()], next_page_token='ghi'), replication.ListReplicationsResponse(replications=[replication.Replication(), replication.Replication()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_replications(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, replication.Replication) for i in results))

def test_list_replications_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_replications), '__call__') as call:
        call.side_effect = (replication.ListReplicationsResponse(replications=[replication.Replication(), replication.Replication(), replication.Replication()], next_page_token='abc'), replication.ListReplicationsResponse(replications=[], next_page_token='def'), replication.ListReplicationsResponse(replications=[replication.Replication()], next_page_token='ghi'), replication.ListReplicationsResponse(replications=[replication.Replication(), replication.Replication()]), RuntimeError)
        pages = list(client.list_replications(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_replications_async_pager():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_replications), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (replication.ListReplicationsResponse(replications=[replication.Replication(), replication.Replication(), replication.Replication()], next_page_token='abc'), replication.ListReplicationsResponse(replications=[], next_page_token='def'), replication.ListReplicationsResponse(replications=[replication.Replication()], next_page_token='ghi'), replication.ListReplicationsResponse(replications=[replication.Replication(), replication.Replication()]), RuntimeError)
        async_pager = await client.list_replications(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, replication.Replication) for i in responses))

@pytest.mark.asyncio
async def test_list_replications_async_pages():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_replications), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (replication.ListReplicationsResponse(replications=[replication.Replication(), replication.Replication(), replication.Replication()], next_page_token='abc'), replication.ListReplicationsResponse(replications=[], next_page_token='def'), replication.ListReplicationsResponse(replications=[replication.Replication()], next_page_token='ghi'), replication.ListReplicationsResponse(replications=[replication.Replication(), replication.Replication()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_replications(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [replication.GetReplicationRequest, dict])
def test_get_replication(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_replication), '__call__') as call:
        call.return_value = replication.Replication(name='name_value', state=replication.Replication.State.CREATING, state_details='state_details_value', role=replication.Replication.ReplicationRole.SOURCE, replication_schedule=replication.Replication.ReplicationSchedule.EVERY_10_MINUTES, mirror_state=replication.Replication.MirrorState.PREPARING, healthy=True, destination_volume='destination_volume_value', description='description_value', source_volume='source_volume_value')
        response = client.get_replication(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.GetReplicationRequest()
    assert isinstance(response, replication.Replication)
    assert response.name == 'name_value'
    assert response.state == replication.Replication.State.CREATING
    assert response.state_details == 'state_details_value'
    assert response.role == replication.Replication.ReplicationRole.SOURCE
    assert response.replication_schedule == replication.Replication.ReplicationSchedule.EVERY_10_MINUTES
    assert response.mirror_state == replication.Replication.MirrorState.PREPARING
    assert response.healthy is True
    assert response.destination_volume == 'destination_volume_value'
    assert response.description == 'description_value'
    assert response.source_volume == 'source_volume_value'

def test_get_replication_empty_call():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_replication), '__call__') as call:
        client.get_replication()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.GetReplicationRequest()

@pytest.mark.asyncio
async def test_get_replication_async(transport: str='grpc_asyncio', request_type=replication.GetReplicationRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_replication), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(replication.Replication(name='name_value', state=replication.Replication.State.CREATING, state_details='state_details_value', role=replication.Replication.ReplicationRole.SOURCE, replication_schedule=replication.Replication.ReplicationSchedule.EVERY_10_MINUTES, mirror_state=replication.Replication.MirrorState.PREPARING, healthy=True, destination_volume='destination_volume_value', description='description_value', source_volume='source_volume_value'))
        response = await client.get_replication(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.GetReplicationRequest()
    assert isinstance(response, replication.Replication)
    assert response.name == 'name_value'
    assert response.state == replication.Replication.State.CREATING
    assert response.state_details == 'state_details_value'
    assert response.role == replication.Replication.ReplicationRole.SOURCE
    assert response.replication_schedule == replication.Replication.ReplicationSchedule.EVERY_10_MINUTES
    assert response.mirror_state == replication.Replication.MirrorState.PREPARING
    assert response.healthy is True
    assert response.destination_volume == 'destination_volume_value'
    assert response.description == 'description_value'
    assert response.source_volume == 'source_volume_value'

@pytest.mark.asyncio
async def test_get_replication_async_from_dict():
    await test_get_replication_async(request_type=dict)

def test_get_replication_field_headers():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = replication.GetReplicationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_replication), '__call__') as call:
        call.return_value = replication.Replication()
        client.get_replication(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_replication_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = replication.GetReplicationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_replication), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(replication.Replication())
        await client.get_replication(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_replication_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_replication), '__call__') as call:
        call.return_value = replication.Replication()
        client.get_replication(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_replication_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_replication(replication.GetReplicationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_replication_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_replication), '__call__') as call:
        call.return_value = replication.Replication()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(replication.Replication())
        response = await client.get_replication(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_replication_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_replication(replication.GetReplicationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_replication.CreateReplicationRequest, dict])
def test_create_replication(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_replication(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_replication.CreateReplicationRequest()
    assert isinstance(response, future.Future)

def test_create_replication_empty_call():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_replication), '__call__') as call:
        client.create_replication()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_replication.CreateReplicationRequest()

@pytest.mark.asyncio
async def test_create_replication_async(transport: str='grpc_asyncio', request_type=gcn_replication.CreateReplicationRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_replication), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_replication(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_replication.CreateReplicationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_replication_async_from_dict():
    await test_create_replication_async(request_type=dict)

def test_create_replication_field_headers():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_replication.CreateReplicationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_replication(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_replication_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_replication.CreateReplicationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_replication), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_replication(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_replication_flattened():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_replication(parent='parent_value', replication=gcn_replication.Replication(name='name_value'), replication_id='replication_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].replication
        mock_val = gcn_replication.Replication(name='name_value')
        assert arg == mock_val
        arg = args[0].replication_id
        mock_val = 'replication_id_value'
        assert arg == mock_val

def test_create_replication_flattened_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_replication(gcn_replication.CreateReplicationRequest(), parent='parent_value', replication=gcn_replication.Replication(name='name_value'), replication_id='replication_id_value')

@pytest.mark.asyncio
async def test_create_replication_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_replication(parent='parent_value', replication=gcn_replication.Replication(name='name_value'), replication_id='replication_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].replication
        mock_val = gcn_replication.Replication(name='name_value')
        assert arg == mock_val
        arg = args[0].replication_id
        mock_val = 'replication_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_replication_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_replication(gcn_replication.CreateReplicationRequest(), parent='parent_value', replication=gcn_replication.Replication(name='name_value'), replication_id='replication_id_value')

@pytest.mark.parametrize('request_type', [replication.DeleteReplicationRequest, dict])
def test_delete_replication(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_replication(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.DeleteReplicationRequest()
    assert isinstance(response, future.Future)

def test_delete_replication_empty_call():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_replication), '__call__') as call:
        client.delete_replication()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.DeleteReplicationRequest()

@pytest.mark.asyncio
async def test_delete_replication_async(transport: str='grpc_asyncio', request_type=replication.DeleteReplicationRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_replication), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_replication(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.DeleteReplicationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_replication_async_from_dict():
    await test_delete_replication_async(request_type=dict)

def test_delete_replication_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = replication.DeleteReplicationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_replication(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_replication_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = replication.DeleteReplicationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_replication), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_replication(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_replication_flattened():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_replication(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_replication_flattened_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_replication(replication.DeleteReplicationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_replication_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_replication(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_replication_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_replication(replication.DeleteReplicationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_replication.UpdateReplicationRequest, dict])
def test_update_replication(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_replication(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_replication.UpdateReplicationRequest()
    assert isinstance(response, future.Future)

def test_update_replication_empty_call():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_replication), '__call__') as call:
        client.update_replication()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_replication.UpdateReplicationRequest()

@pytest.mark.asyncio
async def test_update_replication_async(transport: str='grpc_asyncio', request_type=gcn_replication.UpdateReplicationRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_replication), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_replication(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_replication.UpdateReplicationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_replication_async_from_dict():
    await test_update_replication_async(request_type=dict)

def test_update_replication_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_replication.UpdateReplicationRequest()
    request.replication.name = 'name_value'
    with mock.patch.object(type(client.transport.update_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_replication(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'replication.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_replication_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_replication.UpdateReplicationRequest()
    request.replication.name = 'name_value'
    with mock.patch.object(type(client.transport.update_replication), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_replication(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'replication.name=name_value') in kw['metadata']

def test_update_replication_flattened():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_replication(replication=gcn_replication.Replication(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].replication
        mock_val = gcn_replication.Replication(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_replication_flattened_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_replication(gcn_replication.UpdateReplicationRequest(), replication=gcn_replication.Replication(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_replication_flattened_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_replication(replication=gcn_replication.Replication(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].replication
        mock_val = gcn_replication.Replication(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_replication_flattened_error_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_replication(gcn_replication.UpdateReplicationRequest(), replication=gcn_replication.Replication(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [replication.StopReplicationRequest, dict])
def test_stop_replication(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.stop_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.stop_replication(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.StopReplicationRequest()
    assert isinstance(response, future.Future)

def test_stop_replication_empty_call():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.stop_replication), '__call__') as call:
        client.stop_replication()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.StopReplicationRequest()

@pytest.mark.asyncio
async def test_stop_replication_async(transport: str='grpc_asyncio', request_type=replication.StopReplicationRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.stop_replication), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.stop_replication(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.StopReplicationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_stop_replication_async_from_dict():
    await test_stop_replication_async(request_type=dict)

def test_stop_replication_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = replication.StopReplicationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.stop_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.stop_replication(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_stop_replication_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = replication.StopReplicationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.stop_replication), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.stop_replication(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [replication.ResumeReplicationRequest, dict])
def test_resume_replication(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resume_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.resume_replication(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.ResumeReplicationRequest()
    assert isinstance(response, future.Future)

def test_resume_replication_empty_call():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.resume_replication), '__call__') as call:
        client.resume_replication()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.ResumeReplicationRequest()

@pytest.mark.asyncio
async def test_resume_replication_async(transport: str='grpc_asyncio', request_type=replication.ResumeReplicationRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resume_replication), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.resume_replication(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.ResumeReplicationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_resume_replication_async_from_dict():
    await test_resume_replication_async(request_type=dict)

def test_resume_replication_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = replication.ResumeReplicationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.resume_replication), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.resume_replication(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_resume_replication_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = replication.ResumeReplicationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.resume_replication), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.resume_replication(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [replication.ReverseReplicationDirectionRequest, dict])
def test_reverse_replication_direction(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reverse_replication_direction), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.reverse_replication_direction(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.ReverseReplicationDirectionRequest()
    assert isinstance(response, future.Future)

def test_reverse_replication_direction_empty_call():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.reverse_replication_direction), '__call__') as call:
        client.reverse_replication_direction()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.ReverseReplicationDirectionRequest()

@pytest.mark.asyncio
async def test_reverse_replication_direction_async(transport: str='grpc_asyncio', request_type=replication.ReverseReplicationDirectionRequest):
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reverse_replication_direction), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reverse_replication_direction(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == replication.ReverseReplicationDirectionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_reverse_replication_direction_async_from_dict():
    await test_reverse_replication_direction_async(request_type=dict)

def test_reverse_replication_direction_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    request = replication.ReverseReplicationDirectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.reverse_replication_direction), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reverse_replication_direction(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_reverse_replication_direction_field_headers_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = replication.ReverseReplicationDirectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.reverse_replication_direction), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.reverse_replication_direction(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [storage_pool.ListStoragePoolsRequest, dict])
def test_list_storage_pools_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = storage_pool.ListStoragePoolsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = storage_pool.ListStoragePoolsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_storage_pools(request)
    assert isinstance(response, pagers.ListStoragePoolsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_storage_pools_rest_required_fields(request_type=storage_pool.ListStoragePoolsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_storage_pools._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_storage_pools._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = storage_pool.ListStoragePoolsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = storage_pool.ListStoragePoolsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_storage_pools(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_storage_pools_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_storage_pools._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_storage_pools_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetAppRestInterceptor, 'post_list_storage_pools') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_list_storage_pools') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = storage_pool.ListStoragePoolsRequest.pb(storage_pool.ListStoragePoolsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = storage_pool.ListStoragePoolsResponse.to_json(storage_pool.ListStoragePoolsResponse())
        request = storage_pool.ListStoragePoolsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = storage_pool.ListStoragePoolsResponse()
        client.list_storage_pools(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_storage_pools_rest_bad_request(transport: str='rest', request_type=storage_pool.ListStoragePoolsRequest):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_storage_pools(request)

def test_list_storage_pools_rest_flattened():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = storage_pool.ListStoragePoolsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = storage_pool.ListStoragePoolsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_storage_pools(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/storagePools' % client.transport._host, args[1])

def test_list_storage_pools_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_storage_pools(storage_pool.ListStoragePoolsRequest(), parent='parent_value')

def test_list_storage_pools_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool(), storage_pool.StoragePool(), storage_pool.StoragePool()], next_page_token='abc'), storage_pool.ListStoragePoolsResponse(storage_pools=[], next_page_token='def'), storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool()], next_page_token='ghi'), storage_pool.ListStoragePoolsResponse(storage_pools=[storage_pool.StoragePool(), storage_pool.StoragePool()]))
        response = response + response
        response = tuple((storage_pool.ListStoragePoolsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_storage_pools(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, storage_pool.StoragePool) for i in results))
        pages = list(client.list_storage_pools(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [gcn_storage_pool.CreateStoragePoolRequest, dict])
def test_create_storage_pool_rest(request_type):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['storage_pool'] = {'name': 'name_value', 'service_level': 1, 'capacity_gib': 1247, 'volume_capacity_gib': 2006, 'volume_count': 1312, 'state': 1, 'state_details': 'state_details_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'description': 'description_value', 'labels': {}, 'network': 'network_value', 'active_directory': 'active_directory_value', 'kms_config': 'kms_config_value', 'ldap_enabled': True, 'psa_range': 'psa_range_value', 'encryption_type': 1, 'global_access_allowed': True}
    test_field = gcn_storage_pool.CreateStoragePoolRequest.meta.fields['storage_pool']

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
    for (field, value) in request_init['storage_pool'].items():
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
                for i in range(0, len(request_init['storage_pool'][field])):
                    del request_init['storage_pool'][field][i][subfield]
            else:
                del request_init['storage_pool'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_storage_pool(request)
    assert response.operation.name == 'operations/spam'

def test_create_storage_pool_rest_required_fields(request_type=gcn_storage_pool.CreateStoragePoolRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['storage_pool_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'storagePoolId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_storage_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'storagePoolId' in jsonified_request
    assert jsonified_request['storagePoolId'] == request_init['storage_pool_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['storagePoolId'] = 'storage_pool_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_storage_pool._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('storage_pool_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'storagePoolId' in jsonified_request
    assert jsonified_request['storagePoolId'] == 'storage_pool_id_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_storage_pool(request)
            expected_params = [('storagePoolId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_storage_pool_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_storage_pool._get_unset_required_fields({})
    assert set(unset_fields) == set(('storagePoolId',)) & set(('parent', 'storagePoolId', 'storagePool'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_storage_pool_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_create_storage_pool') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_create_storage_pool') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_storage_pool.CreateStoragePoolRequest.pb(gcn_storage_pool.CreateStoragePoolRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_storage_pool.CreateStoragePoolRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_storage_pool(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_storage_pool_rest_bad_request(transport: str='rest', request_type=gcn_storage_pool.CreateStoragePoolRequest):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_storage_pool(request)

def test_create_storage_pool_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', storage_pool=gcn_storage_pool.StoragePool(name='name_value'), storage_pool_id='storage_pool_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_storage_pool(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/storagePools' % client.transport._host, args[1])

def test_create_storage_pool_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_storage_pool(gcn_storage_pool.CreateStoragePoolRequest(), parent='parent_value', storage_pool=gcn_storage_pool.StoragePool(name='name_value'), storage_pool_id='storage_pool_id_value')

def test_create_storage_pool_rest_error():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [storage_pool.GetStoragePoolRequest, dict])
def test_get_storage_pool_rest(request_type):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/storagePools/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = storage_pool.StoragePool(name='name_value', service_level=common.ServiceLevel.PREMIUM, capacity_gib=1247, volume_capacity_gib=2006, volume_count=1312, state=storage_pool.StoragePool.State.READY, state_details='state_details_value', description='description_value', network='network_value', active_directory='active_directory_value', kms_config='kms_config_value', ldap_enabled=True, psa_range='psa_range_value', encryption_type=common.EncryptionType.SERVICE_MANAGED, global_access_allowed=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = storage_pool.StoragePool.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_storage_pool(request)
    assert isinstance(response, storage_pool.StoragePool)
    assert response.name == 'name_value'
    assert response.service_level == common.ServiceLevel.PREMIUM
    assert response.capacity_gib == 1247
    assert response.volume_capacity_gib == 2006
    assert response.volume_count == 1312
    assert response.state == storage_pool.StoragePool.State.READY
    assert response.state_details == 'state_details_value'
    assert response.description == 'description_value'
    assert response.network == 'network_value'
    assert response.active_directory == 'active_directory_value'
    assert response.kms_config == 'kms_config_value'
    assert response.ldap_enabled is True
    assert response.psa_range == 'psa_range_value'
    assert response.encryption_type == common.EncryptionType.SERVICE_MANAGED
    assert response.global_access_allowed is True

def test_get_storage_pool_rest_required_fields(request_type=storage_pool.GetStoragePoolRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_storage_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_storage_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = storage_pool.StoragePool()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = storage_pool.StoragePool.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_storage_pool(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_storage_pool_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_storage_pool._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_storage_pool_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetAppRestInterceptor, 'post_get_storage_pool') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_get_storage_pool') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = storage_pool.GetStoragePoolRequest.pb(storage_pool.GetStoragePoolRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = storage_pool.StoragePool.to_json(storage_pool.StoragePool())
        request = storage_pool.GetStoragePoolRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = storage_pool.StoragePool()
        client.get_storage_pool(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_storage_pool_rest_bad_request(transport: str='rest', request_type=storage_pool.GetStoragePoolRequest):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/storagePools/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_storage_pool(request)

def test_get_storage_pool_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = storage_pool.StoragePool()
        sample_request = {'name': 'projects/sample1/locations/sample2/storagePools/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = storage_pool.StoragePool.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_storage_pool(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/storagePools/*}' % client.transport._host, args[1])

def test_get_storage_pool_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_storage_pool(storage_pool.GetStoragePoolRequest(), name='name_value')

def test_get_storage_pool_rest_error():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_storage_pool.UpdateStoragePoolRequest, dict])
def test_update_storage_pool_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'storage_pool': {'name': 'projects/sample1/locations/sample2/storagePools/sample3'}}
    request_init['storage_pool'] = {'name': 'projects/sample1/locations/sample2/storagePools/sample3', 'service_level': 1, 'capacity_gib': 1247, 'volume_capacity_gib': 2006, 'volume_count': 1312, 'state': 1, 'state_details': 'state_details_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'description': 'description_value', 'labels': {}, 'network': 'network_value', 'active_directory': 'active_directory_value', 'kms_config': 'kms_config_value', 'ldap_enabled': True, 'psa_range': 'psa_range_value', 'encryption_type': 1, 'global_access_allowed': True}
    test_field = gcn_storage_pool.UpdateStoragePoolRequest.meta.fields['storage_pool']

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
    for (field, value) in request_init['storage_pool'].items():
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
                for i in range(0, len(request_init['storage_pool'][field])):
                    del request_init['storage_pool'][field][i][subfield]
            else:
                del request_init['storage_pool'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_storage_pool(request)
    assert response.operation.name == 'operations/spam'

def test_update_storage_pool_rest_required_fields(request_type=gcn_storage_pool.UpdateStoragePoolRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_storage_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_storage_pool._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_storage_pool(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_storage_pool_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_storage_pool._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('updateMask', 'storagePool'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_storage_pool_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_update_storage_pool') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_update_storage_pool') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_storage_pool.UpdateStoragePoolRequest.pb(gcn_storage_pool.UpdateStoragePoolRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_storage_pool.UpdateStoragePoolRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_storage_pool(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_storage_pool_rest_bad_request(transport: str='rest', request_type=gcn_storage_pool.UpdateStoragePoolRequest):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'storage_pool': {'name': 'projects/sample1/locations/sample2/storagePools/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_storage_pool(request)

def test_update_storage_pool_rest_flattened():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'storage_pool': {'name': 'projects/sample1/locations/sample2/storagePools/sample3'}}
        mock_args = dict(storage_pool=gcn_storage_pool.StoragePool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_storage_pool(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{storage_pool.name=projects/*/locations/*/storagePools/*}' % client.transport._host, args[1])

def test_update_storage_pool_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_storage_pool(gcn_storage_pool.UpdateStoragePoolRequest(), storage_pool=gcn_storage_pool.StoragePool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_storage_pool_rest_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [storage_pool.DeleteStoragePoolRequest, dict])
def test_delete_storage_pool_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/storagePools/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_storage_pool(request)
    assert response.operation.name == 'operations/spam'

def test_delete_storage_pool_rest_required_fields(request_type=storage_pool.DeleteStoragePoolRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_storage_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_storage_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_storage_pool(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_storage_pool_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_storage_pool._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_storage_pool_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_delete_storage_pool') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_delete_storage_pool') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = storage_pool.DeleteStoragePoolRequest.pb(storage_pool.DeleteStoragePoolRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = storage_pool.DeleteStoragePoolRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_storage_pool(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_storage_pool_rest_bad_request(transport: str='rest', request_type=storage_pool.DeleteStoragePoolRequest):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/storagePools/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_storage_pool(request)

def test_delete_storage_pool_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/storagePools/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_storage_pool(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/storagePools/*}' % client.transport._host, args[1])

def test_delete_storage_pool_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_storage_pool(storage_pool.DeleteStoragePoolRequest(), name='name_value')

def test_delete_storage_pool_rest_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [volume.ListVolumesRequest, dict])
def test_list_volumes_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = volume.ListVolumesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = volume.ListVolumesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_volumes(request)
    assert isinstance(response, pagers.ListVolumesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_volumes_rest_required_fields(request_type=volume.ListVolumesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_volumes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_volumes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = volume.ListVolumesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = volume.ListVolumesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_volumes(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_volumes_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_volumes._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_volumes_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetAppRestInterceptor, 'post_list_volumes') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_list_volumes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = volume.ListVolumesRequest.pb(volume.ListVolumesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = volume.ListVolumesResponse.to_json(volume.ListVolumesResponse())
        request = volume.ListVolumesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = volume.ListVolumesResponse()
        client.list_volumes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_volumes_rest_bad_request(transport: str='rest', request_type=volume.ListVolumesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_volumes(request)

def test_list_volumes_rest_flattened():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = volume.ListVolumesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = volume.ListVolumesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_volumes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/volumes' % client.transport._host, args[1])

def test_list_volumes_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_volumes(volume.ListVolumesRequest(), parent='parent_value')

def test_list_volumes_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume(), volume.Volume()], next_page_token='abc'), volume.ListVolumesResponse(volumes=[], next_page_token='def'), volume.ListVolumesResponse(volumes=[volume.Volume()], next_page_token='ghi'), volume.ListVolumesResponse(volumes=[volume.Volume(), volume.Volume()]))
        response = response + response
        response = tuple((volume.ListVolumesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_volumes(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, volume.Volume) for i in results))
        pages = list(client.list_volumes(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [volume.GetVolumeRequest, dict])
def test_get_volume_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = volume.Volume(name='name_value', state=volume.Volume.State.READY, state_details='state_details_value', share_name='share_name_value', psa_range='psa_range_value', storage_pool='storage_pool_value', network='network_value', service_level=common.ServiceLevel.PREMIUM, capacity_gib=1247, protocols=[volume.Protocols.NFSV3], smb_settings=[volume.SMBSettings.ENCRYPT_DATA], unix_permissions='unix_permissions_value', description='description_value', snap_reserve=0.1293, snapshot_directory=True, used_gib=834, security_style=volume.SecurityStyle.NTFS, kerberos_enabled=True, ldap_enabled=True, active_directory='active_directory_value', kms_config='kms_config_value', encryption_type=common.EncryptionType.SERVICE_MANAGED, has_replication=True, restricted_actions=[volume.RestrictedAction.DELETE])
        response_value = Response()
        response_value.status_code = 200
        return_value = volume.Volume.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_volume(request)
    assert isinstance(response, volume.Volume)
    assert response.name == 'name_value'
    assert response.state == volume.Volume.State.READY
    assert response.state_details == 'state_details_value'
    assert response.share_name == 'share_name_value'
    assert response.psa_range == 'psa_range_value'
    assert response.storage_pool == 'storage_pool_value'
    assert response.network == 'network_value'
    assert response.service_level == common.ServiceLevel.PREMIUM
    assert response.capacity_gib == 1247
    assert response.protocols == [volume.Protocols.NFSV3]
    assert response.smb_settings == [volume.SMBSettings.ENCRYPT_DATA]
    assert response.unix_permissions == 'unix_permissions_value'
    assert response.description == 'description_value'
    assert math.isclose(response.snap_reserve, 0.1293, rel_tol=1e-06)
    assert response.snapshot_directory is True
    assert response.used_gib == 834
    assert response.security_style == volume.SecurityStyle.NTFS
    assert response.kerberos_enabled is True
    assert response.ldap_enabled is True
    assert response.active_directory == 'active_directory_value'
    assert response.kms_config == 'kms_config_value'
    assert response.encryption_type == common.EncryptionType.SERVICE_MANAGED
    assert response.has_replication is True
    assert response.restricted_actions == [volume.RestrictedAction.DELETE]

def test_get_volume_rest_required_fields(request_type=volume.GetVolumeRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = volume.Volume()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = volume.Volume.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_volume(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_volume_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_volume._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_volume_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetAppRestInterceptor, 'post_get_volume') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_get_volume') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = volume.GetVolumeRequest.pb(volume.GetVolumeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = volume.Volume.to_json(volume.Volume())
        request = volume.GetVolumeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = volume.Volume()
        client.get_volume(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_volume_rest_bad_request(transport: str='rest', request_type=volume.GetVolumeRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_volume(request)

def test_get_volume_rest_flattened():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = volume.Volume()
        sample_request = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = volume.Volume.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_volume(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/volumes/*}' % client.transport._host, args[1])

def test_get_volume_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_volume(volume.GetVolumeRequest(), name='name_value')

def test_get_volume_rest_error():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_volume.CreateVolumeRequest, dict])
def test_create_volume_rest(request_type):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['volume'] = {'name': 'name_value', 'state': 1, 'state_details': 'state_details_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'share_name': 'share_name_value', 'psa_range': 'psa_range_value', 'storage_pool': 'storage_pool_value', 'network': 'network_value', 'service_level': 1, 'capacity_gib': 1247, 'export_policy': {'rules': [{'allowed_clients': 'allowed_clients_value', 'has_root_access': 'has_root_access_value', 'access_type': 1, 'nfsv3': True, 'nfsv4': True, 'kerberos_5_read_only': True, 'kerberos_5_read_write': True, 'kerberos_5i_read_only': True, 'kerberos_5i_read_write': True, 'kerberos_5p_read_only': True, 'kerberos_5p_read_write': True}]}, 'protocols': [1], 'smb_settings': [1], 'mount_options': [{'export': 'export_value', 'export_full': 'export_full_value', 'protocol': 1, 'instructions': 'instructions_value'}], 'unix_permissions': 'unix_permissions_value', 'labels': {}, 'description': 'description_value', 'snapshot_policy': {'enabled': True, 'hourly_schedule': {'snapshots_to_keep': 0.18330000000000002, 'minute': 0.658}, 'daily_schedule': {'snapshots_to_keep': 0.18330000000000002, 'minute': 0.658, 'hour': 0.446}, 'weekly_schedule': {'snapshots_to_keep': 0.18330000000000002, 'minute': 0.658, 'hour': 0.446, 'day': 'day_value'}, 'monthly_schedule': {'snapshots_to_keep': 0.18330000000000002, 'minute': 0.658, 'hour': 0.446, 'days_of_month': 'days_of_month_value'}}, 'snap_reserve': 0.1293, 'snapshot_directory': True, 'used_gib': 834, 'security_style': 1, 'kerberos_enabled': True, 'ldap_enabled': True, 'active_directory': 'active_directory_value', 'restore_parameters': {'source_snapshot': 'source_snapshot_value'}, 'kms_config': 'kms_config_value', 'encryption_type': 1, 'has_replication': True, 'restricted_actions': [1]}
    test_field = gcn_volume.CreateVolumeRequest.meta.fields['volume']

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
    for (field, value) in request_init['volume'].items():
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
                for i in range(0, len(request_init['volume'][field])):
                    del request_init['volume'][field][i][subfield]
            else:
                del request_init['volume'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_volume(request)
    assert response.operation.name == 'operations/spam'

def test_create_volume_rest_required_fields(request_type=gcn_volume.CreateVolumeRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['volume_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'volumeId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'volumeId' in jsonified_request
    assert jsonified_request['volumeId'] == request_init['volume_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['volumeId'] = 'volume_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_volume._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('volume_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'volumeId' in jsonified_request
    assert jsonified_request['volumeId'] == 'volume_id_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_volume(request)
            expected_params = [('volumeId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_volume_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_volume._get_unset_required_fields({})
    assert set(unset_fields) == set(('volumeId',)) & set(('parent', 'volumeId', 'volume'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_volume_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_create_volume') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_create_volume') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_volume.CreateVolumeRequest.pb(gcn_volume.CreateVolumeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_volume.CreateVolumeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_volume(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_volume_rest_bad_request(transport: str='rest', request_type=gcn_volume.CreateVolumeRequest):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_volume(request)

def test_create_volume_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', volume=gcn_volume.Volume(name='name_value'), volume_id='volume_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_volume(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/volumes' % client.transport._host, args[1])

def test_create_volume_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_volume(gcn_volume.CreateVolumeRequest(), parent='parent_value', volume=gcn_volume.Volume(name='name_value'), volume_id='volume_id_value')

def test_create_volume_rest_error():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_volume.UpdateVolumeRequest, dict])
def test_update_volume_rest(request_type):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'volume': {'name': 'projects/sample1/locations/sample2/volumes/sample3'}}
    request_init['volume'] = {'name': 'projects/sample1/locations/sample2/volumes/sample3', 'state': 1, 'state_details': 'state_details_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'share_name': 'share_name_value', 'psa_range': 'psa_range_value', 'storage_pool': 'storage_pool_value', 'network': 'network_value', 'service_level': 1, 'capacity_gib': 1247, 'export_policy': {'rules': [{'allowed_clients': 'allowed_clients_value', 'has_root_access': 'has_root_access_value', 'access_type': 1, 'nfsv3': True, 'nfsv4': True, 'kerberos_5_read_only': True, 'kerberos_5_read_write': True, 'kerberos_5i_read_only': True, 'kerberos_5i_read_write': True, 'kerberos_5p_read_only': True, 'kerberos_5p_read_write': True}]}, 'protocols': [1], 'smb_settings': [1], 'mount_options': [{'export': 'export_value', 'export_full': 'export_full_value', 'protocol': 1, 'instructions': 'instructions_value'}], 'unix_permissions': 'unix_permissions_value', 'labels': {}, 'description': 'description_value', 'snapshot_policy': {'enabled': True, 'hourly_schedule': {'snapshots_to_keep': 0.18330000000000002, 'minute': 0.658}, 'daily_schedule': {'snapshots_to_keep': 0.18330000000000002, 'minute': 0.658, 'hour': 0.446}, 'weekly_schedule': {'snapshots_to_keep': 0.18330000000000002, 'minute': 0.658, 'hour': 0.446, 'day': 'day_value'}, 'monthly_schedule': {'snapshots_to_keep': 0.18330000000000002, 'minute': 0.658, 'hour': 0.446, 'days_of_month': 'days_of_month_value'}}, 'snap_reserve': 0.1293, 'snapshot_directory': True, 'used_gib': 834, 'security_style': 1, 'kerberos_enabled': True, 'ldap_enabled': True, 'active_directory': 'active_directory_value', 'restore_parameters': {'source_snapshot': 'source_snapshot_value'}, 'kms_config': 'kms_config_value', 'encryption_type': 1, 'has_replication': True, 'restricted_actions': [1]}
    test_field = gcn_volume.UpdateVolumeRequest.meta.fields['volume']

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
    for (field, value) in request_init['volume'].items():
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
                for i in range(0, len(request_init['volume'][field])):
                    del request_init['volume'][field][i][subfield]
            else:
                del request_init['volume'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_volume(request)
    assert response.operation.name == 'operations/spam'

def test_update_volume_rest_required_fields(request_type=gcn_volume.UpdateVolumeRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_volume._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_volume(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_volume_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_volume._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('updateMask', 'volume'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_volume_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_update_volume') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_update_volume') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_volume.UpdateVolumeRequest.pb(gcn_volume.UpdateVolumeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_volume.UpdateVolumeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_volume(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_volume_rest_bad_request(transport: str='rest', request_type=gcn_volume.UpdateVolumeRequest):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'volume': {'name': 'projects/sample1/locations/sample2/volumes/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_volume(request)

def test_update_volume_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'volume': {'name': 'projects/sample1/locations/sample2/volumes/sample3'}}
        mock_args = dict(volume=gcn_volume.Volume(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_volume(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{volume.name=projects/*/locations/*/volumes/*}' % client.transport._host, args[1])

def test_update_volume_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_volume(gcn_volume.UpdateVolumeRequest(), volume=gcn_volume.Volume(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_volume_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [volume.DeleteVolumeRequest, dict])
def test_delete_volume_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_volume(request)
    assert response.operation.name == 'operations/spam'

def test_delete_volume_rest_required_fields(request_type=volume.DeleteVolumeRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_volume._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('force',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_volume(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_volume_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_volume._get_unset_required_fields({})
    assert set(unset_fields) == set(('force',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_volume_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_delete_volume') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_delete_volume') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = volume.DeleteVolumeRequest.pb(volume.DeleteVolumeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = volume.DeleteVolumeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_volume(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_volume_rest_bad_request(transport: str='rest', request_type=volume.DeleteVolumeRequest):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_volume(request)

def test_delete_volume_rest_flattened():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_volume(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/volumes/*}' % client.transport._host, args[1])

def test_delete_volume_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_volume(volume.DeleteVolumeRequest(), name='name_value')

def test_delete_volume_rest_error():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [volume.RevertVolumeRequest, dict])
def test_revert_volume_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.revert_volume(request)
    assert response.operation.name == 'operations/spam'

def test_revert_volume_rest_required_fields(request_type=volume.RevertVolumeRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['snapshot_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).revert_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['snapshotId'] = 'snapshot_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).revert_volume._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'snapshotId' in jsonified_request
    assert jsonified_request['snapshotId'] == 'snapshot_id_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.revert_volume(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_revert_volume_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.revert_volume._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'snapshotId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_revert_volume_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_revert_volume') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_revert_volume') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = volume.RevertVolumeRequest.pb(volume.RevertVolumeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = volume.RevertVolumeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.revert_volume(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_revert_volume_rest_bad_request(transport: str='rest', request_type=volume.RevertVolumeRequest):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.revert_volume(request)

def test_revert_volume_rest_error():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [snapshot.ListSnapshotsRequest, dict])
def test_list_snapshots_rest(request_type):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = snapshot.ListSnapshotsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = snapshot.ListSnapshotsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_snapshots(request)
    assert isinstance(response, pagers.ListSnapshotsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_snapshots_rest_required_fields(request_type=snapshot.ListSnapshotsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_snapshots._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_snapshots._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = snapshot.ListSnapshotsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = snapshot.ListSnapshotsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_snapshots(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_snapshots_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_snapshots._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_snapshots_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetAppRestInterceptor, 'post_list_snapshots') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_list_snapshots') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = snapshot.ListSnapshotsRequest.pb(snapshot.ListSnapshotsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = snapshot.ListSnapshotsResponse.to_json(snapshot.ListSnapshotsResponse())
        request = snapshot.ListSnapshotsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = snapshot.ListSnapshotsResponse()
        client.list_snapshots(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_snapshots_rest_bad_request(transport: str='rest', request_type=snapshot.ListSnapshotsRequest):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_snapshots(request)

def test_list_snapshots_rest_flattened():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = snapshot.ListSnapshotsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = snapshot.ListSnapshotsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_snapshots(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/volumes/*}/snapshots' % client.transport._host, args[1])

def test_list_snapshots_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_snapshots(snapshot.ListSnapshotsRequest(), parent='parent_value')

def test_list_snapshots_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot(), snapshot.Snapshot(), snapshot.Snapshot()], next_page_token='abc'), snapshot.ListSnapshotsResponse(snapshots=[], next_page_token='def'), snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot()], next_page_token='ghi'), snapshot.ListSnapshotsResponse(snapshots=[snapshot.Snapshot(), snapshot.Snapshot()]))
        response = response + response
        response = tuple((snapshot.ListSnapshotsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
        pager = client.list_snapshots(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, snapshot.Snapshot) for i in results))
        pages = list(client.list_snapshots(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [snapshot.GetSnapshotRequest, dict])
def test_get_snapshot_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = snapshot.Snapshot(name='name_value', state=snapshot.Snapshot.State.READY, state_details='state_details_value', description='description_value', used_bytes=0.10790000000000001)
        response_value = Response()
        response_value.status_code = 200
        return_value = snapshot.Snapshot.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_snapshot(request)
    assert isinstance(response, snapshot.Snapshot)
    assert response.name == 'name_value'
    assert response.state == snapshot.Snapshot.State.READY
    assert response.state_details == 'state_details_value'
    assert response.description == 'description_value'
    assert math.isclose(response.used_bytes, 0.10790000000000001, rel_tol=1e-06)

def test_get_snapshot_rest_required_fields(request_type=snapshot.GetSnapshotRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = snapshot.Snapshot()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = snapshot.Snapshot.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_snapshot(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_snapshot_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_snapshot._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_snapshot_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetAppRestInterceptor, 'post_get_snapshot') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_get_snapshot') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = snapshot.GetSnapshotRequest.pb(snapshot.GetSnapshotRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = snapshot.Snapshot.to_json(snapshot.Snapshot())
        request = snapshot.GetSnapshotRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = snapshot.Snapshot()
        client.get_snapshot(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_snapshot_rest_bad_request(transport: str='rest', request_type=snapshot.GetSnapshotRequest):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_snapshot(request)

def test_get_snapshot_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = snapshot.Snapshot()
        sample_request = {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = snapshot.Snapshot.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_snapshot(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/volumes/*/snapshots/*}' % client.transport._host, args[1])

def test_get_snapshot_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_snapshot(snapshot.GetSnapshotRequest(), name='name_value')

def test_get_snapshot_rest_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_snapshot.CreateSnapshotRequest, dict])
def test_create_snapshot_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
    request_init['snapshot'] = {'name': 'name_value', 'state': 1, 'state_details': 'state_details_value', 'description': 'description_value', 'used_bytes': 0.10790000000000001, 'create_time': {'seconds': 751, 'nanos': 543}, 'labels': {}}
    test_field = gcn_snapshot.CreateSnapshotRequest.meta.fields['snapshot']

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
    for (field, value) in request_init['snapshot'].items():
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
                for i in range(0, len(request_init['snapshot'][field])):
                    del request_init['snapshot'][field][i][subfield]
            else:
                del request_init['snapshot'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_snapshot(request)
    assert response.operation.name == 'operations/spam'

def test_create_snapshot_rest_required_fields(request_type=gcn_snapshot.CreateSnapshotRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['snapshot_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'snapshotId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'snapshotId' in jsonified_request
    assert jsonified_request['snapshotId'] == request_init['snapshot_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['snapshotId'] = 'snapshot_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_snapshot._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('snapshot_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'snapshotId' in jsonified_request
    assert jsonified_request['snapshotId'] == 'snapshot_id_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_snapshot(request)
            expected_params = [('snapshotId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_snapshot_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_snapshot._get_unset_required_fields({})
    assert set(unset_fields) == set(('snapshotId',)) & set(('parent', 'snapshot', 'snapshotId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_snapshot_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_create_snapshot') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_create_snapshot') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_snapshot.CreateSnapshotRequest.pb(gcn_snapshot.CreateSnapshotRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_snapshot.CreateSnapshotRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_snapshot(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_snapshot_rest_bad_request(transport: str='rest', request_type=gcn_snapshot.CreateSnapshotRequest):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_snapshot(request)

def test_create_snapshot_rest_flattened():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
        mock_args = dict(parent='parent_value', snapshot=gcn_snapshot.Snapshot(name='name_value'), snapshot_id='snapshot_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_snapshot(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/volumes/*}/snapshots' % client.transport._host, args[1])

def test_create_snapshot_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_snapshot(gcn_snapshot.CreateSnapshotRequest(), parent='parent_value', snapshot=gcn_snapshot.Snapshot(name='name_value'), snapshot_id='snapshot_id_value')

def test_create_snapshot_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [snapshot.DeleteSnapshotRequest, dict])
def test_delete_snapshot_rest(request_type):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_snapshot(request)
    assert response.operation.name == 'operations/spam'

def test_delete_snapshot_rest_required_fields(request_type=snapshot.DeleteSnapshotRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_snapshot(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_snapshot_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_snapshot._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_snapshot_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_delete_snapshot') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_delete_snapshot') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = snapshot.DeleteSnapshotRequest.pb(snapshot.DeleteSnapshotRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = snapshot.DeleteSnapshotRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_snapshot(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_snapshot_rest_bad_request(transport: str='rest', request_type=snapshot.DeleteSnapshotRequest):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_snapshot(request)

def test_delete_snapshot_rest_flattened():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_snapshot(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/volumes/*/snapshots/*}' % client.transport._host, args[1])

def test_delete_snapshot_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_snapshot(snapshot.DeleteSnapshotRequest(), name='name_value')

def test_delete_snapshot_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_snapshot.UpdateSnapshotRequest, dict])
def test_update_snapshot_rest(request_type):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'snapshot': {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}}
    request_init['snapshot'] = {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4', 'state': 1, 'state_details': 'state_details_value', 'description': 'description_value', 'used_bytes': 0.10790000000000001, 'create_time': {'seconds': 751, 'nanos': 543}, 'labels': {}}
    test_field = gcn_snapshot.UpdateSnapshotRequest.meta.fields['snapshot']

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
    for (field, value) in request_init['snapshot'].items():
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
                for i in range(0, len(request_init['snapshot'][field])):
                    del request_init['snapshot'][field][i][subfield]
            else:
                del request_init['snapshot'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_snapshot(request)
    assert response.operation.name == 'operations/spam'

def test_update_snapshot_rest_required_fields(request_type=gcn_snapshot.UpdateSnapshotRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_snapshot._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_snapshot(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_snapshot_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_snapshot._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('updateMask', 'snapshot'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_snapshot_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_update_snapshot') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_update_snapshot') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_snapshot.UpdateSnapshotRequest.pb(gcn_snapshot.UpdateSnapshotRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_snapshot.UpdateSnapshotRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_snapshot(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_snapshot_rest_bad_request(transport: str='rest', request_type=gcn_snapshot.UpdateSnapshotRequest):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'snapshot': {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_snapshot(request)

def test_update_snapshot_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'snapshot': {'name': 'projects/sample1/locations/sample2/volumes/sample3/snapshots/sample4'}}
        mock_args = dict(snapshot=gcn_snapshot.Snapshot(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_snapshot(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{snapshot.name=projects/*/locations/*/volumes/*/snapshots/*}' % client.transport._host, args[1])

def test_update_snapshot_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_snapshot(gcn_snapshot.UpdateSnapshotRequest(), snapshot=gcn_snapshot.Snapshot(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_snapshot_rest_error():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [active_directory.ListActiveDirectoriesRequest, dict])
def test_list_active_directories_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = active_directory.ListActiveDirectoriesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = active_directory.ListActiveDirectoriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_active_directories(request)
    assert isinstance(response, pagers.ListActiveDirectoriesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_active_directories_rest_required_fields(request_type=active_directory.ListActiveDirectoriesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_active_directories._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_active_directories._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = active_directory.ListActiveDirectoriesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = active_directory.ListActiveDirectoriesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_active_directories(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_active_directories_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_active_directories._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_active_directories_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetAppRestInterceptor, 'post_list_active_directories') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_list_active_directories') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = active_directory.ListActiveDirectoriesRequest.pb(active_directory.ListActiveDirectoriesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = active_directory.ListActiveDirectoriesResponse.to_json(active_directory.ListActiveDirectoriesResponse())
        request = active_directory.ListActiveDirectoriesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = active_directory.ListActiveDirectoriesResponse()
        client.list_active_directories(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_active_directories_rest_bad_request(transport: str='rest', request_type=active_directory.ListActiveDirectoriesRequest):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_active_directories(request)

def test_list_active_directories_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = active_directory.ListActiveDirectoriesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = active_directory.ListActiveDirectoriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_active_directories(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/activeDirectories' % client.transport._host, args[1])

def test_list_active_directories_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_active_directories(active_directory.ListActiveDirectoriesRequest(), parent='parent_value')

def test_list_active_directories_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory(), active_directory.ActiveDirectory(), active_directory.ActiveDirectory()], next_page_token='abc'), active_directory.ListActiveDirectoriesResponse(active_directories=[], next_page_token='def'), active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory()], next_page_token='ghi'), active_directory.ListActiveDirectoriesResponse(active_directories=[active_directory.ActiveDirectory(), active_directory.ActiveDirectory()]))
        response = response + response
        response = tuple((active_directory.ListActiveDirectoriesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_active_directories(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, active_directory.ActiveDirectory) for i in results))
        pages = list(client.list_active_directories(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [active_directory.GetActiveDirectoryRequest, dict])
def test_get_active_directory_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/activeDirectories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = active_directory.ActiveDirectory(name='name_value', state=active_directory.ActiveDirectory.State.CREATING, domain='domain_value', site='site_value', dns='dns_value', net_bios_prefix='net_bios_prefix_value', organizational_unit='organizational_unit_value', aes_encryption=True, username='username_value', password='password_value', backup_operators=['backup_operators_value'], security_operators=['security_operators_value'], kdc_hostname='kdc_hostname_value', kdc_ip='kdc_ip_value', nfs_users_with_ldap=True, description='description_value', ldap_signing=True, encrypt_dc_connections=True, state_details='state_details_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = active_directory.ActiveDirectory.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_active_directory(request)
    assert isinstance(response, active_directory.ActiveDirectory)
    assert response.name == 'name_value'
    assert response.state == active_directory.ActiveDirectory.State.CREATING
    assert response.domain == 'domain_value'
    assert response.site == 'site_value'
    assert response.dns == 'dns_value'
    assert response.net_bios_prefix == 'net_bios_prefix_value'
    assert response.organizational_unit == 'organizational_unit_value'
    assert response.aes_encryption is True
    assert response.username == 'username_value'
    assert response.password == 'password_value'
    assert response.backup_operators == ['backup_operators_value']
    assert response.security_operators == ['security_operators_value']
    assert response.kdc_hostname == 'kdc_hostname_value'
    assert response.kdc_ip == 'kdc_ip_value'
    assert response.nfs_users_with_ldap is True
    assert response.description == 'description_value'
    assert response.ldap_signing is True
    assert response.encrypt_dc_connections is True
    assert response.state_details == 'state_details_value'

def test_get_active_directory_rest_required_fields(request_type=active_directory.GetActiveDirectoryRequest):
    if False:
        return 10
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_active_directory._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_active_directory._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = active_directory.ActiveDirectory()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = active_directory.ActiveDirectory.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_active_directory(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_active_directory_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_active_directory._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_active_directory_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetAppRestInterceptor, 'post_get_active_directory') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_get_active_directory') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = active_directory.GetActiveDirectoryRequest.pb(active_directory.GetActiveDirectoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = active_directory.ActiveDirectory.to_json(active_directory.ActiveDirectory())
        request = active_directory.GetActiveDirectoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = active_directory.ActiveDirectory()
        client.get_active_directory(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_active_directory_rest_bad_request(transport: str='rest', request_type=active_directory.GetActiveDirectoryRequest):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/activeDirectories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_active_directory(request)

def test_get_active_directory_rest_flattened():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = active_directory.ActiveDirectory()
        sample_request = {'name': 'projects/sample1/locations/sample2/activeDirectories/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = active_directory.ActiveDirectory.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_active_directory(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/activeDirectories/*}' % client.transport._host, args[1])

def test_get_active_directory_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_active_directory(active_directory.GetActiveDirectoryRequest(), name='name_value')

def test_get_active_directory_rest_error():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_active_directory.CreateActiveDirectoryRequest, dict])
def test_create_active_directory_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['active_directory'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'state': 1, 'domain': 'domain_value', 'site': 'site_value', 'dns': 'dns_value', 'net_bios_prefix': 'net_bios_prefix_value', 'organizational_unit': 'organizational_unit_value', 'aes_encryption': True, 'username': 'username_value', 'password': 'password_value', 'backup_operators': ['backup_operators_value1', 'backup_operators_value2'], 'security_operators': ['security_operators_value1', 'security_operators_value2'], 'kdc_hostname': 'kdc_hostname_value', 'kdc_ip': 'kdc_ip_value', 'nfs_users_with_ldap': True, 'description': 'description_value', 'ldap_signing': True, 'encrypt_dc_connections': True, 'labels': {}, 'state_details': 'state_details_value'}
    test_field = gcn_active_directory.CreateActiveDirectoryRequest.meta.fields['active_directory']

    def get_message_fields(field):
        if False:
            return 10
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
    for (field, value) in request_init['active_directory'].items():
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
                for i in range(0, len(request_init['active_directory'][field])):
                    del request_init['active_directory'][field][i][subfield]
            else:
                del request_init['active_directory'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_active_directory(request)
    assert response.operation.name == 'operations/spam'

def test_create_active_directory_rest_required_fields(request_type=gcn_active_directory.CreateActiveDirectoryRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['active_directory_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'activeDirectoryId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_active_directory._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'activeDirectoryId' in jsonified_request
    assert jsonified_request['activeDirectoryId'] == request_init['active_directory_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['activeDirectoryId'] = 'active_directory_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_active_directory._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('active_directory_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'activeDirectoryId' in jsonified_request
    assert jsonified_request['activeDirectoryId'] == 'active_directory_id_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_active_directory(request)
            expected_params = [('activeDirectoryId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_active_directory_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_active_directory._get_unset_required_fields({})
    assert set(unset_fields) == set(('activeDirectoryId',)) & set(('parent', 'activeDirectory', 'activeDirectoryId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_active_directory_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_create_active_directory') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_create_active_directory') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_active_directory.CreateActiveDirectoryRequest.pb(gcn_active_directory.CreateActiveDirectoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_active_directory.CreateActiveDirectoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_active_directory(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_active_directory_rest_bad_request(transport: str='rest', request_type=gcn_active_directory.CreateActiveDirectoryRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_active_directory(request)

def test_create_active_directory_rest_flattened():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', active_directory=gcn_active_directory.ActiveDirectory(name='name_value'), active_directory_id='active_directory_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_active_directory(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/activeDirectories' % client.transport._host, args[1])

def test_create_active_directory_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_active_directory(gcn_active_directory.CreateActiveDirectoryRequest(), parent='parent_value', active_directory=gcn_active_directory.ActiveDirectory(name='name_value'), active_directory_id='active_directory_id_value')

def test_create_active_directory_rest_error():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_active_directory.UpdateActiveDirectoryRequest, dict])
def test_update_active_directory_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'active_directory': {'name': 'projects/sample1/locations/sample2/activeDirectories/sample3'}}
    request_init['active_directory'] = {'name': 'projects/sample1/locations/sample2/activeDirectories/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'state': 1, 'domain': 'domain_value', 'site': 'site_value', 'dns': 'dns_value', 'net_bios_prefix': 'net_bios_prefix_value', 'organizational_unit': 'organizational_unit_value', 'aes_encryption': True, 'username': 'username_value', 'password': 'password_value', 'backup_operators': ['backup_operators_value1', 'backup_operators_value2'], 'security_operators': ['security_operators_value1', 'security_operators_value2'], 'kdc_hostname': 'kdc_hostname_value', 'kdc_ip': 'kdc_ip_value', 'nfs_users_with_ldap': True, 'description': 'description_value', 'ldap_signing': True, 'encrypt_dc_connections': True, 'labels': {}, 'state_details': 'state_details_value'}
    test_field = gcn_active_directory.UpdateActiveDirectoryRequest.meta.fields['active_directory']

    def get_message_fields(field):
        if False:
            return 10
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
    for (field, value) in request_init['active_directory'].items():
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
                for i in range(0, len(request_init['active_directory'][field])):
                    del request_init['active_directory'][field][i][subfield]
            else:
                del request_init['active_directory'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_active_directory(request)
    assert response.operation.name == 'operations/spam'

def test_update_active_directory_rest_required_fields(request_type=gcn_active_directory.UpdateActiveDirectoryRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_active_directory._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_active_directory._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_active_directory(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_active_directory_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_active_directory._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('updateMask', 'activeDirectory'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_active_directory_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_update_active_directory') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_update_active_directory') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_active_directory.UpdateActiveDirectoryRequest.pb(gcn_active_directory.UpdateActiveDirectoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_active_directory.UpdateActiveDirectoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_active_directory(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_active_directory_rest_bad_request(transport: str='rest', request_type=gcn_active_directory.UpdateActiveDirectoryRequest):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'active_directory': {'name': 'projects/sample1/locations/sample2/activeDirectories/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_active_directory(request)

def test_update_active_directory_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'active_directory': {'name': 'projects/sample1/locations/sample2/activeDirectories/sample3'}}
        mock_args = dict(active_directory=gcn_active_directory.ActiveDirectory(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_active_directory(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{active_directory.name=projects/*/locations/*/activeDirectories/*}' % client.transport._host, args[1])

def test_update_active_directory_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_active_directory(gcn_active_directory.UpdateActiveDirectoryRequest(), active_directory=gcn_active_directory.ActiveDirectory(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_active_directory_rest_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [active_directory.DeleteActiveDirectoryRequest, dict])
def test_delete_active_directory_rest(request_type):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/activeDirectories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_active_directory(request)
    assert response.operation.name == 'operations/spam'

def test_delete_active_directory_rest_required_fields(request_type=active_directory.DeleteActiveDirectoryRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_active_directory._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_active_directory._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_active_directory(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_active_directory_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_active_directory._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_active_directory_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_delete_active_directory') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_delete_active_directory') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = active_directory.DeleteActiveDirectoryRequest.pb(active_directory.DeleteActiveDirectoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = active_directory.DeleteActiveDirectoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_active_directory(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_active_directory_rest_bad_request(transport: str='rest', request_type=active_directory.DeleteActiveDirectoryRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/activeDirectories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_active_directory(request)

def test_delete_active_directory_rest_flattened():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/activeDirectories/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_active_directory(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/activeDirectories/*}' % client.transport._host, args[1])

def test_delete_active_directory_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_active_directory(active_directory.DeleteActiveDirectoryRequest(), name='name_value')

def test_delete_active_directory_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [kms.ListKmsConfigsRequest, dict])
def test_list_kms_configs_rest(request_type):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = kms.ListKmsConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = kms.ListKmsConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_kms_configs(request)
    assert isinstance(response, pagers.ListKmsConfigsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_kms_configs_rest_required_fields(request_type=kms.ListKmsConfigsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_kms_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_kms_configs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = kms.ListKmsConfigsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = kms.ListKmsConfigsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_kms_configs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_kms_configs_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_kms_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_kms_configs_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetAppRestInterceptor, 'post_list_kms_configs') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_list_kms_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = kms.ListKmsConfigsRequest.pb(kms.ListKmsConfigsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = kms.ListKmsConfigsResponse.to_json(kms.ListKmsConfigsResponse())
        request = kms.ListKmsConfigsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = kms.ListKmsConfigsResponse()
        client.list_kms_configs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_kms_configs_rest_bad_request(transport: str='rest', request_type=kms.ListKmsConfigsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_kms_configs(request)

def test_list_kms_configs_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = kms.ListKmsConfigsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = kms.ListKmsConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_kms_configs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/kmsConfigs' % client.transport._host, args[1])

def test_list_kms_configs_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_kms_configs(kms.ListKmsConfigsRequest(), parent='parent_value')

def test_list_kms_configs_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig(), kms.KmsConfig(), kms.KmsConfig()], next_page_token='abc'), kms.ListKmsConfigsResponse(kms_configs=[], next_page_token='def'), kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig()], next_page_token='ghi'), kms.ListKmsConfigsResponse(kms_configs=[kms.KmsConfig(), kms.KmsConfig()]))
        response = response + response
        response = tuple((kms.ListKmsConfigsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_kms_configs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, kms.KmsConfig) for i in results))
        pages = list(client.list_kms_configs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [kms.CreateKmsConfigRequest, dict])
def test_create_kms_config_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['kms_config'] = {'name': 'name_value', 'crypto_key_name': 'crypto_key_name_value', 'state': 1, 'state_details': 'state_details_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'description': 'description_value', 'labels': {}, 'instructions': 'instructions_value', 'service_account': 'service_account_value'}
    test_field = kms.CreateKmsConfigRequest.meta.fields['kms_config']

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
    for (field, value) in request_init['kms_config'].items():
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
                for i in range(0, len(request_init['kms_config'][field])):
                    del request_init['kms_config'][field][i][subfield]
            else:
                del request_init['kms_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_kms_config(request)
    assert response.operation.name == 'operations/spam'

def test_create_kms_config_rest_required_fields(request_type=kms.CreateKmsConfigRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['kms_config_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'kmsConfigId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_kms_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'kmsConfigId' in jsonified_request
    assert jsonified_request['kmsConfigId'] == request_init['kms_config_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['kmsConfigId'] = 'kms_config_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_kms_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('kms_config_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'kmsConfigId' in jsonified_request
    assert jsonified_request['kmsConfigId'] == 'kms_config_id_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_kms_config(request)
            expected_params = [('kmsConfigId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_kms_config_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_kms_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('kmsConfigId',)) & set(('parent', 'kmsConfigId', 'kmsConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_kms_config_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_create_kms_config') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_create_kms_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = kms.CreateKmsConfigRequest.pb(kms.CreateKmsConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = kms.CreateKmsConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_kms_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_kms_config_rest_bad_request(transport: str='rest', request_type=kms.CreateKmsConfigRequest):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_kms_config(request)

def test_create_kms_config_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', kms_config=kms.KmsConfig(name='name_value'), kms_config_id='kms_config_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_kms_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/kmsConfigs' % client.transport._host, args[1])

def test_create_kms_config_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_kms_config(kms.CreateKmsConfigRequest(), parent='parent_value', kms_config=kms.KmsConfig(name='name_value'), kms_config_id='kms_config_id_value')

def test_create_kms_config_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [kms.GetKmsConfigRequest, dict])
def test_get_kms_config_rest(request_type):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/kmsConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = kms.KmsConfig(name='name_value', crypto_key_name='crypto_key_name_value', state=kms.KmsConfig.State.READY, state_details='state_details_value', description='description_value', instructions='instructions_value', service_account='service_account_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = kms.KmsConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_kms_config(request)
    assert isinstance(response, kms.KmsConfig)
    assert response.name == 'name_value'
    assert response.crypto_key_name == 'crypto_key_name_value'
    assert response.state == kms.KmsConfig.State.READY
    assert response.state_details == 'state_details_value'
    assert response.description == 'description_value'
    assert response.instructions == 'instructions_value'
    assert response.service_account == 'service_account_value'

def test_get_kms_config_rest_required_fields(request_type=kms.GetKmsConfigRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_kms_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_kms_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = kms.KmsConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = kms.KmsConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_kms_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_kms_config_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_kms_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_kms_config_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetAppRestInterceptor, 'post_get_kms_config') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_get_kms_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = kms.GetKmsConfigRequest.pb(kms.GetKmsConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = kms.KmsConfig.to_json(kms.KmsConfig())
        request = kms.GetKmsConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = kms.KmsConfig()
        client.get_kms_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_kms_config_rest_bad_request(transport: str='rest', request_type=kms.GetKmsConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/kmsConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_kms_config(request)

def test_get_kms_config_rest_flattened():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = kms.KmsConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/kmsConfigs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = kms.KmsConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_kms_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/kmsConfigs/*}' % client.transport._host, args[1])

def test_get_kms_config_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_kms_config(kms.GetKmsConfigRequest(), name='name_value')

def test_get_kms_config_rest_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [kms.UpdateKmsConfigRequest, dict])
def test_update_kms_config_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'kms_config': {'name': 'projects/sample1/locations/sample2/kmsConfigs/sample3'}}
    request_init['kms_config'] = {'name': 'projects/sample1/locations/sample2/kmsConfigs/sample3', 'crypto_key_name': 'crypto_key_name_value', 'state': 1, 'state_details': 'state_details_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'description': 'description_value', 'labels': {}, 'instructions': 'instructions_value', 'service_account': 'service_account_value'}
    test_field = kms.UpdateKmsConfigRequest.meta.fields['kms_config']

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
    for (field, value) in request_init['kms_config'].items():
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
                for i in range(0, len(request_init['kms_config'][field])):
                    del request_init['kms_config'][field][i][subfield]
            else:
                del request_init['kms_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_kms_config(request)
    assert response.operation.name == 'operations/spam'

def test_update_kms_config_rest_required_fields(request_type=kms.UpdateKmsConfigRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_kms_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_kms_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_kms_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_kms_config_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_kms_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('updateMask', 'kmsConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_kms_config_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_update_kms_config') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_update_kms_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = kms.UpdateKmsConfigRequest.pb(kms.UpdateKmsConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = kms.UpdateKmsConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_kms_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_kms_config_rest_bad_request(transport: str='rest', request_type=kms.UpdateKmsConfigRequest):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'kms_config': {'name': 'projects/sample1/locations/sample2/kmsConfigs/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_kms_config(request)

def test_update_kms_config_rest_flattened():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'kms_config': {'name': 'projects/sample1/locations/sample2/kmsConfigs/sample3'}}
        mock_args = dict(kms_config=kms.KmsConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_kms_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{kms_config.name=projects/*/locations/*/kmsConfigs/*}' % client.transport._host, args[1])

def test_update_kms_config_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_kms_config(kms.UpdateKmsConfigRequest(), kms_config=kms.KmsConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_kms_config_rest_error():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [kms.EncryptVolumesRequest, dict])
def test_encrypt_volumes_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/kmsConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.encrypt_volumes(request)
    assert response.operation.name == 'operations/spam'

def test_encrypt_volumes_rest_required_fields(request_type=kms.EncryptVolumesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).encrypt_volumes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).encrypt_volumes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.encrypt_volumes(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_encrypt_volumes_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.encrypt_volumes._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_encrypt_volumes_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_encrypt_volumes') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_encrypt_volumes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = kms.EncryptVolumesRequest.pb(kms.EncryptVolumesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = kms.EncryptVolumesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.encrypt_volumes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_encrypt_volumes_rest_bad_request(transport: str='rest', request_type=kms.EncryptVolumesRequest):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/kmsConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.encrypt_volumes(request)

def test_encrypt_volumes_rest_error():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [kms.VerifyKmsConfigRequest, dict])
def test_verify_kms_config_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/kmsConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = kms.VerifyKmsConfigResponse(healthy=True, health_error='health_error_value', instructions='instructions_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = kms.VerifyKmsConfigResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.verify_kms_config(request)
    assert isinstance(response, kms.VerifyKmsConfigResponse)
    assert response.healthy is True
    assert response.health_error == 'health_error_value'
    assert response.instructions == 'instructions_value'

def test_verify_kms_config_rest_required_fields(request_type=kms.VerifyKmsConfigRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).verify_kms_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).verify_kms_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = kms.VerifyKmsConfigResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = kms.VerifyKmsConfigResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.verify_kms_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_verify_kms_config_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.verify_kms_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_verify_kms_config_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetAppRestInterceptor, 'post_verify_kms_config') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_verify_kms_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = kms.VerifyKmsConfigRequest.pb(kms.VerifyKmsConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = kms.VerifyKmsConfigResponse.to_json(kms.VerifyKmsConfigResponse())
        request = kms.VerifyKmsConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = kms.VerifyKmsConfigResponse()
        client.verify_kms_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_verify_kms_config_rest_bad_request(transport: str='rest', request_type=kms.VerifyKmsConfigRequest):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/kmsConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.verify_kms_config(request)

def test_verify_kms_config_rest_error():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [kms.DeleteKmsConfigRequest, dict])
def test_delete_kms_config_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/kmsConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_kms_config(request)
    assert response.operation.name == 'operations/spam'

def test_delete_kms_config_rest_required_fields(request_type=kms.DeleteKmsConfigRequest):
    if False:
        return 10
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_kms_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_kms_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_kms_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_kms_config_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_kms_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_kms_config_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_delete_kms_config') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_delete_kms_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = kms.DeleteKmsConfigRequest.pb(kms.DeleteKmsConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = kms.DeleteKmsConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_kms_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_kms_config_rest_bad_request(transport: str='rest', request_type=kms.DeleteKmsConfigRequest):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/kmsConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_kms_config(request)

def test_delete_kms_config_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/kmsConfigs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_kms_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/kmsConfigs/*}' % client.transport._host, args[1])

def test_delete_kms_config_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_kms_config(kms.DeleteKmsConfigRequest(), name='name_value')

def test_delete_kms_config_rest_error():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [replication.ListReplicationsRequest, dict])
def test_list_replications_rest(request_type):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = replication.ListReplicationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = replication.ListReplicationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_replications(request)
    assert isinstance(response, pagers.ListReplicationsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_replications_rest_required_fields(request_type=replication.ListReplicationsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_replications._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_replications._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = replication.ListReplicationsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = replication.ListReplicationsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_replications(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_replications_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_replications._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_replications_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetAppRestInterceptor, 'post_list_replications') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_list_replications') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = replication.ListReplicationsRequest.pb(replication.ListReplicationsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = replication.ListReplicationsResponse.to_json(replication.ListReplicationsResponse())
        request = replication.ListReplicationsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = replication.ListReplicationsResponse()
        client.list_replications(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_replications_rest_bad_request(transport: str='rest', request_type=replication.ListReplicationsRequest):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_replications(request)

def test_list_replications_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = replication.ListReplicationsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = replication.ListReplicationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_replications(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/volumes/*}/replications' % client.transport._host, args[1])

def test_list_replications_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_replications(replication.ListReplicationsRequest(), parent='parent_value')

def test_list_replications_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (replication.ListReplicationsResponse(replications=[replication.Replication(), replication.Replication(), replication.Replication()], next_page_token='abc'), replication.ListReplicationsResponse(replications=[], next_page_token='def'), replication.ListReplicationsResponse(replications=[replication.Replication()], next_page_token='ghi'), replication.ListReplicationsResponse(replications=[replication.Replication(), replication.Replication()]))
        response = response + response
        response = tuple((replication.ListReplicationsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
        pager = client.list_replications(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, replication.Replication) for i in results))
        pages = list(client.list_replications(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [replication.GetReplicationRequest, dict])
def test_get_replication_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = replication.Replication(name='name_value', state=replication.Replication.State.CREATING, state_details='state_details_value', role=replication.Replication.ReplicationRole.SOURCE, replication_schedule=replication.Replication.ReplicationSchedule.EVERY_10_MINUTES, mirror_state=replication.Replication.MirrorState.PREPARING, healthy=True, destination_volume='destination_volume_value', description='description_value', source_volume='source_volume_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = replication.Replication.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_replication(request)
    assert isinstance(response, replication.Replication)
    assert response.name == 'name_value'
    assert response.state == replication.Replication.State.CREATING
    assert response.state_details == 'state_details_value'
    assert response.role == replication.Replication.ReplicationRole.SOURCE
    assert response.replication_schedule == replication.Replication.ReplicationSchedule.EVERY_10_MINUTES
    assert response.mirror_state == replication.Replication.MirrorState.PREPARING
    assert response.healthy is True
    assert response.destination_volume == 'destination_volume_value'
    assert response.description == 'description_value'
    assert response.source_volume == 'source_volume_value'

def test_get_replication_rest_required_fields(request_type=replication.GetReplicationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = replication.Replication()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = replication.Replication.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_replication(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_replication_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_replication._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_replication_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetAppRestInterceptor, 'post_get_replication') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_get_replication') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = replication.GetReplicationRequest.pb(replication.GetReplicationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = replication.Replication.to_json(replication.Replication())
        request = replication.GetReplicationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = replication.Replication()
        client.get_replication(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_replication_rest_bad_request(transport: str='rest', request_type=replication.GetReplicationRequest):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_replication(request)

def test_get_replication_rest_flattened():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = replication.Replication()
        sample_request = {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = replication.Replication.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_replication(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/volumes/*/replications/*}' % client.transport._host, args[1])

def test_get_replication_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_replication(replication.GetReplicationRequest(), name='name_value')

def test_get_replication_rest_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_replication.CreateReplicationRequest, dict])
def test_create_replication_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
    request_init['replication'] = {'name': 'name_value', 'state': 1, 'state_details': 'state_details_value', 'role': 1, 'replication_schedule': 1, 'mirror_state': 1, 'healthy': True, 'create_time': {'seconds': 751, 'nanos': 543}, 'destination_volume': 'destination_volume_value', 'transfer_stats': {'transfer_bytes': 1515, 'total_transfer_duration': {'seconds': 751, 'nanos': 543}, 'last_transfer_bytes': 2046, 'last_transfer_duration': {}, 'lag_duration': {}, 'update_time': {}, 'last_transfer_end_time': {}, 'last_transfer_error': 'last_transfer_error_value'}, 'labels': {}, 'description': 'description_value', 'destination_volume_parameters': {'storage_pool': 'storage_pool_value', 'volume_id': 'volume_id_value', 'share_name': 'share_name_value', 'description': 'description_value'}, 'source_volume': 'source_volume_value'}
    test_field = gcn_replication.CreateReplicationRequest.meta.fields['replication']

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
    for (field, value) in request_init['replication'].items():
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
                for i in range(0, len(request_init['replication'][field])):
                    del request_init['replication'][field][i][subfield]
            else:
                del request_init['replication'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_replication(request)
    assert response.operation.name == 'operations/spam'

def test_create_replication_rest_required_fields(request_type=gcn_replication.CreateReplicationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['replication_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'replicationId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'replicationId' in jsonified_request
    assert jsonified_request['replicationId'] == request_init['replication_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['replicationId'] = 'replication_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_replication._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('replication_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'replicationId' in jsonified_request
    assert jsonified_request['replicationId'] == 'replication_id_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_replication(request)
            expected_params = [('replicationId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_replication_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_replication._get_unset_required_fields({})
    assert set(unset_fields) == set(('replicationId',)) & set(('parent', 'replication', 'replicationId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_replication_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_create_replication') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_create_replication') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_replication.CreateReplicationRequest.pb(gcn_replication.CreateReplicationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_replication.CreateReplicationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_replication(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_replication_rest_bad_request(transport: str='rest', request_type=gcn_replication.CreateReplicationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_replication(request)

def test_create_replication_rest_flattened():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/volumes/sample3'}
        mock_args = dict(parent='parent_value', replication=gcn_replication.Replication(name='name_value'), replication_id='replication_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_replication(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/volumes/*}/replications' % client.transport._host, args[1])

def test_create_replication_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_replication(gcn_replication.CreateReplicationRequest(), parent='parent_value', replication=gcn_replication.Replication(name='name_value'), replication_id='replication_id_value')

def test_create_replication_rest_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [replication.DeleteReplicationRequest, dict])
def test_delete_replication_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_replication(request)
    assert response.operation.name == 'operations/spam'

def test_delete_replication_rest_required_fields(request_type=replication.DeleteReplicationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_replication(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_replication_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_replication._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_replication_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_delete_replication') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_delete_replication') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = replication.DeleteReplicationRequest.pb(replication.DeleteReplicationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = replication.DeleteReplicationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_replication(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_replication_rest_bad_request(transport: str='rest', request_type=replication.DeleteReplicationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_replication(request)

def test_delete_replication_rest_flattened():
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_replication(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/volumes/*/replications/*}' % client.transport._host, args[1])

def test_delete_replication_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_replication(replication.DeleteReplicationRequest(), name='name_value')

def test_delete_replication_rest_error():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_replication.UpdateReplicationRequest, dict])
def test_update_replication_rest(request_type):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'replication': {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}}
    request_init['replication'] = {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4', 'state': 1, 'state_details': 'state_details_value', 'role': 1, 'replication_schedule': 1, 'mirror_state': 1, 'healthy': True, 'create_time': {'seconds': 751, 'nanos': 543}, 'destination_volume': 'destination_volume_value', 'transfer_stats': {'transfer_bytes': 1515, 'total_transfer_duration': {'seconds': 751, 'nanos': 543}, 'last_transfer_bytes': 2046, 'last_transfer_duration': {}, 'lag_duration': {}, 'update_time': {}, 'last_transfer_end_time': {}, 'last_transfer_error': 'last_transfer_error_value'}, 'labels': {}, 'description': 'description_value', 'destination_volume_parameters': {'storage_pool': 'storage_pool_value', 'volume_id': 'volume_id_value', 'share_name': 'share_name_value', 'description': 'description_value'}, 'source_volume': 'source_volume_value'}
    test_field = gcn_replication.UpdateReplicationRequest.meta.fields['replication']

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
    for (field, value) in request_init['replication'].items():
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
                for i in range(0, len(request_init['replication'][field])):
                    del request_init['replication'][field][i][subfield]
            else:
                del request_init['replication'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_replication(request)
    assert response.operation.name == 'operations/spam'

def test_update_replication_rest_required_fields(request_type=gcn_replication.UpdateReplicationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_replication._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_replication(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_replication_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_replication._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('updateMask', 'replication'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_replication_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_update_replication') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_update_replication') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_replication.UpdateReplicationRequest.pb(gcn_replication.UpdateReplicationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_replication.UpdateReplicationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_replication(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_replication_rest_bad_request(transport: str='rest', request_type=gcn_replication.UpdateReplicationRequest):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'replication': {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_replication(request)

def test_update_replication_rest_flattened():
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'replication': {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}}
        mock_args = dict(replication=gcn_replication.Replication(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_replication(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{replication.name=projects/*/locations/*/volumes/*/replications/*}' % client.transport._host, args[1])

def test_update_replication_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_replication(gcn_replication.UpdateReplicationRequest(), replication=gcn_replication.Replication(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_replication_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [replication.StopReplicationRequest, dict])
def test_stop_replication_rest(request_type):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.stop_replication(request)
    assert response.operation.name == 'operations/spam'

def test_stop_replication_rest_required_fields(request_type=replication.StopReplicationRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.stop_replication(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_stop_replication_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.stop_replication._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_stop_replication_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_stop_replication') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_stop_replication') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = replication.StopReplicationRequest.pb(replication.StopReplicationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = replication.StopReplicationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.stop_replication(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_stop_replication_rest_bad_request(transport: str='rest', request_type=replication.StopReplicationRequest):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.stop_replication(request)

def test_stop_replication_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [replication.ResumeReplicationRequest, dict])
def test_resume_replication_rest(request_type):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.resume_replication(request)
    assert response.operation.name == 'operations/spam'

def test_resume_replication_rest_required_fields(request_type=replication.ResumeReplicationRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resume_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resume_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.resume_replication(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_resume_replication_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.resume_replication._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_resume_replication_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_resume_replication') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_resume_replication') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = replication.ResumeReplicationRequest.pb(replication.ResumeReplicationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = replication.ResumeReplicationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.resume_replication(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_resume_replication_rest_bad_request(transport: str='rest', request_type=replication.ResumeReplicationRequest):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.resume_replication(request)

def test_resume_replication_rest_error():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [replication.ReverseReplicationDirectionRequest, dict])
def test_reverse_replication_direction_rest(request_type):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.reverse_replication_direction(request)
    assert response.operation.name == 'operations/spam'

def test_reverse_replication_direction_rest_required_fields(request_type=replication.ReverseReplicationDirectionRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetAppRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).reverse_replication_direction._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).reverse_replication_direction._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.reverse_replication_direction(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_reverse_replication_direction_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.reverse_replication_direction._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_reverse_replication_direction_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetAppRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetAppRestInterceptor())
    client = NetAppClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetAppRestInterceptor, 'post_reverse_replication_direction') as post, mock.patch.object(transports.NetAppRestInterceptor, 'pre_reverse_replication_direction') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = replication.ReverseReplicationDirectionRequest.pb(replication.ReverseReplicationDirectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = replication.ReverseReplicationDirectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.reverse_replication_direction(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_reverse_replication_direction_rest_bad_request(transport: str='rest', request_type=replication.ReverseReplicationDirectionRequest):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/volumes/sample3/replications/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.reverse_replication_direction(request)

def test_reverse_replication_direction_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetAppGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.NetAppGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NetAppClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.NetAppGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = NetAppClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = NetAppClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.NetAppGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NetAppClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.NetAppGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = NetAppClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.NetAppGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.NetAppGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.NetAppGrpcTransport, transports.NetAppGrpcAsyncIOTransport, transports.NetAppRestTransport])
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
    transport = NetAppClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.NetAppGrpcTransport)

def test_net_app_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.NetAppTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_net_app_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.netapp_v1.services.net_app.transports.NetAppTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.NetAppTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_storage_pools', 'create_storage_pool', 'get_storage_pool', 'update_storage_pool', 'delete_storage_pool', 'list_volumes', 'get_volume', 'create_volume', 'update_volume', 'delete_volume', 'revert_volume', 'list_snapshots', 'get_snapshot', 'create_snapshot', 'delete_snapshot', 'update_snapshot', 'list_active_directories', 'get_active_directory', 'create_active_directory', 'update_active_directory', 'delete_active_directory', 'list_kms_configs', 'create_kms_config', 'get_kms_config', 'update_kms_config', 'encrypt_volumes', 'verify_kms_config', 'delete_kms_config', 'list_replications', 'get_replication', 'create_replication', 'delete_replication', 'update_replication', 'stop_replication', 'resume_replication', 'reverse_replication_direction', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_net_app_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.netapp_v1.services.net_app.transports.NetAppTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.NetAppTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_net_app_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.netapp_v1.services.net_app.transports.NetAppTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.NetAppTransport()
        adc.assert_called_once()

def test_net_app_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        NetAppClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.NetAppGrpcTransport, transports.NetAppGrpcAsyncIOTransport])
def test_net_app_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.NetAppGrpcTransport, transports.NetAppGrpcAsyncIOTransport, transports.NetAppRestTransport])
def test_net_app_transport_auth_gdch_credentials(transport_class):
    if False:
        for i in range(10):
            print('nop')
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.NetAppGrpcTransport, grpc_helpers), (transports.NetAppGrpcAsyncIOTransport, grpc_helpers_async)])
def test_net_app_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('netapp.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='netapp.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.NetAppGrpcTransport, transports.NetAppGrpcAsyncIOTransport])
def test_net_app_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_net_app_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.NetAppRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_net_app_rest_lro_client():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_net_app_host_no_port(transport_name):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='netapp.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('netapp.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://netapp.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_net_app_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='netapp.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('netapp.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://netapp.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_net_app_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = NetAppClient(credentials=creds1, transport=transport_name)
    client2 = NetAppClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_storage_pools._session
    session2 = client2.transport.list_storage_pools._session
    assert session1 != session2
    session1 = client1.transport.create_storage_pool._session
    session2 = client2.transport.create_storage_pool._session
    assert session1 != session2
    session1 = client1.transport.get_storage_pool._session
    session2 = client2.transport.get_storage_pool._session
    assert session1 != session2
    session1 = client1.transport.update_storage_pool._session
    session2 = client2.transport.update_storage_pool._session
    assert session1 != session2
    session1 = client1.transport.delete_storage_pool._session
    session2 = client2.transport.delete_storage_pool._session
    assert session1 != session2
    session1 = client1.transport.list_volumes._session
    session2 = client2.transport.list_volumes._session
    assert session1 != session2
    session1 = client1.transport.get_volume._session
    session2 = client2.transport.get_volume._session
    assert session1 != session2
    session1 = client1.transport.create_volume._session
    session2 = client2.transport.create_volume._session
    assert session1 != session2
    session1 = client1.transport.update_volume._session
    session2 = client2.transport.update_volume._session
    assert session1 != session2
    session1 = client1.transport.delete_volume._session
    session2 = client2.transport.delete_volume._session
    assert session1 != session2
    session1 = client1.transport.revert_volume._session
    session2 = client2.transport.revert_volume._session
    assert session1 != session2
    session1 = client1.transport.list_snapshots._session
    session2 = client2.transport.list_snapshots._session
    assert session1 != session2
    session1 = client1.transport.get_snapshot._session
    session2 = client2.transport.get_snapshot._session
    assert session1 != session2
    session1 = client1.transport.create_snapshot._session
    session2 = client2.transport.create_snapshot._session
    assert session1 != session2
    session1 = client1.transport.delete_snapshot._session
    session2 = client2.transport.delete_snapshot._session
    assert session1 != session2
    session1 = client1.transport.update_snapshot._session
    session2 = client2.transport.update_snapshot._session
    assert session1 != session2
    session1 = client1.transport.list_active_directories._session
    session2 = client2.transport.list_active_directories._session
    assert session1 != session2
    session1 = client1.transport.get_active_directory._session
    session2 = client2.transport.get_active_directory._session
    assert session1 != session2
    session1 = client1.transport.create_active_directory._session
    session2 = client2.transport.create_active_directory._session
    assert session1 != session2
    session1 = client1.transport.update_active_directory._session
    session2 = client2.transport.update_active_directory._session
    assert session1 != session2
    session1 = client1.transport.delete_active_directory._session
    session2 = client2.transport.delete_active_directory._session
    assert session1 != session2
    session1 = client1.transport.list_kms_configs._session
    session2 = client2.transport.list_kms_configs._session
    assert session1 != session2
    session1 = client1.transport.create_kms_config._session
    session2 = client2.transport.create_kms_config._session
    assert session1 != session2
    session1 = client1.transport.get_kms_config._session
    session2 = client2.transport.get_kms_config._session
    assert session1 != session2
    session1 = client1.transport.update_kms_config._session
    session2 = client2.transport.update_kms_config._session
    assert session1 != session2
    session1 = client1.transport.encrypt_volumes._session
    session2 = client2.transport.encrypt_volumes._session
    assert session1 != session2
    session1 = client1.transport.verify_kms_config._session
    session2 = client2.transport.verify_kms_config._session
    assert session1 != session2
    session1 = client1.transport.delete_kms_config._session
    session2 = client2.transport.delete_kms_config._session
    assert session1 != session2
    session1 = client1.transport.list_replications._session
    session2 = client2.transport.list_replications._session
    assert session1 != session2
    session1 = client1.transport.get_replication._session
    session2 = client2.transport.get_replication._session
    assert session1 != session2
    session1 = client1.transport.create_replication._session
    session2 = client2.transport.create_replication._session
    assert session1 != session2
    session1 = client1.transport.delete_replication._session
    session2 = client2.transport.delete_replication._session
    assert session1 != session2
    session1 = client1.transport.update_replication._session
    session2 = client2.transport.update_replication._session
    assert session1 != session2
    session1 = client1.transport.stop_replication._session
    session2 = client2.transport.stop_replication._session
    assert session1 != session2
    session1 = client1.transport.resume_replication._session
    session2 = client2.transport.resume_replication._session
    assert session1 != session2
    session1 = client1.transport.reverse_replication_direction._session
    session2 = client2.transport.reverse_replication_direction._session
    assert session1 != session2

def test_net_app_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.NetAppGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_net_app_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.NetAppGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.NetAppGrpcTransport, transports.NetAppGrpcAsyncIOTransport])
def test_net_app_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.NetAppGrpcTransport, transports.NetAppGrpcAsyncIOTransport])
def test_net_app_transport_channel_mtls_with_adc(transport_class):
    if False:
        print('Hello World!')
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

def test_net_app_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_net_app_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_active_directory_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    active_directory = 'whelk'
    expected = 'projects/{project}/locations/{location}/activeDirectories/{active_directory}'.format(project=project, location=location, active_directory=active_directory)
    actual = NetAppClient.active_directory_path(project, location, active_directory)
    assert expected == actual

def test_parse_active_directory_path():
    if False:
        return 10
    expected = {'project': 'octopus', 'location': 'oyster', 'active_directory': 'nudibranch'}
    path = NetAppClient.active_directory_path(**expected)
    actual = NetAppClient.parse_active_directory_path(path)
    assert expected == actual

def test_kms_config_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    kms_config = 'winkle'
    expected = 'projects/{project}/locations/{location}/kmsConfigs/{kms_config}'.format(project=project, location=location, kms_config=kms_config)
    actual = NetAppClient.kms_config_path(project, location, kms_config)
    assert expected == actual

def test_parse_kms_config_path():
    if False:
        return 10
    expected = {'project': 'nautilus', 'location': 'scallop', 'kms_config': 'abalone'}
    path = NetAppClient.kms_config_path(**expected)
    actual = NetAppClient.parse_kms_config_path(path)
    assert expected == actual

def test_network_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    network = 'clam'
    expected = 'projects/{project}/global/networks/{network}'.format(project=project, network=network)
    actual = NetAppClient.network_path(project, network)
    assert expected == actual

def test_parse_network_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'whelk', 'network': 'octopus'}
    path = NetAppClient.network_path(**expected)
    actual = NetAppClient.parse_network_path(path)
    assert expected == actual

def test_replication_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    location = 'nudibranch'
    volume = 'cuttlefish'
    replication = 'mussel'
    expected = 'projects/{project}/locations/{location}/volumes/{volume}/replications/{replication}'.format(project=project, location=location, volume=volume, replication=replication)
    actual = NetAppClient.replication_path(project, location, volume, replication)
    assert expected == actual

def test_parse_replication_path():
    if False:
        return 10
    expected = {'project': 'winkle', 'location': 'nautilus', 'volume': 'scallop', 'replication': 'abalone'}
    path = NetAppClient.replication_path(**expected)
    actual = NetAppClient.parse_replication_path(path)
    assert expected == actual

def test_snapshot_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    volume = 'whelk'
    snapshot = 'octopus'
    expected = 'projects/{project}/locations/{location}/volumes/{volume}/snapshots/{snapshot}'.format(project=project, location=location, volume=volume, snapshot=snapshot)
    actual = NetAppClient.snapshot_path(project, location, volume, snapshot)
    assert expected == actual

def test_parse_snapshot_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'oyster', 'location': 'nudibranch', 'volume': 'cuttlefish', 'snapshot': 'mussel'}
    path = NetAppClient.snapshot_path(**expected)
    actual = NetAppClient.parse_snapshot_path(path)
    assert expected == actual

def test_storage_pool_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    storage_pool = 'scallop'
    expected = 'projects/{project}/locations/{location}/storagePools/{storage_pool}'.format(project=project, location=location, storage_pool=storage_pool)
    actual = NetAppClient.storage_pool_path(project, location, storage_pool)
    assert expected == actual

def test_parse_storage_pool_path():
    if False:
        return 10
    expected = {'project': 'abalone', 'location': 'squid', 'storage_pool': 'clam'}
    path = NetAppClient.storage_pool_path(**expected)
    actual = NetAppClient.parse_storage_pool_path(path)
    assert expected == actual

def test_volume_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    volume = 'oyster'
    expected = 'projects/{project}/locations/{location}/volumes/{volume}'.format(project=project, location=location, volume=volume)
    actual = NetAppClient.volume_path(project, location, volume)
    assert expected == actual

def test_parse_volume_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'volume': 'mussel'}
    path = NetAppClient.volume_path(**expected)
    actual = NetAppClient.parse_volume_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'winkle'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = NetAppClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'nautilus'}
    path = NetAppClient.common_billing_account_path(**expected)
    actual = NetAppClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'scallop'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = NetAppClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'abalone'}
    path = NetAppClient.common_folder_path(**expected)
    actual = NetAppClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'squid'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = NetAppClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'clam'}
    path = NetAppClient.common_organization_path(**expected)
    actual = NetAppClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    expected = 'projects/{project}'.format(project=project)
    actual = NetAppClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus'}
    path = NetAppClient.common_project_path(**expected)
    actual = NetAppClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = NetAppClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = NetAppClient.common_location_path(**expected)
    actual = NetAppClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.NetAppTransport, '_prep_wrapped_messages') as prep:
        client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.NetAppTransport, '_prep_wrapped_messages') as prep:
        transport_class = NetAppClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetAppClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = NetAppAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = NetAppClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(NetAppClient, transports.NetAppGrpcTransport), (NetAppAsyncClient, transports.NetAppGrpcAsyncIOTransport)])
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