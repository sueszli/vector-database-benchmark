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
from google.longrunning import operations_pb2
from google.oauth2 import service_account
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
from google.cloud.asset_v1.services.asset_service import AssetServiceAsyncClient, AssetServiceClient, pagers, transports
from google.cloud.asset_v1.types import asset_service, assets

def client_cert_source_callback():
    if False:
        while True:
            i = 10
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
    assert AssetServiceClient._get_default_mtls_endpoint(None) is None
    assert AssetServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert AssetServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert AssetServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert AssetServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert AssetServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(AssetServiceClient, 'grpc'), (AssetServiceAsyncClient, 'grpc_asyncio'), (AssetServiceClient, 'rest')])
def test_asset_service_client_from_service_account_info(client_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('cloudasset.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudasset.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.AssetServiceGrpcTransport, 'grpc'), (transports.AssetServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.AssetServiceRestTransport, 'rest')])
def test_asset_service_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(AssetServiceClient, 'grpc'), (AssetServiceAsyncClient, 'grpc_asyncio'), (AssetServiceClient, 'rest')])
def test_asset_service_client_from_service_account_file(client_class, transport_name):
    if False:
        i = 10
        return i + 15
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_file') as factory:
        factory.return_value = creds
        client = client_class.from_service_account_file('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        client = client_class.from_service_account_json('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('cloudasset.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudasset.googleapis.com')

def test_asset_service_client_get_transport_class():
    if False:
        return 10
    transport = AssetServiceClient.get_transport_class()
    available_transports = [transports.AssetServiceGrpcTransport, transports.AssetServiceRestTransport]
    assert transport in available_transports
    transport = AssetServiceClient.get_transport_class('grpc')
    assert transport == transports.AssetServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AssetServiceClient, transports.AssetServiceGrpcTransport, 'grpc'), (AssetServiceAsyncClient, transports.AssetServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (AssetServiceClient, transports.AssetServiceRestTransport, 'rest')])
@mock.patch.object(AssetServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AssetServiceClient))
@mock.patch.object(AssetServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AssetServiceAsyncClient))
def test_asset_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(AssetServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(AssetServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(AssetServiceClient, transports.AssetServiceGrpcTransport, 'grpc', 'true'), (AssetServiceAsyncClient, transports.AssetServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (AssetServiceClient, transports.AssetServiceGrpcTransport, 'grpc', 'false'), (AssetServiceAsyncClient, transports.AssetServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (AssetServiceClient, transports.AssetServiceRestTransport, 'rest', 'true'), (AssetServiceClient, transports.AssetServiceRestTransport, 'rest', 'false')])
@mock.patch.object(AssetServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AssetServiceClient))
@mock.patch.object(AssetServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AssetServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_asset_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
    if False:
        for i in range(10):
            print('nop')
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

@pytest.mark.parametrize('client_class', [AssetServiceClient, AssetServiceAsyncClient])
@mock.patch.object(AssetServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AssetServiceClient))
@mock.patch.object(AssetServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AssetServiceAsyncClient))
def test_asset_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AssetServiceClient, transports.AssetServiceGrpcTransport, 'grpc'), (AssetServiceAsyncClient, transports.AssetServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (AssetServiceClient, transports.AssetServiceRestTransport, 'rest')])
def test_asset_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AssetServiceClient, transports.AssetServiceGrpcTransport, 'grpc', grpc_helpers), (AssetServiceAsyncClient, transports.AssetServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (AssetServiceClient, transports.AssetServiceRestTransport, 'rest', None)])
def test_asset_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_asset_service_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.asset_v1.services.asset_service.transports.AssetServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = AssetServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AssetServiceClient, transports.AssetServiceGrpcTransport, 'grpc', grpc_helpers), (AssetServiceAsyncClient, transports.AssetServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_asset_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('cloudasset.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='cloudasset.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [asset_service.ExportAssetsRequest, dict])
def test_export_assets(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.export_assets), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.export_assets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.ExportAssetsRequest()
    assert isinstance(response, future.Future)

def test_export_assets_empty_call():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.export_assets), '__call__') as call:
        client.export_assets()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.ExportAssetsRequest()

@pytest.mark.asyncio
async def test_export_assets_async(transport: str='grpc_asyncio', request_type=asset_service.ExportAssetsRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.export_assets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.export_assets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.ExportAssetsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_export_assets_async_from_dict():
    await test_export_assets_async(request_type=dict)

def test_export_assets_field_headers():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.ExportAssetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.export_assets), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.export_assets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_export_assets_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.ExportAssetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.export_assets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.export_assets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [asset_service.ListAssetsRequest, dict])
def test_list_assets(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.return_value = asset_service.ListAssetsResponse(next_page_token='next_page_token_value')
        response = client.list_assets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.ListAssetsRequest()
    assert isinstance(response, pagers.ListAssetsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_assets_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        client.list_assets()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.ListAssetsRequest()

@pytest.mark.asyncio
async def test_list_assets_async(transport: str='grpc_asyncio', request_type=asset_service.ListAssetsRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.ListAssetsResponse(next_page_token='next_page_token_value'))
        response = await client.list_assets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.ListAssetsRequest()
    assert isinstance(response, pagers.ListAssetsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_assets_async_from_dict():
    await test_list_assets_async(request_type=dict)

def test_list_assets_field_headers():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.ListAssetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.return_value = asset_service.ListAssetsResponse()
        client.list_assets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_assets_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.ListAssetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.ListAssetsResponse())
        await client.list_assets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_assets_flattened():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.return_value = asset_service.ListAssetsResponse()
        client.list_assets(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_assets_flattened_error():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_assets(asset_service.ListAssetsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_assets_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.return_value = asset_service.ListAssetsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.ListAssetsResponse())
        response = await client.list_assets(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_assets_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_assets(asset_service.ListAssetsRequest(), parent='parent_value')

def test_list_assets_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.side_effect = (asset_service.ListAssetsResponse(assets=[assets.Asset(), assets.Asset(), assets.Asset()], next_page_token='abc'), asset_service.ListAssetsResponse(assets=[], next_page_token='def'), asset_service.ListAssetsResponse(assets=[assets.Asset()], next_page_token='ghi'), asset_service.ListAssetsResponse(assets=[assets.Asset(), assets.Asset()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_assets(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, assets.Asset) for i in results))

def test_list_assets_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.side_effect = (asset_service.ListAssetsResponse(assets=[assets.Asset(), assets.Asset(), assets.Asset()], next_page_token='abc'), asset_service.ListAssetsResponse(assets=[], next_page_token='def'), asset_service.ListAssetsResponse(assets=[assets.Asset()], next_page_token='ghi'), asset_service.ListAssetsResponse(assets=[assets.Asset(), assets.Asset()]), RuntimeError)
        pages = list(client.list_assets(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_assets_async_pager():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_assets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (asset_service.ListAssetsResponse(assets=[assets.Asset(), assets.Asset(), assets.Asset()], next_page_token='abc'), asset_service.ListAssetsResponse(assets=[], next_page_token='def'), asset_service.ListAssetsResponse(assets=[assets.Asset()], next_page_token='ghi'), asset_service.ListAssetsResponse(assets=[assets.Asset(), assets.Asset()]), RuntimeError)
        async_pager = await client.list_assets(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, assets.Asset) for i in responses))

@pytest.mark.asyncio
async def test_list_assets_async_pages():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_assets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (asset_service.ListAssetsResponse(assets=[assets.Asset(), assets.Asset(), assets.Asset()], next_page_token='abc'), asset_service.ListAssetsResponse(assets=[], next_page_token='def'), asset_service.ListAssetsResponse(assets=[assets.Asset()], next_page_token='ghi'), asset_service.ListAssetsResponse(assets=[assets.Asset(), assets.Asset()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_assets(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [asset_service.BatchGetAssetsHistoryRequest, dict])
def test_batch_get_assets_history(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_get_assets_history), '__call__') as call:
        call.return_value = asset_service.BatchGetAssetsHistoryResponse()
        response = client.batch_get_assets_history(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.BatchGetAssetsHistoryRequest()
    assert isinstance(response, asset_service.BatchGetAssetsHistoryResponse)

def test_batch_get_assets_history_empty_call():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_get_assets_history), '__call__') as call:
        client.batch_get_assets_history()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.BatchGetAssetsHistoryRequest()

@pytest.mark.asyncio
async def test_batch_get_assets_history_async(transport: str='grpc_asyncio', request_type=asset_service.BatchGetAssetsHistoryRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_get_assets_history), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.BatchGetAssetsHistoryResponse())
        response = await client.batch_get_assets_history(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.BatchGetAssetsHistoryRequest()
    assert isinstance(response, asset_service.BatchGetAssetsHistoryResponse)

@pytest.mark.asyncio
async def test_batch_get_assets_history_async_from_dict():
    await test_batch_get_assets_history_async(request_type=dict)

def test_batch_get_assets_history_field_headers():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.BatchGetAssetsHistoryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_get_assets_history), '__call__') as call:
        call.return_value = asset_service.BatchGetAssetsHistoryResponse()
        client.batch_get_assets_history(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_get_assets_history_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.BatchGetAssetsHistoryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_get_assets_history), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.BatchGetAssetsHistoryResponse())
        await client.batch_get_assets_history(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [asset_service.CreateFeedRequest, dict])
def test_create_feed(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_feed), '__call__') as call:
        call.return_value = asset_service.Feed(name='name_value', asset_names=['asset_names_value'], asset_types=['asset_types_value'], content_type=asset_service.ContentType.RESOURCE, relationship_types=['relationship_types_value'])
        response = client.create_feed(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.CreateFeedRequest()
    assert isinstance(response, asset_service.Feed)
    assert response.name == 'name_value'
    assert response.asset_names == ['asset_names_value']
    assert response.asset_types == ['asset_types_value']
    assert response.content_type == asset_service.ContentType.RESOURCE
    assert response.relationship_types == ['relationship_types_value']

def test_create_feed_empty_call():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_feed), '__call__') as call:
        client.create_feed()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.CreateFeedRequest()

@pytest.mark.asyncio
async def test_create_feed_async(transport: str='grpc_asyncio', request_type=asset_service.CreateFeedRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_feed), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.Feed(name='name_value', asset_names=['asset_names_value'], asset_types=['asset_types_value'], content_type=asset_service.ContentType.RESOURCE, relationship_types=['relationship_types_value']))
        response = await client.create_feed(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.CreateFeedRequest()
    assert isinstance(response, asset_service.Feed)
    assert response.name == 'name_value'
    assert response.asset_names == ['asset_names_value']
    assert response.asset_types == ['asset_types_value']
    assert response.content_type == asset_service.ContentType.RESOURCE
    assert response.relationship_types == ['relationship_types_value']

@pytest.mark.asyncio
async def test_create_feed_async_from_dict():
    await test_create_feed_async(request_type=dict)

def test_create_feed_field_headers():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.CreateFeedRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_feed), '__call__') as call:
        call.return_value = asset_service.Feed()
        client.create_feed(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_feed_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.CreateFeedRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_feed), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.Feed())
        await client.create_feed(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_feed_flattened():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_feed), '__call__') as call:
        call.return_value = asset_service.Feed()
        client.create_feed(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_create_feed_flattened_error():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_feed(asset_service.CreateFeedRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_create_feed_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_feed), '__call__') as call:
        call.return_value = asset_service.Feed()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.Feed())
        response = await client.create_feed(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_feed_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_feed(asset_service.CreateFeedRequest(), parent='parent_value')

@pytest.mark.parametrize('request_type', [asset_service.GetFeedRequest, dict])
def test_get_feed(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_feed), '__call__') as call:
        call.return_value = asset_service.Feed(name='name_value', asset_names=['asset_names_value'], asset_types=['asset_types_value'], content_type=asset_service.ContentType.RESOURCE, relationship_types=['relationship_types_value'])
        response = client.get_feed(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.GetFeedRequest()
    assert isinstance(response, asset_service.Feed)
    assert response.name == 'name_value'
    assert response.asset_names == ['asset_names_value']
    assert response.asset_types == ['asset_types_value']
    assert response.content_type == asset_service.ContentType.RESOURCE
    assert response.relationship_types == ['relationship_types_value']

def test_get_feed_empty_call():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_feed), '__call__') as call:
        client.get_feed()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.GetFeedRequest()

@pytest.mark.asyncio
async def test_get_feed_async(transport: str='grpc_asyncio', request_type=asset_service.GetFeedRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_feed), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.Feed(name='name_value', asset_names=['asset_names_value'], asset_types=['asset_types_value'], content_type=asset_service.ContentType.RESOURCE, relationship_types=['relationship_types_value']))
        response = await client.get_feed(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.GetFeedRequest()
    assert isinstance(response, asset_service.Feed)
    assert response.name == 'name_value'
    assert response.asset_names == ['asset_names_value']
    assert response.asset_types == ['asset_types_value']
    assert response.content_type == asset_service.ContentType.RESOURCE
    assert response.relationship_types == ['relationship_types_value']

@pytest.mark.asyncio
async def test_get_feed_async_from_dict():
    await test_get_feed_async(request_type=dict)

def test_get_feed_field_headers():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.GetFeedRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_feed), '__call__') as call:
        call.return_value = asset_service.Feed()
        client.get_feed(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_feed_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.GetFeedRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_feed), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.Feed())
        await client.get_feed(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_feed_flattened():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_feed), '__call__') as call:
        call.return_value = asset_service.Feed()
        client.get_feed(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_feed_flattened_error():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_feed(asset_service.GetFeedRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_feed_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_feed), '__call__') as call:
        call.return_value = asset_service.Feed()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.Feed())
        response = await client.get_feed(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_feed_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_feed(asset_service.GetFeedRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [asset_service.ListFeedsRequest, dict])
def test_list_feeds(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_feeds), '__call__') as call:
        call.return_value = asset_service.ListFeedsResponse()
        response = client.list_feeds(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.ListFeedsRequest()
    assert isinstance(response, asset_service.ListFeedsResponse)

def test_list_feeds_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_feeds), '__call__') as call:
        client.list_feeds()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.ListFeedsRequest()

@pytest.mark.asyncio
async def test_list_feeds_async(transport: str='grpc_asyncio', request_type=asset_service.ListFeedsRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_feeds), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.ListFeedsResponse())
        response = await client.list_feeds(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.ListFeedsRequest()
    assert isinstance(response, asset_service.ListFeedsResponse)

@pytest.mark.asyncio
async def test_list_feeds_async_from_dict():
    await test_list_feeds_async(request_type=dict)

def test_list_feeds_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.ListFeedsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_feeds), '__call__') as call:
        call.return_value = asset_service.ListFeedsResponse()
        client.list_feeds(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_feeds_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.ListFeedsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_feeds), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.ListFeedsResponse())
        await client.list_feeds(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_feeds_flattened():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_feeds), '__call__') as call:
        call.return_value = asset_service.ListFeedsResponse()
        client.list_feeds(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_feeds_flattened_error():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_feeds(asset_service.ListFeedsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_feeds_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_feeds), '__call__') as call:
        call.return_value = asset_service.ListFeedsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.ListFeedsResponse())
        response = await client.list_feeds(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_feeds_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_feeds(asset_service.ListFeedsRequest(), parent='parent_value')

@pytest.mark.parametrize('request_type', [asset_service.UpdateFeedRequest, dict])
def test_update_feed(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_feed), '__call__') as call:
        call.return_value = asset_service.Feed(name='name_value', asset_names=['asset_names_value'], asset_types=['asset_types_value'], content_type=asset_service.ContentType.RESOURCE, relationship_types=['relationship_types_value'])
        response = client.update_feed(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.UpdateFeedRequest()
    assert isinstance(response, asset_service.Feed)
    assert response.name == 'name_value'
    assert response.asset_names == ['asset_names_value']
    assert response.asset_types == ['asset_types_value']
    assert response.content_type == asset_service.ContentType.RESOURCE
    assert response.relationship_types == ['relationship_types_value']

def test_update_feed_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_feed), '__call__') as call:
        client.update_feed()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.UpdateFeedRequest()

@pytest.mark.asyncio
async def test_update_feed_async(transport: str='grpc_asyncio', request_type=asset_service.UpdateFeedRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_feed), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.Feed(name='name_value', asset_names=['asset_names_value'], asset_types=['asset_types_value'], content_type=asset_service.ContentType.RESOURCE, relationship_types=['relationship_types_value']))
        response = await client.update_feed(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.UpdateFeedRequest()
    assert isinstance(response, asset_service.Feed)
    assert response.name == 'name_value'
    assert response.asset_names == ['asset_names_value']
    assert response.asset_types == ['asset_types_value']
    assert response.content_type == asset_service.ContentType.RESOURCE
    assert response.relationship_types == ['relationship_types_value']

@pytest.mark.asyncio
async def test_update_feed_async_from_dict():
    await test_update_feed_async(request_type=dict)

def test_update_feed_field_headers():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.UpdateFeedRequest()
    request.feed.name = 'name_value'
    with mock.patch.object(type(client.transport.update_feed), '__call__') as call:
        call.return_value = asset_service.Feed()
        client.update_feed(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'feed.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_feed_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.UpdateFeedRequest()
    request.feed.name = 'name_value'
    with mock.patch.object(type(client.transport.update_feed), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.Feed())
        await client.update_feed(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'feed.name=name_value') in kw['metadata']

def test_update_feed_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_feed), '__call__') as call:
        call.return_value = asset_service.Feed()
        client.update_feed(feed=asset_service.Feed(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].feed
        mock_val = asset_service.Feed(name='name_value')
        assert arg == mock_val

def test_update_feed_flattened_error():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_feed(asset_service.UpdateFeedRequest(), feed=asset_service.Feed(name='name_value'))

@pytest.mark.asyncio
async def test_update_feed_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_feed), '__call__') as call:
        call.return_value = asset_service.Feed()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.Feed())
        response = await client.update_feed(feed=asset_service.Feed(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].feed
        mock_val = asset_service.Feed(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_feed_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_feed(asset_service.UpdateFeedRequest(), feed=asset_service.Feed(name='name_value'))

@pytest.mark.parametrize('request_type', [asset_service.DeleteFeedRequest, dict])
def test_delete_feed(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_feed), '__call__') as call:
        call.return_value = None
        response = client.delete_feed(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.DeleteFeedRequest()
    assert response is None

def test_delete_feed_empty_call():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_feed), '__call__') as call:
        client.delete_feed()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.DeleteFeedRequest()

@pytest.mark.asyncio
async def test_delete_feed_async(transport: str='grpc_asyncio', request_type=asset_service.DeleteFeedRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_feed), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_feed(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.DeleteFeedRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_feed_async_from_dict():
    await test_delete_feed_async(request_type=dict)

def test_delete_feed_field_headers():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.DeleteFeedRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_feed), '__call__') as call:
        call.return_value = None
        client.delete_feed(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_feed_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.DeleteFeedRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_feed), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_feed(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_feed_flattened():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_feed), '__call__') as call:
        call.return_value = None
        client.delete_feed(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_feed_flattened_error():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_feed(asset_service.DeleteFeedRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_feed_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_feed), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_feed(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_feed_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_feed(asset_service.DeleteFeedRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [asset_service.SearchAllResourcesRequest, dict])
def test_search_all_resources(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_all_resources), '__call__') as call:
        call.return_value = asset_service.SearchAllResourcesResponse(next_page_token='next_page_token_value')
        response = client.search_all_resources(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.SearchAllResourcesRequest()
    assert isinstance(response, pagers.SearchAllResourcesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_all_resources_empty_call():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_all_resources), '__call__') as call:
        client.search_all_resources()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.SearchAllResourcesRequest()

@pytest.mark.asyncio
async def test_search_all_resources_async(transport: str='grpc_asyncio', request_type=asset_service.SearchAllResourcesRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_all_resources), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SearchAllResourcesResponse(next_page_token='next_page_token_value'))
        response = await client.search_all_resources(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.SearchAllResourcesRequest()
    assert isinstance(response, pagers.SearchAllResourcesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_search_all_resources_async_from_dict():
    await test_search_all_resources_async(request_type=dict)

def test_search_all_resources_field_headers():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.SearchAllResourcesRequest()
    request.scope = 'scope_value'
    with mock.patch.object(type(client.transport.search_all_resources), '__call__') as call:
        call.return_value = asset_service.SearchAllResourcesResponse()
        client.search_all_resources(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'scope=scope_value') in kw['metadata']

@pytest.mark.asyncio
async def test_search_all_resources_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.SearchAllResourcesRequest()
    request.scope = 'scope_value'
    with mock.patch.object(type(client.transport.search_all_resources), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SearchAllResourcesResponse())
        await client.search_all_resources(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'scope=scope_value') in kw['metadata']

def test_search_all_resources_flattened():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_all_resources), '__call__') as call:
        call.return_value = asset_service.SearchAllResourcesResponse()
        client.search_all_resources(scope='scope_value', query='query_value', asset_types=['asset_types_value'])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].scope
        mock_val = 'scope_value'
        assert arg == mock_val
        arg = args[0].query
        mock_val = 'query_value'
        assert arg == mock_val
        arg = args[0].asset_types
        mock_val = ['asset_types_value']
        assert arg == mock_val

def test_search_all_resources_flattened_error():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.search_all_resources(asset_service.SearchAllResourcesRequest(), scope='scope_value', query='query_value', asset_types=['asset_types_value'])

@pytest.mark.asyncio
async def test_search_all_resources_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_all_resources), '__call__') as call:
        call.return_value = asset_service.SearchAllResourcesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SearchAllResourcesResponse())
        response = await client.search_all_resources(scope='scope_value', query='query_value', asset_types=['asset_types_value'])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].scope
        mock_val = 'scope_value'
        assert arg == mock_val
        arg = args[0].query
        mock_val = 'query_value'
        assert arg == mock_val
        arg = args[0].asset_types
        mock_val = ['asset_types_value']
        assert arg == mock_val

@pytest.mark.asyncio
async def test_search_all_resources_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.search_all_resources(asset_service.SearchAllResourcesRequest(), scope='scope_value', query='query_value', asset_types=['asset_types_value'])

def test_search_all_resources_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_all_resources), '__call__') as call:
        call.side_effect = (asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult(), assets.ResourceSearchResult(), assets.ResourceSearchResult()], next_page_token='abc'), asset_service.SearchAllResourcesResponse(results=[], next_page_token='def'), asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult()], next_page_token='ghi'), asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult(), assets.ResourceSearchResult()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('scope', ''),)),)
        pager = client.search_all_resources(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, assets.ResourceSearchResult) for i in results))

def test_search_all_resources_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_all_resources), '__call__') as call:
        call.side_effect = (asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult(), assets.ResourceSearchResult(), assets.ResourceSearchResult()], next_page_token='abc'), asset_service.SearchAllResourcesResponse(results=[], next_page_token='def'), asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult()], next_page_token='ghi'), asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult(), assets.ResourceSearchResult()]), RuntimeError)
        pages = list(client.search_all_resources(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_search_all_resources_async_pager():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_all_resources), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult(), assets.ResourceSearchResult(), assets.ResourceSearchResult()], next_page_token='abc'), asset_service.SearchAllResourcesResponse(results=[], next_page_token='def'), asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult()], next_page_token='ghi'), asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult(), assets.ResourceSearchResult()]), RuntimeError)
        async_pager = await client.search_all_resources(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, assets.ResourceSearchResult) for i in responses))

@pytest.mark.asyncio
async def test_search_all_resources_async_pages():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_all_resources), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult(), assets.ResourceSearchResult(), assets.ResourceSearchResult()], next_page_token='abc'), asset_service.SearchAllResourcesResponse(results=[], next_page_token='def'), asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult()], next_page_token='ghi'), asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult(), assets.ResourceSearchResult()]), RuntimeError)
        pages = []
        async for page_ in (await client.search_all_resources(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [asset_service.SearchAllIamPoliciesRequest, dict])
def test_search_all_iam_policies(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_all_iam_policies), '__call__') as call:
        call.return_value = asset_service.SearchAllIamPoliciesResponse(next_page_token='next_page_token_value')
        response = client.search_all_iam_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.SearchAllIamPoliciesRequest()
    assert isinstance(response, pagers.SearchAllIamPoliciesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_all_iam_policies_empty_call():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_all_iam_policies), '__call__') as call:
        client.search_all_iam_policies()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.SearchAllIamPoliciesRequest()

@pytest.mark.asyncio
async def test_search_all_iam_policies_async(transport: str='grpc_asyncio', request_type=asset_service.SearchAllIamPoliciesRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_all_iam_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SearchAllIamPoliciesResponse(next_page_token='next_page_token_value'))
        response = await client.search_all_iam_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.SearchAllIamPoliciesRequest()
    assert isinstance(response, pagers.SearchAllIamPoliciesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_search_all_iam_policies_async_from_dict():
    await test_search_all_iam_policies_async(request_type=dict)

def test_search_all_iam_policies_field_headers():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.SearchAllIamPoliciesRequest()
    request.scope = 'scope_value'
    with mock.patch.object(type(client.transport.search_all_iam_policies), '__call__') as call:
        call.return_value = asset_service.SearchAllIamPoliciesResponse()
        client.search_all_iam_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'scope=scope_value') in kw['metadata']

@pytest.mark.asyncio
async def test_search_all_iam_policies_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.SearchAllIamPoliciesRequest()
    request.scope = 'scope_value'
    with mock.patch.object(type(client.transport.search_all_iam_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SearchAllIamPoliciesResponse())
        await client.search_all_iam_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'scope=scope_value') in kw['metadata']

def test_search_all_iam_policies_flattened():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_all_iam_policies), '__call__') as call:
        call.return_value = asset_service.SearchAllIamPoliciesResponse()
        client.search_all_iam_policies(scope='scope_value', query='query_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].scope
        mock_val = 'scope_value'
        assert arg == mock_val
        arg = args[0].query
        mock_val = 'query_value'
        assert arg == mock_val

def test_search_all_iam_policies_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.search_all_iam_policies(asset_service.SearchAllIamPoliciesRequest(), scope='scope_value', query='query_value')

@pytest.mark.asyncio
async def test_search_all_iam_policies_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_all_iam_policies), '__call__') as call:
        call.return_value = asset_service.SearchAllIamPoliciesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SearchAllIamPoliciesResponse())
        response = await client.search_all_iam_policies(scope='scope_value', query='query_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].scope
        mock_val = 'scope_value'
        assert arg == mock_val
        arg = args[0].query
        mock_val = 'query_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_search_all_iam_policies_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.search_all_iam_policies(asset_service.SearchAllIamPoliciesRequest(), scope='scope_value', query='query_value')

def test_search_all_iam_policies_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_all_iam_policies), '__call__') as call:
        call.side_effect = (asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult(), assets.IamPolicySearchResult(), assets.IamPolicySearchResult()], next_page_token='abc'), asset_service.SearchAllIamPoliciesResponse(results=[], next_page_token='def'), asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult()], next_page_token='ghi'), asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult(), assets.IamPolicySearchResult()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('scope', ''),)),)
        pager = client.search_all_iam_policies(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, assets.IamPolicySearchResult) for i in results))

def test_search_all_iam_policies_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_all_iam_policies), '__call__') as call:
        call.side_effect = (asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult(), assets.IamPolicySearchResult(), assets.IamPolicySearchResult()], next_page_token='abc'), asset_service.SearchAllIamPoliciesResponse(results=[], next_page_token='def'), asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult()], next_page_token='ghi'), asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult(), assets.IamPolicySearchResult()]), RuntimeError)
        pages = list(client.search_all_iam_policies(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_search_all_iam_policies_async_pager():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_all_iam_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult(), assets.IamPolicySearchResult(), assets.IamPolicySearchResult()], next_page_token='abc'), asset_service.SearchAllIamPoliciesResponse(results=[], next_page_token='def'), asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult()], next_page_token='ghi'), asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult(), assets.IamPolicySearchResult()]), RuntimeError)
        async_pager = await client.search_all_iam_policies(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, assets.IamPolicySearchResult) for i in responses))

@pytest.mark.asyncio
async def test_search_all_iam_policies_async_pages():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_all_iam_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult(), assets.IamPolicySearchResult(), assets.IamPolicySearchResult()], next_page_token='abc'), asset_service.SearchAllIamPoliciesResponse(results=[], next_page_token='def'), asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult()], next_page_token='ghi'), asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult(), assets.IamPolicySearchResult()]), RuntimeError)
        pages = []
        async for page_ in (await client.search_all_iam_policies(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [asset_service.AnalyzeIamPolicyRequest, dict])
def test_analyze_iam_policy(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_iam_policy), '__call__') as call:
        call.return_value = asset_service.AnalyzeIamPolicyResponse(fully_explored=True)
        response = client.analyze_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeIamPolicyRequest()
    assert isinstance(response, asset_service.AnalyzeIamPolicyResponse)
    assert response.fully_explored is True

def test_analyze_iam_policy_empty_call():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.analyze_iam_policy), '__call__') as call:
        client.analyze_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeIamPolicyRequest()

@pytest.mark.asyncio
async def test_analyze_iam_policy_async(transport: str='grpc_asyncio', request_type=asset_service.AnalyzeIamPolicyRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.AnalyzeIamPolicyResponse(fully_explored=True))
        response = await client.analyze_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeIamPolicyRequest()
    assert isinstance(response, asset_service.AnalyzeIamPolicyResponse)
    assert response.fully_explored is True

@pytest.mark.asyncio
async def test_analyze_iam_policy_async_from_dict():
    await test_analyze_iam_policy_async(request_type=dict)

def test_analyze_iam_policy_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.AnalyzeIamPolicyRequest()
    request.analysis_query.scope = 'scope_value'
    with mock.patch.object(type(client.transport.analyze_iam_policy), '__call__') as call:
        call.return_value = asset_service.AnalyzeIamPolicyResponse()
        client.analyze_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'analysis_query.scope=scope_value') in kw['metadata']

@pytest.mark.asyncio
async def test_analyze_iam_policy_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.AnalyzeIamPolicyRequest()
    request.analysis_query.scope = 'scope_value'
    with mock.patch.object(type(client.transport.analyze_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.AnalyzeIamPolicyResponse())
        await client.analyze_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'analysis_query.scope=scope_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [asset_service.AnalyzeIamPolicyLongrunningRequest, dict])
def test_analyze_iam_policy_longrunning(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_iam_policy_longrunning), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.analyze_iam_policy_longrunning(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeIamPolicyLongrunningRequest()
    assert isinstance(response, future.Future)

def test_analyze_iam_policy_longrunning_empty_call():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.analyze_iam_policy_longrunning), '__call__') as call:
        client.analyze_iam_policy_longrunning()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeIamPolicyLongrunningRequest()

@pytest.mark.asyncio
async def test_analyze_iam_policy_longrunning_async(transport: str='grpc_asyncio', request_type=asset_service.AnalyzeIamPolicyLongrunningRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_iam_policy_longrunning), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.analyze_iam_policy_longrunning(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeIamPolicyLongrunningRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_analyze_iam_policy_longrunning_async_from_dict():
    await test_analyze_iam_policy_longrunning_async(request_type=dict)

def test_analyze_iam_policy_longrunning_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.AnalyzeIamPolicyLongrunningRequest()
    request.analysis_query.scope = 'scope_value'
    with mock.patch.object(type(client.transport.analyze_iam_policy_longrunning), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.analyze_iam_policy_longrunning(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'analysis_query.scope=scope_value') in kw['metadata']

@pytest.mark.asyncio
async def test_analyze_iam_policy_longrunning_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.AnalyzeIamPolicyLongrunningRequest()
    request.analysis_query.scope = 'scope_value'
    with mock.patch.object(type(client.transport.analyze_iam_policy_longrunning), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.analyze_iam_policy_longrunning(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'analysis_query.scope=scope_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [asset_service.AnalyzeMoveRequest, dict])
def test_analyze_move(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_move), '__call__') as call:
        call.return_value = asset_service.AnalyzeMoveResponse()
        response = client.analyze_move(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeMoveRequest()
    assert isinstance(response, asset_service.AnalyzeMoveResponse)

def test_analyze_move_empty_call():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.analyze_move), '__call__') as call:
        client.analyze_move()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeMoveRequest()

@pytest.mark.asyncio
async def test_analyze_move_async(transport: str='grpc_asyncio', request_type=asset_service.AnalyzeMoveRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_move), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.AnalyzeMoveResponse())
        response = await client.analyze_move(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeMoveRequest()
    assert isinstance(response, asset_service.AnalyzeMoveResponse)

@pytest.mark.asyncio
async def test_analyze_move_async_from_dict():
    await test_analyze_move_async(request_type=dict)

def test_analyze_move_field_headers():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.AnalyzeMoveRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.analyze_move), '__call__') as call:
        call.return_value = asset_service.AnalyzeMoveResponse()
        client.analyze_move(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_analyze_move_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.AnalyzeMoveRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.analyze_move), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.AnalyzeMoveResponse())
        await client.analyze_move(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [asset_service.QueryAssetsRequest, dict])
def test_query_assets(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.query_assets), '__call__') as call:
        call.return_value = asset_service.QueryAssetsResponse(job_reference='job_reference_value', done=True)
        response = client.query_assets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.QueryAssetsRequest()
    assert isinstance(response, asset_service.QueryAssetsResponse)
    assert response.job_reference == 'job_reference_value'
    assert response.done is True

def test_query_assets_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.query_assets), '__call__') as call:
        client.query_assets()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.QueryAssetsRequest()

@pytest.mark.asyncio
async def test_query_assets_async(transport: str='grpc_asyncio', request_type=asset_service.QueryAssetsRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.query_assets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.QueryAssetsResponse(job_reference='job_reference_value', done=True))
        response = await client.query_assets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.QueryAssetsRequest()
    assert isinstance(response, asset_service.QueryAssetsResponse)
    assert response.job_reference == 'job_reference_value'
    assert response.done is True

@pytest.mark.asyncio
async def test_query_assets_async_from_dict():
    await test_query_assets_async(request_type=dict)

def test_query_assets_field_headers():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.QueryAssetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.query_assets), '__call__') as call:
        call.return_value = asset_service.QueryAssetsResponse()
        client.query_assets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_query_assets_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.QueryAssetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.query_assets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.QueryAssetsResponse())
        await client.query_assets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [asset_service.CreateSavedQueryRequest, dict])
def test_create_saved_query(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_saved_query), '__call__') as call:
        call.return_value = asset_service.SavedQuery(name='name_value', description='description_value', creator='creator_value', last_updater='last_updater_value')
        response = client.create_saved_query(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.CreateSavedQueryRequest()
    assert isinstance(response, asset_service.SavedQuery)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.creator == 'creator_value'
    assert response.last_updater == 'last_updater_value'

def test_create_saved_query_empty_call():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_saved_query), '__call__') as call:
        client.create_saved_query()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.CreateSavedQueryRequest()

@pytest.mark.asyncio
async def test_create_saved_query_async(transport: str='grpc_asyncio', request_type=asset_service.CreateSavedQueryRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_saved_query), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SavedQuery(name='name_value', description='description_value', creator='creator_value', last_updater='last_updater_value'))
        response = await client.create_saved_query(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.CreateSavedQueryRequest()
    assert isinstance(response, asset_service.SavedQuery)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.creator == 'creator_value'
    assert response.last_updater == 'last_updater_value'

@pytest.mark.asyncio
async def test_create_saved_query_async_from_dict():
    await test_create_saved_query_async(request_type=dict)

def test_create_saved_query_field_headers():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.CreateSavedQueryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_saved_query), '__call__') as call:
        call.return_value = asset_service.SavedQuery()
        client.create_saved_query(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_saved_query_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.CreateSavedQueryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_saved_query), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SavedQuery())
        await client.create_saved_query(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_saved_query_flattened():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_saved_query), '__call__') as call:
        call.return_value = asset_service.SavedQuery()
        client.create_saved_query(parent='parent_value', saved_query=asset_service.SavedQuery(name='name_value'), saved_query_id='saved_query_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].saved_query
        mock_val = asset_service.SavedQuery(name='name_value')
        assert arg == mock_val
        arg = args[0].saved_query_id
        mock_val = 'saved_query_id_value'
        assert arg == mock_val

def test_create_saved_query_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_saved_query(asset_service.CreateSavedQueryRequest(), parent='parent_value', saved_query=asset_service.SavedQuery(name='name_value'), saved_query_id='saved_query_id_value')

@pytest.mark.asyncio
async def test_create_saved_query_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_saved_query), '__call__') as call:
        call.return_value = asset_service.SavedQuery()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SavedQuery())
        response = await client.create_saved_query(parent='parent_value', saved_query=asset_service.SavedQuery(name='name_value'), saved_query_id='saved_query_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].saved_query
        mock_val = asset_service.SavedQuery(name='name_value')
        assert arg == mock_val
        arg = args[0].saved_query_id
        mock_val = 'saved_query_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_saved_query_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_saved_query(asset_service.CreateSavedQueryRequest(), parent='parent_value', saved_query=asset_service.SavedQuery(name='name_value'), saved_query_id='saved_query_id_value')

@pytest.mark.parametrize('request_type', [asset_service.GetSavedQueryRequest, dict])
def test_get_saved_query(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_saved_query), '__call__') as call:
        call.return_value = asset_service.SavedQuery(name='name_value', description='description_value', creator='creator_value', last_updater='last_updater_value')
        response = client.get_saved_query(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.GetSavedQueryRequest()
    assert isinstance(response, asset_service.SavedQuery)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.creator == 'creator_value'
    assert response.last_updater == 'last_updater_value'

def test_get_saved_query_empty_call():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_saved_query), '__call__') as call:
        client.get_saved_query()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.GetSavedQueryRequest()

@pytest.mark.asyncio
async def test_get_saved_query_async(transport: str='grpc_asyncio', request_type=asset_service.GetSavedQueryRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_saved_query), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SavedQuery(name='name_value', description='description_value', creator='creator_value', last_updater='last_updater_value'))
        response = await client.get_saved_query(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.GetSavedQueryRequest()
    assert isinstance(response, asset_service.SavedQuery)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.creator == 'creator_value'
    assert response.last_updater == 'last_updater_value'

@pytest.mark.asyncio
async def test_get_saved_query_async_from_dict():
    await test_get_saved_query_async(request_type=dict)

def test_get_saved_query_field_headers():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.GetSavedQueryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_saved_query), '__call__') as call:
        call.return_value = asset_service.SavedQuery()
        client.get_saved_query(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_saved_query_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.GetSavedQueryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_saved_query), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SavedQuery())
        await client.get_saved_query(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_saved_query_flattened():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_saved_query), '__call__') as call:
        call.return_value = asset_service.SavedQuery()
        client.get_saved_query(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_saved_query_flattened_error():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_saved_query(asset_service.GetSavedQueryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_saved_query_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_saved_query), '__call__') as call:
        call.return_value = asset_service.SavedQuery()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SavedQuery())
        response = await client.get_saved_query(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_saved_query_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_saved_query(asset_service.GetSavedQueryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [asset_service.ListSavedQueriesRequest, dict])
def test_list_saved_queries(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_saved_queries), '__call__') as call:
        call.return_value = asset_service.ListSavedQueriesResponse(next_page_token='next_page_token_value')
        response = client.list_saved_queries(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.ListSavedQueriesRequest()
    assert isinstance(response, pagers.ListSavedQueriesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_saved_queries_empty_call():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_saved_queries), '__call__') as call:
        client.list_saved_queries()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.ListSavedQueriesRequest()

@pytest.mark.asyncio
async def test_list_saved_queries_async(transport: str='grpc_asyncio', request_type=asset_service.ListSavedQueriesRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_saved_queries), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.ListSavedQueriesResponse(next_page_token='next_page_token_value'))
        response = await client.list_saved_queries(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.ListSavedQueriesRequest()
    assert isinstance(response, pagers.ListSavedQueriesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_saved_queries_async_from_dict():
    await test_list_saved_queries_async(request_type=dict)

def test_list_saved_queries_field_headers():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.ListSavedQueriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_saved_queries), '__call__') as call:
        call.return_value = asset_service.ListSavedQueriesResponse()
        client.list_saved_queries(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_saved_queries_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.ListSavedQueriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_saved_queries), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.ListSavedQueriesResponse())
        await client.list_saved_queries(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_saved_queries_flattened():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_saved_queries), '__call__') as call:
        call.return_value = asset_service.ListSavedQueriesResponse()
        client.list_saved_queries(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_saved_queries_flattened_error():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_saved_queries(asset_service.ListSavedQueriesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_saved_queries_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_saved_queries), '__call__') as call:
        call.return_value = asset_service.ListSavedQueriesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.ListSavedQueriesResponse())
        response = await client.list_saved_queries(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_saved_queries_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_saved_queries(asset_service.ListSavedQueriesRequest(), parent='parent_value')

def test_list_saved_queries_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_saved_queries), '__call__') as call:
        call.side_effect = (asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery(), asset_service.SavedQuery(), asset_service.SavedQuery()], next_page_token='abc'), asset_service.ListSavedQueriesResponse(saved_queries=[], next_page_token='def'), asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery()], next_page_token='ghi'), asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery(), asset_service.SavedQuery()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_saved_queries(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, asset_service.SavedQuery) for i in results))

def test_list_saved_queries_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_saved_queries), '__call__') as call:
        call.side_effect = (asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery(), asset_service.SavedQuery(), asset_service.SavedQuery()], next_page_token='abc'), asset_service.ListSavedQueriesResponse(saved_queries=[], next_page_token='def'), asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery()], next_page_token='ghi'), asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery(), asset_service.SavedQuery()]), RuntimeError)
        pages = list(client.list_saved_queries(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_saved_queries_async_pager():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_saved_queries), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery(), asset_service.SavedQuery(), asset_service.SavedQuery()], next_page_token='abc'), asset_service.ListSavedQueriesResponse(saved_queries=[], next_page_token='def'), asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery()], next_page_token='ghi'), asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery(), asset_service.SavedQuery()]), RuntimeError)
        async_pager = await client.list_saved_queries(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, asset_service.SavedQuery) for i in responses))

@pytest.mark.asyncio
async def test_list_saved_queries_async_pages():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_saved_queries), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery(), asset_service.SavedQuery(), asset_service.SavedQuery()], next_page_token='abc'), asset_service.ListSavedQueriesResponse(saved_queries=[], next_page_token='def'), asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery()], next_page_token='ghi'), asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery(), asset_service.SavedQuery()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_saved_queries(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [asset_service.UpdateSavedQueryRequest, dict])
def test_update_saved_query(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_saved_query), '__call__') as call:
        call.return_value = asset_service.SavedQuery(name='name_value', description='description_value', creator='creator_value', last_updater='last_updater_value')
        response = client.update_saved_query(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.UpdateSavedQueryRequest()
    assert isinstance(response, asset_service.SavedQuery)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.creator == 'creator_value'
    assert response.last_updater == 'last_updater_value'

def test_update_saved_query_empty_call():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_saved_query), '__call__') as call:
        client.update_saved_query()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.UpdateSavedQueryRequest()

@pytest.mark.asyncio
async def test_update_saved_query_async(transport: str='grpc_asyncio', request_type=asset_service.UpdateSavedQueryRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_saved_query), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SavedQuery(name='name_value', description='description_value', creator='creator_value', last_updater='last_updater_value'))
        response = await client.update_saved_query(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.UpdateSavedQueryRequest()
    assert isinstance(response, asset_service.SavedQuery)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.creator == 'creator_value'
    assert response.last_updater == 'last_updater_value'

@pytest.mark.asyncio
async def test_update_saved_query_async_from_dict():
    await test_update_saved_query_async(request_type=dict)

def test_update_saved_query_field_headers():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.UpdateSavedQueryRequest()
    request.saved_query.name = 'name_value'
    with mock.patch.object(type(client.transport.update_saved_query), '__call__') as call:
        call.return_value = asset_service.SavedQuery()
        client.update_saved_query(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'saved_query.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_saved_query_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.UpdateSavedQueryRequest()
    request.saved_query.name = 'name_value'
    with mock.patch.object(type(client.transport.update_saved_query), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SavedQuery())
        await client.update_saved_query(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'saved_query.name=name_value') in kw['metadata']

def test_update_saved_query_flattened():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_saved_query), '__call__') as call:
        call.return_value = asset_service.SavedQuery()
        client.update_saved_query(saved_query=asset_service.SavedQuery(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].saved_query
        mock_val = asset_service.SavedQuery(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_saved_query_flattened_error():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_saved_query(asset_service.UpdateSavedQueryRequest(), saved_query=asset_service.SavedQuery(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_saved_query_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_saved_query), '__call__') as call:
        call.return_value = asset_service.SavedQuery()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.SavedQuery())
        response = await client.update_saved_query(saved_query=asset_service.SavedQuery(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].saved_query
        mock_val = asset_service.SavedQuery(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_saved_query_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_saved_query(asset_service.UpdateSavedQueryRequest(), saved_query=asset_service.SavedQuery(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [asset_service.DeleteSavedQueryRequest, dict])
def test_delete_saved_query(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_saved_query), '__call__') as call:
        call.return_value = None
        response = client.delete_saved_query(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.DeleteSavedQueryRequest()
    assert response is None

def test_delete_saved_query_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_saved_query), '__call__') as call:
        client.delete_saved_query()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.DeleteSavedQueryRequest()

@pytest.mark.asyncio
async def test_delete_saved_query_async(transport: str='grpc_asyncio', request_type=asset_service.DeleteSavedQueryRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_saved_query), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_saved_query(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.DeleteSavedQueryRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_saved_query_async_from_dict():
    await test_delete_saved_query_async(request_type=dict)

def test_delete_saved_query_field_headers():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.DeleteSavedQueryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_saved_query), '__call__') as call:
        call.return_value = None
        client.delete_saved_query(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_saved_query_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.DeleteSavedQueryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_saved_query), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_saved_query(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_saved_query_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_saved_query), '__call__') as call:
        call.return_value = None
        client.delete_saved_query(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_saved_query_flattened_error():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_saved_query(asset_service.DeleteSavedQueryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_saved_query_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_saved_query), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_saved_query(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_saved_query_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_saved_query(asset_service.DeleteSavedQueryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [asset_service.BatchGetEffectiveIamPoliciesRequest, dict])
def test_batch_get_effective_iam_policies(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_get_effective_iam_policies), '__call__') as call:
        call.return_value = asset_service.BatchGetEffectiveIamPoliciesResponse()
        response = client.batch_get_effective_iam_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.BatchGetEffectiveIamPoliciesRequest()
    assert isinstance(response, asset_service.BatchGetEffectiveIamPoliciesResponse)

def test_batch_get_effective_iam_policies_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_get_effective_iam_policies), '__call__') as call:
        client.batch_get_effective_iam_policies()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.BatchGetEffectiveIamPoliciesRequest()

@pytest.mark.asyncio
async def test_batch_get_effective_iam_policies_async(transport: str='grpc_asyncio', request_type=asset_service.BatchGetEffectiveIamPoliciesRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_get_effective_iam_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.BatchGetEffectiveIamPoliciesResponse())
        response = await client.batch_get_effective_iam_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.BatchGetEffectiveIamPoliciesRequest()
    assert isinstance(response, asset_service.BatchGetEffectiveIamPoliciesResponse)

@pytest.mark.asyncio
async def test_batch_get_effective_iam_policies_async_from_dict():
    await test_batch_get_effective_iam_policies_async(request_type=dict)

def test_batch_get_effective_iam_policies_field_headers():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.BatchGetEffectiveIamPoliciesRequest()
    request.scope = 'scope_value'
    with mock.patch.object(type(client.transport.batch_get_effective_iam_policies), '__call__') as call:
        call.return_value = asset_service.BatchGetEffectiveIamPoliciesResponse()
        client.batch_get_effective_iam_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'scope=scope_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_get_effective_iam_policies_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.BatchGetEffectiveIamPoliciesRequest()
    request.scope = 'scope_value'
    with mock.patch.object(type(client.transport.batch_get_effective_iam_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.BatchGetEffectiveIamPoliciesResponse())
        await client.batch_get_effective_iam_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'scope=scope_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [asset_service.AnalyzeOrgPoliciesRequest, dict])
def test_analyze_org_policies(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_org_policies), '__call__') as call:
        call.return_value = asset_service.AnalyzeOrgPoliciesResponse(next_page_token='next_page_token_value')
        response = client.analyze_org_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeOrgPoliciesRequest()
    assert isinstance(response, pagers.AnalyzeOrgPoliciesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_analyze_org_policies_empty_call():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.analyze_org_policies), '__call__') as call:
        client.analyze_org_policies()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeOrgPoliciesRequest()

@pytest.mark.asyncio
async def test_analyze_org_policies_async(transport: str='grpc_asyncio', request_type=asset_service.AnalyzeOrgPoliciesRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_org_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.AnalyzeOrgPoliciesResponse(next_page_token='next_page_token_value'))
        response = await client.analyze_org_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeOrgPoliciesRequest()
    assert isinstance(response, pagers.AnalyzeOrgPoliciesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_analyze_org_policies_async_from_dict():
    await test_analyze_org_policies_async(request_type=dict)

def test_analyze_org_policies_field_headers():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.AnalyzeOrgPoliciesRequest()
    request.scope = 'scope_value'
    with mock.patch.object(type(client.transport.analyze_org_policies), '__call__') as call:
        call.return_value = asset_service.AnalyzeOrgPoliciesResponse()
        client.analyze_org_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'scope=scope_value') in kw['metadata']

@pytest.mark.asyncio
async def test_analyze_org_policies_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.AnalyzeOrgPoliciesRequest()
    request.scope = 'scope_value'
    with mock.patch.object(type(client.transport.analyze_org_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.AnalyzeOrgPoliciesResponse())
        await client.analyze_org_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'scope=scope_value') in kw['metadata']

def test_analyze_org_policies_flattened():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.analyze_org_policies), '__call__') as call:
        call.return_value = asset_service.AnalyzeOrgPoliciesResponse()
        client.analyze_org_policies(scope='scope_value', constraint='constraint_value', filter='filter_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].scope
        mock_val = 'scope_value'
        assert arg == mock_val
        arg = args[0].constraint
        mock_val = 'constraint_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

def test_analyze_org_policies_flattened_error():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.analyze_org_policies(asset_service.AnalyzeOrgPoliciesRequest(), scope='scope_value', constraint='constraint_value', filter='filter_value')

@pytest.mark.asyncio
async def test_analyze_org_policies_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.analyze_org_policies), '__call__') as call:
        call.return_value = asset_service.AnalyzeOrgPoliciesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.AnalyzeOrgPoliciesResponse())
        response = await client.analyze_org_policies(scope='scope_value', constraint='constraint_value', filter='filter_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].scope
        mock_val = 'scope_value'
        assert arg == mock_val
        arg = args[0].constraint
        mock_val = 'constraint_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_analyze_org_policies_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.analyze_org_policies(asset_service.AnalyzeOrgPoliciesRequest(), scope='scope_value', constraint='constraint_value', filter='filter_value')

def test_analyze_org_policies_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.analyze_org_policies), '__call__') as call:
        call.side_effect = (asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()], next_page_token='abc'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[], next_page_token='def'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()], next_page_token='ghi'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('scope', ''),)),)
        pager = client.analyze_org_policies(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult) for i in results))

def test_analyze_org_policies_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.analyze_org_policies), '__call__') as call:
        call.side_effect = (asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()], next_page_token='abc'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[], next_page_token='def'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()], next_page_token='ghi'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()]), RuntimeError)
        pages = list(client.analyze_org_policies(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_analyze_org_policies_async_pager():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.analyze_org_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()], next_page_token='abc'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[], next_page_token='def'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()], next_page_token='ghi'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()]), RuntimeError)
        async_pager = await client.analyze_org_policies(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult) for i in responses))

@pytest.mark.asyncio
async def test_analyze_org_policies_async_pages():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.analyze_org_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()], next_page_token='abc'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[], next_page_token='def'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()], next_page_token='ghi'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()]), RuntimeError)
        pages = []
        async for page_ in (await client.analyze_org_policies(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [asset_service.AnalyzeOrgPolicyGovernedContainersRequest, dict])
def test_analyze_org_policy_governed_containers(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_containers), '__call__') as call:
        call.return_value = asset_service.AnalyzeOrgPolicyGovernedContainersResponse(next_page_token='next_page_token_value')
        response = client.analyze_org_policy_governed_containers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeOrgPolicyGovernedContainersRequest()
    assert isinstance(response, pagers.AnalyzeOrgPolicyGovernedContainersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_analyze_org_policy_governed_containers_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_containers), '__call__') as call:
        client.analyze_org_policy_governed_containers()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeOrgPolicyGovernedContainersRequest()

@pytest.mark.asyncio
async def test_analyze_org_policy_governed_containers_async(transport: str='grpc_asyncio', request_type=asset_service.AnalyzeOrgPolicyGovernedContainersRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_containers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.AnalyzeOrgPolicyGovernedContainersResponse(next_page_token='next_page_token_value'))
        response = await client.analyze_org_policy_governed_containers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeOrgPolicyGovernedContainersRequest()
    assert isinstance(response, pagers.AnalyzeOrgPolicyGovernedContainersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_analyze_org_policy_governed_containers_async_from_dict():
    await test_analyze_org_policy_governed_containers_async(request_type=dict)

def test_analyze_org_policy_governed_containers_field_headers():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.AnalyzeOrgPolicyGovernedContainersRequest()
    request.scope = 'scope_value'
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_containers), '__call__') as call:
        call.return_value = asset_service.AnalyzeOrgPolicyGovernedContainersResponse()
        client.analyze_org_policy_governed_containers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'scope=scope_value') in kw['metadata']

@pytest.mark.asyncio
async def test_analyze_org_policy_governed_containers_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.AnalyzeOrgPolicyGovernedContainersRequest()
    request.scope = 'scope_value'
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_containers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.AnalyzeOrgPolicyGovernedContainersResponse())
        await client.analyze_org_policy_governed_containers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'scope=scope_value') in kw['metadata']

def test_analyze_org_policy_governed_containers_flattened():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_containers), '__call__') as call:
        call.return_value = asset_service.AnalyzeOrgPolicyGovernedContainersResponse()
        client.analyze_org_policy_governed_containers(scope='scope_value', constraint='constraint_value', filter='filter_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].scope
        mock_val = 'scope_value'
        assert arg == mock_val
        arg = args[0].constraint
        mock_val = 'constraint_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

def test_analyze_org_policy_governed_containers_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.analyze_org_policy_governed_containers(asset_service.AnalyzeOrgPolicyGovernedContainersRequest(), scope='scope_value', constraint='constraint_value', filter='filter_value')

@pytest.mark.asyncio
async def test_analyze_org_policy_governed_containers_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_containers), '__call__') as call:
        call.return_value = asset_service.AnalyzeOrgPolicyGovernedContainersResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.AnalyzeOrgPolicyGovernedContainersResponse())
        response = await client.analyze_org_policy_governed_containers(scope='scope_value', constraint='constraint_value', filter='filter_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].scope
        mock_val = 'scope_value'
        assert arg == mock_val
        arg = args[0].constraint
        mock_val = 'constraint_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_analyze_org_policy_governed_containers_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.analyze_org_policy_governed_containers(asset_service.AnalyzeOrgPolicyGovernedContainersRequest(), scope='scope_value', constraint='constraint_value', filter='filter_value')

def test_analyze_org_policy_governed_containers_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_containers), '__call__') as call:
        call.side_effect = (asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()], next_page_token='abc'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[], next_page_token='def'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()], next_page_token='ghi'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('scope', ''),)),)
        pager = client.analyze_org_policy_governed_containers(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer) for i in results))

def test_analyze_org_policy_governed_containers_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_containers), '__call__') as call:
        call.side_effect = (asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()], next_page_token='abc'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[], next_page_token='def'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()], next_page_token='ghi'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()]), RuntimeError)
        pages = list(client.analyze_org_policy_governed_containers(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_analyze_org_policy_governed_containers_async_pager():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_containers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()], next_page_token='abc'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[], next_page_token='def'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()], next_page_token='ghi'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()]), RuntimeError)
        async_pager = await client.analyze_org_policy_governed_containers(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer) for i in responses))

@pytest.mark.asyncio
async def test_analyze_org_policy_governed_containers_async_pages():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_containers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()], next_page_token='abc'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[], next_page_token='def'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()], next_page_token='ghi'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()]), RuntimeError)
        pages = []
        async for page_ in (await client.analyze_org_policy_governed_containers(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [asset_service.AnalyzeOrgPolicyGovernedAssetsRequest, dict])
def test_analyze_org_policy_governed_assets(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_assets), '__call__') as call:
        call.return_value = asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(next_page_token='next_page_token_value')
        response = client.analyze_org_policy_governed_assets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeOrgPolicyGovernedAssetsRequest()
    assert isinstance(response, pagers.AnalyzeOrgPolicyGovernedAssetsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_analyze_org_policy_governed_assets_empty_call():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_assets), '__call__') as call:
        client.analyze_org_policy_governed_assets()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeOrgPolicyGovernedAssetsRequest()

@pytest.mark.asyncio
async def test_analyze_org_policy_governed_assets_async(transport: str='grpc_asyncio', request_type=asset_service.AnalyzeOrgPolicyGovernedAssetsRequest):
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_assets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(next_page_token='next_page_token_value'))
        response = await client.analyze_org_policy_governed_assets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == asset_service.AnalyzeOrgPolicyGovernedAssetsRequest()
    assert isinstance(response, pagers.AnalyzeOrgPolicyGovernedAssetsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_analyze_org_policy_governed_assets_async_from_dict():
    await test_analyze_org_policy_governed_assets_async(request_type=dict)

def test_analyze_org_policy_governed_assets_field_headers():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.AnalyzeOrgPolicyGovernedAssetsRequest()
    request.scope = 'scope_value'
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_assets), '__call__') as call:
        call.return_value = asset_service.AnalyzeOrgPolicyGovernedAssetsResponse()
        client.analyze_org_policy_governed_assets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'scope=scope_value') in kw['metadata']

@pytest.mark.asyncio
async def test_analyze_org_policy_governed_assets_field_headers_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = asset_service.AnalyzeOrgPolicyGovernedAssetsRequest()
    request.scope = 'scope_value'
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_assets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.AnalyzeOrgPolicyGovernedAssetsResponse())
        await client.analyze_org_policy_governed_assets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'scope=scope_value') in kw['metadata']

def test_analyze_org_policy_governed_assets_flattened():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_assets), '__call__') as call:
        call.return_value = asset_service.AnalyzeOrgPolicyGovernedAssetsResponse()
        client.analyze_org_policy_governed_assets(scope='scope_value', constraint='constraint_value', filter='filter_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].scope
        mock_val = 'scope_value'
        assert arg == mock_val
        arg = args[0].constraint
        mock_val = 'constraint_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

def test_analyze_org_policy_governed_assets_flattened_error():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.analyze_org_policy_governed_assets(asset_service.AnalyzeOrgPolicyGovernedAssetsRequest(), scope='scope_value', constraint='constraint_value', filter='filter_value')

@pytest.mark.asyncio
async def test_analyze_org_policy_governed_assets_flattened_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_assets), '__call__') as call:
        call.return_value = asset_service.AnalyzeOrgPolicyGovernedAssetsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(asset_service.AnalyzeOrgPolicyGovernedAssetsResponse())
        response = await client.analyze_org_policy_governed_assets(scope='scope_value', constraint='constraint_value', filter='filter_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].scope
        mock_val = 'scope_value'
        assert arg == mock_val
        arg = args[0].constraint
        mock_val = 'constraint_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_analyze_org_policy_governed_assets_flattened_error_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.analyze_org_policy_governed_assets(asset_service.AnalyzeOrgPolicyGovernedAssetsRequest(), scope='scope_value', constraint='constraint_value', filter='filter_value')

def test_analyze_org_policy_governed_assets_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_assets), '__call__') as call:
        call.side_effect = (asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()], next_page_token='abc'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[], next_page_token='def'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()], next_page_token='ghi'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('scope', ''),)),)
        pager = client.analyze_org_policy_governed_assets(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset) for i in results))

def test_analyze_org_policy_governed_assets_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_assets), '__call__') as call:
        call.side_effect = (asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()], next_page_token='abc'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[], next_page_token='def'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()], next_page_token='ghi'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()]), RuntimeError)
        pages = list(client.analyze_org_policy_governed_assets(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_analyze_org_policy_governed_assets_async_pager():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_assets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()], next_page_token='abc'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[], next_page_token='def'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()], next_page_token='ghi'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()]), RuntimeError)
        async_pager = await client.analyze_org_policy_governed_assets(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset) for i in responses))

@pytest.mark.asyncio
async def test_analyze_org_policy_governed_assets_async_pages():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.analyze_org_policy_governed_assets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()], next_page_token='abc'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[], next_page_token='def'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()], next_page_token='ghi'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()]), RuntimeError)
        pages = []
        async for page_ in (await client.analyze_org_policy_governed_assets(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [asset_service.ExportAssetsRequest, dict])
def test_export_assets_rest(request_type):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.export_assets(request)
    assert response.operation.name == 'operations/spam'

def test_export_assets_rest_required_fields(request_type=asset_service.ExportAssetsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).export_assets._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).export_assets._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.export_assets(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_export_assets_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.export_assets._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'outputConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_export_assets_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AssetServiceRestInterceptor, 'post_export_assets') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_export_assets') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.ExportAssetsRequest.pb(asset_service.ExportAssetsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = asset_service.ExportAssetsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.export_assets(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_export_assets_rest_bad_request(transport: str='rest', request_type=asset_service.ExportAssetsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.export_assets(request)

def test_export_assets_rest_error():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.ListAssetsRequest, dict])
def test_list_assets_rest(request_type):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.ListAssetsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.ListAssetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_assets(request)
    assert isinstance(response, pagers.ListAssetsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_assets_rest_required_fields(request_type=asset_service.ListAssetsRequest):
    if False:
        return 10
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_assets._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_assets._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('asset_types', 'content_type', 'page_size', 'page_token', 'read_time', 'relationship_types'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.ListAssetsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.ListAssetsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_assets(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_assets_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_assets._get_unset_required_fields({})
    assert set(unset_fields) == set(('assetTypes', 'contentType', 'pageSize', 'pageToken', 'readTime', 'relationshipTypes')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_assets_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_list_assets') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_list_assets') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.ListAssetsRequest.pb(asset_service.ListAssetsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.ListAssetsResponse.to_json(asset_service.ListAssetsResponse())
        request = asset_service.ListAssetsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.ListAssetsResponse()
        client.list_assets(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_assets_rest_bad_request(transport: str='rest', request_type=asset_service.ListAssetsRequest):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_assets(request)

def test_list_assets_rest_flattened():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.ListAssetsResponse()
        sample_request = {'parent': 'sample1/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.ListAssetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_assets(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=*/*}/assets' % client.transport._host, args[1])

def test_list_assets_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_assets(asset_service.ListAssetsRequest(), parent='parent_value')

def test_list_assets_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (asset_service.ListAssetsResponse(assets=[assets.Asset(), assets.Asset(), assets.Asset()], next_page_token='abc'), asset_service.ListAssetsResponse(assets=[], next_page_token='def'), asset_service.ListAssetsResponse(assets=[assets.Asset()], next_page_token='ghi'), asset_service.ListAssetsResponse(assets=[assets.Asset(), assets.Asset()]))
        response = response + response
        response = tuple((asset_service.ListAssetsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'sample1/sample2'}
        pager = client.list_assets(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, assets.Asset) for i in results))
        pages = list(client.list_assets(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [asset_service.BatchGetAssetsHistoryRequest, dict])
def test_batch_get_assets_history_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.BatchGetAssetsHistoryResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.BatchGetAssetsHistoryResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_get_assets_history(request)
    assert isinstance(response, asset_service.BatchGetAssetsHistoryResponse)

def test_batch_get_assets_history_rest_required_fields(request_type=asset_service.BatchGetAssetsHistoryRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_get_assets_history._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_get_assets_history._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('asset_names', 'content_type', 'read_time_window', 'relationship_types'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.BatchGetAssetsHistoryResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.BatchGetAssetsHistoryResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.batch_get_assets_history(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_get_assets_history_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_get_assets_history._get_unset_required_fields({})
    assert set(unset_fields) == set(('assetNames', 'contentType', 'readTimeWindow', 'relationshipTypes')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_get_assets_history_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_batch_get_assets_history') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_batch_get_assets_history') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.BatchGetAssetsHistoryRequest.pb(asset_service.BatchGetAssetsHistoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.BatchGetAssetsHistoryResponse.to_json(asset_service.BatchGetAssetsHistoryResponse())
        request = asset_service.BatchGetAssetsHistoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.BatchGetAssetsHistoryResponse()
        client.batch_get_assets_history(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_get_assets_history_rest_bad_request(transport: str='rest', request_type=asset_service.BatchGetAssetsHistoryRequest):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_get_assets_history(request)

def test_batch_get_assets_history_rest_error():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.CreateFeedRequest, dict])
def test_create_feed_rest(request_type):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.Feed(name='name_value', asset_names=['asset_names_value'], asset_types=['asset_types_value'], content_type=asset_service.ContentType.RESOURCE, relationship_types=['relationship_types_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.Feed.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_feed(request)
    assert isinstance(response, asset_service.Feed)
    assert response.name == 'name_value'
    assert response.asset_names == ['asset_names_value']
    assert response.asset_types == ['asset_types_value']
    assert response.content_type == asset_service.ContentType.RESOURCE
    assert response.relationship_types == ['relationship_types_value']

def test_create_feed_rest_required_fields(request_type=asset_service.CreateFeedRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['feed_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_feed._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['feedId'] = 'feed_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_feed._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'feedId' in jsonified_request
    assert jsonified_request['feedId'] == 'feed_id_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.Feed()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.Feed.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_feed(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_feed_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_feed._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'feedId', 'feed'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_feed_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_create_feed') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_create_feed') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.CreateFeedRequest.pb(asset_service.CreateFeedRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.Feed.to_json(asset_service.Feed())
        request = asset_service.CreateFeedRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.Feed()
        client.create_feed(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_feed_rest_bad_request(transport: str='rest', request_type=asset_service.CreateFeedRequest):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_feed(request)

def test_create_feed_rest_flattened():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.Feed()
        sample_request = {'parent': 'sample1/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.Feed.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_feed(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=*/*}/feeds' % client.transport._host, args[1])

def test_create_feed_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_feed(asset_service.CreateFeedRequest(), parent='parent_value')

def test_create_feed_rest_error():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.GetFeedRequest, dict])
def test_get_feed_rest(request_type):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'sample1/sample2/feeds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.Feed(name='name_value', asset_names=['asset_names_value'], asset_types=['asset_types_value'], content_type=asset_service.ContentType.RESOURCE, relationship_types=['relationship_types_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.Feed.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_feed(request)
    assert isinstance(response, asset_service.Feed)
    assert response.name == 'name_value'
    assert response.asset_names == ['asset_names_value']
    assert response.asset_types == ['asset_types_value']
    assert response.content_type == asset_service.ContentType.RESOURCE
    assert response.relationship_types == ['relationship_types_value']

def test_get_feed_rest_required_fields(request_type=asset_service.GetFeedRequest):
    if False:
        return 10
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_feed._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_feed._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.Feed()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.Feed.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_feed(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_feed_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_feed._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_feed_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_get_feed') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_get_feed') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.GetFeedRequest.pb(asset_service.GetFeedRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.Feed.to_json(asset_service.Feed())
        request = asset_service.GetFeedRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.Feed()
        client.get_feed(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_feed_rest_bad_request(transport: str='rest', request_type=asset_service.GetFeedRequest):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'sample1/sample2/feeds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_feed(request)

def test_get_feed_rest_flattened():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.Feed()
        sample_request = {'name': 'sample1/sample2/feeds/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.Feed.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_feed(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=*/*/feeds/*}' % client.transport._host, args[1])

def test_get_feed_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_feed(asset_service.GetFeedRequest(), name='name_value')

def test_get_feed_rest_error():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.ListFeedsRequest, dict])
def test_list_feeds_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.ListFeedsResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.ListFeedsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_feeds(request)
    assert isinstance(response, asset_service.ListFeedsResponse)

def test_list_feeds_rest_required_fields(request_type=asset_service.ListFeedsRequest):
    if False:
        return 10
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_feeds._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_feeds._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.ListFeedsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.ListFeedsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_feeds(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_feeds_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_feeds._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_feeds_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_list_feeds') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_list_feeds') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.ListFeedsRequest.pb(asset_service.ListFeedsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.ListFeedsResponse.to_json(asset_service.ListFeedsResponse())
        request = asset_service.ListFeedsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.ListFeedsResponse()
        client.list_feeds(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_feeds_rest_bad_request(transport: str='rest', request_type=asset_service.ListFeedsRequest):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_feeds(request)

def test_list_feeds_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.ListFeedsResponse()
        sample_request = {'parent': 'sample1/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.ListFeedsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_feeds(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=*/*}/feeds' % client.transport._host, args[1])

def test_list_feeds_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_feeds(asset_service.ListFeedsRequest(), parent='parent_value')

def test_list_feeds_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.UpdateFeedRequest, dict])
def test_update_feed_rest(request_type):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'feed': {'name': 'sample1/sample2/feeds/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.Feed(name='name_value', asset_names=['asset_names_value'], asset_types=['asset_types_value'], content_type=asset_service.ContentType.RESOURCE, relationship_types=['relationship_types_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.Feed.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_feed(request)
    assert isinstance(response, asset_service.Feed)
    assert response.name == 'name_value'
    assert response.asset_names == ['asset_names_value']
    assert response.asset_types == ['asset_types_value']
    assert response.content_type == asset_service.ContentType.RESOURCE
    assert response.relationship_types == ['relationship_types_value']

def test_update_feed_rest_required_fields(request_type=asset_service.UpdateFeedRequest):
    if False:
        return 10
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_feed._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_feed._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.Feed()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.Feed.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_feed(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_feed_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_feed._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('feed', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_feed_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_update_feed') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_update_feed') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.UpdateFeedRequest.pb(asset_service.UpdateFeedRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.Feed.to_json(asset_service.Feed())
        request = asset_service.UpdateFeedRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.Feed()
        client.update_feed(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_feed_rest_bad_request(transport: str='rest', request_type=asset_service.UpdateFeedRequest):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'feed': {'name': 'sample1/sample2/feeds/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_feed(request)

def test_update_feed_rest_flattened():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.Feed()
        sample_request = {'feed': {'name': 'sample1/sample2/feeds/sample3'}}
        mock_args = dict(feed=asset_service.Feed(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.Feed.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_feed(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{feed.name=*/*/feeds/*}' % client.transport._host, args[1])

def test_update_feed_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_feed(asset_service.UpdateFeedRequest(), feed=asset_service.Feed(name='name_value'))

def test_update_feed_rest_error():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.DeleteFeedRequest, dict])
def test_delete_feed_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'sample1/sample2/feeds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_feed(request)
    assert response is None

def test_delete_feed_rest_required_fields(request_type=asset_service.DeleteFeedRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_feed._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_feed._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_feed(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_feed_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_feed._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_feed_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_delete_feed') as pre:
        pre.assert_not_called()
        pb_message = asset_service.DeleteFeedRequest.pb(asset_service.DeleteFeedRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = asset_service.DeleteFeedRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_feed(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_feed_rest_bad_request(transport: str='rest', request_type=asset_service.DeleteFeedRequest):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'sample1/sample2/feeds/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_feed(request)

def test_delete_feed_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'sample1/sample2/feeds/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_feed(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=*/*/feeds/*}' % client.transport._host, args[1])

def test_delete_feed_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_feed(asset_service.DeleteFeedRequest(), name='name_value')

def test_delete_feed_rest_error():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.SearchAllResourcesRequest, dict])
def test_search_all_resources_rest(request_type):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'scope': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.SearchAllResourcesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.SearchAllResourcesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search_all_resources(request)
    assert isinstance(response, pagers.SearchAllResourcesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_all_resources_rest_required_fields(request_type=asset_service.SearchAllResourcesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['scope'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_all_resources._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['scope'] = 'scope_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_all_resources._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('asset_types', 'order_by', 'page_size', 'page_token', 'query', 'read_mask'))
    jsonified_request.update(unset_fields)
    assert 'scope' in jsonified_request
    assert jsonified_request['scope'] == 'scope_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.SearchAllResourcesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.SearchAllResourcesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.search_all_resources(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_search_all_resources_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search_all_resources._get_unset_required_fields({})
    assert set(unset_fields) == set(('assetTypes', 'orderBy', 'pageSize', 'pageToken', 'query', 'readMask')) & set(('scope',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_all_resources_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_search_all_resources') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_search_all_resources') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.SearchAllResourcesRequest.pb(asset_service.SearchAllResourcesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.SearchAllResourcesResponse.to_json(asset_service.SearchAllResourcesResponse())
        request = asset_service.SearchAllResourcesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.SearchAllResourcesResponse()
        client.search_all_resources(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_all_resources_rest_bad_request(transport: str='rest', request_type=asset_service.SearchAllResourcesRequest):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'scope': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search_all_resources(request)

def test_search_all_resources_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.SearchAllResourcesResponse()
        sample_request = {'scope': 'sample1/sample2'}
        mock_args = dict(scope='scope_value', query='query_value', asset_types=['asset_types_value'])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.SearchAllResourcesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.search_all_resources(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{scope=*/*}:searchAllResources' % client.transport._host, args[1])

def test_search_all_resources_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.search_all_resources(asset_service.SearchAllResourcesRequest(), scope='scope_value', query='query_value', asset_types=['asset_types_value'])

def test_search_all_resources_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult(), assets.ResourceSearchResult(), assets.ResourceSearchResult()], next_page_token='abc'), asset_service.SearchAllResourcesResponse(results=[], next_page_token='def'), asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult()], next_page_token='ghi'), asset_service.SearchAllResourcesResponse(results=[assets.ResourceSearchResult(), assets.ResourceSearchResult()]))
        response = response + response
        response = tuple((asset_service.SearchAllResourcesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'scope': 'sample1/sample2'}
        pager = client.search_all_resources(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, assets.ResourceSearchResult) for i in results))
        pages = list(client.search_all_resources(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [asset_service.SearchAllIamPoliciesRequest, dict])
def test_search_all_iam_policies_rest(request_type):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'scope': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.SearchAllIamPoliciesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.SearchAllIamPoliciesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search_all_iam_policies(request)
    assert isinstance(response, pagers.SearchAllIamPoliciesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_all_iam_policies_rest_required_fields(request_type=asset_service.SearchAllIamPoliciesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['scope'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_all_iam_policies._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['scope'] = 'scope_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_all_iam_policies._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('asset_types', 'order_by', 'page_size', 'page_token', 'query'))
    jsonified_request.update(unset_fields)
    assert 'scope' in jsonified_request
    assert jsonified_request['scope'] == 'scope_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.SearchAllIamPoliciesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.SearchAllIamPoliciesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.search_all_iam_policies(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_search_all_iam_policies_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search_all_iam_policies._get_unset_required_fields({})
    assert set(unset_fields) == set(('assetTypes', 'orderBy', 'pageSize', 'pageToken', 'query')) & set(('scope',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_all_iam_policies_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_search_all_iam_policies') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_search_all_iam_policies') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.SearchAllIamPoliciesRequest.pb(asset_service.SearchAllIamPoliciesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.SearchAllIamPoliciesResponse.to_json(asset_service.SearchAllIamPoliciesResponse())
        request = asset_service.SearchAllIamPoliciesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.SearchAllIamPoliciesResponse()
        client.search_all_iam_policies(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_all_iam_policies_rest_bad_request(transport: str='rest', request_type=asset_service.SearchAllIamPoliciesRequest):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'scope': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search_all_iam_policies(request)

def test_search_all_iam_policies_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.SearchAllIamPoliciesResponse()
        sample_request = {'scope': 'sample1/sample2'}
        mock_args = dict(scope='scope_value', query='query_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.SearchAllIamPoliciesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.search_all_iam_policies(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{scope=*/*}:searchAllIamPolicies' % client.transport._host, args[1])

def test_search_all_iam_policies_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.search_all_iam_policies(asset_service.SearchAllIamPoliciesRequest(), scope='scope_value', query='query_value')

def test_search_all_iam_policies_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult(), assets.IamPolicySearchResult(), assets.IamPolicySearchResult()], next_page_token='abc'), asset_service.SearchAllIamPoliciesResponse(results=[], next_page_token='def'), asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult()], next_page_token='ghi'), asset_service.SearchAllIamPoliciesResponse(results=[assets.IamPolicySearchResult(), assets.IamPolicySearchResult()]))
        response = response + response
        response = tuple((asset_service.SearchAllIamPoliciesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'scope': 'sample1/sample2'}
        pager = client.search_all_iam_policies(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, assets.IamPolicySearchResult) for i in results))
        pages = list(client.search_all_iam_policies(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [asset_service.AnalyzeIamPolicyRequest, dict])
def test_analyze_iam_policy_rest(request_type):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'analysis_query': {'scope': 'sample1/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.AnalyzeIamPolicyResponse(fully_explored=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.AnalyzeIamPolicyResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.analyze_iam_policy(request)
    assert isinstance(response, asset_service.AnalyzeIamPolicyResponse)
    assert response.fully_explored is True

def test_analyze_iam_policy_rest_required_fields(request_type=asset_service.AnalyzeIamPolicyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_iam_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('analysis_query', 'execution_timeout', 'saved_analysis_query'))
    jsonified_request.update(unset_fields)
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.AnalyzeIamPolicyResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.AnalyzeIamPolicyResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.analyze_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_analyze_iam_policy_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.analyze_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('analysisQuery', 'executionTimeout', 'savedAnalysisQuery')) & set(('analysisQuery',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_analyze_iam_policy_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_analyze_iam_policy') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_analyze_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.AnalyzeIamPolicyRequest.pb(asset_service.AnalyzeIamPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.AnalyzeIamPolicyResponse.to_json(asset_service.AnalyzeIamPolicyResponse())
        request = asset_service.AnalyzeIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.AnalyzeIamPolicyResponse()
        client.analyze_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_analyze_iam_policy_rest_bad_request(transport: str='rest', request_type=asset_service.AnalyzeIamPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'analysis_query': {'scope': 'sample1/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.analyze_iam_policy(request)

def test_analyze_iam_policy_rest_error():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.AnalyzeIamPolicyLongrunningRequest, dict])
def test_analyze_iam_policy_longrunning_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'analysis_query': {'scope': 'sample1/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.analyze_iam_policy_longrunning(request)
    assert response.operation.name == 'operations/spam'

def test_analyze_iam_policy_longrunning_rest_required_fields(request_type=asset_service.AnalyzeIamPolicyLongrunningRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_iam_policy_longrunning._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_iam_policy_longrunning._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.analyze_iam_policy_longrunning(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_analyze_iam_policy_longrunning_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.analyze_iam_policy_longrunning._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('analysisQuery', 'outputConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_analyze_iam_policy_longrunning_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AssetServiceRestInterceptor, 'post_analyze_iam_policy_longrunning') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_analyze_iam_policy_longrunning') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.AnalyzeIamPolicyLongrunningRequest.pb(asset_service.AnalyzeIamPolicyLongrunningRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = asset_service.AnalyzeIamPolicyLongrunningRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.analyze_iam_policy_longrunning(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_analyze_iam_policy_longrunning_rest_bad_request(transport: str='rest', request_type=asset_service.AnalyzeIamPolicyLongrunningRequest):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'analysis_query': {'scope': 'sample1/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.analyze_iam_policy_longrunning(request)

def test_analyze_iam_policy_longrunning_rest_error():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.AnalyzeMoveRequest, dict])
def test_analyze_move_rest(request_type):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.AnalyzeMoveResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.AnalyzeMoveResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.analyze_move(request)
    assert isinstance(response, asset_service.AnalyzeMoveResponse)

def test_analyze_move_rest_required_fields(request_type=asset_service.AnalyzeMoveRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['resource'] = ''
    request_init['destination_parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'destinationParent' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_move._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'destinationParent' in jsonified_request
    assert jsonified_request['destinationParent'] == request_init['destination_parent']
    jsonified_request['resource'] = 'resource_value'
    jsonified_request['destinationParent'] = 'destination_parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_move._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('destination_parent', 'view'))
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    assert 'destinationParent' in jsonified_request
    assert jsonified_request['destinationParent'] == 'destination_parent_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.AnalyzeMoveResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.AnalyzeMoveResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.analyze_move(request)
            expected_params = [('destinationParent', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_analyze_move_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.analyze_move._get_unset_required_fields({})
    assert set(unset_fields) == set(('destinationParent', 'view')) & set(('resource', 'destinationParent'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_analyze_move_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_analyze_move') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_analyze_move') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.AnalyzeMoveRequest.pb(asset_service.AnalyzeMoveRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.AnalyzeMoveResponse.to_json(asset_service.AnalyzeMoveResponse())
        request = asset_service.AnalyzeMoveRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.AnalyzeMoveResponse()
        client.analyze_move(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_analyze_move_rest_bad_request(transport: str='rest', request_type=asset_service.AnalyzeMoveRequest):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.analyze_move(request)

def test_analyze_move_rest_error():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.QueryAssetsRequest, dict])
def test_query_assets_rest(request_type):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.QueryAssetsResponse(job_reference='job_reference_value', done=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.QueryAssetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.query_assets(request)
    assert isinstance(response, asset_service.QueryAssetsResponse)
    assert response.job_reference == 'job_reference_value'
    assert response.done is True

def test_query_assets_rest_required_fields(request_type=asset_service.QueryAssetsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).query_assets._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).query_assets._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.QueryAssetsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.QueryAssetsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.query_assets(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_query_assets_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.query_assets._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_query_assets_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_query_assets') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_query_assets') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.QueryAssetsRequest.pb(asset_service.QueryAssetsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.QueryAssetsResponse.to_json(asset_service.QueryAssetsResponse())
        request = asset_service.QueryAssetsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.QueryAssetsResponse()
        client.query_assets(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_query_assets_rest_bad_request(transport: str='rest', request_type=asset_service.QueryAssetsRequest):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.query_assets(request)

def test_query_assets_rest_error():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.CreateSavedQueryRequest, dict])
def test_create_saved_query_rest(request_type):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'sample1/sample2'}
    request_init['saved_query'] = {'name': 'name_value', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'creator': 'creator_value', 'last_update_time': {}, 'last_updater': 'last_updater_value', 'labels': {}, 'content': {'iam_policy_analysis_query': {'scope': 'scope_value', 'resource_selector': {'full_resource_name': 'full_resource_name_value'}, 'identity_selector': {'identity': 'identity_value'}, 'access_selector': {'roles': ['roles_value1', 'roles_value2'], 'permissions': ['permissions_value1', 'permissions_value2']}, 'options': {'expand_groups': True, 'expand_roles': True, 'expand_resources': True, 'output_resource_edges': True, 'output_group_edges': True, 'analyze_service_account_impersonation': True}, 'condition_context': {'access_time': {}}}}}
    test_field = asset_service.CreateSavedQueryRequest.meta.fields['saved_query']

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
    for (field, value) in request_init['saved_query'].items():
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
                for i in range(0, len(request_init['saved_query'][field])):
                    del request_init['saved_query'][field][i][subfield]
            else:
                del request_init['saved_query'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.SavedQuery(name='name_value', description='description_value', creator='creator_value', last_updater='last_updater_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.SavedQuery.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_saved_query(request)
    assert isinstance(response, asset_service.SavedQuery)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.creator == 'creator_value'
    assert response.last_updater == 'last_updater_value'

def test_create_saved_query_rest_required_fields(request_type=asset_service.CreateSavedQueryRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['saved_query_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'savedQueryId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_saved_query._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'savedQueryId' in jsonified_request
    assert jsonified_request['savedQueryId'] == request_init['saved_query_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['savedQueryId'] = 'saved_query_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_saved_query._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('saved_query_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'savedQueryId' in jsonified_request
    assert jsonified_request['savedQueryId'] == 'saved_query_id_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.SavedQuery()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.SavedQuery.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_saved_query(request)
            expected_params = [('savedQueryId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_saved_query_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_saved_query._get_unset_required_fields({})
    assert set(unset_fields) == set(('savedQueryId',)) & set(('parent', 'savedQuery', 'savedQueryId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_saved_query_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_create_saved_query') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_create_saved_query') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.CreateSavedQueryRequest.pb(asset_service.CreateSavedQueryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.SavedQuery.to_json(asset_service.SavedQuery())
        request = asset_service.CreateSavedQueryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.SavedQuery()
        client.create_saved_query(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_saved_query_rest_bad_request(transport: str='rest', request_type=asset_service.CreateSavedQueryRequest):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_saved_query(request)

def test_create_saved_query_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.SavedQuery()
        sample_request = {'parent': 'sample1/sample2'}
        mock_args = dict(parent='parent_value', saved_query=asset_service.SavedQuery(name='name_value'), saved_query_id='saved_query_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.SavedQuery.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_saved_query(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=*/*}/savedQueries' % client.transport._host, args[1])

def test_create_saved_query_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_saved_query(asset_service.CreateSavedQueryRequest(), parent='parent_value', saved_query=asset_service.SavedQuery(name='name_value'), saved_query_id='saved_query_id_value')

def test_create_saved_query_rest_error():
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.GetSavedQueryRequest, dict])
def test_get_saved_query_rest(request_type):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'sample1/sample2/savedQueries/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.SavedQuery(name='name_value', description='description_value', creator='creator_value', last_updater='last_updater_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.SavedQuery.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_saved_query(request)
    assert isinstance(response, asset_service.SavedQuery)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.creator == 'creator_value'
    assert response.last_updater == 'last_updater_value'

def test_get_saved_query_rest_required_fields(request_type=asset_service.GetSavedQueryRequest):
    if False:
        return 10
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_saved_query._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_saved_query._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.SavedQuery()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.SavedQuery.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_saved_query(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_saved_query_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_saved_query._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_saved_query_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_get_saved_query') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_get_saved_query') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.GetSavedQueryRequest.pb(asset_service.GetSavedQueryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.SavedQuery.to_json(asset_service.SavedQuery())
        request = asset_service.GetSavedQueryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.SavedQuery()
        client.get_saved_query(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_saved_query_rest_bad_request(transport: str='rest', request_type=asset_service.GetSavedQueryRequest):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'sample1/sample2/savedQueries/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_saved_query(request)

def test_get_saved_query_rest_flattened():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.SavedQuery()
        sample_request = {'name': 'sample1/sample2/savedQueries/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.SavedQuery.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_saved_query(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=*/*/savedQueries/*}' % client.transport._host, args[1])

def test_get_saved_query_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_saved_query(asset_service.GetSavedQueryRequest(), name='name_value')

def test_get_saved_query_rest_error():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.ListSavedQueriesRequest, dict])
def test_list_saved_queries_rest(request_type):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.ListSavedQueriesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.ListSavedQueriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_saved_queries(request)
    assert isinstance(response, pagers.ListSavedQueriesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_saved_queries_rest_required_fields(request_type=asset_service.ListSavedQueriesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_saved_queries._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_saved_queries._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.ListSavedQueriesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.ListSavedQueriesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_saved_queries(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_saved_queries_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_saved_queries._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_saved_queries_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_list_saved_queries') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_list_saved_queries') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.ListSavedQueriesRequest.pb(asset_service.ListSavedQueriesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.ListSavedQueriesResponse.to_json(asset_service.ListSavedQueriesResponse())
        request = asset_service.ListSavedQueriesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.ListSavedQueriesResponse()
        client.list_saved_queries(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_saved_queries_rest_bad_request(transport: str='rest', request_type=asset_service.ListSavedQueriesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_saved_queries(request)

def test_list_saved_queries_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.ListSavedQueriesResponse()
        sample_request = {'parent': 'sample1/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.ListSavedQueriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_saved_queries(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=*/*}/savedQueries' % client.transport._host, args[1])

def test_list_saved_queries_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_saved_queries(asset_service.ListSavedQueriesRequest(), parent='parent_value')

def test_list_saved_queries_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery(), asset_service.SavedQuery(), asset_service.SavedQuery()], next_page_token='abc'), asset_service.ListSavedQueriesResponse(saved_queries=[], next_page_token='def'), asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery()], next_page_token='ghi'), asset_service.ListSavedQueriesResponse(saved_queries=[asset_service.SavedQuery(), asset_service.SavedQuery()]))
        response = response + response
        response = tuple((asset_service.ListSavedQueriesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'sample1/sample2'}
        pager = client.list_saved_queries(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, asset_service.SavedQuery) for i in results))
        pages = list(client.list_saved_queries(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [asset_service.UpdateSavedQueryRequest, dict])
def test_update_saved_query_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'saved_query': {'name': 'sample1/sample2/savedQueries/sample3'}}
    request_init['saved_query'] = {'name': 'sample1/sample2/savedQueries/sample3', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'creator': 'creator_value', 'last_update_time': {}, 'last_updater': 'last_updater_value', 'labels': {}, 'content': {'iam_policy_analysis_query': {'scope': 'scope_value', 'resource_selector': {'full_resource_name': 'full_resource_name_value'}, 'identity_selector': {'identity': 'identity_value'}, 'access_selector': {'roles': ['roles_value1', 'roles_value2'], 'permissions': ['permissions_value1', 'permissions_value2']}, 'options': {'expand_groups': True, 'expand_roles': True, 'expand_resources': True, 'output_resource_edges': True, 'output_group_edges': True, 'analyze_service_account_impersonation': True}, 'condition_context': {'access_time': {}}}}}
    test_field = asset_service.UpdateSavedQueryRequest.meta.fields['saved_query']

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
    for (field, value) in request_init['saved_query'].items():
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
                for i in range(0, len(request_init['saved_query'][field])):
                    del request_init['saved_query'][field][i][subfield]
            else:
                del request_init['saved_query'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.SavedQuery(name='name_value', description='description_value', creator='creator_value', last_updater='last_updater_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.SavedQuery.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_saved_query(request)
    assert isinstance(response, asset_service.SavedQuery)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.creator == 'creator_value'
    assert response.last_updater == 'last_updater_value'

def test_update_saved_query_rest_required_fields(request_type=asset_service.UpdateSavedQueryRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_saved_query._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_saved_query._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.SavedQuery()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.SavedQuery.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_saved_query(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_saved_query_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_saved_query._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('savedQuery', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_saved_query_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_update_saved_query') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_update_saved_query') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.UpdateSavedQueryRequest.pb(asset_service.UpdateSavedQueryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.SavedQuery.to_json(asset_service.SavedQuery())
        request = asset_service.UpdateSavedQueryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.SavedQuery()
        client.update_saved_query(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_saved_query_rest_bad_request(transport: str='rest', request_type=asset_service.UpdateSavedQueryRequest):
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'saved_query': {'name': 'sample1/sample2/savedQueries/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_saved_query(request)

def test_update_saved_query_rest_flattened():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.SavedQuery()
        sample_request = {'saved_query': {'name': 'sample1/sample2/savedQueries/sample3'}}
        mock_args = dict(saved_query=asset_service.SavedQuery(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.SavedQuery.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_saved_query(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{saved_query.name=*/*/savedQueries/*}' % client.transport._host, args[1])

def test_update_saved_query_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_saved_query(asset_service.UpdateSavedQueryRequest(), saved_query=asset_service.SavedQuery(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_saved_query_rest_error():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.DeleteSavedQueryRequest, dict])
def test_delete_saved_query_rest(request_type):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'sample1/sample2/savedQueries/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_saved_query(request)
    assert response is None

def test_delete_saved_query_rest_required_fields(request_type=asset_service.DeleteSavedQueryRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_saved_query._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_saved_query._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_saved_query(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_saved_query_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_saved_query._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_saved_query_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_delete_saved_query') as pre:
        pre.assert_not_called()
        pb_message = asset_service.DeleteSavedQueryRequest.pb(asset_service.DeleteSavedQueryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = asset_service.DeleteSavedQueryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_saved_query(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_saved_query_rest_bad_request(transport: str='rest', request_type=asset_service.DeleteSavedQueryRequest):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'sample1/sample2/savedQueries/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_saved_query(request)

def test_delete_saved_query_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'sample1/sample2/savedQueries/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_saved_query(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=*/*/savedQueries/*}' % client.transport._host, args[1])

def test_delete_saved_query_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_saved_query(asset_service.DeleteSavedQueryRequest(), name='name_value')

def test_delete_saved_query_rest_error():
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.BatchGetEffectiveIamPoliciesRequest, dict])
def test_batch_get_effective_iam_policies_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'scope': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.BatchGetEffectiveIamPoliciesResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.BatchGetEffectiveIamPoliciesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_get_effective_iam_policies(request)
    assert isinstance(response, asset_service.BatchGetEffectiveIamPoliciesResponse)

def test_batch_get_effective_iam_policies_rest_required_fields(request_type=asset_service.BatchGetEffectiveIamPoliciesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['scope'] = ''
    request_init['names'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'names' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_get_effective_iam_policies._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'names' in jsonified_request
    assert jsonified_request['names'] == request_init['names']
    jsonified_request['scope'] = 'scope_value'
    jsonified_request['names'] = 'names_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_get_effective_iam_policies._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('names',))
    jsonified_request.update(unset_fields)
    assert 'scope' in jsonified_request
    assert jsonified_request['scope'] == 'scope_value'
    assert 'names' in jsonified_request
    assert jsonified_request['names'] == 'names_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.BatchGetEffectiveIamPoliciesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.BatchGetEffectiveIamPoliciesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.batch_get_effective_iam_policies(request)
            expected_params = [('names', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_get_effective_iam_policies_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_get_effective_iam_policies._get_unset_required_fields({})
    assert set(unset_fields) == set(('names',)) & set(('scope', 'names'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_get_effective_iam_policies_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_batch_get_effective_iam_policies') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_batch_get_effective_iam_policies') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.BatchGetEffectiveIamPoliciesRequest.pb(asset_service.BatchGetEffectiveIamPoliciesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.BatchGetEffectiveIamPoliciesResponse.to_json(asset_service.BatchGetEffectiveIamPoliciesResponse())
        request = asset_service.BatchGetEffectiveIamPoliciesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.BatchGetEffectiveIamPoliciesResponse()
        client.batch_get_effective_iam_policies(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_get_effective_iam_policies_rest_bad_request(transport: str='rest', request_type=asset_service.BatchGetEffectiveIamPoliciesRequest):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'scope': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_get_effective_iam_policies(request)

def test_batch_get_effective_iam_policies_rest_error():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [asset_service.AnalyzeOrgPoliciesRequest, dict])
def test_analyze_org_policies_rest(request_type):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'scope': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.AnalyzeOrgPoliciesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.AnalyzeOrgPoliciesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.analyze_org_policies(request)
    assert isinstance(response, pagers.AnalyzeOrgPoliciesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_analyze_org_policies_rest_required_fields(request_type=asset_service.AnalyzeOrgPoliciesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['scope'] = ''
    request_init['constraint'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'constraint' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_org_policies._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'constraint' in jsonified_request
    assert jsonified_request['constraint'] == request_init['constraint']
    jsonified_request['scope'] = 'scope_value'
    jsonified_request['constraint'] = 'constraint_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_org_policies._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('constraint', 'filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'scope' in jsonified_request
    assert jsonified_request['scope'] == 'scope_value'
    assert 'constraint' in jsonified_request
    assert jsonified_request['constraint'] == 'constraint_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.AnalyzeOrgPoliciesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.AnalyzeOrgPoliciesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.analyze_org_policies(request)
            expected_params = [('constraint', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_analyze_org_policies_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.analyze_org_policies._get_unset_required_fields({})
    assert set(unset_fields) == set(('constraint', 'filter', 'pageSize', 'pageToken')) & set(('scope', 'constraint'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_analyze_org_policies_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_analyze_org_policies') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_analyze_org_policies') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.AnalyzeOrgPoliciesRequest.pb(asset_service.AnalyzeOrgPoliciesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.AnalyzeOrgPoliciesResponse.to_json(asset_service.AnalyzeOrgPoliciesResponse())
        request = asset_service.AnalyzeOrgPoliciesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.AnalyzeOrgPoliciesResponse()
        client.analyze_org_policies(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_analyze_org_policies_rest_bad_request(transport: str='rest', request_type=asset_service.AnalyzeOrgPoliciesRequest):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'scope': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.analyze_org_policies(request)

def test_analyze_org_policies_rest_flattened():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.AnalyzeOrgPoliciesResponse()
        sample_request = {'scope': 'sample1/sample2'}
        mock_args = dict(scope='scope_value', constraint='constraint_value', filter='filter_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.AnalyzeOrgPoliciesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.analyze_org_policies(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{scope=*/*}:analyzeOrgPolicies' % client.transport._host, args[1])

def test_analyze_org_policies_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.analyze_org_policies(asset_service.AnalyzeOrgPoliciesRequest(), scope='scope_value', constraint='constraint_value', filter='filter_value')

def test_analyze_org_policies_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()], next_page_token='abc'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[], next_page_token='def'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()], next_page_token='ghi'), asset_service.AnalyzeOrgPoliciesResponse(org_policy_results=[asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult(), asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult()]))
        response = response + response
        response = tuple((asset_service.AnalyzeOrgPoliciesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'scope': 'sample1/sample2'}
        pager = client.analyze_org_policies(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, asset_service.AnalyzeOrgPoliciesResponse.OrgPolicyResult) for i in results))
        pages = list(client.analyze_org_policies(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [asset_service.AnalyzeOrgPolicyGovernedContainersRequest, dict])
def test_analyze_org_policy_governed_containers_rest(request_type):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'scope': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.AnalyzeOrgPolicyGovernedContainersResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.AnalyzeOrgPolicyGovernedContainersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.analyze_org_policy_governed_containers(request)
    assert isinstance(response, pagers.AnalyzeOrgPolicyGovernedContainersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_analyze_org_policy_governed_containers_rest_required_fields(request_type=asset_service.AnalyzeOrgPolicyGovernedContainersRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['scope'] = ''
    request_init['constraint'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'constraint' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_org_policy_governed_containers._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'constraint' in jsonified_request
    assert jsonified_request['constraint'] == request_init['constraint']
    jsonified_request['scope'] = 'scope_value'
    jsonified_request['constraint'] = 'constraint_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_org_policy_governed_containers._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('constraint', 'filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'scope' in jsonified_request
    assert jsonified_request['scope'] == 'scope_value'
    assert 'constraint' in jsonified_request
    assert jsonified_request['constraint'] == 'constraint_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.AnalyzeOrgPolicyGovernedContainersResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.AnalyzeOrgPolicyGovernedContainersResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.analyze_org_policy_governed_containers(request)
            expected_params = [('constraint', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_analyze_org_policy_governed_containers_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.analyze_org_policy_governed_containers._get_unset_required_fields({})
    assert set(unset_fields) == set(('constraint', 'filter', 'pageSize', 'pageToken')) & set(('scope', 'constraint'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_analyze_org_policy_governed_containers_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_analyze_org_policy_governed_containers') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_analyze_org_policy_governed_containers') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.AnalyzeOrgPolicyGovernedContainersRequest.pb(asset_service.AnalyzeOrgPolicyGovernedContainersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.AnalyzeOrgPolicyGovernedContainersResponse.to_json(asset_service.AnalyzeOrgPolicyGovernedContainersResponse())
        request = asset_service.AnalyzeOrgPolicyGovernedContainersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.AnalyzeOrgPolicyGovernedContainersResponse()
        client.analyze_org_policy_governed_containers(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_analyze_org_policy_governed_containers_rest_bad_request(transport: str='rest', request_type=asset_service.AnalyzeOrgPolicyGovernedContainersRequest):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'scope': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.analyze_org_policy_governed_containers(request)

def test_analyze_org_policy_governed_containers_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.AnalyzeOrgPolicyGovernedContainersResponse()
        sample_request = {'scope': 'sample1/sample2'}
        mock_args = dict(scope='scope_value', constraint='constraint_value', filter='filter_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.AnalyzeOrgPolicyGovernedContainersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.analyze_org_policy_governed_containers(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{scope=*/*}:analyzeOrgPolicyGovernedContainers' % client.transport._host, args[1])

def test_analyze_org_policy_governed_containers_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.analyze_org_policy_governed_containers(asset_service.AnalyzeOrgPolicyGovernedContainersRequest(), scope='scope_value', constraint='constraint_value', filter='filter_value')

def test_analyze_org_policy_governed_containers_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()], next_page_token='abc'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[], next_page_token='def'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()], next_page_token='ghi'), asset_service.AnalyzeOrgPolicyGovernedContainersResponse(governed_containers=[asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer(), asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer()]))
        response = response + response
        response = tuple((asset_service.AnalyzeOrgPolicyGovernedContainersResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'scope': 'sample1/sample2'}
        pager = client.analyze_org_policy_governed_containers(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, asset_service.AnalyzeOrgPolicyGovernedContainersResponse.GovernedContainer) for i in results))
        pages = list(client.analyze_org_policy_governed_containers(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [asset_service.AnalyzeOrgPolicyGovernedAssetsRequest, dict])
def test_analyze_org_policy_governed_assets_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'scope': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.analyze_org_policy_governed_assets(request)
    assert isinstance(response, pagers.AnalyzeOrgPolicyGovernedAssetsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_analyze_org_policy_governed_assets_rest_required_fields(request_type=asset_service.AnalyzeOrgPolicyGovernedAssetsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AssetServiceRestTransport
    request_init = {}
    request_init['scope'] = ''
    request_init['constraint'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'constraint' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_org_policy_governed_assets._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'constraint' in jsonified_request
    assert jsonified_request['constraint'] == request_init['constraint']
    jsonified_request['scope'] = 'scope_value'
    jsonified_request['constraint'] = 'constraint_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_org_policy_governed_assets._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('constraint', 'filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'scope' in jsonified_request
    assert jsonified_request['scope'] == 'scope_value'
    assert 'constraint' in jsonified_request
    assert jsonified_request['constraint'] == 'constraint_value'
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = asset_service.AnalyzeOrgPolicyGovernedAssetsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.analyze_org_policy_governed_assets(request)
            expected_params = [('constraint', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_analyze_org_policy_governed_assets_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.analyze_org_policy_governed_assets._get_unset_required_fields({})
    assert set(unset_fields) == set(('constraint', 'filter', 'pageSize', 'pageToken')) & set(('scope', 'constraint'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_analyze_org_policy_governed_assets_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AssetServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssetServiceRestInterceptor())
    client = AssetServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssetServiceRestInterceptor, 'post_analyze_org_policy_governed_assets') as post, mock.patch.object(transports.AssetServiceRestInterceptor, 'pre_analyze_org_policy_governed_assets') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = asset_service.AnalyzeOrgPolicyGovernedAssetsRequest.pb(asset_service.AnalyzeOrgPolicyGovernedAssetsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.to_json(asset_service.AnalyzeOrgPolicyGovernedAssetsResponse())
        request = asset_service.AnalyzeOrgPolicyGovernedAssetsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = asset_service.AnalyzeOrgPolicyGovernedAssetsResponse()
        client.analyze_org_policy_governed_assets(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_analyze_org_policy_governed_assets_rest_bad_request(transport: str='rest', request_type=asset_service.AnalyzeOrgPolicyGovernedAssetsRequest):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'scope': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.analyze_org_policy_governed_assets(request)

def test_analyze_org_policy_governed_assets_rest_flattened():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = asset_service.AnalyzeOrgPolicyGovernedAssetsResponse()
        sample_request = {'scope': 'sample1/sample2'}
        mock_args = dict(scope='scope_value', constraint='constraint_value', filter='filter_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.analyze_org_policy_governed_assets(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{scope=*/*}:analyzeOrgPolicyGovernedAssets' % client.transport._host, args[1])

def test_analyze_org_policy_governed_assets_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.analyze_org_policy_governed_assets(asset_service.AnalyzeOrgPolicyGovernedAssetsRequest(), scope='scope_value', constraint='constraint_value', filter='filter_value')

def test_analyze_org_policy_governed_assets_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()], next_page_token='abc'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[], next_page_token='def'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()], next_page_token='ghi'), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse(governed_assets=[asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset(), asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset()]))
        response = response + response
        response = tuple((asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'scope': 'sample1/sample2'}
        pager = client.analyze_org_policy_governed_assets(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, asset_service.AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset) for i in results))
        pages = list(client.analyze_org_policy_governed_assets(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.AssetServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.AssetServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AssetServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.AssetServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AssetServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AssetServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.AssetServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AssetServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.AssetServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = AssetServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.AssetServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.AssetServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.AssetServiceGrpcTransport, transports.AssetServiceGrpcAsyncIOTransport, transports.AssetServiceRestTransport])
def test_transport_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        print('Hello World!')
    transport = AssetServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.AssetServiceGrpcTransport)

def test_asset_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.AssetServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_asset_service_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.asset_v1.services.asset_service.transports.AssetServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.AssetServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('export_assets', 'list_assets', 'batch_get_assets_history', 'create_feed', 'get_feed', 'list_feeds', 'update_feed', 'delete_feed', 'search_all_resources', 'search_all_iam_policies', 'analyze_iam_policy', 'analyze_iam_policy_longrunning', 'analyze_move', 'query_assets', 'create_saved_query', 'get_saved_query', 'list_saved_queries', 'update_saved_query', 'delete_saved_query', 'batch_get_effective_iam_policies', 'analyze_org_policies', 'analyze_org_policy_governed_containers', 'analyze_org_policy_governed_assets', 'get_operation')
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

def test_asset_service_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.asset_v1.services.asset_service.transports.AssetServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AssetServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_asset_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.asset_v1.services.asset_service.transports.AssetServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AssetServiceTransport()
        adc.assert_called_once()

def test_asset_service_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        AssetServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.AssetServiceGrpcTransport, transports.AssetServiceGrpcAsyncIOTransport])
def test_asset_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.AssetServiceGrpcTransport, transports.AssetServiceGrpcAsyncIOTransport, transports.AssetServiceRestTransport])
def test_asset_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.AssetServiceGrpcTransport, grpc_helpers), (transports.AssetServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_asset_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudasset.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='cloudasset.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.AssetServiceGrpcTransport, transports.AssetServiceGrpcAsyncIOTransport])
def test_asset_service_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        for i in range(10):
            print('nop')
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

def test_asset_service_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.AssetServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_asset_service_rest_lro_client():
    if False:
        print('Hello World!')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_asset_service_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudasset.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudasset.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudasset.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_asset_service_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudasset.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudasset.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudasset.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_asset_service_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = AssetServiceClient(credentials=creds1, transport=transport_name)
    client2 = AssetServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.export_assets._session
    session2 = client2.transport.export_assets._session
    assert session1 != session2
    session1 = client1.transport.list_assets._session
    session2 = client2.transport.list_assets._session
    assert session1 != session2
    session1 = client1.transport.batch_get_assets_history._session
    session2 = client2.transport.batch_get_assets_history._session
    assert session1 != session2
    session1 = client1.transport.create_feed._session
    session2 = client2.transport.create_feed._session
    assert session1 != session2
    session1 = client1.transport.get_feed._session
    session2 = client2.transport.get_feed._session
    assert session1 != session2
    session1 = client1.transport.list_feeds._session
    session2 = client2.transport.list_feeds._session
    assert session1 != session2
    session1 = client1.transport.update_feed._session
    session2 = client2.transport.update_feed._session
    assert session1 != session2
    session1 = client1.transport.delete_feed._session
    session2 = client2.transport.delete_feed._session
    assert session1 != session2
    session1 = client1.transport.search_all_resources._session
    session2 = client2.transport.search_all_resources._session
    assert session1 != session2
    session1 = client1.transport.search_all_iam_policies._session
    session2 = client2.transport.search_all_iam_policies._session
    assert session1 != session2
    session1 = client1.transport.analyze_iam_policy._session
    session2 = client2.transport.analyze_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.analyze_iam_policy_longrunning._session
    session2 = client2.transport.analyze_iam_policy_longrunning._session
    assert session1 != session2
    session1 = client1.transport.analyze_move._session
    session2 = client2.transport.analyze_move._session
    assert session1 != session2
    session1 = client1.transport.query_assets._session
    session2 = client2.transport.query_assets._session
    assert session1 != session2
    session1 = client1.transport.create_saved_query._session
    session2 = client2.transport.create_saved_query._session
    assert session1 != session2
    session1 = client1.transport.get_saved_query._session
    session2 = client2.transport.get_saved_query._session
    assert session1 != session2
    session1 = client1.transport.list_saved_queries._session
    session2 = client2.transport.list_saved_queries._session
    assert session1 != session2
    session1 = client1.transport.update_saved_query._session
    session2 = client2.transport.update_saved_query._session
    assert session1 != session2
    session1 = client1.transport.delete_saved_query._session
    session2 = client2.transport.delete_saved_query._session
    assert session1 != session2
    session1 = client1.transport.batch_get_effective_iam_policies._session
    session2 = client2.transport.batch_get_effective_iam_policies._session
    assert session1 != session2
    session1 = client1.transport.analyze_org_policies._session
    session2 = client2.transport.analyze_org_policies._session
    assert session1 != session2
    session1 = client1.transport.analyze_org_policy_governed_containers._session
    session2 = client2.transport.analyze_org_policy_governed_containers._session
    assert session1 != session2
    session1 = client1.transport.analyze_org_policy_governed_assets._session
    session2 = client2.transport.analyze_org_policy_governed_assets._session
    assert session1 != session2

def test_asset_service_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AssetServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_asset_service_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AssetServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.AssetServiceGrpcTransport, transports.AssetServiceGrpcAsyncIOTransport])
def test_asset_service_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        while True:
            i = 10
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

@pytest.mark.parametrize('transport_class', [transports.AssetServiceGrpcTransport, transports.AssetServiceGrpcAsyncIOTransport])
def test_asset_service_transport_channel_mtls_with_adc(transport_class):
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

def test_asset_service_grpc_lro_client():
    if False:
        return 10
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_asset_service_grpc_lro_async_client():
    if False:
        return 10
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_access_level_path():
    if False:
        for i in range(10):
            print('nop')
    access_policy = 'squid'
    access_level = 'clam'
    expected = 'accessPolicies/{access_policy}/accessLevels/{access_level}'.format(access_policy=access_policy, access_level=access_level)
    actual = AssetServiceClient.access_level_path(access_policy, access_level)
    assert expected == actual

def test_parse_access_level_path():
    if False:
        while True:
            i = 10
    expected = {'access_policy': 'whelk', 'access_level': 'octopus'}
    path = AssetServiceClient.access_level_path(**expected)
    actual = AssetServiceClient.parse_access_level_path(path)
    assert expected == actual

def test_access_policy_path():
    if False:
        print('Hello World!')
    access_policy = 'oyster'
    expected = 'accessPolicies/{access_policy}'.format(access_policy=access_policy)
    actual = AssetServiceClient.access_policy_path(access_policy)
    assert expected == actual

def test_parse_access_policy_path():
    if False:
        i = 10
        return i + 15
    expected = {'access_policy': 'nudibranch'}
    path = AssetServiceClient.access_policy_path(**expected)
    actual = AssetServiceClient.parse_access_policy_path(path)
    assert expected == actual

def test_asset_path():
    if False:
        return 10
    expected = '*'.format()
    actual = AssetServiceClient.asset_path()
    assert expected == actual

def test_parse_asset_path():
    if False:
        i = 10
        return i + 15
    expected = {}
    path = AssetServiceClient.asset_path(**expected)
    actual = AssetServiceClient.parse_asset_path(path)
    assert expected == actual

def test_feed_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    feed = 'mussel'
    expected = 'projects/{project}/feeds/{feed}'.format(project=project, feed=feed)
    actual = AssetServiceClient.feed_path(project, feed)
    assert expected == actual

def test_parse_feed_path():
    if False:
        return 10
    expected = {'project': 'winkle', 'feed': 'nautilus'}
    path = AssetServiceClient.feed_path(**expected)
    actual = AssetServiceClient.parse_feed_path(path)
    assert expected == actual

def test_inventory_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    location = 'abalone'
    instance = 'squid'
    expected = 'projects/{project}/locations/{location}/instances/{instance}/inventory'.format(project=project, location=location, instance=instance)
    actual = AssetServiceClient.inventory_path(project, location, instance)
    assert expected == actual

def test_parse_inventory_path():
    if False:
        print('Hello World!')
    expected = {'project': 'clam', 'location': 'whelk', 'instance': 'octopus'}
    path = AssetServiceClient.inventory_path(**expected)
    actual = AssetServiceClient.parse_inventory_path(path)
    assert expected == actual

def test_saved_query_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    saved_query = 'nudibranch'
    expected = 'projects/{project}/savedQueries/{saved_query}'.format(project=project, saved_query=saved_query)
    actual = AssetServiceClient.saved_query_path(project, saved_query)
    assert expected == actual

def test_parse_saved_query_path():
    if False:
        print('Hello World!')
    expected = {'project': 'cuttlefish', 'saved_query': 'mussel'}
    path = AssetServiceClient.saved_query_path(**expected)
    actual = AssetServiceClient.parse_saved_query_path(path)
    assert expected == actual

def test_service_perimeter_path():
    if False:
        while True:
            i = 10
    access_policy = 'winkle'
    service_perimeter = 'nautilus'
    expected = 'accessPolicies/{access_policy}/servicePerimeters/{service_perimeter}'.format(access_policy=access_policy, service_perimeter=service_perimeter)
    actual = AssetServiceClient.service_perimeter_path(access_policy, service_perimeter)
    assert expected == actual

def test_parse_service_perimeter_path():
    if False:
        print('Hello World!')
    expected = {'access_policy': 'scallop', 'service_perimeter': 'abalone'}
    path = AssetServiceClient.service_perimeter_path(**expected)
    actual = AssetServiceClient.parse_service_perimeter_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = AssetServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'clam'}
    path = AssetServiceClient.common_billing_account_path(**expected)
    actual = AssetServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = AssetServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'octopus'}
    path = AssetServiceClient.common_folder_path(**expected)
    actual = AssetServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = AssetServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'nudibranch'}
    path = AssetServiceClient.common_organization_path(**expected)
    actual = AssetServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = AssetServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'mussel'}
    path = AssetServiceClient.common_project_path(**expected)
    actual = AssetServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = AssetServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = AssetServiceClient.common_location_path(**expected)
    actual = AssetServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.AssetServiceTransport, '_prep_wrapped_messages') as prep:
        client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.AssetServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = AssetServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        i = 10
        return i + 15
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'sample1/sample2/operations/sample3/sample4'}, request)
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
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'sample1/sample2/operations/sample3/sample4'}
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

def test_get_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = AssetServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        i = 10
        return i + 15
    transports = ['rest', 'grpc']
    for transport in transports:
        client = AssetServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(AssetServiceClient, transports.AssetServiceGrpcTransport), (AssetServiceAsyncClient, transports.AssetServiceGrpcAsyncIOTransport)])
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