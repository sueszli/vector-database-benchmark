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
from google.oauth2 import service_account
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.shopping.type.types import types
from google.type import interval_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.shopping.merchant_inventories_v1beta.services.local_inventory_service import LocalInventoryServiceAsyncClient, LocalInventoryServiceClient, pagers, transports
from google.shopping.merchant_inventories_v1beta.types import localinventory

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
        while True:
            i = 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert LocalInventoryServiceClient._get_default_mtls_endpoint(None) is None
    assert LocalInventoryServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert LocalInventoryServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert LocalInventoryServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert LocalInventoryServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert LocalInventoryServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(LocalInventoryServiceClient, 'grpc'), (LocalInventoryServiceAsyncClient, 'grpc_asyncio'), (LocalInventoryServiceClient, 'rest')])
def test_local_inventory_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('merchantapi.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://merchantapi.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.LocalInventoryServiceGrpcTransport, 'grpc'), (transports.LocalInventoryServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.LocalInventoryServiceRestTransport, 'rest')])
def test_local_inventory_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(LocalInventoryServiceClient, 'grpc'), (LocalInventoryServiceAsyncClient, 'grpc_asyncio'), (LocalInventoryServiceClient, 'rest')])
def test_local_inventory_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('merchantapi.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://merchantapi.googleapis.com')

def test_local_inventory_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = LocalInventoryServiceClient.get_transport_class()
    available_transports = [transports.LocalInventoryServiceGrpcTransport, transports.LocalInventoryServiceRestTransport]
    assert transport in available_transports
    transport = LocalInventoryServiceClient.get_transport_class('grpc')
    assert transport == transports.LocalInventoryServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(LocalInventoryServiceClient, transports.LocalInventoryServiceGrpcTransport, 'grpc'), (LocalInventoryServiceAsyncClient, transports.LocalInventoryServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (LocalInventoryServiceClient, transports.LocalInventoryServiceRestTransport, 'rest')])
@mock.patch.object(LocalInventoryServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LocalInventoryServiceClient))
@mock.patch.object(LocalInventoryServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LocalInventoryServiceAsyncClient))
def test_local_inventory_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(LocalInventoryServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(LocalInventoryServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(LocalInventoryServiceClient, transports.LocalInventoryServiceGrpcTransport, 'grpc', 'true'), (LocalInventoryServiceAsyncClient, transports.LocalInventoryServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (LocalInventoryServiceClient, transports.LocalInventoryServiceGrpcTransport, 'grpc', 'false'), (LocalInventoryServiceAsyncClient, transports.LocalInventoryServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (LocalInventoryServiceClient, transports.LocalInventoryServiceRestTransport, 'rest', 'true'), (LocalInventoryServiceClient, transports.LocalInventoryServiceRestTransport, 'rest', 'false')])
@mock.patch.object(LocalInventoryServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LocalInventoryServiceClient))
@mock.patch.object(LocalInventoryServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LocalInventoryServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_local_inventory_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [LocalInventoryServiceClient, LocalInventoryServiceAsyncClient])
@mock.patch.object(LocalInventoryServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LocalInventoryServiceClient))
@mock.patch.object(LocalInventoryServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LocalInventoryServiceAsyncClient))
def test_local_inventory_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(LocalInventoryServiceClient, transports.LocalInventoryServiceGrpcTransport, 'grpc'), (LocalInventoryServiceAsyncClient, transports.LocalInventoryServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (LocalInventoryServiceClient, transports.LocalInventoryServiceRestTransport, 'rest')])
def test_local_inventory_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(LocalInventoryServiceClient, transports.LocalInventoryServiceGrpcTransport, 'grpc', grpc_helpers), (LocalInventoryServiceAsyncClient, transports.LocalInventoryServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (LocalInventoryServiceClient, transports.LocalInventoryServiceRestTransport, 'rest', None)])
def test_local_inventory_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_local_inventory_service_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.shopping.merchant_inventories_v1beta.services.local_inventory_service.transports.LocalInventoryServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = LocalInventoryServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(LocalInventoryServiceClient, transports.LocalInventoryServiceGrpcTransport, 'grpc', grpc_helpers), (LocalInventoryServiceAsyncClient, transports.LocalInventoryServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_local_inventory_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('merchantapi.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/content',), scopes=None, default_host='merchantapi.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [localinventory.ListLocalInventoriesRequest, dict])
def test_list_local_inventories(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_local_inventories), '__call__') as call:
        call.return_value = localinventory.ListLocalInventoriesResponse(next_page_token='next_page_token_value')
        response = client.list_local_inventories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == localinventory.ListLocalInventoriesRequest()
    assert isinstance(response, pagers.ListLocalInventoriesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_local_inventories_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_local_inventories), '__call__') as call:
        client.list_local_inventories()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == localinventory.ListLocalInventoriesRequest()

@pytest.mark.asyncio
async def test_list_local_inventories_async(transport: str='grpc_asyncio', request_type=localinventory.ListLocalInventoriesRequest):
    client = LocalInventoryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_local_inventories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(localinventory.ListLocalInventoriesResponse(next_page_token='next_page_token_value'))
        response = await client.list_local_inventories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == localinventory.ListLocalInventoriesRequest()
    assert isinstance(response, pagers.ListLocalInventoriesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_local_inventories_async_from_dict():
    await test_list_local_inventories_async(request_type=dict)

def test_list_local_inventories_field_headers():
    if False:
        print('Hello World!')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = localinventory.ListLocalInventoriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_local_inventories), '__call__') as call:
        call.return_value = localinventory.ListLocalInventoriesResponse()
        client.list_local_inventories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_local_inventories_field_headers_async():
    client = LocalInventoryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = localinventory.ListLocalInventoriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_local_inventories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(localinventory.ListLocalInventoriesResponse())
        await client.list_local_inventories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_local_inventories_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_local_inventories), '__call__') as call:
        call.return_value = localinventory.ListLocalInventoriesResponse()
        client.list_local_inventories(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_local_inventories_flattened_error():
    if False:
        print('Hello World!')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_local_inventories(localinventory.ListLocalInventoriesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_local_inventories_flattened_async():
    client = LocalInventoryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_local_inventories), '__call__') as call:
        call.return_value = localinventory.ListLocalInventoriesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(localinventory.ListLocalInventoriesResponse())
        response = await client.list_local_inventories(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_local_inventories_flattened_error_async():
    client = LocalInventoryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_local_inventories(localinventory.ListLocalInventoriesRequest(), parent='parent_value')

def test_list_local_inventories_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_local_inventories), '__call__') as call:
        call.side_effect = (localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory(), localinventory.LocalInventory(), localinventory.LocalInventory()], next_page_token='abc'), localinventory.ListLocalInventoriesResponse(local_inventories=[], next_page_token='def'), localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory()], next_page_token='ghi'), localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory(), localinventory.LocalInventory()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_local_inventories(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, localinventory.LocalInventory) for i in results))

def test_list_local_inventories_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_local_inventories), '__call__') as call:
        call.side_effect = (localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory(), localinventory.LocalInventory(), localinventory.LocalInventory()], next_page_token='abc'), localinventory.ListLocalInventoriesResponse(local_inventories=[], next_page_token='def'), localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory()], next_page_token='ghi'), localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory(), localinventory.LocalInventory()]), RuntimeError)
        pages = list(client.list_local_inventories(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_local_inventories_async_pager():
    client = LocalInventoryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_local_inventories), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory(), localinventory.LocalInventory(), localinventory.LocalInventory()], next_page_token='abc'), localinventory.ListLocalInventoriesResponse(local_inventories=[], next_page_token='def'), localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory()], next_page_token='ghi'), localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory(), localinventory.LocalInventory()]), RuntimeError)
        async_pager = await client.list_local_inventories(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, localinventory.LocalInventory) for i in responses))

@pytest.mark.asyncio
async def test_list_local_inventories_async_pages():
    client = LocalInventoryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_local_inventories), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory(), localinventory.LocalInventory(), localinventory.LocalInventory()], next_page_token='abc'), localinventory.ListLocalInventoriesResponse(local_inventories=[], next_page_token='def'), localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory()], next_page_token='ghi'), localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory(), localinventory.LocalInventory()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_local_inventories(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [localinventory.InsertLocalInventoryRequest, dict])
def test_insert_local_inventory(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.insert_local_inventory), '__call__') as call:
        call.return_value = localinventory.LocalInventory(name='name_value', account=749, store_code='store_code_value', availability='availability_value', quantity=895, pickup_method='pickup_method_value', pickup_sla='pickup_sla_value', instore_product_location='instore_product_location_value')
        response = client.insert_local_inventory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == localinventory.InsertLocalInventoryRequest()
    assert isinstance(response, localinventory.LocalInventory)
    assert response.name == 'name_value'
    assert response.account == 749
    assert response.store_code == 'store_code_value'
    assert response.availability == 'availability_value'
    assert response.quantity == 895
    assert response.pickup_method == 'pickup_method_value'
    assert response.pickup_sla == 'pickup_sla_value'
    assert response.instore_product_location == 'instore_product_location_value'

def test_insert_local_inventory_empty_call():
    if False:
        return 10
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.insert_local_inventory), '__call__') as call:
        client.insert_local_inventory()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == localinventory.InsertLocalInventoryRequest()

@pytest.mark.asyncio
async def test_insert_local_inventory_async(transport: str='grpc_asyncio', request_type=localinventory.InsertLocalInventoryRequest):
    client = LocalInventoryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.insert_local_inventory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(localinventory.LocalInventory(name='name_value', account=749, store_code='store_code_value', availability='availability_value', quantity=895, pickup_method='pickup_method_value', pickup_sla='pickup_sla_value', instore_product_location='instore_product_location_value'))
        response = await client.insert_local_inventory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == localinventory.InsertLocalInventoryRequest()
    assert isinstance(response, localinventory.LocalInventory)
    assert response.name == 'name_value'
    assert response.account == 749
    assert response.store_code == 'store_code_value'
    assert response.availability == 'availability_value'
    assert response.quantity == 895
    assert response.pickup_method == 'pickup_method_value'
    assert response.pickup_sla == 'pickup_sla_value'
    assert response.instore_product_location == 'instore_product_location_value'

@pytest.mark.asyncio
async def test_insert_local_inventory_async_from_dict():
    await test_insert_local_inventory_async(request_type=dict)

def test_insert_local_inventory_field_headers():
    if False:
        return 10
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = localinventory.InsertLocalInventoryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.insert_local_inventory), '__call__') as call:
        call.return_value = localinventory.LocalInventory()
        client.insert_local_inventory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_insert_local_inventory_field_headers_async():
    client = LocalInventoryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = localinventory.InsertLocalInventoryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.insert_local_inventory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(localinventory.LocalInventory())
        await client.insert_local_inventory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [localinventory.DeleteLocalInventoryRequest, dict])
def test_delete_local_inventory(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_local_inventory), '__call__') as call:
        call.return_value = None
        response = client.delete_local_inventory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == localinventory.DeleteLocalInventoryRequest()
    assert response is None

def test_delete_local_inventory_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_local_inventory), '__call__') as call:
        client.delete_local_inventory()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == localinventory.DeleteLocalInventoryRequest()

@pytest.mark.asyncio
async def test_delete_local_inventory_async(transport: str='grpc_asyncio', request_type=localinventory.DeleteLocalInventoryRequest):
    client = LocalInventoryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_local_inventory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_local_inventory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == localinventory.DeleteLocalInventoryRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_local_inventory_async_from_dict():
    await test_delete_local_inventory_async(request_type=dict)

def test_delete_local_inventory_field_headers():
    if False:
        while True:
            i = 10
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = localinventory.DeleteLocalInventoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_local_inventory), '__call__') as call:
        call.return_value = None
        client.delete_local_inventory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_local_inventory_field_headers_async():
    client = LocalInventoryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = localinventory.DeleteLocalInventoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_local_inventory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_local_inventory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_local_inventory_flattened():
    if False:
        while True:
            i = 10
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_local_inventory), '__call__') as call:
        call.return_value = None
        client.delete_local_inventory(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_local_inventory_flattened_error():
    if False:
        i = 10
        return i + 15
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_local_inventory(localinventory.DeleteLocalInventoryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_local_inventory_flattened_async():
    client = LocalInventoryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_local_inventory), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_local_inventory(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_local_inventory_flattened_error_async():
    client = LocalInventoryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_local_inventory(localinventory.DeleteLocalInventoryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [localinventory.ListLocalInventoriesRequest, dict])
def test_list_local_inventories_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'accounts/sample1/products/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = localinventory.ListLocalInventoriesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = localinventory.ListLocalInventoriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_local_inventories(request)
    assert isinstance(response, pagers.ListLocalInventoriesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_local_inventories_rest_required_fields(request_type=localinventory.ListLocalInventoriesRequest):
    if False:
        return 10
    transport_class = transports.LocalInventoryServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_local_inventories._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_local_inventories._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = localinventory.ListLocalInventoriesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = localinventory.ListLocalInventoriesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_local_inventories(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_local_inventories_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.LocalInventoryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_local_inventories._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_local_inventories_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.LocalInventoryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LocalInventoryServiceRestInterceptor())
    client = LocalInventoryServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LocalInventoryServiceRestInterceptor, 'post_list_local_inventories') as post, mock.patch.object(transports.LocalInventoryServiceRestInterceptor, 'pre_list_local_inventories') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = localinventory.ListLocalInventoriesRequest.pb(localinventory.ListLocalInventoriesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = localinventory.ListLocalInventoriesResponse.to_json(localinventory.ListLocalInventoriesResponse())
        request = localinventory.ListLocalInventoriesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = localinventory.ListLocalInventoriesResponse()
        client.list_local_inventories(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_local_inventories_rest_bad_request(transport: str='rest', request_type=localinventory.ListLocalInventoriesRequest):
    if False:
        print('Hello World!')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'accounts/sample1/products/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_local_inventories(request)

def test_list_local_inventories_rest_flattened():
    if False:
        while True:
            i = 10
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = localinventory.ListLocalInventoriesResponse()
        sample_request = {'parent': 'accounts/sample1/products/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = localinventory.ListLocalInventoriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_local_inventories(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/inventories/v1beta/{parent=accounts/*/products/*}/localInventories' % client.transport._host, args[1])

def test_list_local_inventories_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_local_inventories(localinventory.ListLocalInventoriesRequest(), parent='parent_value')

def test_list_local_inventories_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory(), localinventory.LocalInventory(), localinventory.LocalInventory()], next_page_token='abc'), localinventory.ListLocalInventoriesResponse(local_inventories=[], next_page_token='def'), localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory()], next_page_token='ghi'), localinventory.ListLocalInventoriesResponse(local_inventories=[localinventory.LocalInventory(), localinventory.LocalInventory()]))
        response = response + response
        response = tuple((localinventory.ListLocalInventoriesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'accounts/sample1/products/sample2'}
        pager = client.list_local_inventories(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, localinventory.LocalInventory) for i in results))
        pages = list(client.list_local_inventories(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [localinventory.InsertLocalInventoryRequest, dict])
def test_insert_local_inventory_rest(request_type):
    if False:
        print('Hello World!')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'accounts/sample1/products/sample2'}
    request_init['local_inventory'] = {'name': 'name_value', 'account': 749, 'store_code': 'store_code_value', 'price': {'amount_micros': 1408, 'currency_code': 'currency_code_value'}, 'sale_price': {}, 'sale_price_effective_date': {'start_time': {'seconds': 751, 'nanos': 543}, 'end_time': {}}, 'availability': 'availability_value', 'quantity': 895, 'pickup_method': 'pickup_method_value', 'pickup_sla': 'pickup_sla_value', 'instore_product_location': 'instore_product_location_value', 'custom_attributes': [{'name': 'name_value', 'value': 'value_value', 'group_values': {}}]}
    test_field = localinventory.InsertLocalInventoryRequest.meta.fields['local_inventory']

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
    for (field, value) in request_init['local_inventory'].items():
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
                for i in range(0, len(request_init['local_inventory'][field])):
                    del request_init['local_inventory'][field][i][subfield]
            else:
                del request_init['local_inventory'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = localinventory.LocalInventory(name='name_value', account=749, store_code='store_code_value', availability='availability_value', quantity=895, pickup_method='pickup_method_value', pickup_sla='pickup_sla_value', instore_product_location='instore_product_location_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = localinventory.LocalInventory.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.insert_local_inventory(request)
    assert isinstance(response, localinventory.LocalInventory)
    assert response.name == 'name_value'
    assert response.account == 749
    assert response.store_code == 'store_code_value'
    assert response.availability == 'availability_value'
    assert response.quantity == 895
    assert response.pickup_method == 'pickup_method_value'
    assert response.pickup_sla == 'pickup_sla_value'
    assert response.instore_product_location == 'instore_product_location_value'

def test_insert_local_inventory_rest_required_fields(request_type=localinventory.InsertLocalInventoryRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.LocalInventoryServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert_local_inventory._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert_local_inventory._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = localinventory.LocalInventory()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = localinventory.LocalInventory.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.insert_local_inventory(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_insert_local_inventory_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.LocalInventoryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert_local_inventory._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'localInventory'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_local_inventory_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.LocalInventoryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LocalInventoryServiceRestInterceptor())
    client = LocalInventoryServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LocalInventoryServiceRestInterceptor, 'post_insert_local_inventory') as post, mock.patch.object(transports.LocalInventoryServiceRestInterceptor, 'pre_insert_local_inventory') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = localinventory.InsertLocalInventoryRequest.pb(localinventory.InsertLocalInventoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = localinventory.LocalInventory.to_json(localinventory.LocalInventory())
        request = localinventory.InsertLocalInventoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = localinventory.LocalInventory()
        client.insert_local_inventory(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_local_inventory_rest_bad_request(transport: str='rest', request_type=localinventory.InsertLocalInventoryRequest):
    if False:
        return 10
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'accounts/sample1/products/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.insert_local_inventory(request)

def test_insert_local_inventory_rest_error():
    if False:
        return 10
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [localinventory.DeleteLocalInventoryRequest, dict])
def test_delete_local_inventory_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'accounts/sample1/products/sample2/localInventories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_local_inventory(request)
    assert response is None

def test_delete_local_inventory_rest_required_fields(request_type=localinventory.DeleteLocalInventoryRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.LocalInventoryServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_local_inventory._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_local_inventory._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_local_inventory(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_local_inventory_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.LocalInventoryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_local_inventory._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_local_inventory_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.LocalInventoryServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LocalInventoryServiceRestInterceptor())
    client = LocalInventoryServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LocalInventoryServiceRestInterceptor, 'pre_delete_local_inventory') as pre:
        pre.assert_not_called()
        pb_message = localinventory.DeleteLocalInventoryRequest.pb(localinventory.DeleteLocalInventoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = localinventory.DeleteLocalInventoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_local_inventory(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_local_inventory_rest_bad_request(transport: str='rest', request_type=localinventory.DeleteLocalInventoryRequest):
    if False:
        while True:
            i = 10
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'accounts/sample1/products/sample2/localInventories/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_local_inventory(request)

def test_delete_local_inventory_rest_flattened():
    if False:
        print('Hello World!')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'accounts/sample1/products/sample2/localInventories/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_local_inventory(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/inventories/v1beta/{name=accounts/*/products/*/localInventories/*}' % client.transport._host, args[1])

def test_delete_local_inventory_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_local_inventory(localinventory.DeleteLocalInventoryRequest(), name='name_value')

def test_delete_local_inventory_rest_error():
    if False:
        print('Hello World!')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.LocalInventoryServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.LocalInventoryServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = LocalInventoryServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.LocalInventoryServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = LocalInventoryServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = LocalInventoryServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.LocalInventoryServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = LocalInventoryServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.LocalInventoryServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = LocalInventoryServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.LocalInventoryServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.LocalInventoryServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.LocalInventoryServiceGrpcTransport, transports.LocalInventoryServiceGrpcAsyncIOTransport, transports.LocalInventoryServiceRestTransport])
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
        return 10
    transport = LocalInventoryServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.LocalInventoryServiceGrpcTransport)

def test_local_inventory_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.LocalInventoryServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_local_inventory_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.shopping.merchant_inventories_v1beta.services.local_inventory_service.transports.LocalInventoryServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.LocalInventoryServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_local_inventories', 'insert_local_inventory', 'delete_local_inventory')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_local_inventory_service_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.shopping.merchant_inventories_v1beta.services.local_inventory_service.transports.LocalInventoryServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.LocalInventoryServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/content',), quota_project_id='octopus')

def test_local_inventory_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.shopping.merchant_inventories_v1beta.services.local_inventory_service.transports.LocalInventoryServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.LocalInventoryServiceTransport()
        adc.assert_called_once()

def test_local_inventory_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        LocalInventoryServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/content',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.LocalInventoryServiceGrpcTransport, transports.LocalInventoryServiceGrpcAsyncIOTransport])
def test_local_inventory_service_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/content',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.LocalInventoryServiceGrpcTransport, transports.LocalInventoryServiceGrpcAsyncIOTransport, transports.LocalInventoryServiceRestTransport])
def test_local_inventory_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.LocalInventoryServiceGrpcTransport, grpc_helpers), (transports.LocalInventoryServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_local_inventory_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('merchantapi.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/content',), scopes=['1', '2'], default_host='merchantapi.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.LocalInventoryServiceGrpcTransport, transports.LocalInventoryServiceGrpcAsyncIOTransport])
def test_local_inventory_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_local_inventory_service_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.LocalInventoryServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_local_inventory_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='merchantapi.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('merchantapi.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://merchantapi.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_local_inventory_service_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='merchantapi.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('merchantapi.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://merchantapi.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_local_inventory_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = LocalInventoryServiceClient(credentials=creds1, transport=transport_name)
    client2 = LocalInventoryServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_local_inventories._session
    session2 = client2.transport.list_local_inventories._session
    assert session1 != session2
    session1 = client1.transport.insert_local_inventory._session
    session2 = client2.transport.insert_local_inventory._session
    assert session1 != session2
    session1 = client1.transport.delete_local_inventory._session
    session2 = client2.transport.delete_local_inventory._session
    assert session1 != session2

def test_local_inventory_service_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.LocalInventoryServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_local_inventory_service_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.LocalInventoryServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.LocalInventoryServiceGrpcTransport, transports.LocalInventoryServiceGrpcAsyncIOTransport])
def test_local_inventory_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.LocalInventoryServiceGrpcTransport, transports.LocalInventoryServiceGrpcAsyncIOTransport])
def test_local_inventory_service_transport_channel_mtls_with_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
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

def test_local_inventory_path():
    if False:
        for i in range(10):
            print('nop')
    account = 'squid'
    product = 'clam'
    store_code = 'whelk'
    expected = 'accounts/{account}/products/{product}/localInventories/{store_code}'.format(account=account, product=product, store_code=store_code)
    actual = LocalInventoryServiceClient.local_inventory_path(account, product, store_code)
    assert expected == actual

def test_parse_local_inventory_path():
    if False:
        return 10
    expected = {'account': 'octopus', 'product': 'oyster', 'store_code': 'nudibranch'}
    path = LocalInventoryServiceClient.local_inventory_path(**expected)
    actual = LocalInventoryServiceClient.parse_local_inventory_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = LocalInventoryServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'mussel'}
    path = LocalInventoryServiceClient.common_billing_account_path(**expected)
    actual = LocalInventoryServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = LocalInventoryServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'nautilus'}
    path = LocalInventoryServiceClient.common_folder_path(**expected)
    actual = LocalInventoryServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = LocalInventoryServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'abalone'}
    path = LocalInventoryServiceClient.common_organization_path(**expected)
    actual = LocalInventoryServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = LocalInventoryServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'clam'}
    path = LocalInventoryServiceClient.common_project_path(**expected)
    actual = LocalInventoryServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = LocalInventoryServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = LocalInventoryServiceClient.common_location_path(**expected)
    actual = LocalInventoryServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.LocalInventoryServiceTransport, '_prep_wrapped_messages') as prep:
        client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.LocalInventoryServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = LocalInventoryServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = LocalInventoryServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = LocalInventoryServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(LocalInventoryServiceClient, transports.LocalInventoryServiceGrpcTransport), (LocalInventoryServiceAsyncClient, transports.LocalInventoryServiceGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)