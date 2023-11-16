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
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import struct_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.area120.tables_v1alpha1.services.tables_service import TablesServiceAsyncClient, TablesServiceClient, pagers, transports
from google.area120.tables_v1alpha1.types import tables

def client_cert_source_callback():
    if False:
        while True:
            i = 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        i = 10
        return i + 15
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert TablesServiceClient._get_default_mtls_endpoint(None) is None
    assert TablesServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert TablesServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert TablesServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert TablesServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert TablesServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(TablesServiceClient, 'grpc'), (TablesServiceAsyncClient, 'grpc_asyncio'), (TablesServiceClient, 'rest')])
def test_tables_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('area120tables.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://area120tables.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.TablesServiceGrpcTransport, 'grpc'), (transports.TablesServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.TablesServiceRestTransport, 'rest')])
def test_tables_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(TablesServiceClient, 'grpc'), (TablesServiceAsyncClient, 'grpc_asyncio'), (TablesServiceClient, 'rest')])
def test_tables_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('area120tables.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://area120tables.googleapis.com')

def test_tables_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = TablesServiceClient.get_transport_class()
    available_transports = [transports.TablesServiceGrpcTransport, transports.TablesServiceRestTransport]
    assert transport in available_transports
    transport = TablesServiceClient.get_transport_class('grpc')
    assert transport == transports.TablesServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TablesServiceClient, transports.TablesServiceGrpcTransport, 'grpc'), (TablesServiceAsyncClient, transports.TablesServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (TablesServiceClient, transports.TablesServiceRestTransport, 'rest')])
@mock.patch.object(TablesServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TablesServiceClient))
@mock.patch.object(TablesServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TablesServiceAsyncClient))
def test_tables_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(TablesServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(TablesServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(TablesServiceClient, transports.TablesServiceGrpcTransport, 'grpc', 'true'), (TablesServiceAsyncClient, transports.TablesServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (TablesServiceClient, transports.TablesServiceGrpcTransport, 'grpc', 'false'), (TablesServiceAsyncClient, transports.TablesServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (TablesServiceClient, transports.TablesServiceRestTransport, 'rest', 'true'), (TablesServiceClient, transports.TablesServiceRestTransport, 'rest', 'false')])
@mock.patch.object(TablesServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TablesServiceClient))
@mock.patch.object(TablesServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TablesServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_tables_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [TablesServiceClient, TablesServiceAsyncClient])
@mock.patch.object(TablesServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TablesServiceClient))
@mock.patch.object(TablesServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TablesServiceAsyncClient))
def test_tables_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TablesServiceClient, transports.TablesServiceGrpcTransport, 'grpc'), (TablesServiceAsyncClient, transports.TablesServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (TablesServiceClient, transports.TablesServiceRestTransport, 'rest')])
def test_tables_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TablesServiceClient, transports.TablesServiceGrpcTransport, 'grpc', grpc_helpers), (TablesServiceAsyncClient, transports.TablesServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (TablesServiceClient, transports.TablesServiceRestTransport, 'rest', None)])
def test_tables_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_tables_service_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.area120.tables_v1alpha1.services.tables_service.transports.TablesServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = TablesServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TablesServiceClient, transports.TablesServiceGrpcTransport, 'grpc', grpc_helpers), (TablesServiceAsyncClient, transports.TablesServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_tables_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('area120tables.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/spreadsheets.readonly', 'https://www.googleapis.com/auth/tables'), scopes=None, default_host='area120tables.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [tables.GetTableRequest, dict])
def test_get_table(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_table), '__call__') as call:
        call.return_value = tables.Table(name='name_value', display_name='display_name_value')
        response = client.get_table(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.GetTableRequest()
    assert isinstance(response, tables.Table)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_get_table_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_table), '__call__') as call:
        client.get_table()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.GetTableRequest()

@pytest.mark.asyncio
async def test_get_table_async(transport: str='grpc_asyncio', request_type=tables.GetTableRequest):
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_table), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Table(name='name_value', display_name='display_name_value'))
        response = await client.get_table(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.GetTableRequest()
    assert isinstance(response, tables.Table)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

@pytest.mark.asyncio
async def test_get_table_async_from_dict():
    await test_get_table_async(request_type=dict)

def test_get_table_field_headers():
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.GetTableRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_table), '__call__') as call:
        call.return_value = tables.Table()
        client.get_table(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_table_field_headers_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.GetTableRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_table), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Table())
        await client.get_table(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_table_flattened():
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_table), '__call__') as call:
        call.return_value = tables.Table()
        client.get_table(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_table_flattened_error():
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_table(tables.GetTableRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_table_flattened_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_table), '__call__') as call:
        call.return_value = tables.Table()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Table())
        response = await client.get_table(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_table_flattened_error_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_table(tables.GetTableRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [tables.ListTablesRequest, dict])
def test_list_tables(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tables), '__call__') as call:
        call.return_value = tables.ListTablesResponse(next_page_token='next_page_token_value')
        response = client.list_tables(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.ListTablesRequest()
    assert isinstance(response, pagers.ListTablesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tables_empty_call():
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_tables), '__call__') as call:
        client.list_tables()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.ListTablesRequest()

@pytest.mark.asyncio
async def test_list_tables_async(transport: str='grpc_asyncio', request_type=tables.ListTablesRequest):
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tables), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.ListTablesResponse(next_page_token='next_page_token_value'))
        response = await client.list_tables(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.ListTablesRequest()
    assert isinstance(response, pagers.ListTablesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_tables_async_from_dict():
    await test_list_tables_async(request_type=dict)

def test_list_tables_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tables), '__call__') as call:
        call.side_effect = (tables.ListTablesResponse(tables=[tables.Table(), tables.Table(), tables.Table()], next_page_token='abc'), tables.ListTablesResponse(tables=[], next_page_token='def'), tables.ListTablesResponse(tables=[tables.Table()], next_page_token='ghi'), tables.ListTablesResponse(tables=[tables.Table(), tables.Table()]), RuntimeError)
        metadata = ()
        pager = client.list_tables(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tables.Table) for i in results))

def test_list_tables_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tables), '__call__') as call:
        call.side_effect = (tables.ListTablesResponse(tables=[tables.Table(), tables.Table(), tables.Table()], next_page_token='abc'), tables.ListTablesResponse(tables=[], next_page_token='def'), tables.ListTablesResponse(tables=[tables.Table()], next_page_token='ghi'), tables.ListTablesResponse(tables=[tables.Table(), tables.Table()]), RuntimeError)
        pages = list(client.list_tables(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_tables_async_pager():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tables), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tables.ListTablesResponse(tables=[tables.Table(), tables.Table(), tables.Table()], next_page_token='abc'), tables.ListTablesResponse(tables=[], next_page_token='def'), tables.ListTablesResponse(tables=[tables.Table()], next_page_token='ghi'), tables.ListTablesResponse(tables=[tables.Table(), tables.Table()]), RuntimeError)
        async_pager = await client.list_tables(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, tables.Table) for i in responses))

@pytest.mark.asyncio
async def test_list_tables_async_pages():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tables), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tables.ListTablesResponse(tables=[tables.Table(), tables.Table(), tables.Table()], next_page_token='abc'), tables.ListTablesResponse(tables=[], next_page_token='def'), tables.ListTablesResponse(tables=[tables.Table()], next_page_token='ghi'), tables.ListTablesResponse(tables=[tables.Table(), tables.Table()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_tables(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tables.GetWorkspaceRequest, dict])
def test_get_workspace(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_workspace), '__call__') as call:
        call.return_value = tables.Workspace(name='name_value', display_name='display_name_value')
        response = client.get_workspace(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.GetWorkspaceRequest()
    assert isinstance(response, tables.Workspace)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_get_workspace_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_workspace), '__call__') as call:
        client.get_workspace()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.GetWorkspaceRequest()

@pytest.mark.asyncio
async def test_get_workspace_async(transport: str='grpc_asyncio', request_type=tables.GetWorkspaceRequest):
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_workspace), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Workspace(name='name_value', display_name='display_name_value'))
        response = await client.get_workspace(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.GetWorkspaceRequest()
    assert isinstance(response, tables.Workspace)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

@pytest.mark.asyncio
async def test_get_workspace_async_from_dict():
    await test_get_workspace_async(request_type=dict)

def test_get_workspace_field_headers():
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.GetWorkspaceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_workspace), '__call__') as call:
        call.return_value = tables.Workspace()
        client.get_workspace(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_workspace_field_headers_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.GetWorkspaceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_workspace), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Workspace())
        await client.get_workspace(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_workspace_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_workspace), '__call__') as call:
        call.return_value = tables.Workspace()
        client.get_workspace(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_workspace_flattened_error():
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_workspace(tables.GetWorkspaceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_workspace_flattened_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_workspace), '__call__') as call:
        call.return_value = tables.Workspace()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Workspace())
        response = await client.get_workspace(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_workspace_flattened_error_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_workspace(tables.GetWorkspaceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [tables.ListWorkspacesRequest, dict])
def test_list_workspaces(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_workspaces), '__call__') as call:
        call.return_value = tables.ListWorkspacesResponse(next_page_token='next_page_token_value')
        response = client.list_workspaces(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.ListWorkspacesRequest()
    assert isinstance(response, pagers.ListWorkspacesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_workspaces_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_workspaces), '__call__') as call:
        client.list_workspaces()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.ListWorkspacesRequest()

@pytest.mark.asyncio
async def test_list_workspaces_async(transport: str='grpc_asyncio', request_type=tables.ListWorkspacesRequest):
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_workspaces), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.ListWorkspacesResponse(next_page_token='next_page_token_value'))
        response = await client.list_workspaces(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.ListWorkspacesRequest()
    assert isinstance(response, pagers.ListWorkspacesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_workspaces_async_from_dict():
    await test_list_workspaces_async(request_type=dict)

def test_list_workspaces_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_workspaces), '__call__') as call:
        call.side_effect = (tables.ListWorkspacesResponse(workspaces=[tables.Workspace(), tables.Workspace(), tables.Workspace()], next_page_token='abc'), tables.ListWorkspacesResponse(workspaces=[], next_page_token='def'), tables.ListWorkspacesResponse(workspaces=[tables.Workspace()], next_page_token='ghi'), tables.ListWorkspacesResponse(workspaces=[tables.Workspace(), tables.Workspace()]), RuntimeError)
        metadata = ()
        pager = client.list_workspaces(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tables.Workspace) for i in results))

def test_list_workspaces_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_workspaces), '__call__') as call:
        call.side_effect = (tables.ListWorkspacesResponse(workspaces=[tables.Workspace(), tables.Workspace(), tables.Workspace()], next_page_token='abc'), tables.ListWorkspacesResponse(workspaces=[], next_page_token='def'), tables.ListWorkspacesResponse(workspaces=[tables.Workspace()], next_page_token='ghi'), tables.ListWorkspacesResponse(workspaces=[tables.Workspace(), tables.Workspace()]), RuntimeError)
        pages = list(client.list_workspaces(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_workspaces_async_pager():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_workspaces), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tables.ListWorkspacesResponse(workspaces=[tables.Workspace(), tables.Workspace(), tables.Workspace()], next_page_token='abc'), tables.ListWorkspacesResponse(workspaces=[], next_page_token='def'), tables.ListWorkspacesResponse(workspaces=[tables.Workspace()], next_page_token='ghi'), tables.ListWorkspacesResponse(workspaces=[tables.Workspace(), tables.Workspace()]), RuntimeError)
        async_pager = await client.list_workspaces(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, tables.Workspace) for i in responses))

@pytest.mark.asyncio
async def test_list_workspaces_async_pages():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_workspaces), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tables.ListWorkspacesResponse(workspaces=[tables.Workspace(), tables.Workspace(), tables.Workspace()], next_page_token='abc'), tables.ListWorkspacesResponse(workspaces=[], next_page_token='def'), tables.ListWorkspacesResponse(workspaces=[tables.Workspace()], next_page_token='ghi'), tables.ListWorkspacesResponse(workspaces=[tables.Workspace(), tables.Workspace()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_workspaces(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tables.GetRowRequest, dict])
def test_get_row(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_row), '__call__') as call:
        call.return_value = tables.Row(name='name_value')
        response = client.get_row(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.GetRowRequest()
    assert isinstance(response, tables.Row)
    assert response.name == 'name_value'

def test_get_row_empty_call():
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_row), '__call__') as call:
        client.get_row()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.GetRowRequest()

@pytest.mark.asyncio
async def test_get_row_async(transport: str='grpc_asyncio', request_type=tables.GetRowRequest):
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_row), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Row(name='name_value'))
        response = await client.get_row(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.GetRowRequest()
    assert isinstance(response, tables.Row)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_row_async_from_dict():
    await test_get_row_async(request_type=dict)

def test_get_row_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.GetRowRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_row), '__call__') as call:
        call.return_value = tables.Row()
        client.get_row(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_row_field_headers_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.GetRowRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_row), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Row())
        await client.get_row(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_row_flattened():
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_row), '__call__') as call:
        call.return_value = tables.Row()
        client.get_row(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_row_flattened_error():
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_row(tables.GetRowRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_row_flattened_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_row), '__call__') as call:
        call.return_value = tables.Row()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Row())
        response = await client.get_row(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_row_flattened_error_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_row(tables.GetRowRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [tables.ListRowsRequest, dict])
def test_list_rows(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_rows), '__call__') as call:
        call.return_value = tables.ListRowsResponse(next_page_token='next_page_token_value')
        response = client.list_rows(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.ListRowsRequest()
    assert isinstance(response, pagers.ListRowsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_rows_empty_call():
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_rows), '__call__') as call:
        client.list_rows()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.ListRowsRequest()

@pytest.mark.asyncio
async def test_list_rows_async(transport: str='grpc_asyncio', request_type=tables.ListRowsRequest):
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_rows), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.ListRowsResponse(next_page_token='next_page_token_value'))
        response = await client.list_rows(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.ListRowsRequest()
    assert isinstance(response, pagers.ListRowsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_rows_async_from_dict():
    await test_list_rows_async(request_type=dict)

def test_list_rows_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.ListRowsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_rows), '__call__') as call:
        call.return_value = tables.ListRowsResponse()
        client.list_rows(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_rows_field_headers_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.ListRowsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_rows), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.ListRowsResponse())
        await client.list_rows(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_rows_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_rows), '__call__') as call:
        call.return_value = tables.ListRowsResponse()
        client.list_rows(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_rows_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_rows(tables.ListRowsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_rows_flattened_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_rows), '__call__') as call:
        call.return_value = tables.ListRowsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.ListRowsResponse())
        response = await client.list_rows(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_rows_flattened_error_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_rows(tables.ListRowsRequest(), parent='parent_value')

def test_list_rows_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_rows), '__call__') as call:
        call.side_effect = (tables.ListRowsResponse(rows=[tables.Row(), tables.Row(), tables.Row()], next_page_token='abc'), tables.ListRowsResponse(rows=[], next_page_token='def'), tables.ListRowsResponse(rows=[tables.Row()], next_page_token='ghi'), tables.ListRowsResponse(rows=[tables.Row(), tables.Row()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_rows(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tables.Row) for i in results))

def test_list_rows_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_rows), '__call__') as call:
        call.side_effect = (tables.ListRowsResponse(rows=[tables.Row(), tables.Row(), tables.Row()], next_page_token='abc'), tables.ListRowsResponse(rows=[], next_page_token='def'), tables.ListRowsResponse(rows=[tables.Row()], next_page_token='ghi'), tables.ListRowsResponse(rows=[tables.Row(), tables.Row()]), RuntimeError)
        pages = list(client.list_rows(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_rows_async_pager():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_rows), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tables.ListRowsResponse(rows=[tables.Row(), tables.Row(), tables.Row()], next_page_token='abc'), tables.ListRowsResponse(rows=[], next_page_token='def'), tables.ListRowsResponse(rows=[tables.Row()], next_page_token='ghi'), tables.ListRowsResponse(rows=[tables.Row(), tables.Row()]), RuntimeError)
        async_pager = await client.list_rows(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, tables.Row) for i in responses))

@pytest.mark.asyncio
async def test_list_rows_async_pages():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_rows), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tables.ListRowsResponse(rows=[tables.Row(), tables.Row(), tables.Row()], next_page_token='abc'), tables.ListRowsResponse(rows=[], next_page_token='def'), tables.ListRowsResponse(rows=[tables.Row()], next_page_token='ghi'), tables.ListRowsResponse(rows=[tables.Row(), tables.Row()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_rows(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tables.CreateRowRequest, dict])
def test_create_row(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_row), '__call__') as call:
        call.return_value = tables.Row(name='name_value')
        response = client.create_row(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.CreateRowRequest()
    assert isinstance(response, tables.Row)
    assert response.name == 'name_value'

def test_create_row_empty_call():
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_row), '__call__') as call:
        client.create_row()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.CreateRowRequest()

@pytest.mark.asyncio
async def test_create_row_async(transport: str='grpc_asyncio', request_type=tables.CreateRowRequest):
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_row), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Row(name='name_value'))
        response = await client.create_row(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.CreateRowRequest()
    assert isinstance(response, tables.Row)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_create_row_async_from_dict():
    await test_create_row_async(request_type=dict)

def test_create_row_field_headers():
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.CreateRowRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_row), '__call__') as call:
        call.return_value = tables.Row()
        client.create_row(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_row_field_headers_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.CreateRowRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_row), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Row())
        await client.create_row(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_row_flattened():
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_row), '__call__') as call:
        call.return_value = tables.Row()
        client.create_row(parent='parent_value', row=tables.Row(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].row
        mock_val = tables.Row(name='name_value')
        assert arg == mock_val

def test_create_row_flattened_error():
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_row(tables.CreateRowRequest(), parent='parent_value', row=tables.Row(name='name_value'))

@pytest.mark.asyncio
async def test_create_row_flattened_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_row), '__call__') as call:
        call.return_value = tables.Row()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Row())
        response = await client.create_row(parent='parent_value', row=tables.Row(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].row
        mock_val = tables.Row(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_row_flattened_error_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_row(tables.CreateRowRequest(), parent='parent_value', row=tables.Row(name='name_value'))

@pytest.mark.parametrize('request_type', [tables.BatchCreateRowsRequest, dict])
def test_batch_create_rows(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_rows), '__call__') as call:
        call.return_value = tables.BatchCreateRowsResponse()
        response = client.batch_create_rows(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.BatchCreateRowsRequest()
    assert isinstance(response, tables.BatchCreateRowsResponse)

def test_batch_create_rows_empty_call():
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_create_rows), '__call__') as call:
        client.batch_create_rows()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.BatchCreateRowsRequest()

@pytest.mark.asyncio
async def test_batch_create_rows_async(transport: str='grpc_asyncio', request_type=tables.BatchCreateRowsRequest):
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_rows), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.BatchCreateRowsResponse())
        response = await client.batch_create_rows(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.BatchCreateRowsRequest()
    assert isinstance(response, tables.BatchCreateRowsResponse)

@pytest.mark.asyncio
async def test_batch_create_rows_async_from_dict():
    await test_batch_create_rows_async(request_type=dict)

def test_batch_create_rows_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.BatchCreateRowsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_create_rows), '__call__') as call:
        call.return_value = tables.BatchCreateRowsResponse()
        client.batch_create_rows(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_create_rows_field_headers_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.BatchCreateRowsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_create_rows), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.BatchCreateRowsResponse())
        await client.batch_create_rows(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [tables.UpdateRowRequest, dict])
def test_update_row(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_row), '__call__') as call:
        call.return_value = tables.Row(name='name_value')
        response = client.update_row(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.UpdateRowRequest()
    assert isinstance(response, tables.Row)
    assert response.name == 'name_value'

def test_update_row_empty_call():
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_row), '__call__') as call:
        client.update_row()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.UpdateRowRequest()

@pytest.mark.asyncio
async def test_update_row_async(transport: str='grpc_asyncio', request_type=tables.UpdateRowRequest):
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_row), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Row(name='name_value'))
        response = await client.update_row(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.UpdateRowRequest()
    assert isinstance(response, tables.Row)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_update_row_async_from_dict():
    await test_update_row_async(request_type=dict)

def test_update_row_field_headers():
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.UpdateRowRequest()
    request.row.name = 'name_value'
    with mock.patch.object(type(client.transport.update_row), '__call__') as call:
        call.return_value = tables.Row()
        client.update_row(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'row.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_row_field_headers_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.UpdateRowRequest()
    request.row.name = 'name_value'
    with mock.patch.object(type(client.transport.update_row), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Row())
        await client.update_row(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'row.name=name_value') in kw['metadata']

def test_update_row_flattened():
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_row), '__call__') as call:
        call.return_value = tables.Row()
        client.update_row(row=tables.Row(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].row
        mock_val = tables.Row(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_row_flattened_error():
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_row(tables.UpdateRowRequest(), row=tables.Row(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_row_flattened_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_row), '__call__') as call:
        call.return_value = tables.Row()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.Row())
        response = await client.update_row(row=tables.Row(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].row
        mock_val = tables.Row(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_row_flattened_error_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_row(tables.UpdateRowRequest(), row=tables.Row(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [tables.BatchUpdateRowsRequest, dict])
def test_batch_update_rows(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_update_rows), '__call__') as call:
        call.return_value = tables.BatchUpdateRowsResponse()
        response = client.batch_update_rows(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.BatchUpdateRowsRequest()
    assert isinstance(response, tables.BatchUpdateRowsResponse)

def test_batch_update_rows_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_update_rows), '__call__') as call:
        client.batch_update_rows()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.BatchUpdateRowsRequest()

@pytest.mark.asyncio
async def test_batch_update_rows_async(transport: str='grpc_asyncio', request_type=tables.BatchUpdateRowsRequest):
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_update_rows), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.BatchUpdateRowsResponse())
        response = await client.batch_update_rows(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.BatchUpdateRowsRequest()
    assert isinstance(response, tables.BatchUpdateRowsResponse)

@pytest.mark.asyncio
async def test_batch_update_rows_async_from_dict():
    await test_batch_update_rows_async(request_type=dict)

def test_batch_update_rows_field_headers():
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.BatchUpdateRowsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_update_rows), '__call__') as call:
        call.return_value = tables.BatchUpdateRowsResponse()
        client.batch_update_rows(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_update_rows_field_headers_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.BatchUpdateRowsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_update_rows), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tables.BatchUpdateRowsResponse())
        await client.batch_update_rows(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [tables.DeleteRowRequest, dict])
def test_delete_row(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_row), '__call__') as call:
        call.return_value = None
        response = client.delete_row(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.DeleteRowRequest()
    assert response is None

def test_delete_row_empty_call():
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_row), '__call__') as call:
        client.delete_row()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.DeleteRowRequest()

@pytest.mark.asyncio
async def test_delete_row_async(transport: str='grpc_asyncio', request_type=tables.DeleteRowRequest):
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_row), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_row(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.DeleteRowRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_row_async_from_dict():
    await test_delete_row_async(request_type=dict)

def test_delete_row_field_headers():
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.DeleteRowRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_row), '__call__') as call:
        call.return_value = None
        client.delete_row(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_row_field_headers_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.DeleteRowRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_row), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_row(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_row_flattened():
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_row), '__call__') as call:
        call.return_value = None
        client.delete_row(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_row_flattened_error():
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_row(tables.DeleteRowRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_row_flattened_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_row), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_row(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_row_flattened_error_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_row(tables.DeleteRowRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [tables.BatchDeleteRowsRequest, dict])
def test_batch_delete_rows(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_delete_rows), '__call__') as call:
        call.return_value = None
        response = client.batch_delete_rows(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.BatchDeleteRowsRequest()
    assert response is None

def test_batch_delete_rows_empty_call():
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_delete_rows), '__call__') as call:
        client.batch_delete_rows()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.BatchDeleteRowsRequest()

@pytest.mark.asyncio
async def test_batch_delete_rows_async(transport: str='grpc_asyncio', request_type=tables.BatchDeleteRowsRequest):
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_delete_rows), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.batch_delete_rows(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tables.BatchDeleteRowsRequest()
    assert response is None

@pytest.mark.asyncio
async def test_batch_delete_rows_async_from_dict():
    await test_batch_delete_rows_async(request_type=dict)

def test_batch_delete_rows_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.BatchDeleteRowsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_delete_rows), '__call__') as call:
        call.return_value = None
        client.batch_delete_rows(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_delete_rows_field_headers_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tables.BatchDeleteRowsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_delete_rows), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.batch_delete_rows(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [tables.GetTableRequest, dict])
def test_get_table_rest(request_type):
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'tables/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.Table(name='name_value', display_name='display_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.Table.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_table(request)
    assert isinstance(response, tables.Table)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_get_table_rest_required_fields(request_type=tables.GetTableRequest):
    if False:
        return 10
    transport_class = transports.TablesServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_table._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_table._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tables.Table()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tables.Table.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_table(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_table_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_table._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_table_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TablesServiceRestInterceptor())
    client = TablesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TablesServiceRestInterceptor, 'post_get_table') as post, mock.patch.object(transports.TablesServiceRestInterceptor, 'pre_get_table') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tables.GetTableRequest.pb(tables.GetTableRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tables.Table.to_json(tables.Table())
        request = tables.GetTableRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tables.Table()
        client.get_table(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_table_rest_bad_request(transport: str='rest', request_type=tables.GetTableRequest):
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'tables/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_table(request)

def test_get_table_rest_flattened():
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.Table()
        sample_request = {'name': 'tables/sample1'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.Table.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_table(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{name=tables/*}' % client.transport._host, args[1])

def test_get_table_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_table(tables.GetTableRequest(), name='name_value')

def test_get_table_rest_error():
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tables.ListTablesRequest, dict])
def test_list_tables_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.ListTablesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.ListTablesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_tables(request)
    assert isinstance(response, pagers.ListTablesPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_tables_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TablesServiceRestInterceptor())
    client = TablesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TablesServiceRestInterceptor, 'post_list_tables') as post, mock.patch.object(transports.TablesServiceRestInterceptor, 'pre_list_tables') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tables.ListTablesRequest.pb(tables.ListTablesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tables.ListTablesResponse.to_json(tables.ListTablesResponse())
        request = tables.ListTablesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tables.ListTablesResponse()
        client.list_tables(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_tables_rest_bad_request(transport: str='rest', request_type=tables.ListTablesRequest):
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_tables(request)

def test_list_tables_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (tables.ListTablesResponse(tables=[tables.Table(), tables.Table(), tables.Table()], next_page_token='abc'), tables.ListTablesResponse(tables=[], next_page_token='def'), tables.ListTablesResponse(tables=[tables.Table()], next_page_token='ghi'), tables.ListTablesResponse(tables=[tables.Table(), tables.Table()]))
        response = response + response
        response = tuple((tables.ListTablesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {}
        pager = client.list_tables(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tables.Table) for i in results))
        pages = list(client.list_tables(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tables.GetWorkspaceRequest, dict])
def test_get_workspace_rest(request_type):
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'workspaces/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.Workspace(name='name_value', display_name='display_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.Workspace.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_workspace(request)
    assert isinstance(response, tables.Workspace)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_get_workspace_rest_required_fields(request_type=tables.GetWorkspaceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.TablesServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_workspace._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_workspace._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tables.Workspace()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tables.Workspace.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_workspace(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_workspace_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_workspace._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_workspace_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TablesServiceRestInterceptor())
    client = TablesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TablesServiceRestInterceptor, 'post_get_workspace') as post, mock.patch.object(transports.TablesServiceRestInterceptor, 'pre_get_workspace') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tables.GetWorkspaceRequest.pb(tables.GetWorkspaceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tables.Workspace.to_json(tables.Workspace())
        request = tables.GetWorkspaceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tables.Workspace()
        client.get_workspace(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_workspace_rest_bad_request(transport: str='rest', request_type=tables.GetWorkspaceRequest):
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'workspaces/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_workspace(request)

def test_get_workspace_rest_flattened():
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.Workspace()
        sample_request = {'name': 'workspaces/sample1'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.Workspace.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_workspace(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{name=workspaces/*}' % client.transport._host, args[1])

def test_get_workspace_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_workspace(tables.GetWorkspaceRequest(), name='name_value')

def test_get_workspace_rest_error():
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tables.ListWorkspacesRequest, dict])
def test_list_workspaces_rest(request_type):
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.ListWorkspacesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.ListWorkspacesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_workspaces(request)
    assert isinstance(response, pagers.ListWorkspacesPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_workspaces_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TablesServiceRestInterceptor())
    client = TablesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TablesServiceRestInterceptor, 'post_list_workspaces') as post, mock.patch.object(transports.TablesServiceRestInterceptor, 'pre_list_workspaces') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tables.ListWorkspacesRequest.pb(tables.ListWorkspacesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tables.ListWorkspacesResponse.to_json(tables.ListWorkspacesResponse())
        request = tables.ListWorkspacesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tables.ListWorkspacesResponse()
        client.list_workspaces(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_workspaces_rest_bad_request(transport: str='rest', request_type=tables.ListWorkspacesRequest):
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_workspaces(request)

def test_list_workspaces_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (tables.ListWorkspacesResponse(workspaces=[tables.Workspace(), tables.Workspace(), tables.Workspace()], next_page_token='abc'), tables.ListWorkspacesResponse(workspaces=[], next_page_token='def'), tables.ListWorkspacesResponse(workspaces=[tables.Workspace()], next_page_token='ghi'), tables.ListWorkspacesResponse(workspaces=[tables.Workspace(), tables.Workspace()]))
        response = response + response
        response = tuple((tables.ListWorkspacesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {}
        pager = client.list_workspaces(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tables.Workspace) for i in results))
        pages = list(client.list_workspaces(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tables.GetRowRequest, dict])
def test_get_row_rest(request_type):
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'tables/sample1/rows/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.Row(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.Row.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_row(request)
    assert isinstance(response, tables.Row)
    assert response.name == 'name_value'

def test_get_row_rest_required_fields(request_type=tables.GetRowRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.TablesServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_row._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_row._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('view',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tables.Row()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tables.Row.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_row(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_row_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_row._get_unset_required_fields({})
    assert set(unset_fields) == set(('view',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_row_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TablesServiceRestInterceptor())
    client = TablesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TablesServiceRestInterceptor, 'post_get_row') as post, mock.patch.object(transports.TablesServiceRestInterceptor, 'pre_get_row') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tables.GetRowRequest.pb(tables.GetRowRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tables.Row.to_json(tables.Row())
        request = tables.GetRowRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tables.Row()
        client.get_row(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_row_rest_bad_request(transport: str='rest', request_type=tables.GetRowRequest):
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'tables/sample1/rows/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_row(request)

def test_get_row_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.Row()
        sample_request = {'name': 'tables/sample1/rows/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.Row.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_row(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{name=tables/*/rows/*}' % client.transport._host, args[1])

def test_get_row_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_row(tables.GetRowRequest(), name='name_value')

def test_get_row_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tables.ListRowsRequest, dict])
def test_list_rows_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'tables/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.ListRowsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.ListRowsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_rows(request)
    assert isinstance(response, pagers.ListRowsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_rows_rest_required_fields(request_type=tables.ListRowsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.TablesServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_rows._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_rows._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token', 'view'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tables.ListRowsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tables.ListRowsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_rows(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_rows_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_rows._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken', 'view')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_rows_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TablesServiceRestInterceptor())
    client = TablesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TablesServiceRestInterceptor, 'post_list_rows') as post, mock.patch.object(transports.TablesServiceRestInterceptor, 'pre_list_rows') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tables.ListRowsRequest.pb(tables.ListRowsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tables.ListRowsResponse.to_json(tables.ListRowsResponse())
        request = tables.ListRowsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tables.ListRowsResponse()
        client.list_rows(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_rows_rest_bad_request(transport: str='rest', request_type=tables.ListRowsRequest):
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'tables/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_rows(request)

def test_list_rows_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.ListRowsResponse()
        sample_request = {'parent': 'tables/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.ListRowsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_rows(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{parent=tables/*}/rows' % client.transport._host, args[1])

def test_list_rows_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_rows(tables.ListRowsRequest(), parent='parent_value')

def test_list_rows_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (tables.ListRowsResponse(rows=[tables.Row(), tables.Row(), tables.Row()], next_page_token='abc'), tables.ListRowsResponse(rows=[], next_page_token='def'), tables.ListRowsResponse(rows=[tables.Row()], next_page_token='ghi'), tables.ListRowsResponse(rows=[tables.Row(), tables.Row()]))
        response = response + response
        response = tuple((tables.ListRowsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'tables/sample1'}
        pager = client.list_rows(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tables.Row) for i in results))
        pages = list(client.list_rows(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tables.CreateRowRequest, dict])
def test_create_row_rest(request_type):
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'tables/sample1'}
    request_init['row'] = {'name': 'name_value', 'values': {}}
    test_field = tables.CreateRowRequest.meta.fields['row']

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
    for (field, value) in request_init['row'].items():
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
                for i in range(0, len(request_init['row'][field])):
                    del request_init['row'][field][i][subfield]
            else:
                del request_init['row'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.Row(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.Row.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_row(request)
    assert isinstance(response, tables.Row)
    assert response.name == 'name_value'

def test_create_row_rest_required_fields(request_type=tables.CreateRowRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.TablesServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_row._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_row._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('view',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tables.Row()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tables.Row.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_row(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_row_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_row._get_unset_required_fields({})
    assert set(unset_fields) == set(('view',)) & set(('parent', 'row'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_row_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TablesServiceRestInterceptor())
    client = TablesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TablesServiceRestInterceptor, 'post_create_row') as post, mock.patch.object(transports.TablesServiceRestInterceptor, 'pre_create_row') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tables.CreateRowRequest.pb(tables.CreateRowRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tables.Row.to_json(tables.Row())
        request = tables.CreateRowRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tables.Row()
        client.create_row(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_row_rest_bad_request(transport: str='rest', request_type=tables.CreateRowRequest):
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'tables/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_row(request)

def test_create_row_rest_flattened():
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.Row()
        sample_request = {'parent': 'tables/sample1'}
        mock_args = dict(parent='parent_value', row=tables.Row(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.Row.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_row(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{parent=tables/*}/rows' % client.transport._host, args[1])

def test_create_row_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_row(tables.CreateRowRequest(), parent='parent_value', row=tables.Row(name='name_value'))

def test_create_row_rest_error():
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tables.BatchCreateRowsRequest, dict])
def test_batch_create_rows_rest(request_type):
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'tables/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.BatchCreateRowsResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.BatchCreateRowsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_create_rows(request)
    assert isinstance(response, tables.BatchCreateRowsResponse)

def test_batch_create_rows_rest_required_fields(request_type=tables.BatchCreateRowsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.TablesServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_rows._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_rows._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tables.BatchCreateRowsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tables.BatchCreateRowsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.batch_create_rows(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_create_rows_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_create_rows._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'requests'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_create_rows_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TablesServiceRestInterceptor())
    client = TablesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TablesServiceRestInterceptor, 'post_batch_create_rows') as post, mock.patch.object(transports.TablesServiceRestInterceptor, 'pre_batch_create_rows') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tables.BatchCreateRowsRequest.pb(tables.BatchCreateRowsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tables.BatchCreateRowsResponse.to_json(tables.BatchCreateRowsResponse())
        request = tables.BatchCreateRowsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tables.BatchCreateRowsResponse()
        client.batch_create_rows(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_create_rows_rest_bad_request(transport: str='rest', request_type=tables.BatchCreateRowsRequest):
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'tables/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_create_rows(request)

def test_batch_create_rows_rest_error():
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tables.UpdateRowRequest, dict])
def test_update_row_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'row': {'name': 'tables/sample1/rows/sample2'}}
    request_init['row'] = {'name': 'tables/sample1/rows/sample2', 'values': {}}
    test_field = tables.UpdateRowRequest.meta.fields['row']

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
    for (field, value) in request_init['row'].items():
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
                for i in range(0, len(request_init['row'][field])):
                    del request_init['row'][field][i][subfield]
            else:
                del request_init['row'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.Row(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.Row.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_row(request)
    assert isinstance(response, tables.Row)
    assert response.name == 'name_value'

def test_update_row_rest_required_fields(request_type=tables.UpdateRowRequest):
    if False:
        return 10
    transport_class = transports.TablesServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_row._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_row._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask', 'view'))
    jsonified_request.update(unset_fields)
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tables.Row()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tables.Row.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_row(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_row_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_row._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask', 'view')) & set(('row',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_row_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TablesServiceRestInterceptor())
    client = TablesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TablesServiceRestInterceptor, 'post_update_row') as post, mock.patch.object(transports.TablesServiceRestInterceptor, 'pre_update_row') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tables.UpdateRowRequest.pb(tables.UpdateRowRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tables.Row.to_json(tables.Row())
        request = tables.UpdateRowRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tables.Row()
        client.update_row(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_row_rest_bad_request(transport: str='rest', request_type=tables.UpdateRowRequest):
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'row': {'name': 'tables/sample1/rows/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_row(request)

def test_update_row_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.Row()
        sample_request = {'row': {'name': 'tables/sample1/rows/sample2'}}
        mock_args = dict(row=tables.Row(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.Row.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_row(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{row.name=tables/*/rows/*}' % client.transport._host, args[1])

def test_update_row_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_row(tables.UpdateRowRequest(), row=tables.Row(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_row_rest_error():
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tables.BatchUpdateRowsRequest, dict])
def test_batch_update_rows_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'tables/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tables.BatchUpdateRowsResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = tables.BatchUpdateRowsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_update_rows(request)
    assert isinstance(response, tables.BatchUpdateRowsResponse)

def test_batch_update_rows_rest_required_fields(request_type=tables.BatchUpdateRowsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.TablesServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_update_rows._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_update_rows._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tables.BatchUpdateRowsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tables.BatchUpdateRowsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.batch_update_rows(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_update_rows_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_update_rows._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'requests'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_update_rows_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TablesServiceRestInterceptor())
    client = TablesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TablesServiceRestInterceptor, 'post_batch_update_rows') as post, mock.patch.object(transports.TablesServiceRestInterceptor, 'pre_batch_update_rows') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tables.BatchUpdateRowsRequest.pb(tables.BatchUpdateRowsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tables.BatchUpdateRowsResponse.to_json(tables.BatchUpdateRowsResponse())
        request = tables.BatchUpdateRowsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tables.BatchUpdateRowsResponse()
        client.batch_update_rows(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_update_rows_rest_bad_request(transport: str='rest', request_type=tables.BatchUpdateRowsRequest):
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'tables/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_update_rows(request)

def test_batch_update_rows_rest_error():
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tables.DeleteRowRequest, dict])
def test_delete_row_rest(request_type):
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'tables/sample1/rows/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_row(request)
    assert response is None

def test_delete_row_rest_required_fields(request_type=tables.DeleteRowRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.TablesServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_row._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_row._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_row(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_row_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_row._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_row_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TablesServiceRestInterceptor())
    client = TablesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TablesServiceRestInterceptor, 'pre_delete_row') as pre:
        pre.assert_not_called()
        pb_message = tables.DeleteRowRequest.pb(tables.DeleteRowRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = tables.DeleteRowRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_row(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_row_rest_bad_request(transport: str='rest', request_type=tables.DeleteRowRequest):
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'tables/sample1/rows/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_row(request)

def test_delete_row_rest_flattened():
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'tables/sample1/rows/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_row(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha1/{name=tables/*/rows/*}' % client.transport._host, args[1])

def test_delete_row_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_row(tables.DeleteRowRequest(), name='name_value')

def test_delete_row_rest_error():
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tables.BatchDeleteRowsRequest, dict])
def test_batch_delete_rows_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'tables/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_delete_rows(request)
    assert response is None

def test_batch_delete_rows_rest_required_fields(request_type=tables.BatchDeleteRowsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.TablesServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['names'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_delete_rows._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['names'] = 'names_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_delete_rows._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'names' in jsonified_request
    assert jsonified_request['names'] == 'names_value'
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = None
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = ''
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.batch_delete_rows(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_delete_rows_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_delete_rows._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'names'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_delete_rows_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TablesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TablesServiceRestInterceptor())
    client = TablesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TablesServiceRestInterceptor, 'pre_batch_delete_rows') as pre:
        pre.assert_not_called()
        pb_message = tables.BatchDeleteRowsRequest.pb(tables.BatchDeleteRowsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = tables.BatchDeleteRowsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.batch_delete_rows(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_batch_delete_rows_rest_bad_request(transport: str='rest', request_type=tables.BatchDeleteRowsRequest):
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'tables/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_delete_rows(request)

def test_batch_delete_rows_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TablesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.TablesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TablesServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.TablesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TablesServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TablesServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.TablesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TablesServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.TablesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = TablesServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TablesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.TablesServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.TablesServiceGrpcTransport, transports.TablesServiceGrpcAsyncIOTransport, transports.TablesServiceRestTransport])
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
        while True:
            i = 10
    transport = TablesServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.TablesServiceGrpcTransport)

def test_tables_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.TablesServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_tables_service_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.area120.tables_v1alpha1.services.tables_service.transports.TablesServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.TablesServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_table', 'list_tables', 'get_workspace', 'list_workspaces', 'get_row', 'list_rows', 'create_row', 'batch_create_rows', 'update_row', 'batch_update_rows', 'delete_row', 'batch_delete_rows')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_tables_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.area120.tables_v1alpha1.services.tables_service.transports.TablesServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TablesServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/spreadsheets.readonly', 'https://www.googleapis.com/auth/tables'), quota_project_id='octopus')

def test_tables_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.area120.tables_v1alpha1.services.tables_service.transports.TablesServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TablesServiceTransport()
        adc.assert_called_once()

def test_tables_service_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        TablesServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/spreadsheets.readonly', 'https://www.googleapis.com/auth/tables'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.TablesServiceGrpcTransport, transports.TablesServiceGrpcAsyncIOTransport])
def test_tables_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/spreadsheets.readonly', 'https://www.googleapis.com/auth/tables'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.TablesServiceGrpcTransport, transports.TablesServiceGrpcAsyncIOTransport, transports.TablesServiceRestTransport])
def test_tables_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.TablesServiceGrpcTransport, grpc_helpers), (transports.TablesServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_tables_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('area120tables.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/spreadsheets.readonly', 'https://www.googleapis.com/auth/tables'), scopes=['1', '2'], default_host='area120tables.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.TablesServiceGrpcTransport, transports.TablesServiceGrpcAsyncIOTransport])
def test_tables_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_tables_service_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.TablesServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_tables_service_host_no_port(transport_name):
    if False:
        return 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='area120tables.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('area120tables.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://area120tables.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_tables_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='area120tables.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('area120tables.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://area120tables.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_tables_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = TablesServiceClient(credentials=creds1, transport=transport_name)
    client2 = TablesServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.get_table._session
    session2 = client2.transport.get_table._session
    assert session1 != session2
    session1 = client1.transport.list_tables._session
    session2 = client2.transport.list_tables._session
    assert session1 != session2
    session1 = client1.transport.get_workspace._session
    session2 = client2.transport.get_workspace._session
    assert session1 != session2
    session1 = client1.transport.list_workspaces._session
    session2 = client2.transport.list_workspaces._session
    assert session1 != session2
    session1 = client1.transport.get_row._session
    session2 = client2.transport.get_row._session
    assert session1 != session2
    session1 = client1.transport.list_rows._session
    session2 = client2.transport.list_rows._session
    assert session1 != session2
    session1 = client1.transport.create_row._session
    session2 = client2.transport.create_row._session
    assert session1 != session2
    session1 = client1.transport.batch_create_rows._session
    session2 = client2.transport.batch_create_rows._session
    assert session1 != session2
    session1 = client1.transport.update_row._session
    session2 = client2.transport.update_row._session
    assert session1 != session2
    session1 = client1.transport.batch_update_rows._session
    session2 = client2.transport.batch_update_rows._session
    assert session1 != session2
    session1 = client1.transport.delete_row._session
    session2 = client2.transport.delete_row._session
    assert session1 != session2
    session1 = client1.transport.batch_delete_rows._session
    session2 = client2.transport.batch_delete_rows._session
    assert session1 != session2

def test_tables_service_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TablesServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_tables_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TablesServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.TablesServiceGrpcTransport, transports.TablesServiceGrpcAsyncIOTransport])
def test_tables_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.TablesServiceGrpcTransport, transports.TablesServiceGrpcAsyncIOTransport])
def test_tables_service_transport_channel_mtls_with_adc(transport_class):
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

def test_row_path():
    if False:
        return 10
    table = 'squid'
    row = 'clam'
    expected = 'tables/{table}/rows/{row}'.format(table=table, row=row)
    actual = TablesServiceClient.row_path(table, row)
    assert expected == actual

def test_parse_row_path():
    if False:
        return 10
    expected = {'table': 'whelk', 'row': 'octopus'}
    path = TablesServiceClient.row_path(**expected)
    actual = TablesServiceClient.parse_row_path(path)
    assert expected == actual

def test_table_path():
    if False:
        while True:
            i = 10
    table = 'oyster'
    expected = 'tables/{table}'.format(table=table)
    actual = TablesServiceClient.table_path(table)
    assert expected == actual

def test_parse_table_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'table': 'nudibranch'}
    path = TablesServiceClient.table_path(**expected)
    actual = TablesServiceClient.parse_table_path(path)
    assert expected == actual

def test_workspace_path():
    if False:
        print('Hello World!')
    workspace = 'cuttlefish'
    expected = 'workspaces/{workspace}'.format(workspace=workspace)
    actual = TablesServiceClient.workspace_path(workspace)
    assert expected == actual

def test_parse_workspace_path():
    if False:
        i = 10
        return i + 15
    expected = {'workspace': 'mussel'}
    path = TablesServiceClient.workspace_path(**expected)
    actual = TablesServiceClient.parse_workspace_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'winkle'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = TablesServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'nautilus'}
    path = TablesServiceClient.common_billing_account_path(**expected)
    actual = TablesServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'scallop'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = TablesServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'abalone'}
    path = TablesServiceClient.common_folder_path(**expected)
    actual = TablesServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'squid'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = TablesServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'clam'}
    path = TablesServiceClient.common_organization_path(**expected)
    actual = TablesServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    expected = 'projects/{project}'.format(project=project)
    actual = TablesServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus'}
    path = TablesServiceClient.common_project_path(**expected)
    actual = TablesServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = TablesServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = TablesServiceClient.common_location_path(**expected)
    actual = TablesServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.TablesServiceTransport, '_prep_wrapped_messages') as prep:
        client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.TablesServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = TablesServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = TablesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = TablesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(TablesServiceClient, transports.TablesServiceGrpcTransport), (TablesServiceAsyncClient, transports.TablesServiceGrpcAsyncIOTransport)])
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