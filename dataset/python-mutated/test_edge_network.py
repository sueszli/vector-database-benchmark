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
from google.cloud.edgenetwork_v1.services.edge_network import EdgeNetworkAsyncClient, EdgeNetworkClient, pagers, transports
from google.cloud.edgenetwork_v1.types import resources, service

def client_cert_source_callback():
    if False:
        print('Hello World!')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        while True:
            i = 10
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert EdgeNetworkClient._get_default_mtls_endpoint(None) is None
    assert EdgeNetworkClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert EdgeNetworkClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert EdgeNetworkClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert EdgeNetworkClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert EdgeNetworkClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(EdgeNetworkClient, 'grpc'), (EdgeNetworkAsyncClient, 'grpc_asyncio'), (EdgeNetworkClient, 'rest')])
def test_edge_network_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('edgenetwork.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://edgenetwork.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.EdgeNetworkGrpcTransport, 'grpc'), (transports.EdgeNetworkGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.EdgeNetworkRestTransport, 'rest')])
def test_edge_network_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(EdgeNetworkClient, 'grpc'), (EdgeNetworkAsyncClient, 'grpc_asyncio'), (EdgeNetworkClient, 'rest')])
def test_edge_network_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('edgenetwork.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://edgenetwork.googleapis.com')

def test_edge_network_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = EdgeNetworkClient.get_transport_class()
    available_transports = [transports.EdgeNetworkGrpcTransport, transports.EdgeNetworkRestTransport]
    assert transport in available_transports
    transport = EdgeNetworkClient.get_transport_class('grpc')
    assert transport == transports.EdgeNetworkGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(EdgeNetworkClient, transports.EdgeNetworkGrpcTransport, 'grpc'), (EdgeNetworkAsyncClient, transports.EdgeNetworkGrpcAsyncIOTransport, 'grpc_asyncio'), (EdgeNetworkClient, transports.EdgeNetworkRestTransport, 'rest')])
@mock.patch.object(EdgeNetworkClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EdgeNetworkClient))
@mock.patch.object(EdgeNetworkAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EdgeNetworkAsyncClient))
def test_edge_network_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(EdgeNetworkClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(EdgeNetworkClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(EdgeNetworkClient, transports.EdgeNetworkGrpcTransport, 'grpc', 'true'), (EdgeNetworkAsyncClient, transports.EdgeNetworkGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (EdgeNetworkClient, transports.EdgeNetworkGrpcTransport, 'grpc', 'false'), (EdgeNetworkAsyncClient, transports.EdgeNetworkGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (EdgeNetworkClient, transports.EdgeNetworkRestTransport, 'rest', 'true'), (EdgeNetworkClient, transports.EdgeNetworkRestTransport, 'rest', 'false')])
@mock.patch.object(EdgeNetworkClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EdgeNetworkClient))
@mock.patch.object(EdgeNetworkAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EdgeNetworkAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_edge_network_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [EdgeNetworkClient, EdgeNetworkAsyncClient])
@mock.patch.object(EdgeNetworkClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EdgeNetworkClient))
@mock.patch.object(EdgeNetworkAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EdgeNetworkAsyncClient))
def test_edge_network_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        for i in range(10):
            print('nop')
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(EdgeNetworkClient, transports.EdgeNetworkGrpcTransport, 'grpc'), (EdgeNetworkAsyncClient, transports.EdgeNetworkGrpcAsyncIOTransport, 'grpc_asyncio'), (EdgeNetworkClient, transports.EdgeNetworkRestTransport, 'rest')])
def test_edge_network_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(EdgeNetworkClient, transports.EdgeNetworkGrpcTransport, 'grpc', grpc_helpers), (EdgeNetworkAsyncClient, transports.EdgeNetworkGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (EdgeNetworkClient, transports.EdgeNetworkRestTransport, 'rest', None)])
def test_edge_network_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_edge_network_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.edgenetwork_v1.services.edge_network.transports.EdgeNetworkGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = EdgeNetworkClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(EdgeNetworkClient, transports.EdgeNetworkGrpcTransport, 'grpc', grpc_helpers), (EdgeNetworkAsyncClient, transports.EdgeNetworkGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_edge_network_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
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
        create_channel.assert_called_with('edgenetwork.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='edgenetwork.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [service.InitializeZoneRequest, dict])
def test_initialize_zone(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.initialize_zone), '__call__') as call:
        call.return_value = service.InitializeZoneResponse()
        response = client.initialize_zone(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.InitializeZoneRequest()
    assert isinstance(response, service.InitializeZoneResponse)

def test_initialize_zone_empty_call():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.initialize_zone), '__call__') as call:
        client.initialize_zone()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.InitializeZoneRequest()

@pytest.mark.asyncio
async def test_initialize_zone_async(transport: str='grpc_asyncio', request_type=service.InitializeZoneRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.initialize_zone), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.InitializeZoneResponse())
        response = await client.initialize_zone(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.InitializeZoneRequest()
    assert isinstance(response, service.InitializeZoneResponse)

@pytest.mark.asyncio
async def test_initialize_zone_async_from_dict():
    await test_initialize_zone_async(request_type=dict)

def test_initialize_zone_field_headers():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.InitializeZoneRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.initialize_zone), '__call__') as call:
        call.return_value = service.InitializeZoneResponse()
        client.initialize_zone(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_initialize_zone_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.InitializeZoneRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.initialize_zone), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.InitializeZoneResponse())
        await client.initialize_zone(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_initialize_zone_flattened():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.initialize_zone), '__call__') as call:
        call.return_value = service.InitializeZoneResponse()
        client.initialize_zone(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_initialize_zone_flattened_error():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.initialize_zone(service.InitializeZoneRequest(), name='name_value')

@pytest.mark.asyncio
async def test_initialize_zone_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.initialize_zone), '__call__') as call:
        call.return_value = service.InitializeZoneResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.InitializeZoneResponse())
        response = await client.initialize_zone(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_initialize_zone_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.initialize_zone(service.InitializeZoneRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListZonesRequest, dict])
def test_list_zones(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_zones), '__call__') as call:
        call.return_value = service.ListZonesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_zones(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListZonesRequest()
    assert isinstance(response, pagers.ListZonesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_zones_empty_call():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_zones), '__call__') as call:
        client.list_zones()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListZonesRequest()

@pytest.mark.asyncio
async def test_list_zones_async(transport: str='grpc_asyncio', request_type=service.ListZonesRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_zones), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListZonesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_zones(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListZonesRequest()
    assert isinstance(response, pagers.ListZonesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_zones_async_from_dict():
    await test_list_zones_async(request_type=dict)

def test_list_zones_field_headers():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListZonesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_zones), '__call__') as call:
        call.return_value = service.ListZonesResponse()
        client.list_zones(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_zones_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListZonesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_zones), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListZonesResponse())
        await client.list_zones(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_zones_flattened():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_zones), '__call__') as call:
        call.return_value = service.ListZonesResponse()
        client.list_zones(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_zones_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_zones(service.ListZonesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_zones_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_zones), '__call__') as call:
        call.return_value = service.ListZonesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListZonesResponse())
        response = await client.list_zones(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_zones_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_zones(service.ListZonesRequest(), parent='parent_value')

def test_list_zones_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_zones), '__call__') as call:
        call.side_effect = (service.ListZonesResponse(zones=[resources.Zone(), resources.Zone(), resources.Zone()], next_page_token='abc'), service.ListZonesResponse(zones=[], next_page_token='def'), service.ListZonesResponse(zones=[resources.Zone()], next_page_token='ghi'), service.ListZonesResponse(zones=[resources.Zone(), resources.Zone()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_zones(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Zone) for i in results))

def test_list_zones_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_zones), '__call__') as call:
        call.side_effect = (service.ListZonesResponse(zones=[resources.Zone(), resources.Zone(), resources.Zone()], next_page_token='abc'), service.ListZonesResponse(zones=[], next_page_token='def'), service.ListZonesResponse(zones=[resources.Zone()], next_page_token='ghi'), service.ListZonesResponse(zones=[resources.Zone(), resources.Zone()]), RuntimeError)
        pages = list(client.list_zones(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_zones_async_pager():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_zones), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListZonesResponse(zones=[resources.Zone(), resources.Zone(), resources.Zone()], next_page_token='abc'), service.ListZonesResponse(zones=[], next_page_token='def'), service.ListZonesResponse(zones=[resources.Zone()], next_page_token='ghi'), service.ListZonesResponse(zones=[resources.Zone(), resources.Zone()]), RuntimeError)
        async_pager = await client.list_zones(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Zone) for i in responses))

@pytest.mark.asyncio
async def test_list_zones_async_pages():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_zones), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListZonesResponse(zones=[resources.Zone(), resources.Zone(), resources.Zone()], next_page_token='abc'), service.ListZonesResponse(zones=[], next_page_token='def'), service.ListZonesResponse(zones=[resources.Zone()], next_page_token='ghi'), service.ListZonesResponse(zones=[resources.Zone(), resources.Zone()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_zones(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetZoneRequest, dict])
def test_get_zone(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_zone), '__call__') as call:
        call.return_value = resources.Zone(name='name_value', layout_name='layout_name_value')
        response = client.get_zone(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetZoneRequest()
    assert isinstance(response, resources.Zone)
    assert response.name == 'name_value'
    assert response.layout_name == 'layout_name_value'

def test_get_zone_empty_call():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_zone), '__call__') as call:
        client.get_zone()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetZoneRequest()

@pytest.mark.asyncio
async def test_get_zone_async(transport: str='grpc_asyncio', request_type=service.GetZoneRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_zone), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Zone(name='name_value', layout_name='layout_name_value'))
        response = await client.get_zone(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetZoneRequest()
    assert isinstance(response, resources.Zone)
    assert response.name == 'name_value'
    assert response.layout_name == 'layout_name_value'

@pytest.mark.asyncio
async def test_get_zone_async_from_dict():
    await test_get_zone_async(request_type=dict)

def test_get_zone_field_headers():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetZoneRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_zone), '__call__') as call:
        call.return_value = resources.Zone()
        client.get_zone(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_zone_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetZoneRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_zone), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Zone())
        await client.get_zone(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_zone_flattened():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_zone), '__call__') as call:
        call.return_value = resources.Zone()
        client.get_zone(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_zone_flattened_error():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_zone(service.GetZoneRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_zone_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_zone), '__call__') as call:
        call.return_value = resources.Zone()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Zone())
        response = await client.get_zone(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_zone_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_zone(service.GetZoneRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListNetworksRequest, dict])
def test_list_networks(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.return_value = service.ListNetworksResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_networks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListNetworksRequest()
    assert isinstance(response, pagers.ListNetworksPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_networks_empty_call():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        client.list_networks()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListNetworksRequest()

@pytest.mark.asyncio
async def test_list_networks_async(transport: str='grpc_asyncio', request_type=service.ListNetworksRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListNetworksResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_networks(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListNetworksRequest()
    assert isinstance(response, pagers.ListNetworksAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_networks_async_from_dict():
    await test_list_networks_async(request_type=dict)

def test_list_networks_field_headers():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListNetworksRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.return_value = service.ListNetworksResponse()
        client.list_networks(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_networks_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListNetworksRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListNetworksResponse())
        await client.list_networks(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_networks_flattened():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.return_value = service.ListNetworksResponse()
        client.list_networks(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_networks_flattened_error():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_networks(service.ListNetworksRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_networks_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.return_value = service.ListNetworksResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListNetworksResponse())
        response = await client.list_networks(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_networks_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_networks(service.ListNetworksRequest(), parent='parent_value')

def test_list_networks_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.side_effect = (service.ListNetworksResponse(networks=[resources.Network(), resources.Network(), resources.Network()], next_page_token='abc'), service.ListNetworksResponse(networks=[], next_page_token='def'), service.ListNetworksResponse(networks=[resources.Network()], next_page_token='ghi'), service.ListNetworksResponse(networks=[resources.Network(), resources.Network()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_networks(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Network) for i in results))

def test_list_networks_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_networks), '__call__') as call:
        call.side_effect = (service.ListNetworksResponse(networks=[resources.Network(), resources.Network(), resources.Network()], next_page_token='abc'), service.ListNetworksResponse(networks=[], next_page_token='def'), service.ListNetworksResponse(networks=[resources.Network()], next_page_token='ghi'), service.ListNetworksResponse(networks=[resources.Network(), resources.Network()]), RuntimeError)
        pages = list(client.list_networks(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_networks_async_pager():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_networks), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListNetworksResponse(networks=[resources.Network(), resources.Network(), resources.Network()], next_page_token='abc'), service.ListNetworksResponse(networks=[], next_page_token='def'), service.ListNetworksResponse(networks=[resources.Network()], next_page_token='ghi'), service.ListNetworksResponse(networks=[resources.Network(), resources.Network()]), RuntimeError)
        async_pager = await client.list_networks(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Network) for i in responses))

@pytest.mark.asyncio
async def test_list_networks_async_pages():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_networks), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListNetworksResponse(networks=[resources.Network(), resources.Network(), resources.Network()], next_page_token='abc'), service.ListNetworksResponse(networks=[], next_page_token='def'), service.ListNetworksResponse(networks=[resources.Network()], next_page_token='ghi'), service.ListNetworksResponse(networks=[resources.Network(), resources.Network()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_networks(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetNetworkRequest, dict])
def test_get_network(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_network), '__call__') as call:
        call.return_value = resources.Network(name='name_value', description='description_value', mtu=342)
        response = client.get_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetNetworkRequest()
    assert isinstance(response, resources.Network)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.mtu == 342

def test_get_network_empty_call():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_network), '__call__') as call:
        client.get_network()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetNetworkRequest()

@pytest.mark.asyncio
async def test_get_network_async(transport: str='grpc_asyncio', request_type=service.GetNetworkRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Network(name='name_value', description='description_value', mtu=342))
        response = await client.get_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetNetworkRequest()
    assert isinstance(response, resources.Network)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.mtu == 342

@pytest.mark.asyncio
async def test_get_network_async_from_dict():
    await test_get_network_async(request_type=dict)

def test_get_network_field_headers():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetNetworkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_network), '__call__') as call:
        call.return_value = resources.Network()
        client.get_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_network_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetNetworkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Network())
        await client.get_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_network_flattened():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_network), '__call__') as call:
        call.return_value = resources.Network()
        client.get_network(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_network_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_network(service.GetNetworkRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_network_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_network), '__call__') as call:
        call.return_value = resources.Network()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Network())
        response = await client.get_network(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_network_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_network(service.GetNetworkRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.DiagnoseNetworkRequest, dict])
def test_diagnose_network(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.diagnose_network), '__call__') as call:
        call.return_value = service.DiagnoseNetworkResponse()
        response = client.diagnose_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DiagnoseNetworkRequest()
    assert isinstance(response, service.DiagnoseNetworkResponse)

def test_diagnose_network_empty_call():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.diagnose_network), '__call__') as call:
        client.diagnose_network()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DiagnoseNetworkRequest()

@pytest.mark.asyncio
async def test_diagnose_network_async(transport: str='grpc_asyncio', request_type=service.DiagnoseNetworkRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.diagnose_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.DiagnoseNetworkResponse())
        response = await client.diagnose_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DiagnoseNetworkRequest()
    assert isinstance(response, service.DiagnoseNetworkResponse)

@pytest.mark.asyncio
async def test_diagnose_network_async_from_dict():
    await test_diagnose_network_async(request_type=dict)

def test_diagnose_network_field_headers():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DiagnoseNetworkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.diagnose_network), '__call__') as call:
        call.return_value = service.DiagnoseNetworkResponse()
        client.diagnose_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_diagnose_network_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DiagnoseNetworkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.diagnose_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.DiagnoseNetworkResponse())
        await client.diagnose_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_diagnose_network_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.diagnose_network), '__call__') as call:
        call.return_value = service.DiagnoseNetworkResponse()
        client.diagnose_network(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_diagnose_network_flattened_error():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.diagnose_network(service.DiagnoseNetworkRequest(), name='name_value')

@pytest.mark.asyncio
async def test_diagnose_network_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.diagnose_network), '__call__') as call:
        call.return_value = service.DiagnoseNetworkResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.DiagnoseNetworkResponse())
        response = await client.diagnose_network(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_diagnose_network_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.diagnose_network(service.DiagnoseNetworkRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.CreateNetworkRequest, dict])
def test_create_network(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateNetworkRequest()
    assert isinstance(response, future.Future)

def test_create_network_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_network), '__call__') as call:
        client.create_network()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateNetworkRequest()

@pytest.mark.asyncio
async def test_create_network_async(transport: str='grpc_asyncio', request_type=service.CreateNetworkRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateNetworkRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_network_async_from_dict():
    await test_create_network_async(request_type=dict)

def test_create_network_field_headers():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateNetworkRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_network_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateNetworkRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_network_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_network(parent='parent_value', network=resources.Network(name='name_value'), network_id='network_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].network
        mock_val = resources.Network(name='name_value')
        assert arg == mock_val
        arg = args[0].network_id
        mock_val = 'network_id_value'
        assert arg == mock_val

def test_create_network_flattened_error():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_network(service.CreateNetworkRequest(), parent='parent_value', network=resources.Network(name='name_value'), network_id='network_id_value')

@pytest.mark.asyncio
async def test_create_network_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_network(parent='parent_value', network=resources.Network(name='name_value'), network_id='network_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].network
        mock_val = resources.Network(name='name_value')
        assert arg == mock_val
        arg = args[0].network_id
        mock_val = 'network_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_network_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_network(service.CreateNetworkRequest(), parent='parent_value', network=resources.Network(name='name_value'), network_id='network_id_value')

@pytest.mark.parametrize('request_type', [service.DeleteNetworkRequest, dict])
def test_delete_network(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteNetworkRequest()
    assert isinstance(response, future.Future)

def test_delete_network_empty_call():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_network), '__call__') as call:
        client.delete_network()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteNetworkRequest()

@pytest.mark.asyncio
async def test_delete_network_async(transport: str='grpc_asyncio', request_type=service.DeleteNetworkRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteNetworkRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_network_async_from_dict():
    await test_delete_network_async(request_type=dict)

def test_delete_network_field_headers():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteNetworkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_network(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_network_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteNetworkRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_network), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_network(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_network_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_network(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_network_flattened_error():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_network(service.DeleteNetworkRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_network_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_network), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_network(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_network_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_network(service.DeleteNetworkRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListSubnetsRequest, dict])
def test_list_subnets(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.return_value = service.ListSubnetsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_subnets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSubnetsRequest()
    assert isinstance(response, pagers.ListSubnetsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_subnets_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        client.list_subnets()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSubnetsRequest()

@pytest.mark.asyncio
async def test_list_subnets_async(transport: str='grpc_asyncio', request_type=service.ListSubnetsRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSubnetsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_subnets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSubnetsRequest()
    assert isinstance(response, pagers.ListSubnetsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_subnets_async_from_dict():
    await test_list_subnets_async(request_type=dict)

def test_list_subnets_field_headers():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSubnetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.return_value = service.ListSubnetsResponse()
        client.list_subnets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_subnets_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSubnetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSubnetsResponse())
        await client.list_subnets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_subnets_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.return_value = service.ListSubnetsResponse()
        client.list_subnets(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_subnets_flattened_error():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_subnets(service.ListSubnetsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_subnets_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.return_value = service.ListSubnetsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSubnetsResponse())
        response = await client.list_subnets(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_subnets_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_subnets(service.ListSubnetsRequest(), parent='parent_value')

def test_list_subnets_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.side_effect = (service.ListSubnetsResponse(subnets=[resources.Subnet(), resources.Subnet(), resources.Subnet()], next_page_token='abc'), service.ListSubnetsResponse(subnets=[], next_page_token='def'), service.ListSubnetsResponse(subnets=[resources.Subnet()], next_page_token='ghi'), service.ListSubnetsResponse(subnets=[resources.Subnet(), resources.Subnet()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_subnets(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Subnet) for i in results))

def test_list_subnets_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_subnets), '__call__') as call:
        call.side_effect = (service.ListSubnetsResponse(subnets=[resources.Subnet(), resources.Subnet(), resources.Subnet()], next_page_token='abc'), service.ListSubnetsResponse(subnets=[], next_page_token='def'), service.ListSubnetsResponse(subnets=[resources.Subnet()], next_page_token='ghi'), service.ListSubnetsResponse(subnets=[resources.Subnet(), resources.Subnet()]), RuntimeError)
        pages = list(client.list_subnets(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_subnets_async_pager():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_subnets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSubnetsResponse(subnets=[resources.Subnet(), resources.Subnet(), resources.Subnet()], next_page_token='abc'), service.ListSubnetsResponse(subnets=[], next_page_token='def'), service.ListSubnetsResponse(subnets=[resources.Subnet()], next_page_token='ghi'), service.ListSubnetsResponse(subnets=[resources.Subnet(), resources.Subnet()]), RuntimeError)
        async_pager = await client.list_subnets(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Subnet) for i in responses))

@pytest.mark.asyncio
async def test_list_subnets_async_pages():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_subnets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSubnetsResponse(subnets=[resources.Subnet(), resources.Subnet(), resources.Subnet()], next_page_token='abc'), service.ListSubnetsResponse(subnets=[], next_page_token='def'), service.ListSubnetsResponse(subnets=[resources.Subnet()], next_page_token='ghi'), service.ListSubnetsResponse(subnets=[resources.Subnet(), resources.Subnet()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_subnets(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetSubnetRequest, dict])
def test_get_subnet(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_subnet), '__call__') as call:
        call.return_value = resources.Subnet(name='name_value', description='description_value', network='network_value', ipv4_cidr=['ipv4_cidr_value'], ipv6_cidr=['ipv6_cidr_value'], vlan_id=733, state=resources.ResourceState.STATE_PENDING)
        response = client.get_subnet(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetSubnetRequest()
    assert isinstance(response, resources.Subnet)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.network == 'network_value'
    assert response.ipv4_cidr == ['ipv4_cidr_value']
    assert response.ipv6_cidr == ['ipv6_cidr_value']
    assert response.vlan_id == 733
    assert response.state == resources.ResourceState.STATE_PENDING

def test_get_subnet_empty_call():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_subnet), '__call__') as call:
        client.get_subnet()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetSubnetRequest()

@pytest.mark.asyncio
async def test_get_subnet_async(transport: str='grpc_asyncio', request_type=service.GetSubnetRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_subnet), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Subnet(name='name_value', description='description_value', network='network_value', ipv4_cidr=['ipv4_cidr_value'], ipv6_cidr=['ipv6_cidr_value'], vlan_id=733, state=resources.ResourceState.STATE_PENDING))
        response = await client.get_subnet(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetSubnetRequest()
    assert isinstance(response, resources.Subnet)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.network == 'network_value'
    assert response.ipv4_cidr == ['ipv4_cidr_value']
    assert response.ipv6_cidr == ['ipv6_cidr_value']
    assert response.vlan_id == 733
    assert response.state == resources.ResourceState.STATE_PENDING

@pytest.mark.asyncio
async def test_get_subnet_async_from_dict():
    await test_get_subnet_async(request_type=dict)

def test_get_subnet_field_headers():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetSubnetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_subnet), '__call__') as call:
        call.return_value = resources.Subnet()
        client.get_subnet(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_subnet_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetSubnetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_subnet), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Subnet())
        await client.get_subnet(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_subnet_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_subnet), '__call__') as call:
        call.return_value = resources.Subnet()
        client.get_subnet(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_subnet_flattened_error():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_subnet(service.GetSubnetRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_subnet_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_subnet), '__call__') as call:
        call.return_value = resources.Subnet()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Subnet())
        response = await client.get_subnet(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_subnet_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_subnet(service.GetSubnetRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.CreateSubnetRequest, dict])
def test_create_subnet(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_subnet(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateSubnetRequest()
    assert isinstance(response, future.Future)

def test_create_subnet_empty_call():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_subnet), '__call__') as call:
        client.create_subnet()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateSubnetRequest()

@pytest.mark.asyncio
async def test_create_subnet_async(transport: str='grpc_asyncio', request_type=service.CreateSubnetRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_subnet), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_subnet(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateSubnetRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_subnet_async_from_dict():
    await test_create_subnet_async(request_type=dict)

def test_create_subnet_field_headers():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateSubnetRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_subnet(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_subnet_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateSubnetRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_subnet), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_subnet(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_subnet_flattened():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_subnet(parent='parent_value', subnet=resources.Subnet(name='name_value'), subnet_id='subnet_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].subnet
        mock_val = resources.Subnet(name='name_value')
        assert arg == mock_val
        arg = args[0].subnet_id
        mock_val = 'subnet_id_value'
        assert arg == mock_val

def test_create_subnet_flattened_error():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_subnet(service.CreateSubnetRequest(), parent='parent_value', subnet=resources.Subnet(name='name_value'), subnet_id='subnet_id_value')

@pytest.mark.asyncio
async def test_create_subnet_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_subnet(parent='parent_value', subnet=resources.Subnet(name='name_value'), subnet_id='subnet_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].subnet
        mock_val = resources.Subnet(name='name_value')
        assert arg == mock_val
        arg = args[0].subnet_id
        mock_val = 'subnet_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_subnet_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_subnet(service.CreateSubnetRequest(), parent='parent_value', subnet=resources.Subnet(name='name_value'), subnet_id='subnet_id_value')

@pytest.mark.parametrize('request_type', [service.UpdateSubnetRequest, dict])
def test_update_subnet(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_subnet(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateSubnetRequest()
    assert isinstance(response, future.Future)

def test_update_subnet_empty_call():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_subnet), '__call__') as call:
        client.update_subnet()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateSubnetRequest()

@pytest.mark.asyncio
async def test_update_subnet_async(transport: str='grpc_asyncio', request_type=service.UpdateSubnetRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_subnet), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_subnet(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateSubnetRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_subnet_async_from_dict():
    await test_update_subnet_async(request_type=dict)

def test_update_subnet_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateSubnetRequest()
    request.subnet.name = 'name_value'
    with mock.patch.object(type(client.transport.update_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_subnet(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'subnet.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_subnet_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateSubnetRequest()
    request.subnet.name = 'name_value'
    with mock.patch.object(type(client.transport.update_subnet), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_subnet(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'subnet.name=name_value') in kw['metadata']

def test_update_subnet_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_subnet(subnet=resources.Subnet(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].subnet
        mock_val = resources.Subnet(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_subnet_flattened_error():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_subnet(service.UpdateSubnetRequest(), subnet=resources.Subnet(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_subnet_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_subnet(subnet=resources.Subnet(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].subnet
        mock_val = resources.Subnet(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_subnet_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_subnet(service.UpdateSubnetRequest(), subnet=resources.Subnet(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.DeleteSubnetRequest, dict])
def test_delete_subnet(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_subnet(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteSubnetRequest()
    assert isinstance(response, future.Future)

def test_delete_subnet_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_subnet), '__call__') as call:
        client.delete_subnet()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteSubnetRequest()

@pytest.mark.asyncio
async def test_delete_subnet_async(transport: str='grpc_asyncio', request_type=service.DeleteSubnetRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_subnet), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_subnet(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteSubnetRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_subnet_async_from_dict():
    await test_delete_subnet_async(request_type=dict)

def test_delete_subnet_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteSubnetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_subnet(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_subnet_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteSubnetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_subnet), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_subnet(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_subnet_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_subnet(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_subnet_flattened_error():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_subnet(service.DeleteSubnetRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_subnet_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_subnet), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_subnet(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_subnet_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_subnet(service.DeleteSubnetRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListInterconnectsRequest, dict])
def test_list_interconnects(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_interconnects), '__call__') as call:
        call.return_value = service.ListInterconnectsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_interconnects(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListInterconnectsRequest()
    assert isinstance(response, pagers.ListInterconnectsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_interconnects_empty_call():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_interconnects), '__call__') as call:
        client.list_interconnects()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListInterconnectsRequest()

@pytest.mark.asyncio
async def test_list_interconnects_async(transport: str='grpc_asyncio', request_type=service.ListInterconnectsRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_interconnects), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListInterconnectsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_interconnects(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListInterconnectsRequest()
    assert isinstance(response, pagers.ListInterconnectsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_interconnects_async_from_dict():
    await test_list_interconnects_async(request_type=dict)

def test_list_interconnects_field_headers():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListInterconnectsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_interconnects), '__call__') as call:
        call.return_value = service.ListInterconnectsResponse()
        client.list_interconnects(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_interconnects_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListInterconnectsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_interconnects), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListInterconnectsResponse())
        await client.list_interconnects(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_interconnects_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_interconnects), '__call__') as call:
        call.return_value = service.ListInterconnectsResponse()
        client.list_interconnects(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_interconnects_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_interconnects(service.ListInterconnectsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_interconnects_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_interconnects), '__call__') as call:
        call.return_value = service.ListInterconnectsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListInterconnectsResponse())
        response = await client.list_interconnects(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_interconnects_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_interconnects(service.ListInterconnectsRequest(), parent='parent_value')

def test_list_interconnects_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_interconnects), '__call__') as call:
        call.side_effect = (service.ListInterconnectsResponse(interconnects=[resources.Interconnect(), resources.Interconnect(), resources.Interconnect()], next_page_token='abc'), service.ListInterconnectsResponse(interconnects=[], next_page_token='def'), service.ListInterconnectsResponse(interconnects=[resources.Interconnect()], next_page_token='ghi'), service.ListInterconnectsResponse(interconnects=[resources.Interconnect(), resources.Interconnect()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_interconnects(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Interconnect) for i in results))

def test_list_interconnects_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_interconnects), '__call__') as call:
        call.side_effect = (service.ListInterconnectsResponse(interconnects=[resources.Interconnect(), resources.Interconnect(), resources.Interconnect()], next_page_token='abc'), service.ListInterconnectsResponse(interconnects=[], next_page_token='def'), service.ListInterconnectsResponse(interconnects=[resources.Interconnect()], next_page_token='ghi'), service.ListInterconnectsResponse(interconnects=[resources.Interconnect(), resources.Interconnect()]), RuntimeError)
        pages = list(client.list_interconnects(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_interconnects_async_pager():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_interconnects), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListInterconnectsResponse(interconnects=[resources.Interconnect(), resources.Interconnect(), resources.Interconnect()], next_page_token='abc'), service.ListInterconnectsResponse(interconnects=[], next_page_token='def'), service.ListInterconnectsResponse(interconnects=[resources.Interconnect()], next_page_token='ghi'), service.ListInterconnectsResponse(interconnects=[resources.Interconnect(), resources.Interconnect()]), RuntimeError)
        async_pager = await client.list_interconnects(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Interconnect) for i in responses))

@pytest.mark.asyncio
async def test_list_interconnects_async_pages():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_interconnects), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListInterconnectsResponse(interconnects=[resources.Interconnect(), resources.Interconnect(), resources.Interconnect()], next_page_token='abc'), service.ListInterconnectsResponse(interconnects=[], next_page_token='def'), service.ListInterconnectsResponse(interconnects=[resources.Interconnect()], next_page_token='ghi'), service.ListInterconnectsResponse(interconnects=[resources.Interconnect(), resources.Interconnect()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_interconnects(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetInterconnectRequest, dict])
def test_get_interconnect(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_interconnect), '__call__') as call:
        call.return_value = resources.Interconnect(name='name_value', description='description_value', interconnect_type=resources.Interconnect.InterconnectType.DEDICATED, uuid='uuid_value', device_cloud_resource_name='device_cloud_resource_name_value', physical_ports=['physical_ports_value'])
        response = client.get_interconnect(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetInterconnectRequest()
    assert isinstance(response, resources.Interconnect)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.interconnect_type == resources.Interconnect.InterconnectType.DEDICATED
    assert response.uuid == 'uuid_value'
    assert response.device_cloud_resource_name == 'device_cloud_resource_name_value'
    assert response.physical_ports == ['physical_ports_value']

def test_get_interconnect_empty_call():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_interconnect), '__call__') as call:
        client.get_interconnect()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetInterconnectRequest()

@pytest.mark.asyncio
async def test_get_interconnect_async(transport: str='grpc_asyncio', request_type=service.GetInterconnectRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_interconnect), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Interconnect(name='name_value', description='description_value', interconnect_type=resources.Interconnect.InterconnectType.DEDICATED, uuid='uuid_value', device_cloud_resource_name='device_cloud_resource_name_value', physical_ports=['physical_ports_value']))
        response = await client.get_interconnect(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetInterconnectRequest()
    assert isinstance(response, resources.Interconnect)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.interconnect_type == resources.Interconnect.InterconnectType.DEDICATED
    assert response.uuid == 'uuid_value'
    assert response.device_cloud_resource_name == 'device_cloud_resource_name_value'
    assert response.physical_ports == ['physical_ports_value']

@pytest.mark.asyncio
async def test_get_interconnect_async_from_dict():
    await test_get_interconnect_async(request_type=dict)

def test_get_interconnect_field_headers():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetInterconnectRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_interconnect), '__call__') as call:
        call.return_value = resources.Interconnect()
        client.get_interconnect(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_interconnect_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetInterconnectRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_interconnect), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Interconnect())
        await client.get_interconnect(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_interconnect_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_interconnect), '__call__') as call:
        call.return_value = resources.Interconnect()
        client.get_interconnect(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_interconnect_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_interconnect(service.GetInterconnectRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_interconnect_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_interconnect), '__call__') as call:
        call.return_value = resources.Interconnect()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Interconnect())
        response = await client.get_interconnect(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_interconnect_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_interconnect(service.GetInterconnectRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.DiagnoseInterconnectRequest, dict])
def test_diagnose_interconnect(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.diagnose_interconnect), '__call__') as call:
        call.return_value = service.DiagnoseInterconnectResponse()
        response = client.diagnose_interconnect(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DiagnoseInterconnectRequest()
    assert isinstance(response, service.DiagnoseInterconnectResponse)

def test_diagnose_interconnect_empty_call():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.diagnose_interconnect), '__call__') as call:
        client.diagnose_interconnect()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DiagnoseInterconnectRequest()

@pytest.mark.asyncio
async def test_diagnose_interconnect_async(transport: str='grpc_asyncio', request_type=service.DiagnoseInterconnectRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.diagnose_interconnect), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.DiagnoseInterconnectResponse())
        response = await client.diagnose_interconnect(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DiagnoseInterconnectRequest()
    assert isinstance(response, service.DiagnoseInterconnectResponse)

@pytest.mark.asyncio
async def test_diagnose_interconnect_async_from_dict():
    await test_diagnose_interconnect_async(request_type=dict)

def test_diagnose_interconnect_field_headers():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DiagnoseInterconnectRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.diagnose_interconnect), '__call__') as call:
        call.return_value = service.DiagnoseInterconnectResponse()
        client.diagnose_interconnect(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_diagnose_interconnect_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DiagnoseInterconnectRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.diagnose_interconnect), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.DiagnoseInterconnectResponse())
        await client.diagnose_interconnect(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_diagnose_interconnect_flattened():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.diagnose_interconnect), '__call__') as call:
        call.return_value = service.DiagnoseInterconnectResponse()
        client.diagnose_interconnect(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_diagnose_interconnect_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.diagnose_interconnect(service.DiagnoseInterconnectRequest(), name='name_value')

@pytest.mark.asyncio
async def test_diagnose_interconnect_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.diagnose_interconnect), '__call__') as call:
        call.return_value = service.DiagnoseInterconnectResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.DiagnoseInterconnectResponse())
        response = await client.diagnose_interconnect(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_diagnose_interconnect_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.diagnose_interconnect(service.DiagnoseInterconnectRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListInterconnectAttachmentsRequest, dict])
def test_list_interconnect_attachments(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_interconnect_attachments), '__call__') as call:
        call.return_value = service.ListInterconnectAttachmentsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_interconnect_attachments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListInterconnectAttachmentsRequest()
    assert isinstance(response, pagers.ListInterconnectAttachmentsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_interconnect_attachments_empty_call():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_interconnect_attachments), '__call__') as call:
        client.list_interconnect_attachments()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListInterconnectAttachmentsRequest()

@pytest.mark.asyncio
async def test_list_interconnect_attachments_async(transport: str='grpc_asyncio', request_type=service.ListInterconnectAttachmentsRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_interconnect_attachments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListInterconnectAttachmentsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_interconnect_attachments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListInterconnectAttachmentsRequest()
    assert isinstance(response, pagers.ListInterconnectAttachmentsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_interconnect_attachments_async_from_dict():
    await test_list_interconnect_attachments_async(request_type=dict)

def test_list_interconnect_attachments_field_headers():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListInterconnectAttachmentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_interconnect_attachments), '__call__') as call:
        call.return_value = service.ListInterconnectAttachmentsResponse()
        client.list_interconnect_attachments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_interconnect_attachments_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListInterconnectAttachmentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_interconnect_attachments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListInterconnectAttachmentsResponse())
        await client.list_interconnect_attachments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_interconnect_attachments_flattened():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_interconnect_attachments), '__call__') as call:
        call.return_value = service.ListInterconnectAttachmentsResponse()
        client.list_interconnect_attachments(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_interconnect_attachments_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_interconnect_attachments(service.ListInterconnectAttachmentsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_interconnect_attachments_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_interconnect_attachments), '__call__') as call:
        call.return_value = service.ListInterconnectAttachmentsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListInterconnectAttachmentsResponse())
        response = await client.list_interconnect_attachments(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_interconnect_attachments_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_interconnect_attachments(service.ListInterconnectAttachmentsRequest(), parent='parent_value')

def test_list_interconnect_attachments_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_interconnect_attachments), '__call__') as call:
        call.side_effect = (service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment(), resources.InterconnectAttachment(), resources.InterconnectAttachment()], next_page_token='abc'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[], next_page_token='def'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment()], next_page_token='ghi'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment(), resources.InterconnectAttachment()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_interconnect_attachments(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.InterconnectAttachment) for i in results))

def test_list_interconnect_attachments_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_interconnect_attachments), '__call__') as call:
        call.side_effect = (service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment(), resources.InterconnectAttachment(), resources.InterconnectAttachment()], next_page_token='abc'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[], next_page_token='def'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment()], next_page_token='ghi'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment(), resources.InterconnectAttachment()]), RuntimeError)
        pages = list(client.list_interconnect_attachments(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_interconnect_attachments_async_pager():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_interconnect_attachments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment(), resources.InterconnectAttachment(), resources.InterconnectAttachment()], next_page_token='abc'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[], next_page_token='def'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment()], next_page_token='ghi'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment(), resources.InterconnectAttachment()]), RuntimeError)
        async_pager = await client.list_interconnect_attachments(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.InterconnectAttachment) for i in responses))

@pytest.mark.asyncio
async def test_list_interconnect_attachments_async_pages():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_interconnect_attachments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment(), resources.InterconnectAttachment(), resources.InterconnectAttachment()], next_page_token='abc'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[], next_page_token='def'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment()], next_page_token='ghi'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment(), resources.InterconnectAttachment()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_interconnect_attachments(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetInterconnectAttachmentRequest, dict])
def test_get_interconnect_attachment(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_interconnect_attachment), '__call__') as call:
        call.return_value = resources.InterconnectAttachment(name='name_value', description='description_value', interconnect='interconnect_value', network='network_value', vlan_id=733, mtu=342, state=resources.ResourceState.STATE_PENDING)
        response = client.get_interconnect_attachment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetInterconnectAttachmentRequest()
    assert isinstance(response, resources.InterconnectAttachment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.interconnect == 'interconnect_value'
    assert response.network == 'network_value'
    assert response.vlan_id == 733
    assert response.mtu == 342
    assert response.state == resources.ResourceState.STATE_PENDING

def test_get_interconnect_attachment_empty_call():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_interconnect_attachment), '__call__') as call:
        client.get_interconnect_attachment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetInterconnectAttachmentRequest()

@pytest.mark.asyncio
async def test_get_interconnect_attachment_async(transport: str='grpc_asyncio', request_type=service.GetInterconnectAttachmentRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_interconnect_attachment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.InterconnectAttachment(name='name_value', description='description_value', interconnect='interconnect_value', network='network_value', vlan_id=733, mtu=342, state=resources.ResourceState.STATE_PENDING))
        response = await client.get_interconnect_attachment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetInterconnectAttachmentRequest()
    assert isinstance(response, resources.InterconnectAttachment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.interconnect == 'interconnect_value'
    assert response.network == 'network_value'
    assert response.vlan_id == 733
    assert response.mtu == 342
    assert response.state == resources.ResourceState.STATE_PENDING

@pytest.mark.asyncio
async def test_get_interconnect_attachment_async_from_dict():
    await test_get_interconnect_attachment_async(request_type=dict)

def test_get_interconnect_attachment_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetInterconnectAttachmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_interconnect_attachment), '__call__') as call:
        call.return_value = resources.InterconnectAttachment()
        client.get_interconnect_attachment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_interconnect_attachment_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetInterconnectAttachmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_interconnect_attachment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.InterconnectAttachment())
        await client.get_interconnect_attachment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_interconnect_attachment_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_interconnect_attachment), '__call__') as call:
        call.return_value = resources.InterconnectAttachment()
        client.get_interconnect_attachment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_interconnect_attachment_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_interconnect_attachment(service.GetInterconnectAttachmentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_interconnect_attachment_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_interconnect_attachment), '__call__') as call:
        call.return_value = resources.InterconnectAttachment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.InterconnectAttachment())
        response = await client.get_interconnect_attachment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_interconnect_attachment_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_interconnect_attachment(service.GetInterconnectAttachmentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.CreateInterconnectAttachmentRequest, dict])
def test_create_interconnect_attachment(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_interconnect_attachment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_interconnect_attachment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateInterconnectAttachmentRequest()
    assert isinstance(response, future.Future)

def test_create_interconnect_attachment_empty_call():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_interconnect_attachment), '__call__') as call:
        client.create_interconnect_attachment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateInterconnectAttachmentRequest()

@pytest.mark.asyncio
async def test_create_interconnect_attachment_async(transport: str='grpc_asyncio', request_type=service.CreateInterconnectAttachmentRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_interconnect_attachment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_interconnect_attachment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateInterconnectAttachmentRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_interconnect_attachment_async_from_dict():
    await test_create_interconnect_attachment_async(request_type=dict)

def test_create_interconnect_attachment_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateInterconnectAttachmentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_interconnect_attachment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_interconnect_attachment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_interconnect_attachment_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateInterconnectAttachmentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_interconnect_attachment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_interconnect_attachment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_interconnect_attachment_flattened():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_interconnect_attachment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_interconnect_attachment(parent='parent_value', interconnect_attachment=resources.InterconnectAttachment(name='name_value'), interconnect_attachment_id='interconnect_attachment_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].interconnect_attachment
        mock_val = resources.InterconnectAttachment(name='name_value')
        assert arg == mock_val
        arg = args[0].interconnect_attachment_id
        mock_val = 'interconnect_attachment_id_value'
        assert arg == mock_val

def test_create_interconnect_attachment_flattened_error():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_interconnect_attachment(service.CreateInterconnectAttachmentRequest(), parent='parent_value', interconnect_attachment=resources.InterconnectAttachment(name='name_value'), interconnect_attachment_id='interconnect_attachment_id_value')

@pytest.mark.asyncio
async def test_create_interconnect_attachment_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_interconnect_attachment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_interconnect_attachment(parent='parent_value', interconnect_attachment=resources.InterconnectAttachment(name='name_value'), interconnect_attachment_id='interconnect_attachment_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].interconnect_attachment
        mock_val = resources.InterconnectAttachment(name='name_value')
        assert arg == mock_val
        arg = args[0].interconnect_attachment_id
        mock_val = 'interconnect_attachment_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_interconnect_attachment_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_interconnect_attachment(service.CreateInterconnectAttachmentRequest(), parent='parent_value', interconnect_attachment=resources.InterconnectAttachment(name='name_value'), interconnect_attachment_id='interconnect_attachment_id_value')

@pytest.mark.parametrize('request_type', [service.DeleteInterconnectAttachmentRequest, dict])
def test_delete_interconnect_attachment(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_interconnect_attachment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_interconnect_attachment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteInterconnectAttachmentRequest()
    assert isinstance(response, future.Future)

def test_delete_interconnect_attachment_empty_call():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_interconnect_attachment), '__call__') as call:
        client.delete_interconnect_attachment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteInterconnectAttachmentRequest()

@pytest.mark.asyncio
async def test_delete_interconnect_attachment_async(transport: str='grpc_asyncio', request_type=service.DeleteInterconnectAttachmentRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_interconnect_attachment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_interconnect_attachment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteInterconnectAttachmentRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_interconnect_attachment_async_from_dict():
    await test_delete_interconnect_attachment_async(request_type=dict)

def test_delete_interconnect_attachment_field_headers():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteInterconnectAttachmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_interconnect_attachment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_interconnect_attachment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_interconnect_attachment_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteInterconnectAttachmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_interconnect_attachment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_interconnect_attachment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_interconnect_attachment_flattened():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_interconnect_attachment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_interconnect_attachment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_interconnect_attachment_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_interconnect_attachment(service.DeleteInterconnectAttachmentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_interconnect_attachment_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_interconnect_attachment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_interconnect_attachment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_interconnect_attachment_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_interconnect_attachment(service.DeleteInterconnectAttachmentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListRoutersRequest, dict])
def test_list_routers(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_routers), '__call__') as call:
        call.return_value = service.ListRoutersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_routers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListRoutersRequest()
    assert isinstance(response, pagers.ListRoutersPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_routers_empty_call():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_routers), '__call__') as call:
        client.list_routers()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListRoutersRequest()

@pytest.mark.asyncio
async def test_list_routers_async(transport: str='grpc_asyncio', request_type=service.ListRoutersRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_routers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListRoutersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_routers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListRoutersRequest()
    assert isinstance(response, pagers.ListRoutersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_routers_async_from_dict():
    await test_list_routers_async(request_type=dict)

def test_list_routers_field_headers():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListRoutersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_routers), '__call__') as call:
        call.return_value = service.ListRoutersResponse()
        client.list_routers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_routers_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListRoutersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_routers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListRoutersResponse())
        await client.list_routers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_routers_flattened():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_routers), '__call__') as call:
        call.return_value = service.ListRoutersResponse()
        client.list_routers(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_routers_flattened_error():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_routers(service.ListRoutersRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_routers_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_routers), '__call__') as call:
        call.return_value = service.ListRoutersResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListRoutersResponse())
        response = await client.list_routers(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_routers_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_routers(service.ListRoutersRequest(), parent='parent_value')

def test_list_routers_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_routers), '__call__') as call:
        call.side_effect = (service.ListRoutersResponse(routers=[resources.Router(), resources.Router(), resources.Router()], next_page_token='abc'), service.ListRoutersResponse(routers=[], next_page_token='def'), service.ListRoutersResponse(routers=[resources.Router()], next_page_token='ghi'), service.ListRoutersResponse(routers=[resources.Router(), resources.Router()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_routers(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Router) for i in results))

def test_list_routers_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_routers), '__call__') as call:
        call.side_effect = (service.ListRoutersResponse(routers=[resources.Router(), resources.Router(), resources.Router()], next_page_token='abc'), service.ListRoutersResponse(routers=[], next_page_token='def'), service.ListRoutersResponse(routers=[resources.Router()], next_page_token='ghi'), service.ListRoutersResponse(routers=[resources.Router(), resources.Router()]), RuntimeError)
        pages = list(client.list_routers(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_routers_async_pager():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_routers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListRoutersResponse(routers=[resources.Router(), resources.Router(), resources.Router()], next_page_token='abc'), service.ListRoutersResponse(routers=[], next_page_token='def'), service.ListRoutersResponse(routers=[resources.Router()], next_page_token='ghi'), service.ListRoutersResponse(routers=[resources.Router(), resources.Router()]), RuntimeError)
        async_pager = await client.list_routers(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Router) for i in responses))

@pytest.mark.asyncio
async def test_list_routers_async_pages():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_routers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListRoutersResponse(routers=[resources.Router(), resources.Router(), resources.Router()], next_page_token='abc'), service.ListRoutersResponse(routers=[], next_page_token='def'), service.ListRoutersResponse(routers=[resources.Router()], next_page_token='ghi'), service.ListRoutersResponse(routers=[resources.Router(), resources.Router()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_routers(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetRouterRequest, dict])
def test_get_router(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_router), '__call__') as call:
        call.return_value = resources.Router(name='name_value', description='description_value', network='network_value', state=resources.ResourceState.STATE_PENDING, route_advertisements=['route_advertisements_value'])
        response = client.get_router(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetRouterRequest()
    assert isinstance(response, resources.Router)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.network == 'network_value'
    assert response.state == resources.ResourceState.STATE_PENDING
    assert response.route_advertisements == ['route_advertisements_value']

def test_get_router_empty_call():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_router), '__call__') as call:
        client.get_router()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetRouterRequest()

@pytest.mark.asyncio
async def test_get_router_async(transport: str='grpc_asyncio', request_type=service.GetRouterRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_router), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Router(name='name_value', description='description_value', network='network_value', state=resources.ResourceState.STATE_PENDING, route_advertisements=['route_advertisements_value']))
        response = await client.get_router(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetRouterRequest()
    assert isinstance(response, resources.Router)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.network == 'network_value'
    assert response.state == resources.ResourceState.STATE_PENDING
    assert response.route_advertisements == ['route_advertisements_value']

@pytest.mark.asyncio
async def test_get_router_async_from_dict():
    await test_get_router_async(request_type=dict)

def test_get_router_field_headers():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetRouterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_router), '__call__') as call:
        call.return_value = resources.Router()
        client.get_router(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_router_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetRouterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_router), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Router())
        await client.get_router(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_router_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_router), '__call__') as call:
        call.return_value = resources.Router()
        client.get_router(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_router_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_router(service.GetRouterRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_router_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_router), '__call__') as call:
        call.return_value = resources.Router()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Router())
        response = await client.get_router(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_router_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_router(service.GetRouterRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.DiagnoseRouterRequest, dict])
def test_diagnose_router(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.diagnose_router), '__call__') as call:
        call.return_value = service.DiagnoseRouterResponse()
        response = client.diagnose_router(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DiagnoseRouterRequest()
    assert isinstance(response, service.DiagnoseRouterResponse)

def test_diagnose_router_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.diagnose_router), '__call__') as call:
        client.diagnose_router()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DiagnoseRouterRequest()

@pytest.mark.asyncio
async def test_diagnose_router_async(transport: str='grpc_asyncio', request_type=service.DiagnoseRouterRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.diagnose_router), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.DiagnoseRouterResponse())
        response = await client.diagnose_router(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DiagnoseRouterRequest()
    assert isinstance(response, service.DiagnoseRouterResponse)

@pytest.mark.asyncio
async def test_diagnose_router_async_from_dict():
    await test_diagnose_router_async(request_type=dict)

def test_diagnose_router_field_headers():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DiagnoseRouterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.diagnose_router), '__call__') as call:
        call.return_value = service.DiagnoseRouterResponse()
        client.diagnose_router(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_diagnose_router_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DiagnoseRouterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.diagnose_router), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.DiagnoseRouterResponse())
        await client.diagnose_router(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_diagnose_router_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.diagnose_router), '__call__') as call:
        call.return_value = service.DiagnoseRouterResponse()
        client.diagnose_router(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_diagnose_router_flattened_error():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.diagnose_router(service.DiagnoseRouterRequest(), name='name_value')

@pytest.mark.asyncio
async def test_diagnose_router_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.diagnose_router), '__call__') as call:
        call.return_value = service.DiagnoseRouterResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.DiagnoseRouterResponse())
        response = await client.diagnose_router(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_diagnose_router_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.diagnose_router(service.DiagnoseRouterRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.CreateRouterRequest, dict])
def test_create_router(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_router), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_router(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateRouterRequest()
    assert isinstance(response, future.Future)

def test_create_router_empty_call():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_router), '__call__') as call:
        client.create_router()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateRouterRequest()

@pytest.mark.asyncio
async def test_create_router_async(transport: str='grpc_asyncio', request_type=service.CreateRouterRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_router), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_router(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateRouterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_router_async_from_dict():
    await test_create_router_async(request_type=dict)

def test_create_router_field_headers():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateRouterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_router), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_router(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_router_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateRouterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_router), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_router(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_router_flattened():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_router), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_router(parent='parent_value', router=resources.Router(name='name_value'), router_id='router_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].router
        mock_val = resources.Router(name='name_value')
        assert arg == mock_val
        arg = args[0].router_id
        mock_val = 'router_id_value'
        assert arg == mock_val

def test_create_router_flattened_error():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_router(service.CreateRouterRequest(), parent='parent_value', router=resources.Router(name='name_value'), router_id='router_id_value')

@pytest.mark.asyncio
async def test_create_router_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_router), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_router(parent='parent_value', router=resources.Router(name='name_value'), router_id='router_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].router
        mock_val = resources.Router(name='name_value')
        assert arg == mock_val
        arg = args[0].router_id
        mock_val = 'router_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_router_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_router(service.CreateRouterRequest(), parent='parent_value', router=resources.Router(name='name_value'), router_id='router_id_value')

@pytest.mark.parametrize('request_type', [service.UpdateRouterRequest, dict])
def test_update_router(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_router), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_router(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateRouterRequest()
    assert isinstance(response, future.Future)

def test_update_router_empty_call():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_router), '__call__') as call:
        client.update_router()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateRouterRequest()

@pytest.mark.asyncio
async def test_update_router_async(transport: str='grpc_asyncio', request_type=service.UpdateRouterRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_router), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_router(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateRouterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_router_async_from_dict():
    await test_update_router_async(request_type=dict)

def test_update_router_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateRouterRequest()
    request.router.name = 'name_value'
    with mock.patch.object(type(client.transport.update_router), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_router(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'router.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_router_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateRouterRequest()
    request.router.name = 'name_value'
    with mock.patch.object(type(client.transport.update_router), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_router(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'router.name=name_value') in kw['metadata']

def test_update_router_flattened():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_router), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_router(router=resources.Router(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].router
        mock_val = resources.Router(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_router_flattened_error():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_router(service.UpdateRouterRequest(), router=resources.Router(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_router_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_router), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_router(router=resources.Router(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].router
        mock_val = resources.Router(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_router_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_router(service.UpdateRouterRequest(), router=resources.Router(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.DeleteRouterRequest, dict])
def test_delete_router(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_router), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_router(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteRouterRequest()
    assert isinstance(response, future.Future)

def test_delete_router_empty_call():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_router), '__call__') as call:
        client.delete_router()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteRouterRequest()

@pytest.mark.asyncio
async def test_delete_router_async(transport: str='grpc_asyncio', request_type=service.DeleteRouterRequest):
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_router), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_router(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteRouterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_router_async_from_dict():
    await test_delete_router_async(request_type=dict)

def test_delete_router_field_headers():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteRouterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_router), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_router(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_router_field_headers_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteRouterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_router), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_router(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_router_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_router), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_router(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_router_flattened_error():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_router(service.DeleteRouterRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_router_flattened_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_router), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_router(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_router_flattened_error_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_router(service.DeleteRouterRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.InitializeZoneRequest, dict])
def test_initialize_zone_rest(request_type):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.InitializeZoneResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = service.InitializeZoneResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.initialize_zone(request)
    assert isinstance(response, service.InitializeZoneResponse)

def test_initialize_zone_rest_required_fields(request_type=service.InitializeZoneRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).initialize_zone._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).initialize_zone._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.InitializeZoneResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.InitializeZoneResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.initialize_zone(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_initialize_zone_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.initialize_zone._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_initialize_zone_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_initialize_zone') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_initialize_zone') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.InitializeZoneRequest.pb(service.InitializeZoneRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.InitializeZoneResponse.to_json(service.InitializeZoneResponse())
        request = service.InitializeZoneRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.InitializeZoneResponse()
        client.initialize_zone(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_initialize_zone_rest_bad_request(transport: str='rest', request_type=service.InitializeZoneRequest):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.initialize_zone(request)

def test_initialize_zone_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.InitializeZoneResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/zones/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.InitializeZoneResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.initialize_zone(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/zones/*}:initialize' % client.transport._host, args[1])

def test_initialize_zone_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.initialize_zone(service.InitializeZoneRequest(), name='name_value')

def test_initialize_zone_rest_error():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListZonesRequest, dict])
def test_list_zones_rest(request_type):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListZonesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListZonesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_zones(request)
    assert isinstance(response, pagers.ListZonesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_zones_rest_required_fields(request_type=service.ListZonesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_zones._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_zones._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListZonesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListZonesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_zones(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_zones_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_zones._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_zones_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_list_zones') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_list_zones') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListZonesRequest.pb(service.ListZonesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListZonesResponse.to_json(service.ListZonesResponse())
        request = service.ListZonesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListZonesResponse()
        client.list_zones(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_zones_rest_bad_request(transport: str='rest', request_type=service.ListZonesRequest):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_zones(request)

def test_list_zones_rest_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListZonesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListZonesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_zones(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/zones' % client.transport._host, args[1])

def test_list_zones_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_zones(service.ListZonesRequest(), parent='parent_value')

def test_list_zones_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListZonesResponse(zones=[resources.Zone(), resources.Zone(), resources.Zone()], next_page_token='abc'), service.ListZonesResponse(zones=[], next_page_token='def'), service.ListZonesResponse(zones=[resources.Zone()], next_page_token='ghi'), service.ListZonesResponse(zones=[resources.Zone(), resources.Zone()]))
        response = response + response
        response = tuple((service.ListZonesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_zones(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Zone) for i in results))
        pages = list(client.list_zones(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetZoneRequest, dict])
def test_get_zone_rest(request_type):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Zone(name='name_value', layout_name='layout_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Zone.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_zone(request)
    assert isinstance(response, resources.Zone)
    assert response.name == 'name_value'
    assert response.layout_name == 'layout_name_value'

def test_get_zone_rest_required_fields(request_type=service.GetZoneRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_zone._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_zone._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Zone()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Zone.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_zone(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_zone_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_zone._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_zone_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_get_zone') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_get_zone') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetZoneRequest.pb(service.GetZoneRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Zone.to_json(resources.Zone())
        request = service.GetZoneRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Zone()
        client.get_zone(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_zone_rest_bad_request(transport: str='rest', request_type=service.GetZoneRequest):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_zone(request)

def test_get_zone_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Zone()
        sample_request = {'name': 'projects/sample1/locations/sample2/zones/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Zone.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_zone(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/zones/*}' % client.transport._host, args[1])

def test_get_zone_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_zone(service.GetZoneRequest(), name='name_value')

def test_get_zone_rest_error():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListNetworksRequest, dict])
def test_list_networks_rest(request_type):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListNetworksResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListNetworksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_networks(request)
    assert isinstance(response, pagers.ListNetworksPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_networks_rest_required_fields(request_type=service.ListNetworksRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_networks._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_networks._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListNetworksResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListNetworksResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_networks(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_networks_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_networks._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_networks_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_list_networks') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_list_networks') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListNetworksRequest.pb(service.ListNetworksRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListNetworksResponse.to_json(service.ListNetworksResponse())
        request = service.ListNetworksRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListNetworksResponse()
        client.list_networks(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_networks_rest_bad_request(transport: str='rest', request_type=service.ListNetworksRequest):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_networks(request)

def test_list_networks_rest_flattened():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListNetworksResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListNetworksResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_networks(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/zones/*}/networks' % client.transport._host, args[1])

def test_list_networks_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_networks(service.ListNetworksRequest(), parent='parent_value')

def test_list_networks_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListNetworksResponse(networks=[resources.Network(), resources.Network(), resources.Network()], next_page_token='abc'), service.ListNetworksResponse(networks=[], next_page_token='def'), service.ListNetworksResponse(networks=[resources.Network()], next_page_token='ghi'), service.ListNetworksResponse(networks=[resources.Network(), resources.Network()]))
        response = response + response
        response = tuple((service.ListNetworksResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
        pager = client.list_networks(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Network) for i in results))
        pages = list(client.list_networks(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetNetworkRequest, dict])
def test_get_network_rest(request_type):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/networks/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Network(name='name_value', description='description_value', mtu=342)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Network.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_network(request)
    assert isinstance(response, resources.Network)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.mtu == 342

def test_get_network_rest_required_fields(request_type=service.GetNetworkRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Network()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Network.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_network(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_network_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_network._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_network_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_get_network') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_get_network') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetNetworkRequest.pb(service.GetNetworkRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Network.to_json(resources.Network())
        request = service.GetNetworkRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Network()
        client.get_network(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_network_rest_bad_request(transport: str='rest', request_type=service.GetNetworkRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/networks/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_network(request)

def test_get_network_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Network()
        sample_request = {'name': 'projects/sample1/locations/sample2/zones/sample3/networks/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Network.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_network(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/zones/*/networks/*}' % client.transport._host, args[1])

def test_get_network_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_network(service.GetNetworkRequest(), name='name_value')

def test_get_network_rest_error():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DiagnoseNetworkRequest, dict])
def test_diagnose_network_rest(request_type):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/networks/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.DiagnoseNetworkResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = service.DiagnoseNetworkResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.diagnose_network(request)
    assert isinstance(response, service.DiagnoseNetworkResponse)

def test_diagnose_network_rest_required_fields(request_type=service.DiagnoseNetworkRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).diagnose_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).diagnose_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.DiagnoseNetworkResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.DiagnoseNetworkResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.diagnose_network(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_diagnose_network_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.diagnose_network._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_diagnose_network_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_diagnose_network') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_diagnose_network') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DiagnoseNetworkRequest.pb(service.DiagnoseNetworkRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.DiagnoseNetworkResponse.to_json(service.DiagnoseNetworkResponse())
        request = service.DiagnoseNetworkRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.DiagnoseNetworkResponse()
        client.diagnose_network(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_diagnose_network_rest_bad_request(transport: str='rest', request_type=service.DiagnoseNetworkRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/networks/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.diagnose_network(request)

def test_diagnose_network_rest_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.DiagnoseNetworkResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/zones/sample3/networks/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.DiagnoseNetworkResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.diagnose_network(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/zones/*/networks/*}:diagnose' % client.transport._host, args[1])

def test_diagnose_network_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.diagnose_network(service.DiagnoseNetworkRequest(), name='name_value')

def test_diagnose_network_rest_error():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateNetworkRequest, dict])
def test_create_network_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request_init['network'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'mtu': 342}
    test_field = service.CreateNetworkRequest.meta.fields['network']

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
    for (field, value) in request_init['network'].items():
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
                for i in range(0, len(request_init['network'][field])):
                    del request_init['network'][field][i][subfield]
            else:
                del request_init['network'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_network(request)
    assert response.operation.name == 'operations/spam'

def test_create_network_rest_required_fields(request_type=service.CreateNetworkRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['network_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'networkId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'networkId' in jsonified_request
    assert jsonified_request['networkId'] == request_init['network_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['networkId'] = 'network_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_network._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('network_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'networkId' in jsonified_request
    assert jsonified_request['networkId'] == 'network_id_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_network(request)
            expected_params = [('networkId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_network_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_network._get_unset_required_fields({})
    assert set(unset_fields) == set(('networkId', 'requestId')) & set(('parent', 'networkId', 'network'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_network_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_create_network') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_create_network') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateNetworkRequest.pb(service.CreateNetworkRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateNetworkRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_network(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_network_rest_bad_request(transport: str='rest', request_type=service.CreateNetworkRequest):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_network(request)

def test_create_network_rest_flattened():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
        mock_args = dict(parent='parent_value', network=resources.Network(name='name_value'), network_id='network_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_network(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/zones/*}/networks' % client.transport._host, args[1])

def test_create_network_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_network(service.CreateNetworkRequest(), parent='parent_value', network=resources.Network(name='name_value'), network_id='network_id_value')

def test_create_network_rest_error():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteNetworkRequest, dict])
def test_delete_network_rest(request_type):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/networks/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_network(request)
    assert response.operation.name == 'operations/spam'

def test_delete_network_rest_required_fields(request_type=service.DeleteNetworkRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_network._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_network._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_network(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_network_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_network._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_network_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_delete_network') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_delete_network') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DeleteNetworkRequest.pb(service.DeleteNetworkRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.DeleteNetworkRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_network(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_network_rest_bad_request(transport: str='rest', request_type=service.DeleteNetworkRequest):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/networks/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_network(request)

def test_delete_network_rest_flattened():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/zones/sample3/networks/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_network(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/zones/*/networks/*}' % client.transport._host, args[1])

def test_delete_network_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_network(service.DeleteNetworkRequest(), name='name_value')

def test_delete_network_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListSubnetsRequest, dict])
def test_list_subnets_rest(request_type):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListSubnetsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListSubnetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_subnets(request)
    assert isinstance(response, pagers.ListSubnetsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_subnets_rest_required_fields(request_type=service.ListSubnetsRequest):
    if False:
        return 10
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_subnets._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_subnets._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListSubnetsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListSubnetsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_subnets(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_subnets_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_subnets._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_subnets_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_list_subnets') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_list_subnets') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListSubnetsRequest.pb(service.ListSubnetsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListSubnetsResponse.to_json(service.ListSubnetsResponse())
        request = service.ListSubnetsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListSubnetsResponse()
        client.list_subnets(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_subnets_rest_bad_request(transport: str='rest', request_type=service.ListSubnetsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_subnets(request)

def test_list_subnets_rest_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListSubnetsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListSubnetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_subnets(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/zones/*}/subnets' % client.transport._host, args[1])

def test_list_subnets_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_subnets(service.ListSubnetsRequest(), parent='parent_value')

def test_list_subnets_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListSubnetsResponse(subnets=[resources.Subnet(), resources.Subnet(), resources.Subnet()], next_page_token='abc'), service.ListSubnetsResponse(subnets=[], next_page_token='def'), service.ListSubnetsResponse(subnets=[resources.Subnet()], next_page_token='ghi'), service.ListSubnetsResponse(subnets=[resources.Subnet(), resources.Subnet()]))
        response = response + response
        response = tuple((service.ListSubnetsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
        pager = client.list_subnets(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Subnet) for i in results))
        pages = list(client.list_subnets(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetSubnetRequest, dict])
def test_get_subnet_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/subnets/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Subnet(name='name_value', description='description_value', network='network_value', ipv4_cidr=['ipv4_cidr_value'], ipv6_cidr=['ipv6_cidr_value'], vlan_id=733, state=resources.ResourceState.STATE_PENDING)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Subnet.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_subnet(request)
    assert isinstance(response, resources.Subnet)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.network == 'network_value'
    assert response.ipv4_cidr == ['ipv4_cidr_value']
    assert response.ipv6_cidr == ['ipv6_cidr_value']
    assert response.vlan_id == 733
    assert response.state == resources.ResourceState.STATE_PENDING

def test_get_subnet_rest_required_fields(request_type=service.GetSubnetRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_subnet._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_subnet._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Subnet()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Subnet.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_subnet(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_subnet_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_subnet._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_subnet_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_get_subnet') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_get_subnet') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetSubnetRequest.pb(service.GetSubnetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Subnet.to_json(resources.Subnet())
        request = service.GetSubnetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Subnet()
        client.get_subnet(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_subnet_rest_bad_request(transport: str='rest', request_type=service.GetSubnetRequest):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/subnets/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_subnet(request)

def test_get_subnet_rest_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Subnet()
        sample_request = {'name': 'projects/sample1/locations/sample2/zones/sample3/subnets/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Subnet.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_subnet(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/zones/*/subnets/*}' % client.transport._host, args[1])

def test_get_subnet_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_subnet(service.GetSubnetRequest(), name='name_value')

def test_get_subnet_rest_error():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateSubnetRequest, dict])
def test_create_subnet_rest(request_type):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request_init['subnet'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'network': 'network_value', 'ipv4_cidr': ['ipv4_cidr_value1', 'ipv4_cidr_value2'], 'ipv6_cidr': ['ipv6_cidr_value1', 'ipv6_cidr_value2'], 'vlan_id': 733, 'state': 1}
    test_field = service.CreateSubnetRequest.meta.fields['subnet']

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
    for (field, value) in request_init['subnet'].items():
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
                for i in range(0, len(request_init['subnet'][field])):
                    del request_init['subnet'][field][i][subfield]
            else:
                del request_init['subnet'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_subnet(request)
    assert response.operation.name == 'operations/spam'

def test_create_subnet_rest_required_fields(request_type=service.CreateSubnetRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['subnet_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'subnetId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_subnet._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'subnetId' in jsonified_request
    assert jsonified_request['subnetId'] == request_init['subnet_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['subnetId'] = 'subnet_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_subnet._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'subnet_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'subnetId' in jsonified_request
    assert jsonified_request['subnetId'] == 'subnet_id_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_subnet(request)
            expected_params = [('subnetId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_subnet_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_subnet._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'subnetId')) & set(('parent', 'subnetId', 'subnet'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_subnet_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_create_subnet') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_create_subnet') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateSubnetRequest.pb(service.CreateSubnetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateSubnetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_subnet(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_subnet_rest_bad_request(transport: str='rest', request_type=service.CreateSubnetRequest):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_subnet(request)

def test_create_subnet_rest_flattened():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
        mock_args = dict(parent='parent_value', subnet=resources.Subnet(name='name_value'), subnet_id='subnet_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_subnet(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/zones/*}/subnets' % client.transport._host, args[1])

def test_create_subnet_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_subnet(service.CreateSubnetRequest(), parent='parent_value', subnet=resources.Subnet(name='name_value'), subnet_id='subnet_id_value')

def test_create_subnet_rest_error():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateSubnetRequest, dict])
def test_update_subnet_rest(request_type):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'subnet': {'name': 'projects/sample1/locations/sample2/zones/sample3/subnets/sample4'}}
    request_init['subnet'] = {'name': 'projects/sample1/locations/sample2/zones/sample3/subnets/sample4', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'network': 'network_value', 'ipv4_cidr': ['ipv4_cidr_value1', 'ipv4_cidr_value2'], 'ipv6_cidr': ['ipv6_cidr_value1', 'ipv6_cidr_value2'], 'vlan_id': 733, 'state': 1}
    test_field = service.UpdateSubnetRequest.meta.fields['subnet']

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
    for (field, value) in request_init['subnet'].items():
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
                for i in range(0, len(request_init['subnet'][field])):
                    del request_init['subnet'][field][i][subfield]
            else:
                del request_init['subnet'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_subnet(request)
    assert response.operation.name == 'operations/spam'

def test_update_subnet_rest_required_fields(request_type=service.UpdateSubnetRequest):
    if False:
        return 10
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_subnet._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_subnet._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_subnet(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_subnet_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_subnet._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('updateMask', 'subnet'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_subnet_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_update_subnet') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_update_subnet') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateSubnetRequest.pb(service.UpdateSubnetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.UpdateSubnetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_subnet(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_subnet_rest_bad_request(transport: str='rest', request_type=service.UpdateSubnetRequest):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'subnet': {'name': 'projects/sample1/locations/sample2/zones/sample3/subnets/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_subnet(request)

def test_update_subnet_rest_flattened():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'subnet': {'name': 'projects/sample1/locations/sample2/zones/sample3/subnets/sample4'}}
        mock_args = dict(subnet=resources.Subnet(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_subnet(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{subnet.name=projects/*/locations/*/zones/*/subnets/*}' % client.transport._host, args[1])

def test_update_subnet_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_subnet(service.UpdateSubnetRequest(), subnet=resources.Subnet(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_subnet_rest_error():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteSubnetRequest, dict])
def test_delete_subnet_rest(request_type):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/subnets/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_subnet(request)
    assert response.operation.name == 'operations/spam'

def test_delete_subnet_rest_required_fields(request_type=service.DeleteSubnetRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_subnet._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_subnet._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_subnet(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_subnet_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_subnet._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_subnet_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_delete_subnet') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_delete_subnet') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DeleteSubnetRequest.pb(service.DeleteSubnetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.DeleteSubnetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_subnet(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_subnet_rest_bad_request(transport: str='rest', request_type=service.DeleteSubnetRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/subnets/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_subnet(request)

def test_delete_subnet_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/zones/sample3/subnets/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_subnet(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/zones/*/subnets/*}' % client.transport._host, args[1])

def test_delete_subnet_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_subnet(service.DeleteSubnetRequest(), name='name_value')

def test_delete_subnet_rest_error():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListInterconnectsRequest, dict])
def test_list_interconnects_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListInterconnectsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListInterconnectsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_interconnects(request)
    assert isinstance(response, pagers.ListInterconnectsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_interconnects_rest_required_fields(request_type=service.ListInterconnectsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_interconnects._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_interconnects._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListInterconnectsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListInterconnectsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_interconnects(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_interconnects_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_interconnects._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_interconnects_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_list_interconnects') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_list_interconnects') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListInterconnectsRequest.pb(service.ListInterconnectsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListInterconnectsResponse.to_json(service.ListInterconnectsResponse())
        request = service.ListInterconnectsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListInterconnectsResponse()
        client.list_interconnects(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_interconnects_rest_bad_request(transport: str='rest', request_type=service.ListInterconnectsRequest):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_interconnects(request)

def test_list_interconnects_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListInterconnectsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListInterconnectsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_interconnects(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/zones/*}/interconnects' % client.transport._host, args[1])

def test_list_interconnects_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_interconnects(service.ListInterconnectsRequest(), parent='parent_value')

def test_list_interconnects_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListInterconnectsResponse(interconnects=[resources.Interconnect(), resources.Interconnect(), resources.Interconnect()], next_page_token='abc'), service.ListInterconnectsResponse(interconnects=[], next_page_token='def'), service.ListInterconnectsResponse(interconnects=[resources.Interconnect()], next_page_token='ghi'), service.ListInterconnectsResponse(interconnects=[resources.Interconnect(), resources.Interconnect()]))
        response = response + response
        response = tuple((service.ListInterconnectsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
        pager = client.list_interconnects(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Interconnect) for i in results))
        pages = list(client.list_interconnects(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetInterconnectRequest, dict])
def test_get_interconnect_rest(request_type):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/interconnects/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Interconnect(name='name_value', description='description_value', interconnect_type=resources.Interconnect.InterconnectType.DEDICATED, uuid='uuid_value', device_cloud_resource_name='device_cloud_resource_name_value', physical_ports=['physical_ports_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Interconnect.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_interconnect(request)
    assert isinstance(response, resources.Interconnect)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.interconnect_type == resources.Interconnect.InterconnectType.DEDICATED
    assert response.uuid == 'uuid_value'
    assert response.device_cloud_resource_name == 'device_cloud_resource_name_value'
    assert response.physical_ports == ['physical_ports_value']

def test_get_interconnect_rest_required_fields(request_type=service.GetInterconnectRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_interconnect._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_interconnect._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Interconnect()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Interconnect.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_interconnect(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_interconnect_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_interconnect._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_interconnect_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_get_interconnect') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_get_interconnect') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetInterconnectRequest.pb(service.GetInterconnectRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Interconnect.to_json(resources.Interconnect())
        request = service.GetInterconnectRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Interconnect()
        client.get_interconnect(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_interconnect_rest_bad_request(transport: str='rest', request_type=service.GetInterconnectRequest):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/interconnects/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_interconnect(request)

def test_get_interconnect_rest_flattened():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Interconnect()
        sample_request = {'name': 'projects/sample1/locations/sample2/zones/sample3/interconnects/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Interconnect.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_interconnect(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/zones/*/interconnects/*}' % client.transport._host, args[1])

def test_get_interconnect_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_interconnect(service.GetInterconnectRequest(), name='name_value')

def test_get_interconnect_rest_error():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DiagnoseInterconnectRequest, dict])
def test_diagnose_interconnect_rest(request_type):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/interconnects/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.DiagnoseInterconnectResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = service.DiagnoseInterconnectResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.diagnose_interconnect(request)
    assert isinstance(response, service.DiagnoseInterconnectResponse)

def test_diagnose_interconnect_rest_required_fields(request_type=service.DiagnoseInterconnectRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).diagnose_interconnect._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).diagnose_interconnect._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.DiagnoseInterconnectResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.DiagnoseInterconnectResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.diagnose_interconnect(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_diagnose_interconnect_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.diagnose_interconnect._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_diagnose_interconnect_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_diagnose_interconnect') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_diagnose_interconnect') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DiagnoseInterconnectRequest.pb(service.DiagnoseInterconnectRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.DiagnoseInterconnectResponse.to_json(service.DiagnoseInterconnectResponse())
        request = service.DiagnoseInterconnectRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.DiagnoseInterconnectResponse()
        client.diagnose_interconnect(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_diagnose_interconnect_rest_bad_request(transport: str='rest', request_type=service.DiagnoseInterconnectRequest):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/interconnects/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.diagnose_interconnect(request)

def test_diagnose_interconnect_rest_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.DiagnoseInterconnectResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/zones/sample3/interconnects/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.DiagnoseInterconnectResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.diagnose_interconnect(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/zones/*/interconnects/*}:diagnose' % client.transport._host, args[1])

def test_diagnose_interconnect_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.diagnose_interconnect(service.DiagnoseInterconnectRequest(), name='name_value')

def test_diagnose_interconnect_rest_error():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListInterconnectAttachmentsRequest, dict])
def test_list_interconnect_attachments_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListInterconnectAttachmentsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListInterconnectAttachmentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_interconnect_attachments(request)
    assert isinstance(response, pagers.ListInterconnectAttachmentsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_interconnect_attachments_rest_required_fields(request_type=service.ListInterconnectAttachmentsRequest):
    if False:
        return 10
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_interconnect_attachments._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_interconnect_attachments._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListInterconnectAttachmentsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListInterconnectAttachmentsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_interconnect_attachments(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_interconnect_attachments_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_interconnect_attachments._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_interconnect_attachments_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_list_interconnect_attachments') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_list_interconnect_attachments') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListInterconnectAttachmentsRequest.pb(service.ListInterconnectAttachmentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListInterconnectAttachmentsResponse.to_json(service.ListInterconnectAttachmentsResponse())
        request = service.ListInterconnectAttachmentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListInterconnectAttachmentsResponse()
        client.list_interconnect_attachments(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_interconnect_attachments_rest_bad_request(transport: str='rest', request_type=service.ListInterconnectAttachmentsRequest):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_interconnect_attachments(request)

def test_list_interconnect_attachments_rest_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListInterconnectAttachmentsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListInterconnectAttachmentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_interconnect_attachments(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/zones/*}/interconnectAttachments' % client.transport._host, args[1])

def test_list_interconnect_attachments_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_interconnect_attachments(service.ListInterconnectAttachmentsRequest(), parent='parent_value')

def test_list_interconnect_attachments_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment(), resources.InterconnectAttachment(), resources.InterconnectAttachment()], next_page_token='abc'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[], next_page_token='def'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment()], next_page_token='ghi'), service.ListInterconnectAttachmentsResponse(interconnect_attachments=[resources.InterconnectAttachment(), resources.InterconnectAttachment()]))
        response = response + response
        response = tuple((service.ListInterconnectAttachmentsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
        pager = client.list_interconnect_attachments(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.InterconnectAttachment) for i in results))
        pages = list(client.list_interconnect_attachments(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetInterconnectAttachmentRequest, dict])
def test_get_interconnect_attachment_rest(request_type):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/interconnectAttachments/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.InterconnectAttachment(name='name_value', description='description_value', interconnect='interconnect_value', network='network_value', vlan_id=733, mtu=342, state=resources.ResourceState.STATE_PENDING)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.InterconnectAttachment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_interconnect_attachment(request)
    assert isinstance(response, resources.InterconnectAttachment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.interconnect == 'interconnect_value'
    assert response.network == 'network_value'
    assert response.vlan_id == 733
    assert response.mtu == 342
    assert response.state == resources.ResourceState.STATE_PENDING

def test_get_interconnect_attachment_rest_required_fields(request_type=service.GetInterconnectAttachmentRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_interconnect_attachment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_interconnect_attachment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.InterconnectAttachment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.InterconnectAttachment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_interconnect_attachment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_interconnect_attachment_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_interconnect_attachment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_interconnect_attachment_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_get_interconnect_attachment') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_get_interconnect_attachment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetInterconnectAttachmentRequest.pb(service.GetInterconnectAttachmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.InterconnectAttachment.to_json(resources.InterconnectAttachment())
        request = service.GetInterconnectAttachmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.InterconnectAttachment()
        client.get_interconnect_attachment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_interconnect_attachment_rest_bad_request(transport: str='rest', request_type=service.GetInterconnectAttachmentRequest):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/interconnectAttachments/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_interconnect_attachment(request)

def test_get_interconnect_attachment_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.InterconnectAttachment()
        sample_request = {'name': 'projects/sample1/locations/sample2/zones/sample3/interconnectAttachments/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.InterconnectAttachment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_interconnect_attachment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/zones/*/interconnectAttachments/*}' % client.transport._host, args[1])

def test_get_interconnect_attachment_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_interconnect_attachment(service.GetInterconnectAttachmentRequest(), name='name_value')

def test_get_interconnect_attachment_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateInterconnectAttachmentRequest, dict])
def test_create_interconnect_attachment_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request_init['interconnect_attachment'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'interconnect': 'interconnect_value', 'network': 'network_value', 'vlan_id': 733, 'mtu': 342, 'state': 1}
    test_field = service.CreateInterconnectAttachmentRequest.meta.fields['interconnect_attachment']

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
    for (field, value) in request_init['interconnect_attachment'].items():
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
                for i in range(0, len(request_init['interconnect_attachment'][field])):
                    del request_init['interconnect_attachment'][field][i][subfield]
            else:
                del request_init['interconnect_attachment'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_interconnect_attachment(request)
    assert response.operation.name == 'operations/spam'

def test_create_interconnect_attachment_rest_required_fields(request_type=service.CreateInterconnectAttachmentRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['interconnect_attachment_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'interconnectAttachmentId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_interconnect_attachment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'interconnectAttachmentId' in jsonified_request
    assert jsonified_request['interconnectAttachmentId'] == request_init['interconnect_attachment_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['interconnectAttachmentId'] = 'interconnect_attachment_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_interconnect_attachment._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('interconnect_attachment_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'interconnectAttachmentId' in jsonified_request
    assert jsonified_request['interconnectAttachmentId'] == 'interconnect_attachment_id_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_interconnect_attachment(request)
            expected_params = [('interconnectAttachmentId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_interconnect_attachment_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_interconnect_attachment._get_unset_required_fields({})
    assert set(unset_fields) == set(('interconnectAttachmentId', 'requestId')) & set(('parent', 'interconnectAttachmentId', 'interconnectAttachment'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_interconnect_attachment_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_create_interconnect_attachment') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_create_interconnect_attachment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateInterconnectAttachmentRequest.pb(service.CreateInterconnectAttachmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateInterconnectAttachmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_interconnect_attachment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_interconnect_attachment_rest_bad_request(transport: str='rest', request_type=service.CreateInterconnectAttachmentRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_interconnect_attachment(request)

def test_create_interconnect_attachment_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
        mock_args = dict(parent='parent_value', interconnect_attachment=resources.InterconnectAttachment(name='name_value'), interconnect_attachment_id='interconnect_attachment_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_interconnect_attachment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/zones/*}/interconnectAttachments' % client.transport._host, args[1])

def test_create_interconnect_attachment_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_interconnect_attachment(service.CreateInterconnectAttachmentRequest(), parent='parent_value', interconnect_attachment=resources.InterconnectAttachment(name='name_value'), interconnect_attachment_id='interconnect_attachment_id_value')

def test_create_interconnect_attachment_rest_error():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteInterconnectAttachmentRequest, dict])
def test_delete_interconnect_attachment_rest(request_type):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/interconnectAttachments/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_interconnect_attachment(request)
    assert response.operation.name == 'operations/spam'

def test_delete_interconnect_attachment_rest_required_fields(request_type=service.DeleteInterconnectAttachmentRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_interconnect_attachment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_interconnect_attachment._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_interconnect_attachment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_interconnect_attachment_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_interconnect_attachment._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_interconnect_attachment_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_delete_interconnect_attachment') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_delete_interconnect_attachment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DeleteInterconnectAttachmentRequest.pb(service.DeleteInterconnectAttachmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.DeleteInterconnectAttachmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_interconnect_attachment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_interconnect_attachment_rest_bad_request(transport: str='rest', request_type=service.DeleteInterconnectAttachmentRequest):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/interconnectAttachments/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_interconnect_attachment(request)

def test_delete_interconnect_attachment_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/zones/sample3/interconnectAttachments/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_interconnect_attachment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/zones/*/interconnectAttachments/*}' % client.transport._host, args[1])

def test_delete_interconnect_attachment_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_interconnect_attachment(service.DeleteInterconnectAttachmentRequest(), name='name_value')

def test_delete_interconnect_attachment_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListRoutersRequest, dict])
def test_list_routers_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListRoutersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListRoutersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_routers(request)
    assert isinstance(response, pagers.ListRoutersPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_routers_rest_required_fields(request_type=service.ListRoutersRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_routers._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_routers._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListRoutersResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListRoutersResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_routers(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_routers_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_routers._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_routers_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_list_routers') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_list_routers') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListRoutersRequest.pb(service.ListRoutersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListRoutersResponse.to_json(service.ListRoutersResponse())
        request = service.ListRoutersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListRoutersResponse()
        client.list_routers(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_routers_rest_bad_request(transport: str='rest', request_type=service.ListRoutersRequest):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_routers(request)

def test_list_routers_rest_flattened():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListRoutersResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListRoutersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_routers(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/zones/*}/routers' % client.transport._host, args[1])

def test_list_routers_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_routers(service.ListRoutersRequest(), parent='parent_value')

def test_list_routers_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListRoutersResponse(routers=[resources.Router(), resources.Router(), resources.Router()], next_page_token='abc'), service.ListRoutersResponse(routers=[], next_page_token='def'), service.ListRoutersResponse(routers=[resources.Router()], next_page_token='ghi'), service.ListRoutersResponse(routers=[resources.Router(), resources.Router()]))
        response = response + response
        response = tuple((service.ListRoutersResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
        pager = client.list_routers(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Router) for i in results))
        pages = list(client.list_routers(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetRouterRequest, dict])
def test_get_router_rest(request_type):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/routers/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Router(name='name_value', description='description_value', network='network_value', state=resources.ResourceState.STATE_PENDING, route_advertisements=['route_advertisements_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Router.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_router(request)
    assert isinstance(response, resources.Router)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.network == 'network_value'
    assert response.state == resources.ResourceState.STATE_PENDING
    assert response.route_advertisements == ['route_advertisements_value']

def test_get_router_rest_required_fields(request_type=service.GetRouterRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_router._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_router._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Router()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Router.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_router(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_router_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_router._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_router_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_get_router') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_get_router') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetRouterRequest.pb(service.GetRouterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Router.to_json(resources.Router())
        request = service.GetRouterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Router()
        client.get_router(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_router_rest_bad_request(transport: str='rest', request_type=service.GetRouterRequest):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/routers/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_router(request)

def test_get_router_rest_flattened():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Router()
        sample_request = {'name': 'projects/sample1/locations/sample2/zones/sample3/routers/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Router.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_router(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/zones/*/routers/*}' % client.transport._host, args[1])

def test_get_router_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_router(service.GetRouterRequest(), name='name_value')

def test_get_router_rest_error():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DiagnoseRouterRequest, dict])
def test_diagnose_router_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/routers/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.DiagnoseRouterResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = service.DiagnoseRouterResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.diagnose_router(request)
    assert isinstance(response, service.DiagnoseRouterResponse)

def test_diagnose_router_rest_required_fields(request_type=service.DiagnoseRouterRequest):
    if False:
        return 10
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).diagnose_router._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).diagnose_router._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.DiagnoseRouterResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.DiagnoseRouterResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.diagnose_router(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_diagnose_router_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.diagnose_router._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_diagnose_router_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_diagnose_router') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_diagnose_router') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DiagnoseRouterRequest.pb(service.DiagnoseRouterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.DiagnoseRouterResponse.to_json(service.DiagnoseRouterResponse())
        request = service.DiagnoseRouterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.DiagnoseRouterResponse()
        client.diagnose_router(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_diagnose_router_rest_bad_request(transport: str='rest', request_type=service.DiagnoseRouterRequest):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/routers/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.diagnose_router(request)

def test_diagnose_router_rest_flattened():
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.DiagnoseRouterResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/zones/sample3/routers/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.DiagnoseRouterResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.diagnose_router(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/zones/*/routers/*}:diagnose' % client.transport._host, args[1])

def test_diagnose_router_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.diagnose_router(service.DiagnoseRouterRequest(), name='name_value')

def test_diagnose_router_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateRouterRequest, dict])
def test_create_router_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request_init['router'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'network': 'network_value', 'interface': [{'name': 'name_value', 'ipv4_cidr': 'ipv4_cidr_value', 'ipv6_cidr': 'ipv6_cidr_value', 'linked_interconnect_attachment': 'linked_interconnect_attachment_value', 'subnetwork': 'subnetwork_value', 'loopback_ip_addresses': ['loopback_ip_addresses_value1', 'loopback_ip_addresses_value2']}], 'bgp_peer': [{'name': 'name_value', 'interface': 'interface_value', 'interface_ipv4_cidr': 'interface_ipv4_cidr_value', 'interface_ipv6_cidr': 'interface_ipv6_cidr_value', 'peer_ipv4_cidr': 'peer_ipv4_cidr_value', 'peer_ipv6_cidr': 'peer_ipv6_cidr_value', 'peer_asn': 845, 'local_asn': 940}], 'bgp': {'asn': 322, 'keepalive_interval_in_seconds': 3070}, 'state': 1, 'route_advertisements': ['route_advertisements_value1', 'route_advertisements_value2']}
    test_field = service.CreateRouterRequest.meta.fields['router']

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
    for (field, value) in request_init['router'].items():
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
                for i in range(0, len(request_init['router'][field])):
                    del request_init['router'][field][i][subfield]
            else:
                del request_init['router'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_router(request)
    assert response.operation.name == 'operations/spam'

def test_create_router_rest_required_fields(request_type=service.CreateRouterRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['router_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'routerId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_router._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'routerId' in jsonified_request
    assert jsonified_request['routerId'] == request_init['router_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['routerId'] = 'router_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_router._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'router_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'routerId' in jsonified_request
    assert jsonified_request['routerId'] == 'router_id_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_router(request)
            expected_params = [('routerId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_router_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_router._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'routerId')) & set(('parent', 'routerId', 'router'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_router_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_create_router') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_create_router') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateRouterRequest.pb(service.CreateRouterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateRouterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_router(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_router_rest_bad_request(transport: str='rest', request_type=service.CreateRouterRequest):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_router(request)

def test_create_router_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/zones/sample3'}
        mock_args = dict(parent='parent_value', router=resources.Router(name='name_value'), router_id='router_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_router(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/zones/*}/routers' % client.transport._host, args[1])

def test_create_router_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_router(service.CreateRouterRequest(), parent='parent_value', router=resources.Router(name='name_value'), router_id='router_id_value')

def test_create_router_rest_error():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateRouterRequest, dict])
def test_update_router_rest(request_type):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'router': {'name': 'projects/sample1/locations/sample2/zones/sample3/routers/sample4'}}
    request_init['router'] = {'name': 'projects/sample1/locations/sample2/zones/sample3/routers/sample4', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'network': 'network_value', 'interface': [{'name': 'name_value', 'ipv4_cidr': 'ipv4_cidr_value', 'ipv6_cidr': 'ipv6_cidr_value', 'linked_interconnect_attachment': 'linked_interconnect_attachment_value', 'subnetwork': 'subnetwork_value', 'loopback_ip_addresses': ['loopback_ip_addresses_value1', 'loopback_ip_addresses_value2']}], 'bgp_peer': [{'name': 'name_value', 'interface': 'interface_value', 'interface_ipv4_cidr': 'interface_ipv4_cidr_value', 'interface_ipv6_cidr': 'interface_ipv6_cidr_value', 'peer_ipv4_cidr': 'peer_ipv4_cidr_value', 'peer_ipv6_cidr': 'peer_ipv6_cidr_value', 'peer_asn': 845, 'local_asn': 940}], 'bgp': {'asn': 322, 'keepalive_interval_in_seconds': 3070}, 'state': 1, 'route_advertisements': ['route_advertisements_value1', 'route_advertisements_value2']}
    test_field = service.UpdateRouterRequest.meta.fields['router']

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
    for (field, value) in request_init['router'].items():
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
                for i in range(0, len(request_init['router'][field])):
                    del request_init['router'][field][i][subfield]
            else:
                del request_init['router'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_router(request)
    assert response.operation.name == 'operations/spam'

def test_update_router_rest_required_fields(request_type=service.UpdateRouterRequest):
    if False:
        return 10
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_router._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_router._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_router(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_router_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_router._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('updateMask', 'router'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_router_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_update_router') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_update_router') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateRouterRequest.pb(service.UpdateRouterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.UpdateRouterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_router(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_router_rest_bad_request(transport: str='rest', request_type=service.UpdateRouterRequest):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'router': {'name': 'projects/sample1/locations/sample2/zones/sample3/routers/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_router(request)

def test_update_router_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'router': {'name': 'projects/sample1/locations/sample2/zones/sample3/routers/sample4'}}
        mock_args = dict(router=resources.Router(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_router(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{router.name=projects/*/locations/*/zones/*/routers/*}' % client.transport._host, args[1])

def test_update_router_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_router(service.UpdateRouterRequest(), router=resources.Router(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_router_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteRouterRequest, dict])
def test_delete_router_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/routers/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_router(request)
    assert response.operation.name == 'operations/spam'

def test_delete_router_rest_required_fields(request_type=service.DeleteRouterRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.EdgeNetworkRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_router._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_router._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_router(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_router_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_router._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_router_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EdgeNetworkRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EdgeNetworkRestInterceptor())
    client = EdgeNetworkClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EdgeNetworkRestInterceptor, 'post_delete_router') as post, mock.patch.object(transports.EdgeNetworkRestInterceptor, 'pre_delete_router') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DeleteRouterRequest.pb(service.DeleteRouterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.DeleteRouterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_router(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_router_rest_bad_request(transport: str='rest', request_type=service.DeleteRouterRequest):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/zones/sample3/routers/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_router(request)

def test_delete_router_rest_flattened():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/zones/sample3/routers/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_router(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/zones/*/routers/*}' % client.transport._host, args[1])

def test_delete_router_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_router(service.DeleteRouterRequest(), name='name_value')

def test_delete_router_rest_error():
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.EdgeNetworkGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.EdgeNetworkGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EdgeNetworkClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.EdgeNetworkGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = EdgeNetworkClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = EdgeNetworkClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.EdgeNetworkGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EdgeNetworkClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.EdgeNetworkGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = EdgeNetworkClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.EdgeNetworkGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.EdgeNetworkGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.EdgeNetworkGrpcTransport, transports.EdgeNetworkGrpcAsyncIOTransport, transports.EdgeNetworkRestTransport])
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
        while True:
            i = 10
    transport = EdgeNetworkClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.EdgeNetworkGrpcTransport)

def test_edge_network_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.EdgeNetworkTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_edge_network_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.edgenetwork_v1.services.edge_network.transports.EdgeNetworkTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.EdgeNetworkTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('initialize_zone', 'list_zones', 'get_zone', 'list_networks', 'get_network', 'diagnose_network', 'create_network', 'delete_network', 'list_subnets', 'get_subnet', 'create_subnet', 'update_subnet', 'delete_subnet', 'list_interconnects', 'get_interconnect', 'diagnose_interconnect', 'list_interconnect_attachments', 'get_interconnect_attachment', 'create_interconnect_attachment', 'delete_interconnect_attachment', 'list_routers', 'get_router', 'diagnose_router', 'create_router', 'update_router', 'delete_router', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_edge_network_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.edgenetwork_v1.services.edge_network.transports.EdgeNetworkTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.EdgeNetworkTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_edge_network_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.edgenetwork_v1.services.edge_network.transports.EdgeNetworkTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.EdgeNetworkTransport()
        adc.assert_called_once()

def test_edge_network_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        EdgeNetworkClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.EdgeNetworkGrpcTransport, transports.EdgeNetworkGrpcAsyncIOTransport])
def test_edge_network_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.EdgeNetworkGrpcTransport, transports.EdgeNetworkGrpcAsyncIOTransport, transports.EdgeNetworkRestTransport])
def test_edge_network_transport_auth_gdch_credentials(transport_class):
    if False:
        print('Hello World!')
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.EdgeNetworkGrpcTransport, grpc_helpers), (transports.EdgeNetworkGrpcAsyncIOTransport, grpc_helpers_async)])
def test_edge_network_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('edgenetwork.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='edgenetwork.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.EdgeNetworkGrpcTransport, transports.EdgeNetworkGrpcAsyncIOTransport])
def test_edge_network_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_edge_network_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.EdgeNetworkRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_edge_network_rest_lro_client():
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_edge_network_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='edgenetwork.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('edgenetwork.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://edgenetwork.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_edge_network_host_with_port(transport_name):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='edgenetwork.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('edgenetwork.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://edgenetwork.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_edge_network_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = EdgeNetworkClient(credentials=creds1, transport=transport_name)
    client2 = EdgeNetworkClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.initialize_zone._session
    session2 = client2.transport.initialize_zone._session
    assert session1 != session2
    session1 = client1.transport.list_zones._session
    session2 = client2.transport.list_zones._session
    assert session1 != session2
    session1 = client1.transport.get_zone._session
    session2 = client2.transport.get_zone._session
    assert session1 != session2
    session1 = client1.transport.list_networks._session
    session2 = client2.transport.list_networks._session
    assert session1 != session2
    session1 = client1.transport.get_network._session
    session2 = client2.transport.get_network._session
    assert session1 != session2
    session1 = client1.transport.diagnose_network._session
    session2 = client2.transport.diagnose_network._session
    assert session1 != session2
    session1 = client1.transport.create_network._session
    session2 = client2.transport.create_network._session
    assert session1 != session2
    session1 = client1.transport.delete_network._session
    session2 = client2.transport.delete_network._session
    assert session1 != session2
    session1 = client1.transport.list_subnets._session
    session2 = client2.transport.list_subnets._session
    assert session1 != session2
    session1 = client1.transport.get_subnet._session
    session2 = client2.transport.get_subnet._session
    assert session1 != session2
    session1 = client1.transport.create_subnet._session
    session2 = client2.transport.create_subnet._session
    assert session1 != session2
    session1 = client1.transport.update_subnet._session
    session2 = client2.transport.update_subnet._session
    assert session1 != session2
    session1 = client1.transport.delete_subnet._session
    session2 = client2.transport.delete_subnet._session
    assert session1 != session2
    session1 = client1.transport.list_interconnects._session
    session2 = client2.transport.list_interconnects._session
    assert session1 != session2
    session1 = client1.transport.get_interconnect._session
    session2 = client2.transport.get_interconnect._session
    assert session1 != session2
    session1 = client1.transport.diagnose_interconnect._session
    session2 = client2.transport.diagnose_interconnect._session
    assert session1 != session2
    session1 = client1.transport.list_interconnect_attachments._session
    session2 = client2.transport.list_interconnect_attachments._session
    assert session1 != session2
    session1 = client1.transport.get_interconnect_attachment._session
    session2 = client2.transport.get_interconnect_attachment._session
    assert session1 != session2
    session1 = client1.transport.create_interconnect_attachment._session
    session2 = client2.transport.create_interconnect_attachment._session
    assert session1 != session2
    session1 = client1.transport.delete_interconnect_attachment._session
    session2 = client2.transport.delete_interconnect_attachment._session
    assert session1 != session2
    session1 = client1.transport.list_routers._session
    session2 = client2.transport.list_routers._session
    assert session1 != session2
    session1 = client1.transport.get_router._session
    session2 = client2.transport.get_router._session
    assert session1 != session2
    session1 = client1.transport.diagnose_router._session
    session2 = client2.transport.diagnose_router._session
    assert session1 != session2
    session1 = client1.transport.create_router._session
    session2 = client2.transport.create_router._session
    assert session1 != session2
    session1 = client1.transport.update_router._session
    session2 = client2.transport.update_router._session
    assert session1 != session2
    session1 = client1.transport.delete_router._session
    session2 = client2.transport.delete_router._session
    assert session1 != session2

def test_edge_network_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.EdgeNetworkGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_edge_network_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.EdgeNetworkGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.EdgeNetworkGrpcTransport, transports.EdgeNetworkGrpcAsyncIOTransport])
def test_edge_network_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        i = 10
        return i + 15
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

@pytest.mark.parametrize('transport_class', [transports.EdgeNetworkGrpcTransport, transports.EdgeNetworkGrpcAsyncIOTransport])
def test_edge_network_transport_channel_mtls_with_adc(transport_class):
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

def test_edge_network_grpc_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_edge_network_grpc_lro_async_client():
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_interconnect_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    zone = 'whelk'
    interconnect = 'octopus'
    expected = 'projects/{project}/locations/{location}/zones/{zone}/interconnects/{interconnect}'.format(project=project, location=location, zone=zone, interconnect=interconnect)
    actual = EdgeNetworkClient.interconnect_path(project, location, zone, interconnect)
    assert expected == actual

def test_parse_interconnect_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'oyster', 'location': 'nudibranch', 'zone': 'cuttlefish', 'interconnect': 'mussel'}
    path = EdgeNetworkClient.interconnect_path(**expected)
    actual = EdgeNetworkClient.parse_interconnect_path(path)
    assert expected == actual

def test_interconnect_attachment_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    zone = 'scallop'
    interconnect_attachment = 'abalone'
    expected = 'projects/{project}/locations/{location}/zones/{zone}/interconnectAttachments/{interconnect_attachment}'.format(project=project, location=location, zone=zone, interconnect_attachment=interconnect_attachment)
    actual = EdgeNetworkClient.interconnect_attachment_path(project, location, zone, interconnect_attachment)
    assert expected == actual

def test_parse_interconnect_attachment_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'squid', 'location': 'clam', 'zone': 'whelk', 'interconnect_attachment': 'octopus'}
    path = EdgeNetworkClient.interconnect_attachment_path(**expected)
    actual = EdgeNetworkClient.parse_interconnect_attachment_path(path)
    assert expected == actual

def test_network_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    location = 'nudibranch'
    zone = 'cuttlefish'
    network = 'mussel'
    expected = 'projects/{project}/locations/{location}/zones/{zone}/networks/{network}'.format(project=project, location=location, zone=zone, network=network)
    actual = EdgeNetworkClient.network_path(project, location, zone, network)
    assert expected == actual

def test_parse_network_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'winkle', 'location': 'nautilus', 'zone': 'scallop', 'network': 'abalone'}
    path = EdgeNetworkClient.network_path(**expected)
    actual = EdgeNetworkClient.parse_network_path(path)
    assert expected == actual

def test_router_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    zone = 'whelk'
    router = 'octopus'
    expected = 'projects/{project}/locations/{location}/zones/{zone}/routers/{router}'.format(project=project, location=location, zone=zone, router=router)
    actual = EdgeNetworkClient.router_path(project, location, zone, router)
    assert expected == actual

def test_parse_router_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'oyster', 'location': 'nudibranch', 'zone': 'cuttlefish', 'router': 'mussel'}
    path = EdgeNetworkClient.router_path(**expected)
    actual = EdgeNetworkClient.parse_router_path(path)
    assert expected == actual

def test_subnet_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    location = 'nautilus'
    zone = 'scallop'
    subnet = 'abalone'
    expected = 'projects/{project}/locations/{location}/zones/{zone}/subnets/{subnet}'.format(project=project, location=location, zone=zone, subnet=subnet)
    actual = EdgeNetworkClient.subnet_path(project, location, zone, subnet)
    assert expected == actual

def test_parse_subnet_path():
    if False:
        return 10
    expected = {'project': 'squid', 'location': 'clam', 'zone': 'whelk', 'subnet': 'octopus'}
    path = EdgeNetworkClient.subnet_path(**expected)
    actual = EdgeNetworkClient.parse_subnet_path(path)
    assert expected == actual

def test_zone_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    location = 'nudibranch'
    zone = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/zones/{zone}'.format(project=project, location=location, zone=zone)
    actual = EdgeNetworkClient.zone_path(project, location, zone)
    assert expected == actual

def test_parse_zone_path():
    if False:
        return 10
    expected = {'project': 'mussel', 'location': 'winkle', 'zone': 'nautilus'}
    path = EdgeNetworkClient.zone_path(**expected)
    actual = EdgeNetworkClient.parse_zone_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = EdgeNetworkClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'abalone'}
    path = EdgeNetworkClient.common_billing_account_path(**expected)
    actual = EdgeNetworkClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = EdgeNetworkClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'clam'}
    path = EdgeNetworkClient.common_folder_path(**expected)
    actual = EdgeNetworkClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = EdgeNetworkClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'octopus'}
    path = EdgeNetworkClient.common_organization_path(**expected)
    actual = EdgeNetworkClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = EdgeNetworkClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nudibranch'}
    path = EdgeNetworkClient.common_project_path(**expected)
    actual = EdgeNetworkClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = EdgeNetworkClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = EdgeNetworkClient.common_location_path(**expected)
    actual = EdgeNetworkClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.EdgeNetworkTransport, '_prep_wrapped_messages') as prep:
        client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.EdgeNetworkTransport, '_prep_wrapped_messages') as prep:
        transport_class = EdgeNetworkClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = EdgeNetworkAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = EdgeNetworkClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(EdgeNetworkClient, transports.EdgeNetworkGrpcTransport), (EdgeNetworkAsyncClient, transports.EdgeNetworkGrpcAsyncIOTransport)])
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