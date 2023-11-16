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
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
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
from google.cloud.beyondcorp_appconnections_v1.services.app_connections_service import AppConnectionsServiceAsyncClient, AppConnectionsServiceClient, pagers, transports
from google.cloud.beyondcorp_appconnections_v1.types import app_connections_service

def client_cert_source_callback():
    if False:
        print('Hello World!')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        print('Hello World!')
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
    assert AppConnectionsServiceClient._get_default_mtls_endpoint(None) is None
    assert AppConnectionsServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert AppConnectionsServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert AppConnectionsServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert AppConnectionsServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert AppConnectionsServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(AppConnectionsServiceClient, 'grpc'), (AppConnectionsServiceAsyncClient, 'grpc_asyncio'), (AppConnectionsServiceClient, 'rest')])
def test_app_connections_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('beyondcorp.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://beyondcorp.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.AppConnectionsServiceGrpcTransport, 'grpc'), (transports.AppConnectionsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.AppConnectionsServiceRestTransport, 'rest')])
def test_app_connections_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(AppConnectionsServiceClient, 'grpc'), (AppConnectionsServiceAsyncClient, 'grpc_asyncio'), (AppConnectionsServiceClient, 'rest')])
def test_app_connections_service_client_from_service_account_file(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_file') as factory:
        factory.return_value = creds
        client = client_class.from_service_account_file('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        client = client_class.from_service_account_json('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('beyondcorp.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://beyondcorp.googleapis.com')

def test_app_connections_service_client_get_transport_class():
    if False:
        return 10
    transport = AppConnectionsServiceClient.get_transport_class()
    available_transports = [transports.AppConnectionsServiceGrpcTransport, transports.AppConnectionsServiceRestTransport]
    assert transport in available_transports
    transport = AppConnectionsServiceClient.get_transport_class('grpc')
    assert transport == transports.AppConnectionsServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AppConnectionsServiceClient, transports.AppConnectionsServiceGrpcTransport, 'grpc'), (AppConnectionsServiceAsyncClient, transports.AppConnectionsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (AppConnectionsServiceClient, transports.AppConnectionsServiceRestTransport, 'rest')])
@mock.patch.object(AppConnectionsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AppConnectionsServiceClient))
@mock.patch.object(AppConnectionsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AppConnectionsServiceAsyncClient))
def test_app_connections_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(AppConnectionsServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(AppConnectionsServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(AppConnectionsServiceClient, transports.AppConnectionsServiceGrpcTransport, 'grpc', 'true'), (AppConnectionsServiceAsyncClient, transports.AppConnectionsServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (AppConnectionsServiceClient, transports.AppConnectionsServiceGrpcTransport, 'grpc', 'false'), (AppConnectionsServiceAsyncClient, transports.AppConnectionsServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (AppConnectionsServiceClient, transports.AppConnectionsServiceRestTransport, 'rest', 'true'), (AppConnectionsServiceClient, transports.AppConnectionsServiceRestTransport, 'rest', 'false')])
@mock.patch.object(AppConnectionsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AppConnectionsServiceClient))
@mock.patch.object(AppConnectionsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AppConnectionsServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_app_connections_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [AppConnectionsServiceClient, AppConnectionsServiceAsyncClient])
@mock.patch.object(AppConnectionsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AppConnectionsServiceClient))
@mock.patch.object(AppConnectionsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AppConnectionsServiceAsyncClient))
def test_app_connections_service_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        return 10
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AppConnectionsServiceClient, transports.AppConnectionsServiceGrpcTransport, 'grpc'), (AppConnectionsServiceAsyncClient, transports.AppConnectionsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (AppConnectionsServiceClient, transports.AppConnectionsServiceRestTransport, 'rest')])
def test_app_connections_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AppConnectionsServiceClient, transports.AppConnectionsServiceGrpcTransport, 'grpc', grpc_helpers), (AppConnectionsServiceAsyncClient, transports.AppConnectionsServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (AppConnectionsServiceClient, transports.AppConnectionsServiceRestTransport, 'rest', None)])
def test_app_connections_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_app_connections_service_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.beyondcorp_appconnections_v1.services.app_connections_service.transports.AppConnectionsServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = AppConnectionsServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AppConnectionsServiceClient, transports.AppConnectionsServiceGrpcTransport, 'grpc', grpc_helpers), (AppConnectionsServiceAsyncClient, transports.AppConnectionsServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_app_connections_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('beyondcorp.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='beyondcorp.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [app_connections_service.ListAppConnectionsRequest, dict])
def test_list_app_connections(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_app_connections), '__call__') as call:
        call.return_value = app_connections_service.ListAppConnectionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_app_connections(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.ListAppConnectionsRequest()
    assert isinstance(response, pagers.ListAppConnectionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_app_connections_empty_call():
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_app_connections), '__call__') as call:
        client.list_app_connections()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.ListAppConnectionsRequest()

@pytest.mark.asyncio
async def test_list_app_connections_async(transport: str='grpc_asyncio', request_type=app_connections_service.ListAppConnectionsRequest):
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_app_connections), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(app_connections_service.ListAppConnectionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_app_connections(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.ListAppConnectionsRequest()
    assert isinstance(response, pagers.ListAppConnectionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_app_connections_async_from_dict():
    await test_list_app_connections_async(request_type=dict)

def test_list_app_connections_field_headers():
    if False:
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = app_connections_service.ListAppConnectionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_app_connections), '__call__') as call:
        call.return_value = app_connections_service.ListAppConnectionsResponse()
        client.list_app_connections(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_app_connections_field_headers_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = app_connections_service.ListAppConnectionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_app_connections), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(app_connections_service.ListAppConnectionsResponse())
        await client.list_app_connections(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_app_connections_flattened():
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_app_connections), '__call__') as call:
        call.return_value = app_connections_service.ListAppConnectionsResponse()
        client.list_app_connections(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_app_connections_flattened_error():
    if False:
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_app_connections(app_connections_service.ListAppConnectionsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_app_connections_flattened_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_app_connections), '__call__') as call:
        call.return_value = app_connections_service.ListAppConnectionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(app_connections_service.ListAppConnectionsResponse())
        response = await client.list_app_connections(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_app_connections_flattened_error_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_app_connections(app_connections_service.ListAppConnectionsRequest(), parent='parent_value')

def test_list_app_connections_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_app_connections), '__call__') as call:
        call.side_effect = (app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection(), app_connections_service.AppConnection(), app_connections_service.AppConnection()], next_page_token='abc'), app_connections_service.ListAppConnectionsResponse(app_connections=[], next_page_token='def'), app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection()], next_page_token='ghi'), app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection(), app_connections_service.AppConnection()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_app_connections(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, app_connections_service.AppConnection) for i in results))

def test_list_app_connections_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_app_connections), '__call__') as call:
        call.side_effect = (app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection(), app_connections_service.AppConnection(), app_connections_service.AppConnection()], next_page_token='abc'), app_connections_service.ListAppConnectionsResponse(app_connections=[], next_page_token='def'), app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection()], next_page_token='ghi'), app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection(), app_connections_service.AppConnection()]), RuntimeError)
        pages = list(client.list_app_connections(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_app_connections_async_pager():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_app_connections), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection(), app_connections_service.AppConnection(), app_connections_service.AppConnection()], next_page_token='abc'), app_connections_service.ListAppConnectionsResponse(app_connections=[], next_page_token='def'), app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection()], next_page_token='ghi'), app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection(), app_connections_service.AppConnection()]), RuntimeError)
        async_pager = await client.list_app_connections(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, app_connections_service.AppConnection) for i in responses))

@pytest.mark.asyncio
async def test_list_app_connections_async_pages():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_app_connections), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection(), app_connections_service.AppConnection(), app_connections_service.AppConnection()], next_page_token='abc'), app_connections_service.ListAppConnectionsResponse(app_connections=[], next_page_token='def'), app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection()], next_page_token='ghi'), app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection(), app_connections_service.AppConnection()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_app_connections(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [app_connections_service.GetAppConnectionRequest, dict])
def test_get_app_connection(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_app_connection), '__call__') as call:
        call.return_value = app_connections_service.AppConnection(name='name_value', display_name='display_name_value', uid='uid_value', type_=app_connections_service.AppConnection.Type.TCP_PROXY, connectors=['connectors_value'], state=app_connections_service.AppConnection.State.CREATING)
        response = client.get_app_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.GetAppConnectionRequest()
    assert isinstance(response, app_connections_service.AppConnection)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.type_ == app_connections_service.AppConnection.Type.TCP_PROXY
    assert response.connectors == ['connectors_value']
    assert response.state == app_connections_service.AppConnection.State.CREATING

def test_get_app_connection_empty_call():
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_app_connection), '__call__') as call:
        client.get_app_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.GetAppConnectionRequest()

@pytest.mark.asyncio
async def test_get_app_connection_async(transport: str='grpc_asyncio', request_type=app_connections_service.GetAppConnectionRequest):
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_app_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(app_connections_service.AppConnection(name='name_value', display_name='display_name_value', uid='uid_value', type_=app_connections_service.AppConnection.Type.TCP_PROXY, connectors=['connectors_value'], state=app_connections_service.AppConnection.State.CREATING))
        response = await client.get_app_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.GetAppConnectionRequest()
    assert isinstance(response, app_connections_service.AppConnection)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.type_ == app_connections_service.AppConnection.Type.TCP_PROXY
    assert response.connectors == ['connectors_value']
    assert response.state == app_connections_service.AppConnection.State.CREATING

@pytest.mark.asyncio
async def test_get_app_connection_async_from_dict():
    await test_get_app_connection_async(request_type=dict)

def test_get_app_connection_field_headers():
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = app_connections_service.GetAppConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_app_connection), '__call__') as call:
        call.return_value = app_connections_service.AppConnection()
        client.get_app_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_app_connection_field_headers_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = app_connections_service.GetAppConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_app_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(app_connections_service.AppConnection())
        await client.get_app_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_app_connection_flattened():
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_app_connection), '__call__') as call:
        call.return_value = app_connections_service.AppConnection()
        client.get_app_connection(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_app_connection_flattened_error():
    if False:
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_app_connection(app_connections_service.GetAppConnectionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_app_connection_flattened_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_app_connection), '__call__') as call:
        call.return_value = app_connections_service.AppConnection()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(app_connections_service.AppConnection())
        response = await client.get_app_connection(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_app_connection_flattened_error_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_app_connection(app_connections_service.GetAppConnectionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [app_connections_service.CreateAppConnectionRequest, dict])
def test_create_app_connection(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_app_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_app_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.CreateAppConnectionRequest()
    assert isinstance(response, future.Future)

def test_create_app_connection_empty_call():
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_app_connection), '__call__') as call:
        client.create_app_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.CreateAppConnectionRequest()

@pytest.mark.asyncio
async def test_create_app_connection_async(transport: str='grpc_asyncio', request_type=app_connections_service.CreateAppConnectionRequest):
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_app_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_app_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.CreateAppConnectionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_app_connection_async_from_dict():
    await test_create_app_connection_async(request_type=dict)

def test_create_app_connection_field_headers():
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = app_connections_service.CreateAppConnectionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_app_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_app_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_app_connection_field_headers_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = app_connections_service.CreateAppConnectionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_app_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_app_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_app_connection_flattened():
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_app_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_app_connection(parent='parent_value', app_connection=app_connections_service.AppConnection(name='name_value'), app_connection_id='app_connection_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].app_connection
        mock_val = app_connections_service.AppConnection(name='name_value')
        assert arg == mock_val
        arg = args[0].app_connection_id
        mock_val = 'app_connection_id_value'
        assert arg == mock_val

def test_create_app_connection_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_app_connection(app_connections_service.CreateAppConnectionRequest(), parent='parent_value', app_connection=app_connections_service.AppConnection(name='name_value'), app_connection_id='app_connection_id_value')

@pytest.mark.asyncio
async def test_create_app_connection_flattened_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_app_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_app_connection(parent='parent_value', app_connection=app_connections_service.AppConnection(name='name_value'), app_connection_id='app_connection_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].app_connection
        mock_val = app_connections_service.AppConnection(name='name_value')
        assert arg == mock_val
        arg = args[0].app_connection_id
        mock_val = 'app_connection_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_app_connection_flattened_error_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_app_connection(app_connections_service.CreateAppConnectionRequest(), parent='parent_value', app_connection=app_connections_service.AppConnection(name='name_value'), app_connection_id='app_connection_id_value')

@pytest.mark.parametrize('request_type', [app_connections_service.UpdateAppConnectionRequest, dict])
def test_update_app_connection(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_app_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_app_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.UpdateAppConnectionRequest()
    assert isinstance(response, future.Future)

def test_update_app_connection_empty_call():
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_app_connection), '__call__') as call:
        client.update_app_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.UpdateAppConnectionRequest()

@pytest.mark.asyncio
async def test_update_app_connection_async(transport: str='grpc_asyncio', request_type=app_connections_service.UpdateAppConnectionRequest):
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_app_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_app_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.UpdateAppConnectionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_app_connection_async_from_dict():
    await test_update_app_connection_async(request_type=dict)

def test_update_app_connection_field_headers():
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = app_connections_service.UpdateAppConnectionRequest()
    request.app_connection.name = 'name_value'
    with mock.patch.object(type(client.transport.update_app_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_app_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'app_connection.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_app_connection_field_headers_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = app_connections_service.UpdateAppConnectionRequest()
    request.app_connection.name = 'name_value'
    with mock.patch.object(type(client.transport.update_app_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_app_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'app_connection.name=name_value') in kw['metadata']

def test_update_app_connection_flattened():
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_app_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_app_connection(app_connection=app_connections_service.AppConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].app_connection
        mock_val = app_connections_service.AppConnection(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_app_connection_flattened_error():
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_app_connection(app_connections_service.UpdateAppConnectionRequest(), app_connection=app_connections_service.AppConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_app_connection_flattened_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_app_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_app_connection(app_connection=app_connections_service.AppConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].app_connection
        mock_val = app_connections_service.AppConnection(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_app_connection_flattened_error_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_app_connection(app_connections_service.UpdateAppConnectionRequest(), app_connection=app_connections_service.AppConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [app_connections_service.DeleteAppConnectionRequest, dict])
def test_delete_app_connection(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_app_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_app_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.DeleteAppConnectionRequest()
    assert isinstance(response, future.Future)

def test_delete_app_connection_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_app_connection), '__call__') as call:
        client.delete_app_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.DeleteAppConnectionRequest()

@pytest.mark.asyncio
async def test_delete_app_connection_async(transport: str='grpc_asyncio', request_type=app_connections_service.DeleteAppConnectionRequest):
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_app_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_app_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.DeleteAppConnectionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_app_connection_async_from_dict():
    await test_delete_app_connection_async(request_type=dict)

def test_delete_app_connection_field_headers():
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = app_connections_service.DeleteAppConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_app_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_app_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_app_connection_field_headers_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = app_connections_service.DeleteAppConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_app_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_app_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_app_connection_flattened():
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_app_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_app_connection(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_app_connection_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_app_connection(app_connections_service.DeleteAppConnectionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_app_connection_flattened_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_app_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_app_connection(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_app_connection_flattened_error_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_app_connection(app_connections_service.DeleteAppConnectionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [app_connections_service.ResolveAppConnectionsRequest, dict])
def test_resolve_app_connections(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resolve_app_connections), '__call__') as call:
        call.return_value = app_connections_service.ResolveAppConnectionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.resolve_app_connections(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.ResolveAppConnectionsRequest()
    assert isinstance(response, pagers.ResolveAppConnectionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_resolve_app_connections_empty_call():
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.resolve_app_connections), '__call__') as call:
        client.resolve_app_connections()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.ResolveAppConnectionsRequest()

@pytest.mark.asyncio
async def test_resolve_app_connections_async(transport: str='grpc_asyncio', request_type=app_connections_service.ResolveAppConnectionsRequest):
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resolve_app_connections), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(app_connections_service.ResolveAppConnectionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.resolve_app_connections(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == app_connections_service.ResolveAppConnectionsRequest()
    assert isinstance(response, pagers.ResolveAppConnectionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_resolve_app_connections_async_from_dict():
    await test_resolve_app_connections_async(request_type=dict)

def test_resolve_app_connections_field_headers():
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = app_connections_service.ResolveAppConnectionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.resolve_app_connections), '__call__') as call:
        call.return_value = app_connections_service.ResolveAppConnectionsResponse()
        client.resolve_app_connections(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_resolve_app_connections_field_headers_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = app_connections_service.ResolveAppConnectionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.resolve_app_connections), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(app_connections_service.ResolveAppConnectionsResponse())
        await client.resolve_app_connections(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_resolve_app_connections_flattened():
    if False:
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.resolve_app_connections), '__call__') as call:
        call.return_value = app_connections_service.ResolveAppConnectionsResponse()
        client.resolve_app_connections(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_resolve_app_connections_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.resolve_app_connections(app_connections_service.ResolveAppConnectionsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_resolve_app_connections_flattened_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.resolve_app_connections), '__call__') as call:
        call.return_value = app_connections_service.ResolveAppConnectionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(app_connections_service.ResolveAppConnectionsResponse())
        response = await client.resolve_app_connections(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_resolve_app_connections_flattened_error_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.resolve_app_connections(app_connections_service.ResolveAppConnectionsRequest(), parent='parent_value')

def test_resolve_app_connections_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.resolve_app_connections), '__call__') as call:
        call.side_effect = (app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()], next_page_token='abc'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[], next_page_token='def'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()], next_page_token='ghi'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.resolve_app_connections(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails) for i in results))

def test_resolve_app_connections_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.resolve_app_connections), '__call__') as call:
        call.side_effect = (app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()], next_page_token='abc'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[], next_page_token='def'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()], next_page_token='ghi'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()]), RuntimeError)
        pages = list(client.resolve_app_connections(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_resolve_app_connections_async_pager():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.resolve_app_connections), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()], next_page_token='abc'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[], next_page_token='def'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()], next_page_token='ghi'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()]), RuntimeError)
        async_pager = await client.resolve_app_connections(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails) for i in responses))

@pytest.mark.asyncio
async def test_resolve_app_connections_async_pages():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.resolve_app_connections), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()], next_page_token='abc'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[], next_page_token='def'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()], next_page_token='ghi'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()]), RuntimeError)
        pages = []
        async for page_ in (await client.resolve_app_connections(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [app_connections_service.ListAppConnectionsRequest, dict])
def test_list_app_connections_rest(request_type):
    if False:
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = app_connections_service.ListAppConnectionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = app_connections_service.ListAppConnectionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_app_connections(request)
    assert isinstance(response, pagers.ListAppConnectionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_app_connections_rest_required_fields(request_type=app_connections_service.ListAppConnectionsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AppConnectionsServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_app_connections._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_app_connections._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = app_connections_service.ListAppConnectionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = app_connections_service.ListAppConnectionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_app_connections(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_app_connections_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AppConnectionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_app_connections._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_app_connections_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AppConnectionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AppConnectionsServiceRestInterceptor())
    client = AppConnectionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AppConnectionsServiceRestInterceptor, 'post_list_app_connections') as post, mock.patch.object(transports.AppConnectionsServiceRestInterceptor, 'pre_list_app_connections') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = app_connections_service.ListAppConnectionsRequest.pb(app_connections_service.ListAppConnectionsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = app_connections_service.ListAppConnectionsResponse.to_json(app_connections_service.ListAppConnectionsResponse())
        request = app_connections_service.ListAppConnectionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = app_connections_service.ListAppConnectionsResponse()
        client.list_app_connections(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_app_connections_rest_bad_request(transport: str='rest', request_type=app_connections_service.ListAppConnectionsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_app_connections(request)

def test_list_app_connections_rest_flattened():
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = app_connections_service.ListAppConnectionsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = app_connections_service.ListAppConnectionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_app_connections(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/appConnections' % client.transport._host, args[1])

def test_list_app_connections_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_app_connections(app_connections_service.ListAppConnectionsRequest(), parent='parent_value')

def test_list_app_connections_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection(), app_connections_service.AppConnection(), app_connections_service.AppConnection()], next_page_token='abc'), app_connections_service.ListAppConnectionsResponse(app_connections=[], next_page_token='def'), app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection()], next_page_token='ghi'), app_connections_service.ListAppConnectionsResponse(app_connections=[app_connections_service.AppConnection(), app_connections_service.AppConnection()]))
        response = response + response
        response = tuple((app_connections_service.ListAppConnectionsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_app_connections(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, app_connections_service.AppConnection) for i in results))
        pages = list(client.list_app_connections(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [app_connections_service.GetAppConnectionRequest, dict])
def test_get_app_connection_rest(request_type):
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/appConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = app_connections_service.AppConnection(name='name_value', display_name='display_name_value', uid='uid_value', type_=app_connections_service.AppConnection.Type.TCP_PROXY, connectors=['connectors_value'], state=app_connections_service.AppConnection.State.CREATING)
        response_value = Response()
        response_value.status_code = 200
        return_value = app_connections_service.AppConnection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_app_connection(request)
    assert isinstance(response, app_connections_service.AppConnection)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.type_ == app_connections_service.AppConnection.Type.TCP_PROXY
    assert response.connectors == ['connectors_value']
    assert response.state == app_connections_service.AppConnection.State.CREATING

def test_get_app_connection_rest_required_fields(request_type=app_connections_service.GetAppConnectionRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AppConnectionsServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_app_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_app_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = app_connections_service.AppConnection()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = app_connections_service.AppConnection.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_app_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_app_connection_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AppConnectionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_app_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_app_connection_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AppConnectionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AppConnectionsServiceRestInterceptor())
    client = AppConnectionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AppConnectionsServiceRestInterceptor, 'post_get_app_connection') as post, mock.patch.object(transports.AppConnectionsServiceRestInterceptor, 'pre_get_app_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = app_connections_service.GetAppConnectionRequest.pb(app_connections_service.GetAppConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = app_connections_service.AppConnection.to_json(app_connections_service.AppConnection())
        request = app_connections_service.GetAppConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = app_connections_service.AppConnection()
        client.get_app_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_app_connection_rest_bad_request(transport: str='rest', request_type=app_connections_service.GetAppConnectionRequest):
    if False:
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/appConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_app_connection(request)

def test_get_app_connection_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = app_connections_service.AppConnection()
        sample_request = {'name': 'projects/sample1/locations/sample2/appConnections/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = app_connections_service.AppConnection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_app_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/appConnections/*}' % client.transport._host, args[1])

def test_get_app_connection_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_app_connection(app_connections_service.GetAppConnectionRequest(), name='name_value')

def test_get_app_connection_rest_error():
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [app_connections_service.CreateAppConnectionRequest, dict])
def test_create_app_connection_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['app_connection'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'display_name': 'display_name_value', 'uid': 'uid_value', 'type_': 1, 'application_endpoint': {'host': 'host_value', 'port': 453}, 'connectors': ['connectors_value1', 'connectors_value2'], 'state': 1, 'gateway': {'type_': 1, 'uri': 'uri_value', 'ingress_port': 1311, 'app_gateway': 'app_gateway_value'}}
    test_field = app_connections_service.CreateAppConnectionRequest.meta.fields['app_connection']

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
    for (field, value) in request_init['app_connection'].items():
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
                for i in range(0, len(request_init['app_connection'][field])):
                    del request_init['app_connection'][field][i][subfield]
            else:
                del request_init['app_connection'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_app_connection(request)
    assert response.operation.name == 'operations/spam'

def test_create_app_connection_rest_required_fields(request_type=app_connections_service.CreateAppConnectionRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AppConnectionsServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_app_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_app_connection._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('app_connection_id', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_app_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_app_connection_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AppConnectionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_app_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(('appConnectionId', 'requestId', 'validateOnly')) & set(('parent', 'appConnection'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_app_connection_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AppConnectionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AppConnectionsServiceRestInterceptor())
    client = AppConnectionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AppConnectionsServiceRestInterceptor, 'post_create_app_connection') as post, mock.patch.object(transports.AppConnectionsServiceRestInterceptor, 'pre_create_app_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = app_connections_service.CreateAppConnectionRequest.pb(app_connections_service.CreateAppConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = app_connections_service.CreateAppConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_app_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_app_connection_rest_bad_request(transport: str='rest', request_type=app_connections_service.CreateAppConnectionRequest):
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_app_connection(request)

def test_create_app_connection_rest_flattened():
    if False:
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', app_connection=app_connections_service.AppConnection(name='name_value'), app_connection_id='app_connection_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_app_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/appConnections' % client.transport._host, args[1])

def test_create_app_connection_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_app_connection(app_connections_service.CreateAppConnectionRequest(), parent='parent_value', app_connection=app_connections_service.AppConnection(name='name_value'), app_connection_id='app_connection_id_value')

def test_create_app_connection_rest_error():
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [app_connections_service.UpdateAppConnectionRequest, dict])
def test_update_app_connection_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'app_connection': {'name': 'projects/sample1/locations/sample2/appConnections/sample3'}}
    request_init['app_connection'] = {'name': 'projects/sample1/locations/sample2/appConnections/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'display_name': 'display_name_value', 'uid': 'uid_value', 'type_': 1, 'application_endpoint': {'host': 'host_value', 'port': 453}, 'connectors': ['connectors_value1', 'connectors_value2'], 'state': 1, 'gateway': {'type_': 1, 'uri': 'uri_value', 'ingress_port': 1311, 'app_gateway': 'app_gateway_value'}}
    test_field = app_connections_service.UpdateAppConnectionRequest.meta.fields['app_connection']

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
    for (field, value) in request_init['app_connection'].items():
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
                for i in range(0, len(request_init['app_connection'][field])):
                    del request_init['app_connection'][field][i][subfield]
            else:
                del request_init['app_connection'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_app_connection(request)
    assert response.operation.name == 'operations/spam'

def test_update_app_connection_rest_required_fields(request_type=app_connections_service.UpdateAppConnectionRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AppConnectionsServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_app_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_app_connection._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'request_id', 'update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_app_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_app_connection_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AppConnectionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_app_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'requestId', 'updateMask', 'validateOnly')) & set(('updateMask', 'appConnection'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_app_connection_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AppConnectionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AppConnectionsServiceRestInterceptor())
    client = AppConnectionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AppConnectionsServiceRestInterceptor, 'post_update_app_connection') as post, mock.patch.object(transports.AppConnectionsServiceRestInterceptor, 'pre_update_app_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = app_connections_service.UpdateAppConnectionRequest.pb(app_connections_service.UpdateAppConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = app_connections_service.UpdateAppConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_app_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_app_connection_rest_bad_request(transport: str='rest', request_type=app_connections_service.UpdateAppConnectionRequest):
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'app_connection': {'name': 'projects/sample1/locations/sample2/appConnections/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_app_connection(request)

def test_update_app_connection_rest_flattened():
    if False:
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'app_connection': {'name': 'projects/sample1/locations/sample2/appConnections/sample3'}}
        mock_args = dict(app_connection=app_connections_service.AppConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_app_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{app_connection.name=projects/*/locations/*/appConnections/*}' % client.transport._host, args[1])

def test_update_app_connection_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_app_connection(app_connections_service.UpdateAppConnectionRequest(), app_connection=app_connections_service.AppConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_app_connection_rest_error():
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [app_connections_service.DeleteAppConnectionRequest, dict])
def test_delete_app_connection_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/appConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_app_connection(request)
    assert response.operation.name == 'operations/spam'

def test_delete_app_connection_rest_required_fields(request_type=app_connections_service.DeleteAppConnectionRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AppConnectionsServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_app_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_app_connection._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_app_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_app_connection_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AppConnectionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_app_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_app_connection_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AppConnectionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AppConnectionsServiceRestInterceptor())
    client = AppConnectionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AppConnectionsServiceRestInterceptor, 'post_delete_app_connection') as post, mock.patch.object(transports.AppConnectionsServiceRestInterceptor, 'pre_delete_app_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = app_connections_service.DeleteAppConnectionRequest.pb(app_connections_service.DeleteAppConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = app_connections_service.DeleteAppConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_app_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_app_connection_rest_bad_request(transport: str='rest', request_type=app_connections_service.DeleteAppConnectionRequest):
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/appConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_app_connection(request)

def test_delete_app_connection_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/appConnections/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_app_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/appConnections/*}' % client.transport._host, args[1])

def test_delete_app_connection_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_app_connection(app_connections_service.DeleteAppConnectionRequest(), name='name_value')

def test_delete_app_connection_rest_error():
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [app_connections_service.ResolveAppConnectionsRequest, dict])
def test_resolve_app_connections_rest(request_type):
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = app_connections_service.ResolveAppConnectionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = app_connections_service.ResolveAppConnectionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.resolve_app_connections(request)
    assert isinstance(response, pagers.ResolveAppConnectionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_resolve_app_connections_rest_required_fields(request_type=app_connections_service.ResolveAppConnectionsRequest):
    if False:
        return 10
    transport_class = transports.AppConnectionsServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['app_connector_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'appConnectorId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resolve_app_connections._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'appConnectorId' in jsonified_request
    assert jsonified_request['appConnectorId'] == request_init['app_connector_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['appConnectorId'] = 'app_connector_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resolve_app_connections._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('app_connector_id', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'appConnectorId' in jsonified_request
    assert jsonified_request['appConnectorId'] == 'app_connector_id_value'
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = app_connections_service.ResolveAppConnectionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = app_connections_service.ResolveAppConnectionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.resolve_app_connections(request)
            expected_params = [('appConnectorId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_resolve_app_connections_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AppConnectionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.resolve_app_connections._get_unset_required_fields({})
    assert set(unset_fields) == set(('appConnectorId', 'pageSize', 'pageToken')) & set(('parent', 'appConnectorId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_resolve_app_connections_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AppConnectionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AppConnectionsServiceRestInterceptor())
    client = AppConnectionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AppConnectionsServiceRestInterceptor, 'post_resolve_app_connections') as post, mock.patch.object(transports.AppConnectionsServiceRestInterceptor, 'pre_resolve_app_connections') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = app_connections_service.ResolveAppConnectionsRequest.pb(app_connections_service.ResolveAppConnectionsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = app_connections_service.ResolveAppConnectionsResponse.to_json(app_connections_service.ResolveAppConnectionsResponse())
        request = app_connections_service.ResolveAppConnectionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = app_connections_service.ResolveAppConnectionsResponse()
        client.resolve_app_connections(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_resolve_app_connections_rest_bad_request(transport: str='rest', request_type=app_connections_service.ResolveAppConnectionsRequest):
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.resolve_app_connections(request)

def test_resolve_app_connections_rest_flattened():
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = app_connections_service.ResolveAppConnectionsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = app_connections_service.ResolveAppConnectionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.resolve_app_connections(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/appConnections:resolve' % client.transport._host, args[1])

def test_resolve_app_connections_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.resolve_app_connections(app_connections_service.ResolveAppConnectionsRequest(), parent='parent_value')

def test_resolve_app_connections_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()], next_page_token='abc'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[], next_page_token='def'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()], next_page_token='ghi'), app_connections_service.ResolveAppConnectionsResponse(app_connection_details=[app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails(), app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails()]))
        response = response + response
        response = tuple((app_connections_service.ResolveAppConnectionsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.resolve_app_connections(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, app_connections_service.ResolveAppConnectionsResponse.AppConnectionDetails) for i in results))
        pages = list(client.resolve_app_connections(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.AppConnectionsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.AppConnectionsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AppConnectionsServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.AppConnectionsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AppConnectionsServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AppConnectionsServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.AppConnectionsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AppConnectionsServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AppConnectionsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = AppConnectionsServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.AppConnectionsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.AppConnectionsServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.AppConnectionsServiceGrpcTransport, transports.AppConnectionsServiceGrpcAsyncIOTransport, transports.AppConnectionsServiceRestTransport])
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
        print('Hello World!')
    transport = AppConnectionsServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.AppConnectionsServiceGrpcTransport)

def test_app_connections_service_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.AppConnectionsServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_app_connections_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.beyondcorp_appconnections_v1.services.app_connections_service.transports.AppConnectionsServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.AppConnectionsServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_app_connections', 'get_app_connection', 'create_app_connection', 'update_app_connection', 'delete_app_connection', 'resolve_app_connections', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_app_connections_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.beyondcorp_appconnections_v1.services.app_connections_service.transports.AppConnectionsServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AppConnectionsServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_app_connections_service_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.beyondcorp_appconnections_v1.services.app_connections_service.transports.AppConnectionsServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AppConnectionsServiceTransport()
        adc.assert_called_once()

def test_app_connections_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        AppConnectionsServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.AppConnectionsServiceGrpcTransport, transports.AppConnectionsServiceGrpcAsyncIOTransport])
def test_app_connections_service_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.AppConnectionsServiceGrpcTransport, transports.AppConnectionsServiceGrpcAsyncIOTransport, transports.AppConnectionsServiceRestTransport])
def test_app_connections_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.AppConnectionsServiceGrpcTransport, grpc_helpers), (transports.AppConnectionsServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_app_connections_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('beyondcorp.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='beyondcorp.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.AppConnectionsServiceGrpcTransport, transports.AppConnectionsServiceGrpcAsyncIOTransport])
def test_app_connections_service_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        return 10
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

def test_app_connections_service_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.AppConnectionsServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_app_connections_service_rest_lro_client():
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_app_connections_service_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='beyondcorp.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('beyondcorp.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://beyondcorp.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_app_connections_service_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='beyondcorp.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('beyondcorp.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://beyondcorp.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_app_connections_service_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = AppConnectionsServiceClient(credentials=creds1, transport=transport_name)
    client2 = AppConnectionsServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_app_connections._session
    session2 = client2.transport.list_app_connections._session
    assert session1 != session2
    session1 = client1.transport.get_app_connection._session
    session2 = client2.transport.get_app_connection._session
    assert session1 != session2
    session1 = client1.transport.create_app_connection._session
    session2 = client2.transport.create_app_connection._session
    assert session1 != session2
    session1 = client1.transport.update_app_connection._session
    session2 = client2.transport.update_app_connection._session
    assert session1 != session2
    session1 = client1.transport.delete_app_connection._session
    session2 = client2.transport.delete_app_connection._session
    assert session1 != session2
    session1 = client1.transport.resolve_app_connections._session
    session2 = client2.transport.resolve_app_connections._session
    assert session1 != session2

def test_app_connections_service_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AppConnectionsServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_app_connections_service_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AppConnectionsServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.AppConnectionsServiceGrpcTransport, transports.AppConnectionsServiceGrpcAsyncIOTransport])
def test_app_connections_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.AppConnectionsServiceGrpcTransport, transports.AppConnectionsServiceGrpcAsyncIOTransport])
def test_app_connections_service_transport_channel_mtls_with_adc(transport_class):
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

def test_app_connections_service_grpc_lro_client():
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_app_connections_service_grpc_lro_async_client():
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_app_connection_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    app_connection = 'whelk'
    expected = 'projects/{project}/locations/{location}/appConnections/{app_connection}'.format(project=project, location=location, app_connection=app_connection)
    actual = AppConnectionsServiceClient.app_connection_path(project, location, app_connection)
    assert expected == actual

def test_parse_app_connection_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus', 'location': 'oyster', 'app_connection': 'nudibranch'}
    path = AppConnectionsServiceClient.app_connection_path(**expected)
    actual = AppConnectionsServiceClient.parse_app_connection_path(path)
    assert expected == actual

def test_app_connector_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    location = 'mussel'
    app_connector = 'winkle'
    expected = 'projects/{project}/locations/{location}/appConnectors/{app_connector}'.format(project=project, location=location, app_connector=app_connector)
    actual = AppConnectionsServiceClient.app_connector_path(project, location, app_connector)
    assert expected == actual

def test_parse_app_connector_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nautilus', 'location': 'scallop', 'app_connector': 'abalone'}
    path = AppConnectionsServiceClient.app_connector_path(**expected)
    actual = AppConnectionsServiceClient.parse_app_connector_path(path)
    assert expected == actual

def test_app_gateway_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    app_gateway = 'whelk'
    expected = 'projects/{project}/locations/{location}/appGateways/{app_gateway}'.format(project=project, location=location, app_gateway=app_gateway)
    actual = AppConnectionsServiceClient.app_gateway_path(project, location, app_gateway)
    assert expected == actual

def test_parse_app_gateway_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'app_gateway': 'nudibranch'}
    path = AppConnectionsServiceClient.app_gateway_path(**expected)
    actual = AppConnectionsServiceClient.parse_app_gateway_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = AppConnectionsServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'mussel'}
    path = AppConnectionsServiceClient.common_billing_account_path(**expected)
    actual = AppConnectionsServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = AppConnectionsServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'nautilus'}
    path = AppConnectionsServiceClient.common_folder_path(**expected)
    actual = AppConnectionsServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = AppConnectionsServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'abalone'}
    path = AppConnectionsServiceClient.common_organization_path(**expected)
    actual = AppConnectionsServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = AppConnectionsServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'clam'}
    path = AppConnectionsServiceClient.common_project_path(**expected)
    actual = AppConnectionsServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = AppConnectionsServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = AppConnectionsServiceClient.common_location_path(**expected)
    actual = AppConnectionsServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.AppConnectionsServiceTransport, '_prep_wrapped_messages') as prep:
        client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.AppConnectionsServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = AppConnectionsServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/appConnections/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/appConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/appConnections/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/appConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)

def test_test_iam_permissions_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/appConnections/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/appConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.test_iam_permissions(request)
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)

def test_cancel_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.CancelOperationRequest):
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.SetIamPolicyRequest()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_set_iam_policy_async(transport: str='grpc_asyncio'):
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.SetIamPolicyRequest()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_field_headers():
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_iam_policy_field_headers_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

def test_set_iam_policy_from_dict():
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.GetIamPolicyRequest()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_get_iam_policy_async(transport: str='grpc_asyncio'):
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.GetIamPolicyRequest()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_iam_policy_field_headers_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

def test_get_iam_policy_from_dict():
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.TestIamPermissionsRequest()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response = client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

@pytest.mark.asyncio
async def test_test_iam_permissions_async(transport: str='grpc_asyncio'):
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.TestIamPermissionsRequest()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value']))
        response = await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_field_headers():
    if False:
        print('Hello World!')
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

@pytest.mark.asyncio
async def test_test_iam_permissions_field_headers_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

def test_test_iam_permissions_from_dict():
    if False:
        return 10
    client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = AppConnectionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = AppConnectionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(AppConnectionsServiceClient, transports.AppConnectionsServiceGrpcTransport), (AppConnectionsServiceAsyncClient, transports.AppConnectionsServiceGrpcAsyncIOTransport)])
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