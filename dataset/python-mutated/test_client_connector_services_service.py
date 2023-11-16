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
from google.cloud.beyondcorp_clientconnectorservices_v1.services.client_connector_services_service import ClientConnectorServicesServiceAsyncClient, ClientConnectorServicesServiceClient, pagers, transports
from google.cloud.beyondcorp_clientconnectorservices_v1.types import client_connector_services_service

def client_cert_source_callback():
    if False:
        while True:
            i = 10
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
    assert ClientConnectorServicesServiceClient._get_default_mtls_endpoint(None) is None
    assert ClientConnectorServicesServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ClientConnectorServicesServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ClientConnectorServicesServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ClientConnectorServicesServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ClientConnectorServicesServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ClientConnectorServicesServiceClient, 'grpc'), (ClientConnectorServicesServiceAsyncClient, 'grpc_asyncio'), (ClientConnectorServicesServiceClient, 'rest')])
def test_client_connector_services_service_client_from_service_account_info(client_class, transport_name):
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

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ClientConnectorServicesServiceGrpcTransport, 'grpc'), (transports.ClientConnectorServicesServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.ClientConnectorServicesServiceRestTransport, 'rest')])
def test_client_connector_services_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ClientConnectorServicesServiceClient, 'grpc'), (ClientConnectorServicesServiceAsyncClient, 'grpc_asyncio'), (ClientConnectorServicesServiceClient, 'rest')])
def test_client_connector_services_service_client_from_service_account_file(client_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
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

def test_client_connector_services_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = ClientConnectorServicesServiceClient.get_transport_class()
    available_transports = [transports.ClientConnectorServicesServiceGrpcTransport, transports.ClientConnectorServicesServiceRestTransport]
    assert transport in available_transports
    transport = ClientConnectorServicesServiceClient.get_transport_class('grpc')
    assert transport == transports.ClientConnectorServicesServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ClientConnectorServicesServiceClient, transports.ClientConnectorServicesServiceGrpcTransport, 'grpc'), (ClientConnectorServicesServiceAsyncClient, transports.ClientConnectorServicesServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (ClientConnectorServicesServiceClient, transports.ClientConnectorServicesServiceRestTransport, 'rest')])
@mock.patch.object(ClientConnectorServicesServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ClientConnectorServicesServiceClient))
@mock.patch.object(ClientConnectorServicesServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ClientConnectorServicesServiceAsyncClient))
def test_client_connector_services_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(ClientConnectorServicesServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ClientConnectorServicesServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ClientConnectorServicesServiceClient, transports.ClientConnectorServicesServiceGrpcTransport, 'grpc', 'true'), (ClientConnectorServicesServiceAsyncClient, transports.ClientConnectorServicesServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ClientConnectorServicesServiceClient, transports.ClientConnectorServicesServiceGrpcTransport, 'grpc', 'false'), (ClientConnectorServicesServiceAsyncClient, transports.ClientConnectorServicesServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (ClientConnectorServicesServiceClient, transports.ClientConnectorServicesServiceRestTransport, 'rest', 'true'), (ClientConnectorServicesServiceClient, transports.ClientConnectorServicesServiceRestTransport, 'rest', 'false')])
@mock.patch.object(ClientConnectorServicesServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ClientConnectorServicesServiceClient))
@mock.patch.object(ClientConnectorServicesServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ClientConnectorServicesServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_client_connector_services_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [ClientConnectorServicesServiceClient, ClientConnectorServicesServiceAsyncClient])
@mock.patch.object(ClientConnectorServicesServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ClientConnectorServicesServiceClient))
@mock.patch.object(ClientConnectorServicesServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ClientConnectorServicesServiceAsyncClient))
def test_client_connector_services_service_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        print('Hello World!')
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ClientConnectorServicesServiceClient, transports.ClientConnectorServicesServiceGrpcTransport, 'grpc'), (ClientConnectorServicesServiceAsyncClient, transports.ClientConnectorServicesServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (ClientConnectorServicesServiceClient, transports.ClientConnectorServicesServiceRestTransport, 'rest')])
def test_client_connector_services_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ClientConnectorServicesServiceClient, transports.ClientConnectorServicesServiceGrpcTransport, 'grpc', grpc_helpers), (ClientConnectorServicesServiceAsyncClient, transports.ClientConnectorServicesServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (ClientConnectorServicesServiceClient, transports.ClientConnectorServicesServiceRestTransport, 'rest', None)])
def test_client_connector_services_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_client_connector_services_service_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.beyondcorp_clientconnectorservices_v1.services.client_connector_services_service.transports.ClientConnectorServicesServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ClientConnectorServicesServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ClientConnectorServicesServiceClient, transports.ClientConnectorServicesServiceGrpcTransport, 'grpc', grpc_helpers), (ClientConnectorServicesServiceAsyncClient, transports.ClientConnectorServicesServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_client_connector_services_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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

@pytest.mark.parametrize('request_type', [client_connector_services_service.ListClientConnectorServicesRequest, dict])
def test_list_client_connector_services(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_client_connector_services), '__call__') as call:
        call.return_value = client_connector_services_service.ListClientConnectorServicesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_client_connector_services(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.ListClientConnectorServicesRequest()
    assert isinstance(response, pagers.ListClientConnectorServicesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_client_connector_services_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_client_connector_services), '__call__') as call:
        client.list_client_connector_services()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.ListClientConnectorServicesRequest()

@pytest.mark.asyncio
async def test_list_client_connector_services_async(transport: str='grpc_asyncio', request_type=client_connector_services_service.ListClientConnectorServicesRequest):
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_client_connector_services), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(client_connector_services_service.ListClientConnectorServicesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_client_connector_services(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.ListClientConnectorServicesRequest()
    assert isinstance(response, pagers.ListClientConnectorServicesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_client_connector_services_async_from_dict():
    await test_list_client_connector_services_async(request_type=dict)

def test_list_client_connector_services_field_headers():
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_connector_services_service.ListClientConnectorServicesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_client_connector_services), '__call__') as call:
        call.return_value = client_connector_services_service.ListClientConnectorServicesResponse()
        client.list_client_connector_services(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_client_connector_services_field_headers_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_connector_services_service.ListClientConnectorServicesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_client_connector_services), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(client_connector_services_service.ListClientConnectorServicesResponse())
        await client.list_client_connector_services(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_client_connector_services_flattened():
    if False:
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_client_connector_services), '__call__') as call:
        call.return_value = client_connector_services_service.ListClientConnectorServicesResponse()
        client.list_client_connector_services(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_client_connector_services_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_client_connector_services(client_connector_services_service.ListClientConnectorServicesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_client_connector_services_flattened_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_client_connector_services), '__call__') as call:
        call.return_value = client_connector_services_service.ListClientConnectorServicesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(client_connector_services_service.ListClientConnectorServicesResponse())
        response = await client.list_client_connector_services(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_client_connector_services_flattened_error_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_client_connector_services(client_connector_services_service.ListClientConnectorServicesRequest(), parent='parent_value')

def test_list_client_connector_services_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_client_connector_services), '__call__') as call:
        call.side_effect = (client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService()], next_page_token='abc'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[], next_page_token='def'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService()], next_page_token='ghi'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_client_connector_services(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, client_connector_services_service.ClientConnectorService) for i in results))

def test_list_client_connector_services_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_client_connector_services), '__call__') as call:
        call.side_effect = (client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService()], next_page_token='abc'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[], next_page_token='def'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService()], next_page_token='ghi'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService()]), RuntimeError)
        pages = list(client.list_client_connector_services(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_client_connector_services_async_pager():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_client_connector_services), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService()], next_page_token='abc'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[], next_page_token='def'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService()], next_page_token='ghi'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService()]), RuntimeError)
        async_pager = await client.list_client_connector_services(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, client_connector_services_service.ClientConnectorService) for i in responses))

@pytest.mark.asyncio
async def test_list_client_connector_services_async_pages():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_client_connector_services), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService()], next_page_token='abc'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[], next_page_token='def'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService()], next_page_token='ghi'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_client_connector_services(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [client_connector_services_service.GetClientConnectorServiceRequest, dict])
def test_get_client_connector_service(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_client_connector_service), '__call__') as call:
        call.return_value = client_connector_services_service.ClientConnectorService(name='name_value', display_name='display_name_value', state=client_connector_services_service.ClientConnectorService.State.CREATING)
        response = client.get_client_connector_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.GetClientConnectorServiceRequest()
    assert isinstance(response, client_connector_services_service.ClientConnectorService)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == client_connector_services_service.ClientConnectorService.State.CREATING

def test_get_client_connector_service_empty_call():
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_client_connector_service), '__call__') as call:
        client.get_client_connector_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.GetClientConnectorServiceRequest()

@pytest.mark.asyncio
async def test_get_client_connector_service_async(transport: str='grpc_asyncio', request_type=client_connector_services_service.GetClientConnectorServiceRequest):
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_client_connector_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(client_connector_services_service.ClientConnectorService(name='name_value', display_name='display_name_value', state=client_connector_services_service.ClientConnectorService.State.CREATING))
        response = await client.get_client_connector_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.GetClientConnectorServiceRequest()
    assert isinstance(response, client_connector_services_service.ClientConnectorService)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == client_connector_services_service.ClientConnectorService.State.CREATING

@pytest.mark.asyncio
async def test_get_client_connector_service_async_from_dict():
    await test_get_client_connector_service_async(request_type=dict)

def test_get_client_connector_service_field_headers():
    if False:
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_connector_services_service.GetClientConnectorServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_client_connector_service), '__call__') as call:
        call.return_value = client_connector_services_service.ClientConnectorService()
        client.get_client_connector_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_client_connector_service_field_headers_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_connector_services_service.GetClientConnectorServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_client_connector_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(client_connector_services_service.ClientConnectorService())
        await client.get_client_connector_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_client_connector_service_flattened():
    if False:
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_client_connector_service), '__call__') as call:
        call.return_value = client_connector_services_service.ClientConnectorService()
        client.get_client_connector_service(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_client_connector_service_flattened_error():
    if False:
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_client_connector_service(client_connector_services_service.GetClientConnectorServiceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_client_connector_service_flattened_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_client_connector_service), '__call__') as call:
        call.return_value = client_connector_services_service.ClientConnectorService()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(client_connector_services_service.ClientConnectorService())
        response = await client.get_client_connector_service(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_client_connector_service_flattened_error_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_client_connector_service(client_connector_services_service.GetClientConnectorServiceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [client_connector_services_service.CreateClientConnectorServiceRequest, dict])
def test_create_client_connector_service(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_client_connector_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_client_connector_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.CreateClientConnectorServiceRequest()
    assert isinstance(response, future.Future)

def test_create_client_connector_service_empty_call():
    if False:
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_client_connector_service), '__call__') as call:
        client.create_client_connector_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.CreateClientConnectorServiceRequest()

@pytest.mark.asyncio
async def test_create_client_connector_service_async(transport: str='grpc_asyncio', request_type=client_connector_services_service.CreateClientConnectorServiceRequest):
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_client_connector_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_client_connector_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.CreateClientConnectorServiceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_client_connector_service_async_from_dict():
    await test_create_client_connector_service_async(request_type=dict)

def test_create_client_connector_service_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_connector_services_service.CreateClientConnectorServiceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_client_connector_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_client_connector_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_client_connector_service_field_headers_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_connector_services_service.CreateClientConnectorServiceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_client_connector_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_client_connector_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_client_connector_service_flattened():
    if False:
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_client_connector_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_client_connector_service(parent='parent_value', client_connector_service=client_connector_services_service.ClientConnectorService(name='name_value'), client_connector_service_id='client_connector_service_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].client_connector_service
        mock_val = client_connector_services_service.ClientConnectorService(name='name_value')
        assert arg == mock_val
        arg = args[0].client_connector_service_id
        mock_val = 'client_connector_service_id_value'
        assert arg == mock_val

def test_create_client_connector_service_flattened_error():
    if False:
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_client_connector_service(client_connector_services_service.CreateClientConnectorServiceRequest(), parent='parent_value', client_connector_service=client_connector_services_service.ClientConnectorService(name='name_value'), client_connector_service_id='client_connector_service_id_value')

@pytest.mark.asyncio
async def test_create_client_connector_service_flattened_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_client_connector_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_client_connector_service(parent='parent_value', client_connector_service=client_connector_services_service.ClientConnectorService(name='name_value'), client_connector_service_id='client_connector_service_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].client_connector_service
        mock_val = client_connector_services_service.ClientConnectorService(name='name_value')
        assert arg == mock_val
        arg = args[0].client_connector_service_id
        mock_val = 'client_connector_service_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_client_connector_service_flattened_error_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_client_connector_service(client_connector_services_service.CreateClientConnectorServiceRequest(), parent='parent_value', client_connector_service=client_connector_services_service.ClientConnectorService(name='name_value'), client_connector_service_id='client_connector_service_id_value')

@pytest.mark.parametrize('request_type', [client_connector_services_service.UpdateClientConnectorServiceRequest, dict])
def test_update_client_connector_service(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_client_connector_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_client_connector_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.UpdateClientConnectorServiceRequest()
    assert isinstance(response, future.Future)

def test_update_client_connector_service_empty_call():
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_client_connector_service), '__call__') as call:
        client.update_client_connector_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.UpdateClientConnectorServiceRequest()

@pytest.mark.asyncio
async def test_update_client_connector_service_async(transport: str='grpc_asyncio', request_type=client_connector_services_service.UpdateClientConnectorServiceRequest):
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_client_connector_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_client_connector_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.UpdateClientConnectorServiceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_client_connector_service_async_from_dict():
    await test_update_client_connector_service_async(request_type=dict)

def test_update_client_connector_service_field_headers():
    if False:
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_connector_services_service.UpdateClientConnectorServiceRequest()
    request.client_connector_service.name = 'name_value'
    with mock.patch.object(type(client.transport.update_client_connector_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_client_connector_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'client_connector_service.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_client_connector_service_field_headers_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_connector_services_service.UpdateClientConnectorServiceRequest()
    request.client_connector_service.name = 'name_value'
    with mock.patch.object(type(client.transport.update_client_connector_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_client_connector_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'client_connector_service.name=name_value') in kw['metadata']

def test_update_client_connector_service_flattened():
    if False:
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_client_connector_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_client_connector_service(client_connector_service=client_connector_services_service.ClientConnectorService(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].client_connector_service
        mock_val = client_connector_services_service.ClientConnectorService(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_client_connector_service_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_client_connector_service(client_connector_services_service.UpdateClientConnectorServiceRequest(), client_connector_service=client_connector_services_service.ClientConnectorService(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_client_connector_service_flattened_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_client_connector_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_client_connector_service(client_connector_service=client_connector_services_service.ClientConnectorService(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].client_connector_service
        mock_val = client_connector_services_service.ClientConnectorService(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_client_connector_service_flattened_error_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_client_connector_service(client_connector_services_service.UpdateClientConnectorServiceRequest(), client_connector_service=client_connector_services_service.ClientConnectorService(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [client_connector_services_service.DeleteClientConnectorServiceRequest, dict])
def test_delete_client_connector_service(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_client_connector_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_client_connector_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.DeleteClientConnectorServiceRequest()
    assert isinstance(response, future.Future)

def test_delete_client_connector_service_empty_call():
    if False:
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_client_connector_service), '__call__') as call:
        client.delete_client_connector_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.DeleteClientConnectorServiceRequest()

@pytest.mark.asyncio
async def test_delete_client_connector_service_async(transport: str='grpc_asyncio', request_type=client_connector_services_service.DeleteClientConnectorServiceRequest):
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_client_connector_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_client_connector_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_connector_services_service.DeleteClientConnectorServiceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_client_connector_service_async_from_dict():
    await test_delete_client_connector_service_async(request_type=dict)

def test_delete_client_connector_service_field_headers():
    if False:
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_connector_services_service.DeleteClientConnectorServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_client_connector_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_client_connector_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_client_connector_service_field_headers_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_connector_services_service.DeleteClientConnectorServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_client_connector_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_client_connector_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_client_connector_service_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_client_connector_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_client_connector_service(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_client_connector_service_flattened_error():
    if False:
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_client_connector_service(client_connector_services_service.DeleteClientConnectorServiceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_client_connector_service_flattened_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_client_connector_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_client_connector_service(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_client_connector_service_flattened_error_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_client_connector_service(client_connector_services_service.DeleteClientConnectorServiceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [client_connector_services_service.ListClientConnectorServicesRequest, dict])
def test_list_client_connector_services_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = client_connector_services_service.ListClientConnectorServicesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = client_connector_services_service.ListClientConnectorServicesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_client_connector_services(request)
    assert isinstance(response, pagers.ListClientConnectorServicesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_client_connector_services_rest_required_fields(request_type=client_connector_services_service.ListClientConnectorServicesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ClientConnectorServicesServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_client_connector_services._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_client_connector_services._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = client_connector_services_service.ListClientConnectorServicesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = client_connector_services_service.ListClientConnectorServicesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_client_connector_services(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_client_connector_services_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ClientConnectorServicesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_client_connector_services._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_client_connector_services_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ClientConnectorServicesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ClientConnectorServicesServiceRestInterceptor())
    client = ClientConnectorServicesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ClientConnectorServicesServiceRestInterceptor, 'post_list_client_connector_services') as post, mock.patch.object(transports.ClientConnectorServicesServiceRestInterceptor, 'pre_list_client_connector_services') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = client_connector_services_service.ListClientConnectorServicesRequest.pb(client_connector_services_service.ListClientConnectorServicesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = client_connector_services_service.ListClientConnectorServicesResponse.to_json(client_connector_services_service.ListClientConnectorServicesResponse())
        request = client_connector_services_service.ListClientConnectorServicesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = client_connector_services_service.ListClientConnectorServicesResponse()
        client.list_client_connector_services(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_client_connector_services_rest_bad_request(transport: str='rest', request_type=client_connector_services_service.ListClientConnectorServicesRequest):
    if False:
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_client_connector_services(request)

def test_list_client_connector_services_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = client_connector_services_service.ListClientConnectorServicesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = client_connector_services_service.ListClientConnectorServicesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_client_connector_services(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/clientConnectorServices' % client.transport._host, args[1])

def test_list_client_connector_services_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_client_connector_services(client_connector_services_service.ListClientConnectorServicesRequest(), parent='parent_value')

def test_list_client_connector_services_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService()], next_page_token='abc'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[], next_page_token='def'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService()], next_page_token='ghi'), client_connector_services_service.ListClientConnectorServicesResponse(client_connector_services=[client_connector_services_service.ClientConnectorService(), client_connector_services_service.ClientConnectorService()]))
        response = response + response
        response = tuple((client_connector_services_service.ListClientConnectorServicesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_client_connector_services(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, client_connector_services_service.ClientConnectorService) for i in results))
        pages = list(client.list_client_connector_services(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [client_connector_services_service.GetClientConnectorServiceRequest, dict])
def test_get_client_connector_service_rest(request_type):
    if False:
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/clientConnectorServices/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = client_connector_services_service.ClientConnectorService(name='name_value', display_name='display_name_value', state=client_connector_services_service.ClientConnectorService.State.CREATING)
        response_value = Response()
        response_value.status_code = 200
        return_value = client_connector_services_service.ClientConnectorService.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_client_connector_service(request)
    assert isinstance(response, client_connector_services_service.ClientConnectorService)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == client_connector_services_service.ClientConnectorService.State.CREATING

def test_get_client_connector_service_rest_required_fields(request_type=client_connector_services_service.GetClientConnectorServiceRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ClientConnectorServicesServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_client_connector_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_client_connector_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = client_connector_services_service.ClientConnectorService()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = client_connector_services_service.ClientConnectorService.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_client_connector_service(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_client_connector_service_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ClientConnectorServicesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_client_connector_service._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_client_connector_service_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ClientConnectorServicesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ClientConnectorServicesServiceRestInterceptor())
    client = ClientConnectorServicesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ClientConnectorServicesServiceRestInterceptor, 'post_get_client_connector_service') as post, mock.patch.object(transports.ClientConnectorServicesServiceRestInterceptor, 'pre_get_client_connector_service') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = client_connector_services_service.GetClientConnectorServiceRequest.pb(client_connector_services_service.GetClientConnectorServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = client_connector_services_service.ClientConnectorService.to_json(client_connector_services_service.ClientConnectorService())
        request = client_connector_services_service.GetClientConnectorServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = client_connector_services_service.ClientConnectorService()
        client.get_client_connector_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_client_connector_service_rest_bad_request(transport: str='rest', request_type=client_connector_services_service.GetClientConnectorServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/clientConnectorServices/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_client_connector_service(request)

def test_get_client_connector_service_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = client_connector_services_service.ClientConnectorService()
        sample_request = {'name': 'projects/sample1/locations/sample2/clientConnectorServices/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = client_connector_services_service.ClientConnectorService.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_client_connector_service(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/clientConnectorServices/*}' % client.transport._host, args[1])

def test_get_client_connector_service_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_client_connector_service(client_connector_services_service.GetClientConnectorServiceRequest(), name='name_value')

def test_get_client_connector_service_rest_error():
    if False:
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [client_connector_services_service.CreateClientConnectorServiceRequest, dict])
def test_create_client_connector_service_rest(request_type):
    if False:
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['client_connector_service'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'display_name': 'display_name_value', 'ingress': {'config': {'transport_protocol': 1, 'destination_routes': [{'address': 'address_value', 'netmask': 'netmask_value'}]}}, 'egress': {'peered_vpc': {'network_vpc': 'network_vpc_value'}}, 'state': 1}
    test_field = client_connector_services_service.CreateClientConnectorServiceRequest.meta.fields['client_connector_service']

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
    for (field, value) in request_init['client_connector_service'].items():
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
                for i in range(0, len(request_init['client_connector_service'][field])):
                    del request_init['client_connector_service'][field][i][subfield]
            else:
                del request_init['client_connector_service'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_client_connector_service(request)
    assert response.operation.name == 'operations/spam'

def test_create_client_connector_service_rest_required_fields(request_type=client_connector_services_service.CreateClientConnectorServiceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ClientConnectorServicesServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_client_connector_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_client_connector_service._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('client_connector_service_id', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_client_connector_service(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_client_connector_service_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ClientConnectorServicesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_client_connector_service._get_unset_required_fields({})
    assert set(unset_fields) == set(('clientConnectorServiceId', 'requestId', 'validateOnly')) & set(('parent', 'clientConnectorService'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_client_connector_service_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ClientConnectorServicesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ClientConnectorServicesServiceRestInterceptor())
    client = ClientConnectorServicesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ClientConnectorServicesServiceRestInterceptor, 'post_create_client_connector_service') as post, mock.patch.object(transports.ClientConnectorServicesServiceRestInterceptor, 'pre_create_client_connector_service') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = client_connector_services_service.CreateClientConnectorServiceRequest.pb(client_connector_services_service.CreateClientConnectorServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = client_connector_services_service.CreateClientConnectorServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_client_connector_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_client_connector_service_rest_bad_request(transport: str='rest', request_type=client_connector_services_service.CreateClientConnectorServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_client_connector_service(request)

def test_create_client_connector_service_rest_flattened():
    if False:
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', client_connector_service=client_connector_services_service.ClientConnectorService(name='name_value'), client_connector_service_id='client_connector_service_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_client_connector_service(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/clientConnectorServices' % client.transport._host, args[1])

def test_create_client_connector_service_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_client_connector_service(client_connector_services_service.CreateClientConnectorServiceRequest(), parent='parent_value', client_connector_service=client_connector_services_service.ClientConnectorService(name='name_value'), client_connector_service_id='client_connector_service_id_value')

def test_create_client_connector_service_rest_error():
    if False:
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [client_connector_services_service.UpdateClientConnectorServiceRequest, dict])
def test_update_client_connector_service_rest(request_type):
    if False:
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'client_connector_service': {'name': 'projects/sample1/locations/sample2/clientConnectorServices/sample3'}}
    request_init['client_connector_service'] = {'name': 'projects/sample1/locations/sample2/clientConnectorServices/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'display_name': 'display_name_value', 'ingress': {'config': {'transport_protocol': 1, 'destination_routes': [{'address': 'address_value', 'netmask': 'netmask_value'}]}}, 'egress': {'peered_vpc': {'network_vpc': 'network_vpc_value'}}, 'state': 1}
    test_field = client_connector_services_service.UpdateClientConnectorServiceRequest.meta.fields['client_connector_service']

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
    for (field, value) in request_init['client_connector_service'].items():
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
                for i in range(0, len(request_init['client_connector_service'][field])):
                    del request_init['client_connector_service'][field][i][subfield]
            else:
                del request_init['client_connector_service'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_client_connector_service(request)
    assert response.operation.name == 'operations/spam'

def test_update_client_connector_service_rest_required_fields(request_type=client_connector_services_service.UpdateClientConnectorServiceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ClientConnectorServicesServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_client_connector_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_client_connector_service._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'request_id', 'update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_client_connector_service(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_client_connector_service_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ClientConnectorServicesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_client_connector_service._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'requestId', 'updateMask', 'validateOnly')) & set(('updateMask', 'clientConnectorService'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_client_connector_service_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ClientConnectorServicesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ClientConnectorServicesServiceRestInterceptor())
    client = ClientConnectorServicesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ClientConnectorServicesServiceRestInterceptor, 'post_update_client_connector_service') as post, mock.patch.object(transports.ClientConnectorServicesServiceRestInterceptor, 'pre_update_client_connector_service') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = client_connector_services_service.UpdateClientConnectorServiceRequest.pb(client_connector_services_service.UpdateClientConnectorServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = client_connector_services_service.UpdateClientConnectorServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_client_connector_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_client_connector_service_rest_bad_request(transport: str='rest', request_type=client_connector_services_service.UpdateClientConnectorServiceRequest):
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'client_connector_service': {'name': 'projects/sample1/locations/sample2/clientConnectorServices/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_client_connector_service(request)

def test_update_client_connector_service_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'client_connector_service': {'name': 'projects/sample1/locations/sample2/clientConnectorServices/sample3'}}
        mock_args = dict(client_connector_service=client_connector_services_service.ClientConnectorService(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_client_connector_service(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{client_connector_service.name=projects/*/locations/*/clientConnectorServices/*}' % client.transport._host, args[1])

def test_update_client_connector_service_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_client_connector_service(client_connector_services_service.UpdateClientConnectorServiceRequest(), client_connector_service=client_connector_services_service.ClientConnectorService(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_client_connector_service_rest_error():
    if False:
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [client_connector_services_service.DeleteClientConnectorServiceRequest, dict])
def test_delete_client_connector_service_rest(request_type):
    if False:
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/clientConnectorServices/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_client_connector_service(request)
    assert response.operation.name == 'operations/spam'

def test_delete_client_connector_service_rest_required_fields(request_type=client_connector_services_service.DeleteClientConnectorServiceRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ClientConnectorServicesServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_client_connector_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_client_connector_service._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_client_connector_service(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_client_connector_service_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ClientConnectorServicesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_client_connector_service._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_client_connector_service_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ClientConnectorServicesServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ClientConnectorServicesServiceRestInterceptor())
    client = ClientConnectorServicesServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ClientConnectorServicesServiceRestInterceptor, 'post_delete_client_connector_service') as post, mock.patch.object(transports.ClientConnectorServicesServiceRestInterceptor, 'pre_delete_client_connector_service') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = client_connector_services_service.DeleteClientConnectorServiceRequest.pb(client_connector_services_service.DeleteClientConnectorServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = client_connector_services_service.DeleteClientConnectorServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_client_connector_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_client_connector_service_rest_bad_request(transport: str='rest', request_type=client_connector_services_service.DeleteClientConnectorServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/clientConnectorServices/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_client_connector_service(request)

def test_delete_client_connector_service_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/clientConnectorServices/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_client_connector_service(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/clientConnectorServices/*}' % client.transport._host, args[1])

def test_delete_client_connector_service_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_client_connector_service(client_connector_services_service.DeleteClientConnectorServiceRequest(), name='name_value')

def test_delete_client_connector_service_rest_error():
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.ClientConnectorServicesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ClientConnectorServicesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ClientConnectorServicesServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ClientConnectorServicesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ClientConnectorServicesServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ClientConnectorServicesServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ClientConnectorServicesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ClientConnectorServicesServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.ClientConnectorServicesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ClientConnectorServicesServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.ClientConnectorServicesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ClientConnectorServicesServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ClientConnectorServicesServiceGrpcTransport, transports.ClientConnectorServicesServiceGrpcAsyncIOTransport, transports.ClientConnectorServicesServiceRestTransport])
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
        i = 10
        return i + 15
    transport = ClientConnectorServicesServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ClientConnectorServicesServiceGrpcTransport)

def test_client_connector_services_service_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ClientConnectorServicesServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_client_connector_services_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.beyondcorp_clientconnectorservices_v1.services.client_connector_services_service.transports.ClientConnectorServicesServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ClientConnectorServicesServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_client_connector_services', 'get_client_connector_service', 'create_client_connector_service', 'update_client_connector_service', 'delete_client_connector_service', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_client_connector_services_service_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.beyondcorp_clientconnectorservices_v1.services.client_connector_services_service.transports.ClientConnectorServicesServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ClientConnectorServicesServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_client_connector_services_service_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.beyondcorp_clientconnectorservices_v1.services.client_connector_services_service.transports.ClientConnectorServicesServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ClientConnectorServicesServiceTransport()
        adc.assert_called_once()

def test_client_connector_services_service_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ClientConnectorServicesServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ClientConnectorServicesServiceGrpcTransport, transports.ClientConnectorServicesServiceGrpcAsyncIOTransport])
def test_client_connector_services_service_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ClientConnectorServicesServiceGrpcTransport, transports.ClientConnectorServicesServiceGrpcAsyncIOTransport, transports.ClientConnectorServicesServiceRestTransport])
def test_client_connector_services_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ClientConnectorServicesServiceGrpcTransport, grpc_helpers), (transports.ClientConnectorServicesServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_client_connector_services_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('beyondcorp.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='beyondcorp.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ClientConnectorServicesServiceGrpcTransport, transports.ClientConnectorServicesServiceGrpcAsyncIOTransport])
def test_client_connector_services_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_client_connector_services_service_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.ClientConnectorServicesServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_client_connector_services_service_rest_lro_client():
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_client_connector_services_service_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='beyondcorp.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('beyondcorp.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://beyondcorp.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_client_connector_services_service_host_with_port(transport_name):
    if False:
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='beyondcorp.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('beyondcorp.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://beyondcorp.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_client_connector_services_service_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = ClientConnectorServicesServiceClient(credentials=creds1, transport=transport_name)
    client2 = ClientConnectorServicesServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_client_connector_services._session
    session2 = client2.transport.list_client_connector_services._session
    assert session1 != session2
    session1 = client1.transport.get_client_connector_service._session
    session2 = client2.transport.get_client_connector_service._session
    assert session1 != session2
    session1 = client1.transport.create_client_connector_service._session
    session2 = client2.transport.create_client_connector_service._session
    assert session1 != session2
    session1 = client1.transport.update_client_connector_service._session
    session2 = client2.transport.update_client_connector_service._session
    assert session1 != session2
    session1 = client1.transport.delete_client_connector_service._session
    session2 = client2.transport.delete_client_connector_service._session
    assert session1 != session2

def test_client_connector_services_service_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ClientConnectorServicesServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_client_connector_services_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ClientConnectorServicesServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ClientConnectorServicesServiceGrpcTransport, transports.ClientConnectorServicesServiceGrpcAsyncIOTransport])
def test_client_connector_services_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.ClientConnectorServicesServiceGrpcTransport, transports.ClientConnectorServicesServiceGrpcAsyncIOTransport])
def test_client_connector_services_service_transport_channel_mtls_with_adc(transport_class):
    if False:
        i = 10
        return i + 15
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

def test_client_connector_services_service_grpc_lro_client():
    if False:
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_client_connector_services_service_grpc_lro_async_client():
    if False:
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_client_connector_service_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    client_connector_service = 'whelk'
    expected = 'projects/{project}/locations/{location}/clientConnectorServices/{client_connector_service}'.format(project=project, location=location, client_connector_service=client_connector_service)
    actual = ClientConnectorServicesServiceClient.client_connector_service_path(project, location, client_connector_service)
    assert expected == actual

def test_parse_client_connector_service_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'client_connector_service': 'nudibranch'}
    path = ClientConnectorServicesServiceClient.client_connector_service_path(**expected)
    actual = ClientConnectorServicesServiceClient.parse_client_connector_service_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ClientConnectorServicesServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'mussel'}
    path = ClientConnectorServicesServiceClient.common_billing_account_path(**expected)
    actual = ClientConnectorServicesServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ClientConnectorServicesServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'nautilus'}
    path = ClientConnectorServicesServiceClient.common_folder_path(**expected)
    actual = ClientConnectorServicesServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ClientConnectorServicesServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'abalone'}
    path = ClientConnectorServicesServiceClient.common_organization_path(**expected)
    actual = ClientConnectorServicesServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = ClientConnectorServicesServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'clam'}
    path = ClientConnectorServicesServiceClient.common_project_path(**expected)
    actual = ClientConnectorServicesServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ClientConnectorServicesServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = ClientConnectorServicesServiceClient.common_location_path(**expected)
    actual = ClientConnectorServicesServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ClientConnectorServicesServiceTransport, '_prep_wrapped_messages') as prep:
        client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ClientConnectorServicesServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = ClientConnectorServicesServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = ClientConnectorServicesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = ClientConnectorServicesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ClientConnectorServicesServiceClient, transports.ClientConnectorServicesServiceGrpcTransport), (ClientConnectorServicesServiceAsyncClient, transports.ClientConnectorServicesServiceGrpcAsyncIOTransport)])
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