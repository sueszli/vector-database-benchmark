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
from google.cloud.apigateway_v1.services.api_gateway_service import ApiGatewayServiceAsyncClient, ApiGatewayServiceClient, pagers, transports
from google.cloud.apigateway_v1.types import apigateway

def client_cert_source_callback():
    if False:
        for i in range(10):
            print('nop')
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
    assert ApiGatewayServiceClient._get_default_mtls_endpoint(None) is None
    assert ApiGatewayServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ApiGatewayServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ApiGatewayServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ApiGatewayServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ApiGatewayServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ApiGatewayServiceClient, 'grpc'), (ApiGatewayServiceAsyncClient, 'grpc_asyncio'), (ApiGatewayServiceClient, 'rest')])
def test_api_gateway_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('apigateway.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://apigateway.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ApiGatewayServiceGrpcTransport, 'grpc'), (transports.ApiGatewayServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.ApiGatewayServiceRestTransport, 'rest')])
def test_api_gateway_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ApiGatewayServiceClient, 'grpc'), (ApiGatewayServiceAsyncClient, 'grpc_asyncio'), (ApiGatewayServiceClient, 'rest')])
def test_api_gateway_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('apigateway.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://apigateway.googleapis.com')

def test_api_gateway_service_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = ApiGatewayServiceClient.get_transport_class()
    available_transports = [transports.ApiGatewayServiceGrpcTransport, transports.ApiGatewayServiceRestTransport]
    assert transport in available_transports
    transport = ApiGatewayServiceClient.get_transport_class('grpc')
    assert transport == transports.ApiGatewayServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ApiGatewayServiceClient, transports.ApiGatewayServiceGrpcTransport, 'grpc'), (ApiGatewayServiceAsyncClient, transports.ApiGatewayServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (ApiGatewayServiceClient, transports.ApiGatewayServiceRestTransport, 'rest')])
@mock.patch.object(ApiGatewayServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ApiGatewayServiceClient))
@mock.patch.object(ApiGatewayServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ApiGatewayServiceAsyncClient))
def test_api_gateway_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(ApiGatewayServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ApiGatewayServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ApiGatewayServiceClient, transports.ApiGatewayServiceGrpcTransport, 'grpc', 'true'), (ApiGatewayServiceAsyncClient, transports.ApiGatewayServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ApiGatewayServiceClient, transports.ApiGatewayServiceGrpcTransport, 'grpc', 'false'), (ApiGatewayServiceAsyncClient, transports.ApiGatewayServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (ApiGatewayServiceClient, transports.ApiGatewayServiceRestTransport, 'rest', 'true'), (ApiGatewayServiceClient, transports.ApiGatewayServiceRestTransport, 'rest', 'false')])
@mock.patch.object(ApiGatewayServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ApiGatewayServiceClient))
@mock.patch.object(ApiGatewayServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ApiGatewayServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_api_gateway_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
    if False:
        return 10
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

@pytest.mark.parametrize('client_class', [ApiGatewayServiceClient, ApiGatewayServiceAsyncClient])
@mock.patch.object(ApiGatewayServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ApiGatewayServiceClient))
@mock.patch.object(ApiGatewayServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ApiGatewayServiceAsyncClient))
def test_api_gateway_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ApiGatewayServiceClient, transports.ApiGatewayServiceGrpcTransport, 'grpc'), (ApiGatewayServiceAsyncClient, transports.ApiGatewayServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (ApiGatewayServiceClient, transports.ApiGatewayServiceRestTransport, 'rest')])
def test_api_gateway_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ApiGatewayServiceClient, transports.ApiGatewayServiceGrpcTransport, 'grpc', grpc_helpers), (ApiGatewayServiceAsyncClient, transports.ApiGatewayServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (ApiGatewayServiceClient, transports.ApiGatewayServiceRestTransport, 'rest', None)])
def test_api_gateway_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_api_gateway_service_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.apigateway_v1.services.api_gateway_service.transports.ApiGatewayServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ApiGatewayServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ApiGatewayServiceClient, transports.ApiGatewayServiceGrpcTransport, 'grpc', grpc_helpers), (ApiGatewayServiceAsyncClient, transports.ApiGatewayServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_api_gateway_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('apigateway.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='apigateway.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [apigateway.ListGatewaysRequest, dict])
def test_list_gateways(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.return_value = apigateway.ListGatewaysResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value'])
        response = client.list_gateways(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.ListGatewaysRequest()
    assert isinstance(response, pagers.ListGatewaysPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

def test_list_gateways_empty_call():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        client.list_gateways()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.ListGatewaysRequest()

@pytest.mark.asyncio
async def test_list_gateways_async(transport: str='grpc_asyncio', request_type=apigateway.ListGatewaysRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.ListGatewaysResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value']))
        response = await client.list_gateways(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.ListGatewaysRequest()
    assert isinstance(response, pagers.ListGatewaysAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

@pytest.mark.asyncio
async def test_list_gateways_async_from_dict():
    await test_list_gateways_async(request_type=dict)

def test_list_gateways_field_headers():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.ListGatewaysRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.return_value = apigateway.ListGatewaysResponse()
        client.list_gateways(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_gateways_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.ListGatewaysRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.ListGatewaysResponse())
        await client.list_gateways(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_gateways_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.return_value = apigateway.ListGatewaysResponse()
        client.list_gateways(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_gateways_flattened_error():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_gateways(apigateway.ListGatewaysRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_gateways_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.return_value = apigateway.ListGatewaysResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.ListGatewaysResponse())
        response = await client.list_gateways(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_gateways_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_gateways(apigateway.ListGatewaysRequest(), parent='parent_value')

def test_list_gateways_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.side_effect = (apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway(), apigateway.Gateway(), apigateway.Gateway()], next_page_token='abc'), apigateway.ListGatewaysResponse(gateways=[], next_page_token='def'), apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway()], next_page_token='ghi'), apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway(), apigateway.Gateway()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_gateways(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, apigateway.Gateway) for i in results))

def test_list_gateways_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.side_effect = (apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway(), apigateway.Gateway(), apigateway.Gateway()], next_page_token='abc'), apigateway.ListGatewaysResponse(gateways=[], next_page_token='def'), apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway()], next_page_token='ghi'), apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway(), apigateway.Gateway()]), RuntimeError)
        pages = list(client.list_gateways(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_gateways_async_pager():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_gateways), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway(), apigateway.Gateway(), apigateway.Gateway()], next_page_token='abc'), apigateway.ListGatewaysResponse(gateways=[], next_page_token='def'), apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway()], next_page_token='ghi'), apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway(), apigateway.Gateway()]), RuntimeError)
        async_pager = await client.list_gateways(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, apigateway.Gateway) for i in responses))

@pytest.mark.asyncio
async def test_list_gateways_async_pages():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_gateways), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway(), apigateway.Gateway(), apigateway.Gateway()], next_page_token='abc'), apigateway.ListGatewaysResponse(gateways=[], next_page_token='def'), apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway()], next_page_token='ghi'), apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway(), apigateway.Gateway()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_gateways(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [apigateway.GetGatewayRequest, dict])
def test_get_gateway(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_gateway), '__call__') as call:
        call.return_value = apigateway.Gateway(name='name_value', display_name='display_name_value', api_config='api_config_value', state=apigateway.Gateway.State.CREATING, default_hostname='default_hostname_value')
        response = client.get_gateway(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.GetGatewayRequest()
    assert isinstance(response, apigateway.Gateway)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.api_config == 'api_config_value'
    assert response.state == apigateway.Gateway.State.CREATING
    assert response.default_hostname == 'default_hostname_value'

def test_get_gateway_empty_call():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_gateway), '__call__') as call:
        client.get_gateway()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.GetGatewayRequest()

@pytest.mark.asyncio
async def test_get_gateway_async(transport: str='grpc_asyncio', request_type=apigateway.GetGatewayRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_gateway), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.Gateway(name='name_value', display_name='display_name_value', api_config='api_config_value', state=apigateway.Gateway.State.CREATING, default_hostname='default_hostname_value'))
        response = await client.get_gateway(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.GetGatewayRequest()
    assert isinstance(response, apigateway.Gateway)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.api_config == 'api_config_value'
    assert response.state == apigateway.Gateway.State.CREATING
    assert response.default_hostname == 'default_hostname_value'

@pytest.mark.asyncio
async def test_get_gateway_async_from_dict():
    await test_get_gateway_async(request_type=dict)

def test_get_gateway_field_headers():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.GetGatewayRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_gateway), '__call__') as call:
        call.return_value = apigateway.Gateway()
        client.get_gateway(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_gateway_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.GetGatewayRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_gateway), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.Gateway())
        await client.get_gateway(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_gateway_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_gateway), '__call__') as call:
        call.return_value = apigateway.Gateway()
        client.get_gateway(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_gateway_flattened_error():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_gateway(apigateway.GetGatewayRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_gateway_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_gateway), '__call__') as call:
        call.return_value = apigateway.Gateway()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.Gateway())
        response = await client.get_gateway(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_gateway_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_gateway(apigateway.GetGatewayRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [apigateway.CreateGatewayRequest, dict])
def test_create_gateway(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_gateway(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.CreateGatewayRequest()
    assert isinstance(response, future.Future)

def test_create_gateway_empty_call():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_gateway), '__call__') as call:
        client.create_gateway()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.CreateGatewayRequest()

@pytest.mark.asyncio
async def test_create_gateway_async(transport: str='grpc_asyncio', request_type=apigateway.CreateGatewayRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_gateway), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_gateway(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.CreateGatewayRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_gateway_async_from_dict():
    await test_create_gateway_async(request_type=dict)

def test_create_gateway_field_headers():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.CreateGatewayRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_gateway(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_gateway_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.CreateGatewayRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_gateway), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_gateway(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_gateway_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_gateway(parent='parent_value', gateway=apigateway.Gateway(name='name_value'), gateway_id='gateway_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].gateway
        mock_val = apigateway.Gateway(name='name_value')
        assert arg == mock_val
        arg = args[0].gateway_id
        mock_val = 'gateway_id_value'
        assert arg == mock_val

def test_create_gateway_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_gateway(apigateway.CreateGatewayRequest(), parent='parent_value', gateway=apigateway.Gateway(name='name_value'), gateway_id='gateway_id_value')

@pytest.mark.asyncio
async def test_create_gateway_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_gateway(parent='parent_value', gateway=apigateway.Gateway(name='name_value'), gateway_id='gateway_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].gateway
        mock_val = apigateway.Gateway(name='name_value')
        assert arg == mock_val
        arg = args[0].gateway_id
        mock_val = 'gateway_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_gateway_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_gateway(apigateway.CreateGatewayRequest(), parent='parent_value', gateway=apigateway.Gateway(name='name_value'), gateway_id='gateway_id_value')

@pytest.mark.parametrize('request_type', [apigateway.UpdateGatewayRequest, dict])
def test_update_gateway(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_gateway(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.UpdateGatewayRequest()
    assert isinstance(response, future.Future)

def test_update_gateway_empty_call():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_gateway), '__call__') as call:
        client.update_gateway()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.UpdateGatewayRequest()

@pytest.mark.asyncio
async def test_update_gateway_async(transport: str='grpc_asyncio', request_type=apigateway.UpdateGatewayRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_gateway), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_gateway(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.UpdateGatewayRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_gateway_async_from_dict():
    await test_update_gateway_async(request_type=dict)

def test_update_gateway_field_headers():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.UpdateGatewayRequest()
    request.gateway.name = 'name_value'
    with mock.patch.object(type(client.transport.update_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_gateway(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'gateway.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_gateway_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.UpdateGatewayRequest()
    request.gateway.name = 'name_value'
    with mock.patch.object(type(client.transport.update_gateway), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_gateway(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'gateway.name=name_value') in kw['metadata']

def test_update_gateway_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_gateway(gateway=apigateway.Gateway(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].gateway
        mock_val = apigateway.Gateway(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_gateway_flattened_error():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_gateway(apigateway.UpdateGatewayRequest(), gateway=apigateway.Gateway(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_gateway_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_gateway(gateway=apigateway.Gateway(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].gateway
        mock_val = apigateway.Gateway(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_gateway_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_gateway(apigateway.UpdateGatewayRequest(), gateway=apigateway.Gateway(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [apigateway.DeleteGatewayRequest, dict])
def test_delete_gateway(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_gateway(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.DeleteGatewayRequest()
    assert isinstance(response, future.Future)

def test_delete_gateway_empty_call():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_gateway), '__call__') as call:
        client.delete_gateway()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.DeleteGatewayRequest()

@pytest.mark.asyncio
async def test_delete_gateway_async(transport: str='grpc_asyncio', request_type=apigateway.DeleteGatewayRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_gateway), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_gateway(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.DeleteGatewayRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_gateway_async_from_dict():
    await test_delete_gateway_async(request_type=dict)

def test_delete_gateway_field_headers():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.DeleteGatewayRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_gateway(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_gateway_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.DeleteGatewayRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_gateway), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_gateway(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_gateway_flattened():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_gateway(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_gateway_flattened_error():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_gateway(apigateway.DeleteGatewayRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_gateway_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_gateway(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_gateway_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_gateway(apigateway.DeleteGatewayRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [apigateway.ListApisRequest, dict])
def test_list_apis(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_apis), '__call__') as call:
        call.return_value = apigateway.ListApisResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value'])
        response = client.list_apis(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.ListApisRequest()
    assert isinstance(response, pagers.ListApisPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

def test_list_apis_empty_call():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_apis), '__call__') as call:
        client.list_apis()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.ListApisRequest()

@pytest.mark.asyncio
async def test_list_apis_async(transport: str='grpc_asyncio', request_type=apigateway.ListApisRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_apis), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.ListApisResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value']))
        response = await client.list_apis(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.ListApisRequest()
    assert isinstance(response, pagers.ListApisAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

@pytest.mark.asyncio
async def test_list_apis_async_from_dict():
    await test_list_apis_async(request_type=dict)

def test_list_apis_field_headers():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.ListApisRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_apis), '__call__') as call:
        call.return_value = apigateway.ListApisResponse()
        client.list_apis(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_apis_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.ListApisRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_apis), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.ListApisResponse())
        await client.list_apis(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_apis_flattened():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_apis), '__call__') as call:
        call.return_value = apigateway.ListApisResponse()
        client.list_apis(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_apis_flattened_error():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_apis(apigateway.ListApisRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_apis_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_apis), '__call__') as call:
        call.return_value = apigateway.ListApisResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.ListApisResponse())
        response = await client.list_apis(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_apis_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_apis(apigateway.ListApisRequest(), parent='parent_value')

def test_list_apis_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_apis), '__call__') as call:
        call.side_effect = (apigateway.ListApisResponse(apis=[apigateway.Api(), apigateway.Api(), apigateway.Api()], next_page_token='abc'), apigateway.ListApisResponse(apis=[], next_page_token='def'), apigateway.ListApisResponse(apis=[apigateway.Api()], next_page_token='ghi'), apigateway.ListApisResponse(apis=[apigateway.Api(), apigateway.Api()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_apis(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, apigateway.Api) for i in results))

def test_list_apis_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_apis), '__call__') as call:
        call.side_effect = (apigateway.ListApisResponse(apis=[apigateway.Api(), apigateway.Api(), apigateway.Api()], next_page_token='abc'), apigateway.ListApisResponse(apis=[], next_page_token='def'), apigateway.ListApisResponse(apis=[apigateway.Api()], next_page_token='ghi'), apigateway.ListApisResponse(apis=[apigateway.Api(), apigateway.Api()]), RuntimeError)
        pages = list(client.list_apis(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_apis_async_pager():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_apis), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (apigateway.ListApisResponse(apis=[apigateway.Api(), apigateway.Api(), apigateway.Api()], next_page_token='abc'), apigateway.ListApisResponse(apis=[], next_page_token='def'), apigateway.ListApisResponse(apis=[apigateway.Api()], next_page_token='ghi'), apigateway.ListApisResponse(apis=[apigateway.Api(), apigateway.Api()]), RuntimeError)
        async_pager = await client.list_apis(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, apigateway.Api) for i in responses))

@pytest.mark.asyncio
async def test_list_apis_async_pages():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_apis), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (apigateway.ListApisResponse(apis=[apigateway.Api(), apigateway.Api(), apigateway.Api()], next_page_token='abc'), apigateway.ListApisResponse(apis=[], next_page_token='def'), apigateway.ListApisResponse(apis=[apigateway.Api()], next_page_token='ghi'), apigateway.ListApisResponse(apis=[apigateway.Api(), apigateway.Api()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_apis(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [apigateway.GetApiRequest, dict])
def test_get_api(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_api), '__call__') as call:
        call.return_value = apigateway.Api(name='name_value', display_name='display_name_value', managed_service='managed_service_value', state=apigateway.Api.State.CREATING)
        response = client.get_api(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.GetApiRequest()
    assert isinstance(response, apigateway.Api)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.managed_service == 'managed_service_value'
    assert response.state == apigateway.Api.State.CREATING

def test_get_api_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_api), '__call__') as call:
        client.get_api()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.GetApiRequest()

@pytest.mark.asyncio
async def test_get_api_async(transport: str='grpc_asyncio', request_type=apigateway.GetApiRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_api), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.Api(name='name_value', display_name='display_name_value', managed_service='managed_service_value', state=apigateway.Api.State.CREATING))
        response = await client.get_api(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.GetApiRequest()
    assert isinstance(response, apigateway.Api)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.managed_service == 'managed_service_value'
    assert response.state == apigateway.Api.State.CREATING

@pytest.mark.asyncio
async def test_get_api_async_from_dict():
    await test_get_api_async(request_type=dict)

def test_get_api_field_headers():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.GetApiRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_api), '__call__') as call:
        call.return_value = apigateway.Api()
        client.get_api(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_api_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.GetApiRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_api), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.Api())
        await client.get_api(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_api_flattened():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_api), '__call__') as call:
        call.return_value = apigateway.Api()
        client.get_api(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_api_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_api(apigateway.GetApiRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_api_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_api), '__call__') as call:
        call.return_value = apigateway.Api()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.Api())
        response = await client.get_api(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_api_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_api(apigateway.GetApiRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [apigateway.CreateApiRequest, dict])
def test_create_api(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_api), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_api(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.CreateApiRequest()
    assert isinstance(response, future.Future)

def test_create_api_empty_call():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_api), '__call__') as call:
        client.create_api()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.CreateApiRequest()

@pytest.mark.asyncio
async def test_create_api_async(transport: str='grpc_asyncio', request_type=apigateway.CreateApiRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_api), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_api(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.CreateApiRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_api_async_from_dict():
    await test_create_api_async(request_type=dict)

def test_create_api_field_headers():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.CreateApiRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_api), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_api(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_api_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.CreateApiRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_api), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_api(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_api_flattened():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_api), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_api(parent='parent_value', api=apigateway.Api(name='name_value'), api_id='api_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].api
        mock_val = apigateway.Api(name='name_value')
        assert arg == mock_val
        arg = args[0].api_id
        mock_val = 'api_id_value'
        assert arg == mock_val

def test_create_api_flattened_error():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_api(apigateway.CreateApiRequest(), parent='parent_value', api=apigateway.Api(name='name_value'), api_id='api_id_value')

@pytest.mark.asyncio
async def test_create_api_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_api), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_api(parent='parent_value', api=apigateway.Api(name='name_value'), api_id='api_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].api
        mock_val = apigateway.Api(name='name_value')
        assert arg == mock_val
        arg = args[0].api_id
        mock_val = 'api_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_api_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_api(apigateway.CreateApiRequest(), parent='parent_value', api=apigateway.Api(name='name_value'), api_id='api_id_value')

@pytest.mark.parametrize('request_type', [apigateway.UpdateApiRequest, dict])
def test_update_api(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_api), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_api(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.UpdateApiRequest()
    assert isinstance(response, future.Future)

def test_update_api_empty_call():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_api), '__call__') as call:
        client.update_api()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.UpdateApiRequest()

@pytest.mark.asyncio
async def test_update_api_async(transport: str='grpc_asyncio', request_type=apigateway.UpdateApiRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_api), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_api(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.UpdateApiRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_api_async_from_dict():
    await test_update_api_async(request_type=dict)

def test_update_api_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.UpdateApiRequest()
    request.api.name = 'name_value'
    with mock.patch.object(type(client.transport.update_api), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_api(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'api.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_api_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.UpdateApiRequest()
    request.api.name = 'name_value'
    with mock.patch.object(type(client.transport.update_api), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_api(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'api.name=name_value') in kw['metadata']

def test_update_api_flattened():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_api), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_api(api=apigateway.Api(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].api
        mock_val = apigateway.Api(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_api_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_api(apigateway.UpdateApiRequest(), api=apigateway.Api(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_api_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_api), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_api(api=apigateway.Api(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].api
        mock_val = apigateway.Api(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_api_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_api(apigateway.UpdateApiRequest(), api=apigateway.Api(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [apigateway.DeleteApiRequest, dict])
def test_delete_api(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_api), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_api(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.DeleteApiRequest()
    assert isinstance(response, future.Future)

def test_delete_api_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_api), '__call__') as call:
        client.delete_api()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.DeleteApiRequest()

@pytest.mark.asyncio
async def test_delete_api_async(transport: str='grpc_asyncio', request_type=apigateway.DeleteApiRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_api), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_api(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.DeleteApiRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_api_async_from_dict():
    await test_delete_api_async(request_type=dict)

def test_delete_api_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.DeleteApiRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_api), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_api(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_api_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.DeleteApiRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_api), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_api(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_api_flattened():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_api), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_api(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_api_flattened_error():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_api(apigateway.DeleteApiRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_api_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_api), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_api(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_api_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_api(apigateway.DeleteApiRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [apigateway.ListApiConfigsRequest, dict])
def test_list_api_configs(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_api_configs), '__call__') as call:
        call.return_value = apigateway.ListApiConfigsResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value'])
        response = client.list_api_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.ListApiConfigsRequest()
    assert isinstance(response, pagers.ListApiConfigsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

def test_list_api_configs_empty_call():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_api_configs), '__call__') as call:
        client.list_api_configs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.ListApiConfigsRequest()

@pytest.mark.asyncio
async def test_list_api_configs_async(transport: str='grpc_asyncio', request_type=apigateway.ListApiConfigsRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_api_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.ListApiConfigsResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value']))
        response = await client.list_api_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.ListApiConfigsRequest()
    assert isinstance(response, pagers.ListApiConfigsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

@pytest.mark.asyncio
async def test_list_api_configs_async_from_dict():
    await test_list_api_configs_async(request_type=dict)

def test_list_api_configs_field_headers():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.ListApiConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_api_configs), '__call__') as call:
        call.return_value = apigateway.ListApiConfigsResponse()
        client.list_api_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_api_configs_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.ListApiConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_api_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.ListApiConfigsResponse())
        await client.list_api_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_api_configs_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_api_configs), '__call__') as call:
        call.return_value = apigateway.ListApiConfigsResponse()
        client.list_api_configs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_api_configs_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_api_configs(apigateway.ListApiConfigsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_api_configs_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_api_configs), '__call__') as call:
        call.return_value = apigateway.ListApiConfigsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.ListApiConfigsResponse())
        response = await client.list_api_configs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_api_configs_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_api_configs(apigateway.ListApiConfigsRequest(), parent='parent_value')

def test_list_api_configs_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_api_configs), '__call__') as call:
        call.side_effect = (apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig(), apigateway.ApiConfig(), apigateway.ApiConfig()], next_page_token='abc'), apigateway.ListApiConfigsResponse(api_configs=[], next_page_token='def'), apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig()], next_page_token='ghi'), apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig(), apigateway.ApiConfig()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_api_configs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, apigateway.ApiConfig) for i in results))

def test_list_api_configs_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_api_configs), '__call__') as call:
        call.side_effect = (apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig(), apigateway.ApiConfig(), apigateway.ApiConfig()], next_page_token='abc'), apigateway.ListApiConfigsResponse(api_configs=[], next_page_token='def'), apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig()], next_page_token='ghi'), apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig(), apigateway.ApiConfig()]), RuntimeError)
        pages = list(client.list_api_configs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_api_configs_async_pager():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_api_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig(), apigateway.ApiConfig(), apigateway.ApiConfig()], next_page_token='abc'), apigateway.ListApiConfigsResponse(api_configs=[], next_page_token='def'), apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig()], next_page_token='ghi'), apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig(), apigateway.ApiConfig()]), RuntimeError)
        async_pager = await client.list_api_configs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, apigateway.ApiConfig) for i in responses))

@pytest.mark.asyncio
async def test_list_api_configs_async_pages():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_api_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig(), apigateway.ApiConfig(), apigateway.ApiConfig()], next_page_token='abc'), apigateway.ListApiConfigsResponse(api_configs=[], next_page_token='def'), apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig()], next_page_token='ghi'), apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig(), apigateway.ApiConfig()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_api_configs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [apigateway.GetApiConfigRequest, dict])
def test_get_api_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_api_config), '__call__') as call:
        call.return_value = apigateway.ApiConfig(name='name_value', display_name='display_name_value', gateway_service_account='gateway_service_account_value', service_config_id='service_config_id_value', state=apigateway.ApiConfig.State.CREATING)
        response = client.get_api_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.GetApiConfigRequest()
    assert isinstance(response, apigateway.ApiConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.gateway_service_account == 'gateway_service_account_value'
    assert response.service_config_id == 'service_config_id_value'
    assert response.state == apigateway.ApiConfig.State.CREATING

def test_get_api_config_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_api_config), '__call__') as call:
        client.get_api_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.GetApiConfigRequest()

@pytest.mark.asyncio
async def test_get_api_config_async(transport: str='grpc_asyncio', request_type=apigateway.GetApiConfigRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_api_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.ApiConfig(name='name_value', display_name='display_name_value', gateway_service_account='gateway_service_account_value', service_config_id='service_config_id_value', state=apigateway.ApiConfig.State.CREATING))
        response = await client.get_api_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.GetApiConfigRequest()
    assert isinstance(response, apigateway.ApiConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.gateway_service_account == 'gateway_service_account_value'
    assert response.service_config_id == 'service_config_id_value'
    assert response.state == apigateway.ApiConfig.State.CREATING

@pytest.mark.asyncio
async def test_get_api_config_async_from_dict():
    await test_get_api_config_async(request_type=dict)

def test_get_api_config_field_headers():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.GetApiConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_api_config), '__call__') as call:
        call.return_value = apigateway.ApiConfig()
        client.get_api_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_api_config_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.GetApiConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_api_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.ApiConfig())
        await client.get_api_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_api_config_flattened():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_api_config), '__call__') as call:
        call.return_value = apigateway.ApiConfig()
        client.get_api_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_api_config_flattened_error():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_api_config(apigateway.GetApiConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_api_config_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_api_config), '__call__') as call:
        call.return_value = apigateway.ApiConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(apigateway.ApiConfig())
        response = await client.get_api_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_api_config_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_api_config(apigateway.GetApiConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [apigateway.CreateApiConfigRequest, dict])
def test_create_api_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_api_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_api_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.CreateApiConfigRequest()
    assert isinstance(response, future.Future)

def test_create_api_config_empty_call():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_api_config), '__call__') as call:
        client.create_api_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.CreateApiConfigRequest()

@pytest.mark.asyncio
async def test_create_api_config_async(transport: str='grpc_asyncio', request_type=apigateway.CreateApiConfigRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_api_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_api_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.CreateApiConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_api_config_async_from_dict():
    await test_create_api_config_async(request_type=dict)

def test_create_api_config_field_headers():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.CreateApiConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_api_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_api_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_api_config_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.CreateApiConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_api_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_api_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_api_config_flattened():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_api_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_api_config(parent='parent_value', api_config=apigateway.ApiConfig(name='name_value'), api_config_id='api_config_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].api_config
        mock_val = apigateway.ApiConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].api_config_id
        mock_val = 'api_config_id_value'
        assert arg == mock_val

def test_create_api_config_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_api_config(apigateway.CreateApiConfigRequest(), parent='parent_value', api_config=apigateway.ApiConfig(name='name_value'), api_config_id='api_config_id_value')

@pytest.mark.asyncio
async def test_create_api_config_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_api_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_api_config(parent='parent_value', api_config=apigateway.ApiConfig(name='name_value'), api_config_id='api_config_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].api_config
        mock_val = apigateway.ApiConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].api_config_id
        mock_val = 'api_config_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_api_config_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_api_config(apigateway.CreateApiConfigRequest(), parent='parent_value', api_config=apigateway.ApiConfig(name='name_value'), api_config_id='api_config_id_value')

@pytest.mark.parametrize('request_type', [apigateway.UpdateApiConfigRequest, dict])
def test_update_api_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_api_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_api_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.UpdateApiConfigRequest()
    assert isinstance(response, future.Future)

def test_update_api_config_empty_call():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_api_config), '__call__') as call:
        client.update_api_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.UpdateApiConfigRequest()

@pytest.mark.asyncio
async def test_update_api_config_async(transport: str='grpc_asyncio', request_type=apigateway.UpdateApiConfigRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_api_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_api_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.UpdateApiConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_api_config_async_from_dict():
    await test_update_api_config_async(request_type=dict)

def test_update_api_config_field_headers():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.UpdateApiConfigRequest()
    request.api_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_api_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_api_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'api_config.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_api_config_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.UpdateApiConfigRequest()
    request.api_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_api_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_api_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'api_config.name=name_value') in kw['metadata']

def test_update_api_config_flattened():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_api_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_api_config(api_config=apigateway.ApiConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].api_config
        mock_val = apigateway.ApiConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_api_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_api_config(apigateway.UpdateApiConfigRequest(), api_config=apigateway.ApiConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_api_config_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_api_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_api_config(api_config=apigateway.ApiConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].api_config
        mock_val = apigateway.ApiConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_api_config_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_api_config(apigateway.UpdateApiConfigRequest(), api_config=apigateway.ApiConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [apigateway.DeleteApiConfigRequest, dict])
def test_delete_api_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_api_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_api_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.DeleteApiConfigRequest()
    assert isinstance(response, future.Future)

def test_delete_api_config_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_api_config), '__call__') as call:
        client.delete_api_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.DeleteApiConfigRequest()

@pytest.mark.asyncio
async def test_delete_api_config_async(transport: str='grpc_asyncio', request_type=apigateway.DeleteApiConfigRequest):
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_api_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_api_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == apigateway.DeleteApiConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_api_config_async_from_dict():
    await test_delete_api_config_async(request_type=dict)

def test_delete_api_config_field_headers():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.DeleteApiConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_api_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_api_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_api_config_field_headers_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = apigateway.DeleteApiConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_api_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_api_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_api_config_flattened():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_api_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_api_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_api_config_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_api_config(apigateway.DeleteApiConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_api_config_flattened_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_api_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_api_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_api_config_flattened_error_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_api_config(apigateway.DeleteApiConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [apigateway.ListGatewaysRequest, dict])
def test_list_gateways_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = apigateway.ListGatewaysResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = apigateway.ListGatewaysResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_gateways(request)
    assert isinstance(response, pagers.ListGatewaysPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

def test_list_gateways_rest_required_fields(request_type=apigateway.ListGatewaysRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_gateways._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_gateways._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = apigateway.ListGatewaysResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = apigateway.ListGatewaysResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_gateways(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_gateways_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_gateways._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_gateways_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_list_gateways') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_list_gateways') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.ListGatewaysRequest.pb(apigateway.ListGatewaysRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = apigateway.ListGatewaysResponse.to_json(apigateway.ListGatewaysResponse())
        request = apigateway.ListGatewaysRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = apigateway.ListGatewaysResponse()
        client.list_gateways(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_gateways_rest_bad_request(transport: str='rest', request_type=apigateway.ListGatewaysRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_gateways(request)

def test_list_gateways_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = apigateway.ListGatewaysResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = apigateway.ListGatewaysResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_gateways(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/gateways' % client.transport._host, args[1])

def test_list_gateways_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_gateways(apigateway.ListGatewaysRequest(), parent='parent_value')

def test_list_gateways_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway(), apigateway.Gateway(), apigateway.Gateway()], next_page_token='abc'), apigateway.ListGatewaysResponse(gateways=[], next_page_token='def'), apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway()], next_page_token='ghi'), apigateway.ListGatewaysResponse(gateways=[apigateway.Gateway(), apigateway.Gateway()]))
        response = response + response
        response = tuple((apigateway.ListGatewaysResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_gateways(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, apigateway.Gateway) for i in results))
        pages = list(client.list_gateways(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [apigateway.GetGatewayRequest, dict])
def test_get_gateway_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/gateways/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = apigateway.Gateway(name='name_value', display_name='display_name_value', api_config='api_config_value', state=apigateway.Gateway.State.CREATING, default_hostname='default_hostname_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = apigateway.Gateway.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_gateway(request)
    assert isinstance(response, apigateway.Gateway)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.api_config == 'api_config_value'
    assert response.state == apigateway.Gateway.State.CREATING
    assert response.default_hostname == 'default_hostname_value'

def test_get_gateway_rest_required_fields(request_type=apigateway.GetGatewayRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_gateway._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_gateway._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = apigateway.Gateway()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = apigateway.Gateway.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_gateway(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_gateway_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_gateway._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_gateway_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_get_gateway') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_get_gateway') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.GetGatewayRequest.pb(apigateway.GetGatewayRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = apigateway.Gateway.to_json(apigateway.Gateway())
        request = apigateway.GetGatewayRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = apigateway.Gateway()
        client.get_gateway(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_gateway_rest_bad_request(transport: str='rest', request_type=apigateway.GetGatewayRequest):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/gateways/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_gateway(request)

def test_get_gateway_rest_flattened():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = apigateway.Gateway()
        sample_request = {'name': 'projects/sample1/locations/sample2/gateways/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = apigateway.Gateway.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_gateway(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/gateways/*}' % client.transport._host, args[1])

def test_get_gateway_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_gateway(apigateway.GetGatewayRequest(), name='name_value')

def test_get_gateway_rest_error():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [apigateway.CreateGatewayRequest, dict])
def test_create_gateway_rest(request_type):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['gateway'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'display_name': 'display_name_value', 'api_config': 'api_config_value', 'state': 1, 'default_hostname': 'default_hostname_value'}
    test_field = apigateway.CreateGatewayRequest.meta.fields['gateway']

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
    for (field, value) in request_init['gateway'].items():
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
                for i in range(0, len(request_init['gateway'][field])):
                    del request_init['gateway'][field][i][subfield]
            else:
                del request_init['gateway'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_gateway(request)
    assert response.operation.name == 'operations/spam'

def test_create_gateway_rest_required_fields(request_type=apigateway.CreateGatewayRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['gateway_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'gatewayId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_gateway._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'gatewayId' in jsonified_request
    assert jsonified_request['gatewayId'] == request_init['gateway_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['gatewayId'] = 'gateway_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_gateway._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('gateway_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'gatewayId' in jsonified_request
    assert jsonified_request['gatewayId'] == 'gateway_id_value'
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_gateway(request)
            expected_params = [('gatewayId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_gateway_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_gateway._get_unset_required_fields({})
    assert set(unset_fields) == set(('gatewayId',)) & set(('parent', 'gatewayId', 'gateway'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_gateway_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_create_gateway') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_create_gateway') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.CreateGatewayRequest.pb(apigateway.CreateGatewayRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = apigateway.CreateGatewayRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_gateway(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_gateway_rest_bad_request(transport: str='rest', request_type=apigateway.CreateGatewayRequest):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_gateway(request)

def test_create_gateway_rest_flattened():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', gateway=apigateway.Gateway(name='name_value'), gateway_id='gateway_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_gateway(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/gateways' % client.transport._host, args[1])

def test_create_gateway_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_gateway(apigateway.CreateGatewayRequest(), parent='parent_value', gateway=apigateway.Gateway(name='name_value'), gateway_id='gateway_id_value')

def test_create_gateway_rest_error():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [apigateway.UpdateGatewayRequest, dict])
def test_update_gateway_rest(request_type):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'gateway': {'name': 'projects/sample1/locations/sample2/gateways/sample3'}}
    request_init['gateway'] = {'name': 'projects/sample1/locations/sample2/gateways/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'display_name': 'display_name_value', 'api_config': 'api_config_value', 'state': 1, 'default_hostname': 'default_hostname_value'}
    test_field = apigateway.UpdateGatewayRequest.meta.fields['gateway']

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
    for (field, value) in request_init['gateway'].items():
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
                for i in range(0, len(request_init['gateway'][field])):
                    del request_init['gateway'][field][i][subfield]
            else:
                del request_init['gateway'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_gateway(request)
    assert response.operation.name == 'operations/spam'

def test_update_gateway_rest_required_fields(request_type=apigateway.UpdateGatewayRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_gateway._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_gateway._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_gateway(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_gateway_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_gateway._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('gateway',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_gateway_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_update_gateway') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_update_gateway') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.UpdateGatewayRequest.pb(apigateway.UpdateGatewayRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = apigateway.UpdateGatewayRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_gateway(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_gateway_rest_bad_request(transport: str='rest', request_type=apigateway.UpdateGatewayRequest):
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'gateway': {'name': 'projects/sample1/locations/sample2/gateways/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_gateway(request)

def test_update_gateway_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'gateway': {'name': 'projects/sample1/locations/sample2/gateways/sample3'}}
        mock_args = dict(gateway=apigateway.Gateway(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_gateway(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{gateway.name=projects/*/locations/*/gateways/*}' % client.transport._host, args[1])

def test_update_gateway_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_gateway(apigateway.UpdateGatewayRequest(), gateway=apigateway.Gateway(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_gateway_rest_error():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [apigateway.DeleteGatewayRequest, dict])
def test_delete_gateway_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/gateways/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_gateway(request)
    assert response.operation.name == 'operations/spam'

def test_delete_gateway_rest_required_fields(request_type=apigateway.DeleteGatewayRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_gateway._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_gateway._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_gateway(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_gateway_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_gateway._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_gateway_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_delete_gateway') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_delete_gateway') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.DeleteGatewayRequest.pb(apigateway.DeleteGatewayRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = apigateway.DeleteGatewayRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_gateway(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_gateway_rest_bad_request(transport: str='rest', request_type=apigateway.DeleteGatewayRequest):
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/gateways/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_gateway(request)

def test_delete_gateway_rest_flattened():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/gateways/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_gateway(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/gateways/*}' % client.transport._host, args[1])

def test_delete_gateway_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_gateway(apigateway.DeleteGatewayRequest(), name='name_value')

def test_delete_gateway_rest_error():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [apigateway.ListApisRequest, dict])
def test_list_apis_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = apigateway.ListApisResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = apigateway.ListApisResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_apis(request)
    assert isinstance(response, pagers.ListApisPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

def test_list_apis_rest_required_fields(request_type=apigateway.ListApisRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_apis._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_apis._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = apigateway.ListApisResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = apigateway.ListApisResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_apis(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_apis_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_apis._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_apis_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_list_apis') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_list_apis') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.ListApisRequest.pb(apigateway.ListApisRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = apigateway.ListApisResponse.to_json(apigateway.ListApisResponse())
        request = apigateway.ListApisRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = apigateway.ListApisResponse()
        client.list_apis(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_apis_rest_bad_request(transport: str='rest', request_type=apigateway.ListApisRequest):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_apis(request)

def test_list_apis_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = apigateway.ListApisResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = apigateway.ListApisResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_apis(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/apis' % client.transport._host, args[1])

def test_list_apis_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_apis(apigateway.ListApisRequest(), parent='parent_value')

def test_list_apis_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (apigateway.ListApisResponse(apis=[apigateway.Api(), apigateway.Api(), apigateway.Api()], next_page_token='abc'), apigateway.ListApisResponse(apis=[], next_page_token='def'), apigateway.ListApisResponse(apis=[apigateway.Api()], next_page_token='ghi'), apigateway.ListApisResponse(apis=[apigateway.Api(), apigateway.Api()]))
        response = response + response
        response = tuple((apigateway.ListApisResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_apis(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, apigateway.Api) for i in results))
        pages = list(client.list_apis(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [apigateway.GetApiRequest, dict])
def test_get_api_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/apis/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = apigateway.Api(name='name_value', display_name='display_name_value', managed_service='managed_service_value', state=apigateway.Api.State.CREATING)
        response_value = Response()
        response_value.status_code = 200
        return_value = apigateway.Api.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_api(request)
    assert isinstance(response, apigateway.Api)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.managed_service == 'managed_service_value'
    assert response.state == apigateway.Api.State.CREATING

def test_get_api_rest_required_fields(request_type=apigateway.GetApiRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_api._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_api._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = apigateway.Api()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = apigateway.Api.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_api(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_api_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_api._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_api_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_get_api') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_get_api') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.GetApiRequest.pb(apigateway.GetApiRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = apigateway.Api.to_json(apigateway.Api())
        request = apigateway.GetApiRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = apigateway.Api()
        client.get_api(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_api_rest_bad_request(transport: str='rest', request_type=apigateway.GetApiRequest):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/apis/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_api(request)

def test_get_api_rest_flattened():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = apigateway.Api()
        sample_request = {'name': 'projects/sample1/locations/sample2/apis/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = apigateway.Api.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_api(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/apis/*}' % client.transport._host, args[1])

def test_get_api_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_api(apigateway.GetApiRequest(), name='name_value')

def test_get_api_rest_error():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [apigateway.CreateApiRequest, dict])
def test_create_api_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['api'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'display_name': 'display_name_value', 'managed_service': 'managed_service_value', 'state': 1}
    test_field = apigateway.CreateApiRequest.meta.fields['api']

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
    for (field, value) in request_init['api'].items():
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
                for i in range(0, len(request_init['api'][field])):
                    del request_init['api'][field][i][subfield]
            else:
                del request_init['api'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_api(request)
    assert response.operation.name == 'operations/spam'

def test_create_api_rest_required_fields(request_type=apigateway.CreateApiRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['api_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'apiId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_api._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'apiId' in jsonified_request
    assert jsonified_request['apiId'] == request_init['api_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['apiId'] = 'api_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_api._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('api_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'apiId' in jsonified_request
    assert jsonified_request['apiId'] == 'api_id_value'
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_api(request)
            expected_params = [('apiId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_api_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_api._get_unset_required_fields({})
    assert set(unset_fields) == set(('apiId',)) & set(('parent', 'apiId', 'api'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_api_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_create_api') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_create_api') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.CreateApiRequest.pb(apigateway.CreateApiRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = apigateway.CreateApiRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_api(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_api_rest_bad_request(transport: str='rest', request_type=apigateway.CreateApiRequest):
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_api(request)

def test_create_api_rest_flattened():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', api=apigateway.Api(name='name_value'), api_id='api_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_api(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/apis' % client.transport._host, args[1])

def test_create_api_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_api(apigateway.CreateApiRequest(), parent='parent_value', api=apigateway.Api(name='name_value'), api_id='api_id_value')

def test_create_api_rest_error():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [apigateway.UpdateApiRequest, dict])
def test_update_api_rest(request_type):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'api': {'name': 'projects/sample1/locations/sample2/apis/sample3'}}
    request_init['api'] = {'name': 'projects/sample1/locations/sample2/apis/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'display_name': 'display_name_value', 'managed_service': 'managed_service_value', 'state': 1}
    test_field = apigateway.UpdateApiRequest.meta.fields['api']

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
    for (field, value) in request_init['api'].items():
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
                for i in range(0, len(request_init['api'][field])):
                    del request_init['api'][field][i][subfield]
            else:
                del request_init['api'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_api(request)
    assert response.operation.name == 'operations/spam'

def test_update_api_rest_required_fields(request_type=apigateway.UpdateApiRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_api._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_api._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_api(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_api_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_api._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('api',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_api_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_update_api') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_update_api') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.UpdateApiRequest.pb(apigateway.UpdateApiRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = apigateway.UpdateApiRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_api(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_api_rest_bad_request(transport: str='rest', request_type=apigateway.UpdateApiRequest):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'api': {'name': 'projects/sample1/locations/sample2/apis/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_api(request)

def test_update_api_rest_flattened():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'api': {'name': 'projects/sample1/locations/sample2/apis/sample3'}}
        mock_args = dict(api=apigateway.Api(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_api(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{api.name=projects/*/locations/*/apis/*}' % client.transport._host, args[1])

def test_update_api_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_api(apigateway.UpdateApiRequest(), api=apigateway.Api(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_api_rest_error():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [apigateway.DeleteApiRequest, dict])
def test_delete_api_rest(request_type):
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/apis/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_api(request)
    assert response.operation.name == 'operations/spam'

def test_delete_api_rest_required_fields(request_type=apigateway.DeleteApiRequest):
    if False:
        return 10
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_api._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_api._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_api(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_api_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_api._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_api_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_delete_api') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_delete_api') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.DeleteApiRequest.pb(apigateway.DeleteApiRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = apigateway.DeleteApiRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_api(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_api_rest_bad_request(transport: str='rest', request_type=apigateway.DeleteApiRequest):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/apis/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_api(request)

def test_delete_api_rest_flattened():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/apis/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_api(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/apis/*}' % client.transport._host, args[1])

def test_delete_api_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_api(apigateway.DeleteApiRequest(), name='name_value')

def test_delete_api_rest_error():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [apigateway.ListApiConfigsRequest, dict])
def test_list_api_configs_rest(request_type):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/apis/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = apigateway.ListApiConfigsResponse(next_page_token='next_page_token_value', unreachable_locations=['unreachable_locations_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = apigateway.ListApiConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_api_configs(request)
    assert isinstance(response, pagers.ListApiConfigsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable_locations == ['unreachable_locations_value']

def test_list_api_configs_rest_required_fields(request_type=apigateway.ListApiConfigsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_api_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_api_configs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = apigateway.ListApiConfigsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = apigateway.ListApiConfigsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_api_configs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_api_configs_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_api_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_api_configs_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_list_api_configs') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_list_api_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.ListApiConfigsRequest.pb(apigateway.ListApiConfigsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = apigateway.ListApiConfigsResponse.to_json(apigateway.ListApiConfigsResponse())
        request = apigateway.ListApiConfigsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = apigateway.ListApiConfigsResponse()
        client.list_api_configs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_api_configs_rest_bad_request(transport: str='rest', request_type=apigateway.ListApiConfigsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/apis/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_api_configs(request)

def test_list_api_configs_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = apigateway.ListApiConfigsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/apis/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = apigateway.ListApiConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_api_configs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/apis/*}/configs' % client.transport._host, args[1])

def test_list_api_configs_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_api_configs(apigateway.ListApiConfigsRequest(), parent='parent_value')

def test_list_api_configs_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig(), apigateway.ApiConfig(), apigateway.ApiConfig()], next_page_token='abc'), apigateway.ListApiConfigsResponse(api_configs=[], next_page_token='def'), apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig()], next_page_token='ghi'), apigateway.ListApiConfigsResponse(api_configs=[apigateway.ApiConfig(), apigateway.ApiConfig()]))
        response = response + response
        response = tuple((apigateway.ListApiConfigsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/apis/sample3'}
        pager = client.list_api_configs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, apigateway.ApiConfig) for i in results))
        pages = list(client.list_api_configs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [apigateway.GetApiConfigRequest, dict])
def test_get_api_config_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/apis/sample3/configs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = apigateway.ApiConfig(name='name_value', display_name='display_name_value', gateway_service_account='gateway_service_account_value', service_config_id='service_config_id_value', state=apigateway.ApiConfig.State.CREATING)
        response_value = Response()
        response_value.status_code = 200
        return_value = apigateway.ApiConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_api_config(request)
    assert isinstance(response, apigateway.ApiConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.gateway_service_account == 'gateway_service_account_value'
    assert response.service_config_id == 'service_config_id_value'
    assert response.state == apigateway.ApiConfig.State.CREATING

def test_get_api_config_rest_required_fields(request_type=apigateway.GetApiConfigRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_api_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_api_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('view',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = apigateway.ApiConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = apigateway.ApiConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_api_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_api_config_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_api_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('view',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_api_config_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_get_api_config') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_get_api_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.GetApiConfigRequest.pb(apigateway.GetApiConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = apigateway.ApiConfig.to_json(apigateway.ApiConfig())
        request = apigateway.GetApiConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = apigateway.ApiConfig()
        client.get_api_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_api_config_rest_bad_request(transport: str='rest', request_type=apigateway.GetApiConfigRequest):
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/apis/sample3/configs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_api_config(request)

def test_get_api_config_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = apigateway.ApiConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/apis/sample3/configs/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = apigateway.ApiConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_api_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/apis/*/configs/*}' % client.transport._host, args[1])

def test_get_api_config_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_api_config(apigateway.GetApiConfigRequest(), name='name_value')

def test_get_api_config_rest_error():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [apigateway.CreateApiConfigRequest, dict])
def test_create_api_config_rest(request_type):
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/apis/sample3'}
    request_init['api_config'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'display_name': 'display_name_value', 'gateway_service_account': 'gateway_service_account_value', 'service_config_id': 'service_config_id_value', 'state': 1, 'openapi_documents': [{'document': {'path': 'path_value', 'contents': b'contents_blob'}}], 'grpc_services': [{'file_descriptor_set': {}, 'source': {}}], 'managed_service_configs': {}}
    test_field = apigateway.CreateApiConfigRequest.meta.fields['api_config']

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
    for (field, value) in request_init['api_config'].items():
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
                for i in range(0, len(request_init['api_config'][field])):
                    del request_init['api_config'][field][i][subfield]
            else:
                del request_init['api_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_api_config(request)
    assert response.operation.name == 'operations/spam'

def test_create_api_config_rest_required_fields(request_type=apigateway.CreateApiConfigRequest):
    if False:
        return 10
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['api_config_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'apiConfigId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_api_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'apiConfigId' in jsonified_request
    assert jsonified_request['apiConfigId'] == request_init['api_config_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['apiConfigId'] = 'api_config_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_api_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('api_config_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'apiConfigId' in jsonified_request
    assert jsonified_request['apiConfigId'] == 'api_config_id_value'
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_api_config(request)
            expected_params = [('apiConfigId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_api_config_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_api_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('apiConfigId',)) & set(('parent', 'apiConfigId', 'apiConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_api_config_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_create_api_config') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_create_api_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.CreateApiConfigRequest.pb(apigateway.CreateApiConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = apigateway.CreateApiConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_api_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_api_config_rest_bad_request(transport: str='rest', request_type=apigateway.CreateApiConfigRequest):
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/apis/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_api_config(request)

def test_create_api_config_rest_flattened():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/apis/sample3'}
        mock_args = dict(parent='parent_value', api_config=apigateway.ApiConfig(name='name_value'), api_config_id='api_config_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_api_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/apis/*}/configs' % client.transport._host, args[1])

def test_create_api_config_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_api_config(apigateway.CreateApiConfigRequest(), parent='parent_value', api_config=apigateway.ApiConfig(name='name_value'), api_config_id='api_config_id_value')

def test_create_api_config_rest_error():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [apigateway.UpdateApiConfigRequest, dict])
def test_update_api_config_rest(request_type):
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'api_config': {'name': 'projects/sample1/locations/sample2/apis/sample3/configs/sample4'}}
    request_init['api_config'] = {'name': 'projects/sample1/locations/sample2/apis/sample3/configs/sample4', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'display_name': 'display_name_value', 'gateway_service_account': 'gateway_service_account_value', 'service_config_id': 'service_config_id_value', 'state': 1, 'openapi_documents': [{'document': {'path': 'path_value', 'contents': b'contents_blob'}}], 'grpc_services': [{'file_descriptor_set': {}, 'source': {}}], 'managed_service_configs': {}}
    test_field = apigateway.UpdateApiConfigRequest.meta.fields['api_config']

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
    for (field, value) in request_init['api_config'].items():
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
                for i in range(0, len(request_init['api_config'][field])):
                    del request_init['api_config'][field][i][subfield]
            else:
                del request_init['api_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_api_config(request)
    assert response.operation.name == 'operations/spam'

def test_update_api_config_rest_required_fields(request_type=apigateway.UpdateApiConfigRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_api_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_api_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_api_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_api_config_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_api_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('apiConfig',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_api_config_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_update_api_config') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_update_api_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.UpdateApiConfigRequest.pb(apigateway.UpdateApiConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = apigateway.UpdateApiConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_api_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_api_config_rest_bad_request(transport: str='rest', request_type=apigateway.UpdateApiConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'api_config': {'name': 'projects/sample1/locations/sample2/apis/sample3/configs/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_api_config(request)

def test_update_api_config_rest_flattened():
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'api_config': {'name': 'projects/sample1/locations/sample2/apis/sample3/configs/sample4'}}
        mock_args = dict(api_config=apigateway.ApiConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_api_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{api_config.name=projects/*/locations/*/apis/*/configs/*}' % client.transport._host, args[1])

def test_update_api_config_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_api_config(apigateway.UpdateApiConfigRequest(), api_config=apigateway.ApiConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_api_config_rest_error():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [apigateway.DeleteApiConfigRequest, dict])
def test_delete_api_config_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/apis/sample3/configs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_api_config(request)
    assert response.operation.name == 'operations/spam'

def test_delete_api_config_rest_required_fields(request_type=apigateway.DeleteApiConfigRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ApiGatewayServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_api_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_api_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_api_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_api_config_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_api_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_api_config_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ApiGatewayServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ApiGatewayServiceRestInterceptor())
    client = ApiGatewayServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'post_delete_api_config') as post, mock.patch.object(transports.ApiGatewayServiceRestInterceptor, 'pre_delete_api_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = apigateway.DeleteApiConfigRequest.pb(apigateway.DeleteApiConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = apigateway.DeleteApiConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_api_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_api_config_rest_bad_request(transport: str='rest', request_type=apigateway.DeleteApiConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/apis/sample3/configs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_api_config(request)

def test_delete_api_config_rest_flattened():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/apis/sample3/configs/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_api_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/apis/*/configs/*}' % client.transport._host, args[1])

def test_delete_api_config_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_api_config(apigateway.DeleteApiConfigRequest(), name='name_value')

def test_delete_api_config_rest_error():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ApiGatewayServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ApiGatewayServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ApiGatewayServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ApiGatewayServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ApiGatewayServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ApiGatewayServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ApiGatewayServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ApiGatewayServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.ApiGatewayServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ApiGatewayServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ApiGatewayServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ApiGatewayServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ApiGatewayServiceGrpcTransport, transports.ApiGatewayServiceGrpcAsyncIOTransport, transports.ApiGatewayServiceRestTransport])
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
        for i in range(10):
            print('nop')
    transport = ApiGatewayServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ApiGatewayServiceGrpcTransport)

def test_api_gateway_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ApiGatewayServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_api_gateway_service_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.apigateway_v1.services.api_gateway_service.transports.ApiGatewayServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ApiGatewayServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_gateways', 'get_gateway', 'create_gateway', 'update_gateway', 'delete_gateway', 'list_apis', 'get_api', 'create_api', 'update_api', 'delete_api', 'list_api_configs', 'get_api_config', 'create_api_config', 'update_api_config', 'delete_api_config')
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

def test_api_gateway_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.apigateway_v1.services.api_gateway_service.transports.ApiGatewayServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ApiGatewayServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_api_gateway_service_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.apigateway_v1.services.api_gateway_service.transports.ApiGatewayServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ApiGatewayServiceTransport()
        adc.assert_called_once()

def test_api_gateway_service_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ApiGatewayServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ApiGatewayServiceGrpcTransport, transports.ApiGatewayServiceGrpcAsyncIOTransport])
def test_api_gateway_service_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ApiGatewayServiceGrpcTransport, transports.ApiGatewayServiceGrpcAsyncIOTransport, transports.ApiGatewayServiceRestTransport])
def test_api_gateway_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ApiGatewayServiceGrpcTransport, grpc_helpers), (transports.ApiGatewayServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_api_gateway_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('apigateway.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='apigateway.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ApiGatewayServiceGrpcTransport, transports.ApiGatewayServiceGrpcAsyncIOTransport])
def test_api_gateway_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_api_gateway_service_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.ApiGatewayServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_api_gateway_service_rest_lro_client():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_api_gateway_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='apigateway.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('apigateway.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://apigateway.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_api_gateway_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='apigateway.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('apigateway.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://apigateway.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_api_gateway_service_client_transport_session_collision(transport_name):
    if False:
        while True:
            i = 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = ApiGatewayServiceClient(credentials=creds1, transport=transport_name)
    client2 = ApiGatewayServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_gateways._session
    session2 = client2.transport.list_gateways._session
    assert session1 != session2
    session1 = client1.transport.get_gateway._session
    session2 = client2.transport.get_gateway._session
    assert session1 != session2
    session1 = client1.transport.create_gateway._session
    session2 = client2.transport.create_gateway._session
    assert session1 != session2
    session1 = client1.transport.update_gateway._session
    session2 = client2.transport.update_gateway._session
    assert session1 != session2
    session1 = client1.transport.delete_gateway._session
    session2 = client2.transport.delete_gateway._session
    assert session1 != session2
    session1 = client1.transport.list_apis._session
    session2 = client2.transport.list_apis._session
    assert session1 != session2
    session1 = client1.transport.get_api._session
    session2 = client2.transport.get_api._session
    assert session1 != session2
    session1 = client1.transport.create_api._session
    session2 = client2.transport.create_api._session
    assert session1 != session2
    session1 = client1.transport.update_api._session
    session2 = client2.transport.update_api._session
    assert session1 != session2
    session1 = client1.transport.delete_api._session
    session2 = client2.transport.delete_api._session
    assert session1 != session2
    session1 = client1.transport.list_api_configs._session
    session2 = client2.transport.list_api_configs._session
    assert session1 != session2
    session1 = client1.transport.get_api_config._session
    session2 = client2.transport.get_api_config._session
    assert session1 != session2
    session1 = client1.transport.create_api_config._session
    session2 = client2.transport.create_api_config._session
    assert session1 != session2
    session1 = client1.transport.update_api_config._session
    session2 = client2.transport.update_api_config._session
    assert session1 != session2
    session1 = client1.transport.delete_api_config._session
    session2 = client2.transport.delete_api_config._session
    assert session1 != session2

def test_api_gateway_service_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ApiGatewayServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_api_gateway_service_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ApiGatewayServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ApiGatewayServiceGrpcTransport, transports.ApiGatewayServiceGrpcAsyncIOTransport])
def test_api_gateway_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.ApiGatewayServiceGrpcTransport, transports.ApiGatewayServiceGrpcAsyncIOTransport])
def test_api_gateway_service_transport_channel_mtls_with_adc(transport_class):
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

def test_api_gateway_service_grpc_lro_client():
    if False:
        print('Hello World!')
    client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_api_gateway_service_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_api_path():
    if False:
        print('Hello World!')
    project = 'squid'
    api = 'clam'
    expected = 'projects/{project}/locations/global/apis/{api}'.format(project=project, api=api)
    actual = ApiGatewayServiceClient.api_path(project, api)
    assert expected == actual

def test_parse_api_path():
    if False:
        print('Hello World!')
    expected = {'project': 'whelk', 'api': 'octopus'}
    path = ApiGatewayServiceClient.api_path(**expected)
    actual = ApiGatewayServiceClient.parse_api_path(path)
    assert expected == actual

def test_api_config_path():
    if False:
        return 10
    project = 'oyster'
    api = 'nudibranch'
    api_config = 'cuttlefish'
    expected = 'projects/{project}/locations/global/apis/{api}/configs/{api_config}'.format(project=project, api=api, api_config=api_config)
    actual = ApiGatewayServiceClient.api_config_path(project, api, api_config)
    assert expected == actual

def test_parse_api_config_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'mussel', 'api': 'winkle', 'api_config': 'nautilus'}
    path = ApiGatewayServiceClient.api_config_path(**expected)
    actual = ApiGatewayServiceClient.parse_api_config_path(path)
    assert expected == actual

def test_gateway_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    location = 'abalone'
    gateway = 'squid'
    expected = 'projects/{project}/locations/{location}/gateways/{gateway}'.format(project=project, location=location, gateway=gateway)
    actual = ApiGatewayServiceClient.gateway_path(project, location, gateway)
    assert expected == actual

def test_parse_gateway_path():
    if False:
        print('Hello World!')
    expected = {'project': 'clam', 'location': 'whelk', 'gateway': 'octopus'}
    path = ApiGatewayServiceClient.gateway_path(**expected)
    actual = ApiGatewayServiceClient.parse_gateway_path(path)
    assert expected == actual

def test_managed_service_path():
    if False:
        while True:
            i = 10
    service = 'oyster'
    expected = 'services/{service}'.format(service=service)
    actual = ApiGatewayServiceClient.managed_service_path(service)
    assert expected == actual

def test_parse_managed_service_path():
    if False:
        i = 10
        return i + 15
    expected = {'service': 'nudibranch'}
    path = ApiGatewayServiceClient.managed_service_path(**expected)
    actual = ApiGatewayServiceClient.parse_managed_service_path(path)
    assert expected == actual

def test_service_path():
    if False:
        return 10
    service = 'cuttlefish'
    config = 'mussel'
    expected = 'services/{service}/configs/{config}'.format(service=service, config=config)
    actual = ApiGatewayServiceClient.service_path(service, config)
    assert expected == actual

def test_parse_service_path():
    if False:
        return 10
    expected = {'service': 'winkle', 'config': 'nautilus'}
    path = ApiGatewayServiceClient.service_path(**expected)
    actual = ApiGatewayServiceClient.parse_service_path(path)
    assert expected == actual

def test_service_account_path():
    if False:
        print('Hello World!')
    project = 'scallop'
    service_account = 'abalone'
    expected = 'projects/{project}/serviceAccounts/{service_account}'.format(project=project, service_account=service_account)
    actual = ApiGatewayServiceClient.service_account_path(project, service_account)
    assert expected == actual

def test_parse_service_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'squid', 'service_account': 'clam'}
    path = ApiGatewayServiceClient.service_account_path(**expected)
    actual = ApiGatewayServiceClient.parse_service_account_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ApiGatewayServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'octopus'}
    path = ApiGatewayServiceClient.common_billing_account_path(**expected)
    actual = ApiGatewayServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ApiGatewayServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'nudibranch'}
    path = ApiGatewayServiceClient.common_folder_path(**expected)
    actual = ApiGatewayServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ApiGatewayServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'mussel'}
    path = ApiGatewayServiceClient.common_organization_path(**expected)
    actual = ApiGatewayServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = ApiGatewayServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nautilus'}
    path = ApiGatewayServiceClient.common_project_path(**expected)
    actual = ApiGatewayServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ApiGatewayServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'squid', 'location': 'clam'}
    path = ApiGatewayServiceClient.common_location_path(**expected)
    actual = ApiGatewayServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ApiGatewayServiceTransport, '_prep_wrapped_messages') as prep:
        client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ApiGatewayServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = ApiGatewayServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ApiGatewayServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = ApiGatewayServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ApiGatewayServiceClient, transports.ApiGatewayServiceGrpcTransport), (ApiGatewayServiceAsyncClient, transports.ApiGatewayServiceGrpcAsyncIOTransport)])
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