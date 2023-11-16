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
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
from google.oauth2 import service_account
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.type import expr_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.secretmanager_v1beta1.services.secret_manager_service import SecretManagerServiceAsyncClient, SecretManagerServiceClient, pagers, transports
from google.cloud.secretmanager_v1beta1.types import resources, service

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
        for i in range(10):
            print('nop')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert SecretManagerServiceClient._get_default_mtls_endpoint(None) is None
    assert SecretManagerServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert SecretManagerServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert SecretManagerServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert SecretManagerServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert SecretManagerServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(SecretManagerServiceClient, 'grpc'), (SecretManagerServiceAsyncClient, 'grpc_asyncio'), (SecretManagerServiceClient, 'rest')])
def test_secret_manager_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('secretmanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://secretmanager.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.SecretManagerServiceGrpcTransport, 'grpc'), (transports.SecretManagerServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.SecretManagerServiceRestTransport, 'rest')])
def test_secret_manager_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(SecretManagerServiceClient, 'grpc'), (SecretManagerServiceAsyncClient, 'grpc_asyncio'), (SecretManagerServiceClient, 'rest')])
def test_secret_manager_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('secretmanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://secretmanager.googleapis.com')

def test_secret_manager_service_client_get_transport_class():
    if False:
        return 10
    transport = SecretManagerServiceClient.get_transport_class()
    available_transports = [transports.SecretManagerServiceGrpcTransport, transports.SecretManagerServiceRestTransport]
    assert transport in available_transports
    transport = SecretManagerServiceClient.get_transport_class('grpc')
    assert transport == transports.SecretManagerServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(SecretManagerServiceClient, transports.SecretManagerServiceGrpcTransport, 'grpc'), (SecretManagerServiceAsyncClient, transports.SecretManagerServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (SecretManagerServiceClient, transports.SecretManagerServiceRestTransport, 'rest')])
@mock.patch.object(SecretManagerServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecretManagerServiceClient))
@mock.patch.object(SecretManagerServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecretManagerServiceAsyncClient))
def test_secret_manager_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(SecretManagerServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(SecretManagerServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(SecretManagerServiceClient, transports.SecretManagerServiceGrpcTransport, 'grpc', 'true'), (SecretManagerServiceAsyncClient, transports.SecretManagerServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (SecretManagerServiceClient, transports.SecretManagerServiceGrpcTransport, 'grpc', 'false'), (SecretManagerServiceAsyncClient, transports.SecretManagerServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (SecretManagerServiceClient, transports.SecretManagerServiceRestTransport, 'rest', 'true'), (SecretManagerServiceClient, transports.SecretManagerServiceRestTransport, 'rest', 'false')])
@mock.patch.object(SecretManagerServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecretManagerServiceClient))
@mock.patch.object(SecretManagerServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecretManagerServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_secret_manager_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [SecretManagerServiceClient, SecretManagerServiceAsyncClient])
@mock.patch.object(SecretManagerServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecretManagerServiceClient))
@mock.patch.object(SecretManagerServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecretManagerServiceAsyncClient))
def test_secret_manager_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(SecretManagerServiceClient, transports.SecretManagerServiceGrpcTransport, 'grpc'), (SecretManagerServiceAsyncClient, transports.SecretManagerServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (SecretManagerServiceClient, transports.SecretManagerServiceRestTransport, 'rest')])
def test_secret_manager_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(SecretManagerServiceClient, transports.SecretManagerServiceGrpcTransport, 'grpc', grpc_helpers), (SecretManagerServiceAsyncClient, transports.SecretManagerServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (SecretManagerServiceClient, transports.SecretManagerServiceRestTransport, 'rest', None)])
def test_secret_manager_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_secret_manager_service_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.secretmanager_v1beta1.services.secret_manager_service.transports.SecretManagerServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = SecretManagerServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(SecretManagerServiceClient, transports.SecretManagerServiceGrpcTransport, 'grpc', grpc_helpers), (SecretManagerServiceAsyncClient, transports.SecretManagerServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_secret_manager_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('secretmanager.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='secretmanager.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [service.ListSecretsRequest, dict])
def test_list_secrets(request_type, transport: str='grpc'):
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_secrets), '__call__') as call:
        call.return_value = service.ListSecretsResponse(next_page_token='next_page_token_value', total_size=1086)
        response = client.list_secrets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSecretsRequest()
    assert isinstance(response, pagers.ListSecretsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_secrets_empty_call():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_secrets), '__call__') as call:
        client.list_secrets()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSecretsRequest()

@pytest.mark.asyncio
async def test_list_secrets_async(transport: str='grpc_asyncio', request_type=service.ListSecretsRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_secrets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSecretsResponse(next_page_token='next_page_token_value', total_size=1086))
        response = await client.list_secrets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSecretsRequest()
    assert isinstance(response, pagers.ListSecretsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

@pytest.mark.asyncio
async def test_list_secrets_async_from_dict():
    await test_list_secrets_async(request_type=dict)

def test_list_secrets_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSecretsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_secrets), '__call__') as call:
        call.return_value = service.ListSecretsResponse()
        client.list_secrets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_secrets_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSecretsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_secrets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSecretsResponse())
        await client.list_secrets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_secrets_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_secrets), '__call__') as call:
        call.return_value = service.ListSecretsResponse()
        client.list_secrets(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_secrets_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_secrets(service.ListSecretsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_secrets_flattened_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_secrets), '__call__') as call:
        call.return_value = service.ListSecretsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSecretsResponse())
        response = await client.list_secrets(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_secrets_flattened_error_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_secrets(service.ListSecretsRequest(), parent='parent_value')

def test_list_secrets_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_secrets), '__call__') as call:
        call.side_effect = (service.ListSecretsResponse(secrets=[resources.Secret(), resources.Secret(), resources.Secret()], next_page_token='abc'), service.ListSecretsResponse(secrets=[], next_page_token='def'), service.ListSecretsResponse(secrets=[resources.Secret()], next_page_token='ghi'), service.ListSecretsResponse(secrets=[resources.Secret(), resources.Secret()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_secrets(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Secret) for i in results))

def test_list_secrets_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_secrets), '__call__') as call:
        call.side_effect = (service.ListSecretsResponse(secrets=[resources.Secret(), resources.Secret(), resources.Secret()], next_page_token='abc'), service.ListSecretsResponse(secrets=[], next_page_token='def'), service.ListSecretsResponse(secrets=[resources.Secret()], next_page_token='ghi'), service.ListSecretsResponse(secrets=[resources.Secret(), resources.Secret()]), RuntimeError)
        pages = list(client.list_secrets(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_secrets_async_pager():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_secrets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSecretsResponse(secrets=[resources.Secret(), resources.Secret(), resources.Secret()], next_page_token='abc'), service.ListSecretsResponse(secrets=[], next_page_token='def'), service.ListSecretsResponse(secrets=[resources.Secret()], next_page_token='ghi'), service.ListSecretsResponse(secrets=[resources.Secret(), resources.Secret()]), RuntimeError)
        async_pager = await client.list_secrets(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Secret) for i in responses))

@pytest.mark.asyncio
async def test_list_secrets_async_pages():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_secrets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSecretsResponse(secrets=[resources.Secret(), resources.Secret(), resources.Secret()], next_page_token='abc'), service.ListSecretsResponse(secrets=[], next_page_token='def'), service.ListSecretsResponse(secrets=[resources.Secret()], next_page_token='ghi'), service.ListSecretsResponse(secrets=[resources.Secret(), resources.Secret()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_secrets(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.CreateSecretRequest, dict])
def test_create_secret(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_secret), '__call__') as call:
        call.return_value = resources.Secret(name='name_value')
        response = client.create_secret(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateSecretRequest()
    assert isinstance(response, resources.Secret)
    assert response.name == 'name_value'

def test_create_secret_empty_call():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_secret), '__call__') as call:
        client.create_secret()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateSecretRequest()

@pytest.mark.asyncio
async def test_create_secret_async(transport: str='grpc_asyncio', request_type=service.CreateSecretRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_secret), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Secret(name='name_value'))
        response = await client.create_secret(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateSecretRequest()
    assert isinstance(response, resources.Secret)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_create_secret_async_from_dict():
    await test_create_secret_async(request_type=dict)

def test_create_secret_field_headers():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateSecretRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_secret), '__call__') as call:
        call.return_value = resources.Secret()
        client.create_secret(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_secret_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateSecretRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_secret), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Secret())
        await client.create_secret(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_secret_flattened():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_secret), '__call__') as call:
        call.return_value = resources.Secret()
        client.create_secret(parent='parent_value', secret_id='secret_id_value', secret=resources.Secret(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].secret_id
        mock_val = 'secret_id_value'
        assert arg == mock_val
        arg = args[0].secret
        mock_val = resources.Secret(name='name_value')
        assert arg == mock_val

def test_create_secret_flattened_error():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_secret(service.CreateSecretRequest(), parent='parent_value', secret_id='secret_id_value', secret=resources.Secret(name='name_value'))

@pytest.mark.asyncio
async def test_create_secret_flattened_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_secret), '__call__') as call:
        call.return_value = resources.Secret()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Secret())
        response = await client.create_secret(parent='parent_value', secret_id='secret_id_value', secret=resources.Secret(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].secret_id
        mock_val = 'secret_id_value'
        assert arg == mock_val
        arg = args[0].secret
        mock_val = resources.Secret(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_secret_flattened_error_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_secret(service.CreateSecretRequest(), parent='parent_value', secret_id='secret_id_value', secret=resources.Secret(name='name_value'))

@pytest.mark.parametrize('request_type', [service.AddSecretVersionRequest, dict])
def test_add_secret_version(request_type, transport: str='grpc'):
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.add_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED)
        response = client.add_secret_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.AddSecretVersionRequest()
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

def test_add_secret_version_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.add_secret_version), '__call__') as call:
        client.add_secret_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.AddSecretVersionRequest()

@pytest.mark.asyncio
async def test_add_secret_version_async(transport: str='grpc_asyncio', request_type=service.AddSecretVersionRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.add_secret_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED))
        response = await client.add_secret_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.AddSecretVersionRequest()
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

@pytest.mark.asyncio
async def test_add_secret_version_async_from_dict():
    await test_add_secret_version_async(request_type=dict)

def test_add_secret_version_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.AddSecretVersionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.add_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        client.add_secret_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_add_secret_version_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.AddSecretVersionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.add_secret_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion())
        await client.add_secret_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_add_secret_version_flattened():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.add_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        client.add_secret_version(parent='parent_value', payload=resources.SecretPayload(data=b'data_blob'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].payload
        mock_val = resources.SecretPayload(data=b'data_blob')
        assert arg == mock_val

def test_add_secret_version_flattened_error():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.add_secret_version(service.AddSecretVersionRequest(), parent='parent_value', payload=resources.SecretPayload(data=b'data_blob'))

@pytest.mark.asyncio
async def test_add_secret_version_flattened_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.add_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion())
        response = await client.add_secret_version(parent='parent_value', payload=resources.SecretPayload(data=b'data_blob'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].payload
        mock_val = resources.SecretPayload(data=b'data_blob')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_add_secret_version_flattened_error_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.add_secret_version(service.AddSecretVersionRequest(), parent='parent_value', payload=resources.SecretPayload(data=b'data_blob'))

@pytest.mark.parametrize('request_type', [service.GetSecretRequest, dict])
def test_get_secret(request_type, transport: str='grpc'):
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_secret), '__call__') as call:
        call.return_value = resources.Secret(name='name_value')
        response = client.get_secret(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetSecretRequest()
    assert isinstance(response, resources.Secret)
    assert response.name == 'name_value'

def test_get_secret_empty_call():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_secret), '__call__') as call:
        client.get_secret()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetSecretRequest()

@pytest.mark.asyncio
async def test_get_secret_async(transport: str='grpc_asyncio', request_type=service.GetSecretRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_secret), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Secret(name='name_value'))
        response = await client.get_secret(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetSecretRequest()
    assert isinstance(response, resources.Secret)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_secret_async_from_dict():
    await test_get_secret_async(request_type=dict)

def test_get_secret_field_headers():
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetSecretRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_secret), '__call__') as call:
        call.return_value = resources.Secret()
        client.get_secret(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_secret_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetSecretRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_secret), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Secret())
        await client.get_secret(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_secret_flattened():
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_secret), '__call__') as call:
        call.return_value = resources.Secret()
        client.get_secret(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_secret_flattened_error():
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_secret(service.GetSecretRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_secret_flattened_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_secret), '__call__') as call:
        call.return_value = resources.Secret()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Secret())
        response = await client.get_secret(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_secret_flattened_error_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_secret(service.GetSecretRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.UpdateSecretRequest, dict])
def test_update_secret(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_secret), '__call__') as call:
        call.return_value = resources.Secret(name='name_value')
        response = client.update_secret(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateSecretRequest()
    assert isinstance(response, resources.Secret)
    assert response.name == 'name_value'

def test_update_secret_empty_call():
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_secret), '__call__') as call:
        client.update_secret()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateSecretRequest()

@pytest.mark.asyncio
async def test_update_secret_async(transport: str='grpc_asyncio', request_type=service.UpdateSecretRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_secret), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Secret(name='name_value'))
        response = await client.update_secret(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateSecretRequest()
    assert isinstance(response, resources.Secret)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_update_secret_async_from_dict():
    await test_update_secret_async(request_type=dict)

def test_update_secret_field_headers():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateSecretRequest()
    request.secret.name = 'name_value'
    with mock.patch.object(type(client.transport.update_secret), '__call__') as call:
        call.return_value = resources.Secret()
        client.update_secret(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'secret.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_secret_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateSecretRequest()
    request.secret.name = 'name_value'
    with mock.patch.object(type(client.transport.update_secret), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Secret())
        await client.update_secret(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'secret.name=name_value') in kw['metadata']

def test_update_secret_flattened():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_secret), '__call__') as call:
        call.return_value = resources.Secret()
        client.update_secret(secret=resources.Secret(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].secret
        mock_val = resources.Secret(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_secret_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_secret(service.UpdateSecretRequest(), secret=resources.Secret(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_secret_flattened_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_secret), '__call__') as call:
        call.return_value = resources.Secret()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Secret())
        response = await client.update_secret(secret=resources.Secret(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].secret
        mock_val = resources.Secret(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_secret_flattened_error_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_secret(service.UpdateSecretRequest(), secret=resources.Secret(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.DeleteSecretRequest, dict])
def test_delete_secret(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_secret), '__call__') as call:
        call.return_value = None
        response = client.delete_secret(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteSecretRequest()
    assert response is None

def test_delete_secret_empty_call():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_secret), '__call__') as call:
        client.delete_secret()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteSecretRequest()

@pytest.mark.asyncio
async def test_delete_secret_async(transport: str='grpc_asyncio', request_type=service.DeleteSecretRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_secret), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_secret(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteSecretRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_secret_async_from_dict():
    await test_delete_secret_async(request_type=dict)

def test_delete_secret_field_headers():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteSecretRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_secret), '__call__') as call:
        call.return_value = None
        client.delete_secret(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_secret_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteSecretRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_secret), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_secret(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_secret_flattened():
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_secret), '__call__') as call:
        call.return_value = None
        client.delete_secret(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_secret_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_secret(service.DeleteSecretRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_secret_flattened_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_secret), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_secret(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_secret_flattened_error_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_secret(service.DeleteSecretRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListSecretVersionsRequest, dict])
def test_list_secret_versions(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_secret_versions), '__call__') as call:
        call.return_value = service.ListSecretVersionsResponse(next_page_token='next_page_token_value', total_size=1086)
        response = client.list_secret_versions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSecretVersionsRequest()
    assert isinstance(response, pagers.ListSecretVersionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_secret_versions_empty_call():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_secret_versions), '__call__') as call:
        client.list_secret_versions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSecretVersionsRequest()

@pytest.mark.asyncio
async def test_list_secret_versions_async(transport: str='grpc_asyncio', request_type=service.ListSecretVersionsRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_secret_versions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSecretVersionsResponse(next_page_token='next_page_token_value', total_size=1086))
        response = await client.list_secret_versions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSecretVersionsRequest()
    assert isinstance(response, pagers.ListSecretVersionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

@pytest.mark.asyncio
async def test_list_secret_versions_async_from_dict():
    await test_list_secret_versions_async(request_type=dict)

def test_list_secret_versions_field_headers():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSecretVersionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_secret_versions), '__call__') as call:
        call.return_value = service.ListSecretVersionsResponse()
        client.list_secret_versions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_secret_versions_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSecretVersionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_secret_versions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSecretVersionsResponse())
        await client.list_secret_versions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_secret_versions_flattened():
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_secret_versions), '__call__') as call:
        call.return_value = service.ListSecretVersionsResponse()
        client.list_secret_versions(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_secret_versions_flattened_error():
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_secret_versions(service.ListSecretVersionsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_secret_versions_flattened_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_secret_versions), '__call__') as call:
        call.return_value = service.ListSecretVersionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSecretVersionsResponse())
        response = await client.list_secret_versions(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_secret_versions_flattened_error_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_secret_versions(service.ListSecretVersionsRequest(), parent='parent_value')

def test_list_secret_versions_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_secret_versions), '__call__') as call:
        call.side_effect = (service.ListSecretVersionsResponse(versions=[resources.SecretVersion(), resources.SecretVersion(), resources.SecretVersion()], next_page_token='abc'), service.ListSecretVersionsResponse(versions=[], next_page_token='def'), service.ListSecretVersionsResponse(versions=[resources.SecretVersion()], next_page_token='ghi'), service.ListSecretVersionsResponse(versions=[resources.SecretVersion(), resources.SecretVersion()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_secret_versions(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.SecretVersion) for i in results))

def test_list_secret_versions_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_secret_versions), '__call__') as call:
        call.side_effect = (service.ListSecretVersionsResponse(versions=[resources.SecretVersion(), resources.SecretVersion(), resources.SecretVersion()], next_page_token='abc'), service.ListSecretVersionsResponse(versions=[], next_page_token='def'), service.ListSecretVersionsResponse(versions=[resources.SecretVersion()], next_page_token='ghi'), service.ListSecretVersionsResponse(versions=[resources.SecretVersion(), resources.SecretVersion()]), RuntimeError)
        pages = list(client.list_secret_versions(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_secret_versions_async_pager():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_secret_versions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSecretVersionsResponse(versions=[resources.SecretVersion(), resources.SecretVersion(), resources.SecretVersion()], next_page_token='abc'), service.ListSecretVersionsResponse(versions=[], next_page_token='def'), service.ListSecretVersionsResponse(versions=[resources.SecretVersion()], next_page_token='ghi'), service.ListSecretVersionsResponse(versions=[resources.SecretVersion(), resources.SecretVersion()]), RuntimeError)
        async_pager = await client.list_secret_versions(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.SecretVersion) for i in responses))

@pytest.mark.asyncio
async def test_list_secret_versions_async_pages():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_secret_versions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSecretVersionsResponse(versions=[resources.SecretVersion(), resources.SecretVersion(), resources.SecretVersion()], next_page_token='abc'), service.ListSecretVersionsResponse(versions=[], next_page_token='def'), service.ListSecretVersionsResponse(versions=[resources.SecretVersion()], next_page_token='ghi'), service.ListSecretVersionsResponse(versions=[resources.SecretVersion(), resources.SecretVersion()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_secret_versions(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetSecretVersionRequest, dict])
def test_get_secret_version(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED)
        response = client.get_secret_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetSecretVersionRequest()
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

def test_get_secret_version_empty_call():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_secret_version), '__call__') as call:
        client.get_secret_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetSecretVersionRequest()

@pytest.mark.asyncio
async def test_get_secret_version_async(transport: str='grpc_asyncio', request_type=service.GetSecretVersionRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_secret_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED))
        response = await client.get_secret_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetSecretVersionRequest()
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

@pytest.mark.asyncio
async def test_get_secret_version_async_from_dict():
    await test_get_secret_version_async(request_type=dict)

def test_get_secret_version_field_headers():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetSecretVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        client.get_secret_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_secret_version_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetSecretVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_secret_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion())
        await client.get_secret_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_secret_version_flattened():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        client.get_secret_version(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_secret_version_flattened_error():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_secret_version(service.GetSecretVersionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_secret_version_flattened_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion())
        response = await client.get_secret_version(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_secret_version_flattened_error_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_secret_version(service.GetSecretVersionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.AccessSecretVersionRequest, dict])
def test_access_secret_version(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.access_secret_version), '__call__') as call:
        call.return_value = service.AccessSecretVersionResponse(name='name_value')
        response = client.access_secret_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.AccessSecretVersionRequest()
    assert isinstance(response, service.AccessSecretVersionResponse)
    assert response.name == 'name_value'

def test_access_secret_version_empty_call():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.access_secret_version), '__call__') as call:
        client.access_secret_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.AccessSecretVersionRequest()

@pytest.mark.asyncio
async def test_access_secret_version_async(transport: str='grpc_asyncio', request_type=service.AccessSecretVersionRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.access_secret_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.AccessSecretVersionResponse(name='name_value'))
        response = await client.access_secret_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.AccessSecretVersionRequest()
    assert isinstance(response, service.AccessSecretVersionResponse)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_access_secret_version_async_from_dict():
    await test_access_secret_version_async(request_type=dict)

def test_access_secret_version_field_headers():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.AccessSecretVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.access_secret_version), '__call__') as call:
        call.return_value = service.AccessSecretVersionResponse()
        client.access_secret_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_access_secret_version_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.AccessSecretVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.access_secret_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.AccessSecretVersionResponse())
        await client.access_secret_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_access_secret_version_flattened():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.access_secret_version), '__call__') as call:
        call.return_value = service.AccessSecretVersionResponse()
        client.access_secret_version(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_access_secret_version_flattened_error():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.access_secret_version(service.AccessSecretVersionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_access_secret_version_flattened_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.access_secret_version), '__call__') as call:
        call.return_value = service.AccessSecretVersionResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.AccessSecretVersionResponse())
        response = await client.access_secret_version(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_access_secret_version_flattened_error_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.access_secret_version(service.AccessSecretVersionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.DisableSecretVersionRequest, dict])
def test_disable_secret_version(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.disable_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED)
        response = client.disable_secret_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DisableSecretVersionRequest()
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

def test_disable_secret_version_empty_call():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.disable_secret_version), '__call__') as call:
        client.disable_secret_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DisableSecretVersionRequest()

@pytest.mark.asyncio
async def test_disable_secret_version_async(transport: str='grpc_asyncio', request_type=service.DisableSecretVersionRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.disable_secret_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED))
        response = await client.disable_secret_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DisableSecretVersionRequest()
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

@pytest.mark.asyncio
async def test_disable_secret_version_async_from_dict():
    await test_disable_secret_version_async(request_type=dict)

def test_disable_secret_version_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DisableSecretVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.disable_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        client.disable_secret_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_disable_secret_version_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DisableSecretVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.disable_secret_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion())
        await client.disable_secret_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_disable_secret_version_flattened():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.disable_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        client.disable_secret_version(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_disable_secret_version_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.disable_secret_version(service.DisableSecretVersionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_disable_secret_version_flattened_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.disable_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion())
        response = await client.disable_secret_version(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_disable_secret_version_flattened_error_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.disable_secret_version(service.DisableSecretVersionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.EnableSecretVersionRequest, dict])
def test_enable_secret_version(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.enable_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED)
        response = client.enable_secret_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.EnableSecretVersionRequest()
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

def test_enable_secret_version_empty_call():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.enable_secret_version), '__call__') as call:
        client.enable_secret_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.EnableSecretVersionRequest()

@pytest.mark.asyncio
async def test_enable_secret_version_async(transport: str='grpc_asyncio', request_type=service.EnableSecretVersionRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.enable_secret_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED))
        response = await client.enable_secret_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.EnableSecretVersionRequest()
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

@pytest.mark.asyncio
async def test_enable_secret_version_async_from_dict():
    await test_enable_secret_version_async(request_type=dict)

def test_enable_secret_version_field_headers():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.EnableSecretVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.enable_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        client.enable_secret_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_enable_secret_version_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.EnableSecretVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.enable_secret_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion())
        await client.enable_secret_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_enable_secret_version_flattened():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.enable_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        client.enable_secret_version(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_enable_secret_version_flattened_error():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.enable_secret_version(service.EnableSecretVersionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_enable_secret_version_flattened_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.enable_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion())
        response = await client.enable_secret_version(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_enable_secret_version_flattened_error_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.enable_secret_version(service.EnableSecretVersionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.DestroySecretVersionRequest, dict])
def test_destroy_secret_version(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.destroy_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED)
        response = client.destroy_secret_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DestroySecretVersionRequest()
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

def test_destroy_secret_version_empty_call():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.destroy_secret_version), '__call__') as call:
        client.destroy_secret_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DestroySecretVersionRequest()

@pytest.mark.asyncio
async def test_destroy_secret_version_async(transport: str='grpc_asyncio', request_type=service.DestroySecretVersionRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.destroy_secret_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED))
        response = await client.destroy_secret_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DestroySecretVersionRequest()
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

@pytest.mark.asyncio
async def test_destroy_secret_version_async_from_dict():
    await test_destroy_secret_version_async(request_type=dict)

def test_destroy_secret_version_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DestroySecretVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.destroy_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        client.destroy_secret_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_destroy_secret_version_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DestroySecretVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.destroy_secret_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion())
        await client.destroy_secret_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_destroy_secret_version_flattened():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.destroy_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        client.destroy_secret_version(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_destroy_secret_version_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.destroy_secret_version(service.DestroySecretVersionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_destroy_secret_version_flattened_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.destroy_secret_version), '__call__') as call:
        call.return_value = resources.SecretVersion()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.SecretVersion())
        response = await client.destroy_secret_version(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_destroy_secret_version_flattened_error_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.destroy_secret_version(service.DestroySecretVersionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_empty_call():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        client.set_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()

@pytest.mark.asyncio
async def test_set_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.SetIamPolicyRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.set_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_set_iam_policy_async_from_dict():
    await test_set_iam_policy_async(request_type=dict)

def test_set_iam_policy_field_headers():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_iam_policy_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.set_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_set_iam_policy_from_dict_foreign():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774), 'update_mask': field_mask_pb2.FieldMask(paths=['paths_value'])})
        call.assert_called()

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_empty_call():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        client.get_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()

@pytest.mark.asyncio
async def test_get_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.GetIamPolicyRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_get_iam_policy_async_from_dict():
    await test_get_iam_policy_async(request_type=dict)

def test_get_iam_policy_field_headers():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_iam_policy_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_get_iam_policy_from_dict_foreign():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response = client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_empty_call():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        client.test_iam_permissions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()

@pytest.mark.asyncio
async def test_test_iam_permissions_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value']))
        response = await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

@pytest.mark.asyncio
async def test_test_iam_permissions_async_from_dict():
    await test_test_iam_permissions_async(request_type=dict)

def test_test_iam_permissions_field_headers():
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_test_iam_permissions_field_headers_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_test_iam_permissions_from_dict_foreign():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.parametrize('request_type', [service.ListSecretsRequest, dict])
def test_list_secrets_rest(request_type):
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListSecretsResponse(next_page_token='next_page_token_value', total_size=1086)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListSecretsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_secrets(request)
    assert isinstance(response, pagers.ListSecretsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_secrets_rest_required_fields(request_type=service.ListSecretsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_secrets._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_secrets._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListSecretsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListSecretsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_secrets(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_secrets_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_secrets._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_secrets_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'post_list_secrets') as post, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_list_secrets') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListSecretsRequest.pb(service.ListSecretsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListSecretsResponse.to_json(service.ListSecretsResponse())
        request = service.ListSecretsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListSecretsResponse()
        client.list_secrets(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_secrets_rest_bad_request(transport: str='rest', request_type=service.ListSecretsRequest):
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_secrets(request)

def test_list_secrets_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListSecretsResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListSecretsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_secrets(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*}/secrets' % client.transport._host, args[1])

def test_list_secrets_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_secrets(service.ListSecretsRequest(), parent='parent_value')

def test_list_secrets_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListSecretsResponse(secrets=[resources.Secret(), resources.Secret(), resources.Secret()], next_page_token='abc'), service.ListSecretsResponse(secrets=[], next_page_token='def'), service.ListSecretsResponse(secrets=[resources.Secret()], next_page_token='ghi'), service.ListSecretsResponse(secrets=[resources.Secret(), resources.Secret()]))
        response = response + response
        response = tuple((service.ListSecretsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_secrets(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Secret) for i in results))
        pages = list(client.list_secrets(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.CreateSecretRequest, dict])
def test_create_secret_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request_init['secret'] = {'name': 'name_value', 'replication': {'automatic': {}, 'user_managed': {'replicas': [{'location': 'location_value'}]}}, 'create_time': {'seconds': 751, 'nanos': 543}, 'labels': {}}
    test_field = service.CreateSecretRequest.meta.fields['secret']

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
    for (field, value) in request_init['secret'].items():
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
                for i in range(0, len(request_init['secret'][field])):
                    del request_init['secret'][field][i][subfield]
            else:
                del request_init['secret'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Secret(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Secret.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_secret(request)
    assert isinstance(response, resources.Secret)
    assert response.name == 'name_value'

def test_create_secret_rest_required_fields(request_type=service.CreateSecretRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['secret_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'secretId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_secret._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'secretId' in jsonified_request
    assert jsonified_request['secretId'] == request_init['secret_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['secretId'] = 'secret_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_secret._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('secret_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'secretId' in jsonified_request
    assert jsonified_request['secretId'] == 'secret_id_value'
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Secret()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Secret.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_secret(request)
            expected_params = [('secretId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_secret_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_secret._get_unset_required_fields({})
    assert set(unset_fields) == set(('secretId',)) & set(('parent', 'secretId', 'secret'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_secret_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'post_create_secret') as post, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_create_secret') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateSecretRequest.pb(service.CreateSecretRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Secret.to_json(resources.Secret())
        request = service.CreateSecretRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Secret()
        client.create_secret(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_secret_rest_bad_request(transport: str='rest', request_type=service.CreateSecretRequest):
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_secret(request)

def test_create_secret_rest_flattened():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Secret()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', secret_id='secret_id_value', secret=resources.Secret(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Secret.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_secret(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*}/secrets' % client.transport._host, args[1])

def test_create_secret_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_secret(service.CreateSecretRequest(), parent='parent_value', secret_id='secret_id_value', secret=resources.Secret(name='name_value'))

def test_create_secret_rest_error():
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.AddSecretVersionRequest, dict])
def test_add_secret_version_rest(request_type):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/secrets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.SecretVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.add_secret_version(request)
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

def test_add_secret_version_rest_required_fields(request_type=service.AddSecretVersionRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_secret_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_secret_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.SecretVersion()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.SecretVersion.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.add_secret_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_add_secret_version_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.add_secret_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'payload'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_add_secret_version_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'post_add_secret_version') as post, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_add_secret_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.AddSecretVersionRequest.pb(service.AddSecretVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.SecretVersion.to_json(resources.SecretVersion())
        request = service.AddSecretVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.SecretVersion()
        client.add_secret_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_add_secret_version_rest_bad_request(transport: str='rest', request_type=service.AddSecretVersionRequest):
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/secrets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.add_secret_version(request)

def test_add_secret_version_rest_flattened():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.SecretVersion()
        sample_request = {'parent': 'projects/sample1/secrets/sample2'}
        mock_args = dict(parent='parent_value', payload=resources.SecretPayload(data=b'data_blob'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.SecretVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.add_secret_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/secrets/*}:addVersion' % client.transport._host, args[1])

def test_add_secret_version_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.add_secret_version(service.AddSecretVersionRequest(), parent='parent_value', payload=resources.SecretPayload(data=b'data_blob'))

def test_add_secret_version_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetSecretRequest, dict])
def test_get_secret_rest(request_type):
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/secrets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Secret(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Secret.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_secret(request)
    assert isinstance(response, resources.Secret)
    assert response.name == 'name_value'

def test_get_secret_rest_required_fields(request_type=service.GetSecretRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_secret._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_secret._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Secret()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Secret.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_secret(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_secret_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_secret._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_secret_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'post_get_secret') as post, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_get_secret') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetSecretRequest.pb(service.GetSecretRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Secret.to_json(resources.Secret())
        request = service.GetSecretRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Secret()
        client.get_secret(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_secret_rest_bad_request(transport: str='rest', request_type=service.GetSecretRequest):
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/secrets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_secret(request)

def test_get_secret_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Secret()
        sample_request = {'name': 'projects/sample1/secrets/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Secret.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_secret(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/secrets/*}' % client.transport._host, args[1])

def test_get_secret_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_secret(service.GetSecretRequest(), name='name_value')

def test_get_secret_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateSecretRequest, dict])
def test_update_secret_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'secret': {'name': 'projects/sample1/secrets/sample2'}}
    request_init['secret'] = {'name': 'projects/sample1/secrets/sample2', 'replication': {'automatic': {}, 'user_managed': {'replicas': [{'location': 'location_value'}]}}, 'create_time': {'seconds': 751, 'nanos': 543}, 'labels': {}}
    test_field = service.UpdateSecretRequest.meta.fields['secret']

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
    for (field, value) in request_init['secret'].items():
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
                for i in range(0, len(request_init['secret'][field])):
                    del request_init['secret'][field][i][subfield]
            else:
                del request_init['secret'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Secret(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Secret.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_secret(request)
    assert isinstance(response, resources.Secret)
    assert response.name == 'name_value'

def test_update_secret_rest_required_fields(request_type=service.UpdateSecretRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_secret._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_secret._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Secret()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Secret.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_secret(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_secret_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_secret._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('secret', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_secret_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'post_update_secret') as post, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_update_secret') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateSecretRequest.pb(service.UpdateSecretRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Secret.to_json(resources.Secret())
        request = service.UpdateSecretRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Secret()
        client.update_secret(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_secret_rest_bad_request(transport: str='rest', request_type=service.UpdateSecretRequest):
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'secret': {'name': 'projects/sample1/secrets/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_secret(request)

def test_update_secret_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Secret()
        sample_request = {'secret': {'name': 'projects/sample1/secrets/sample2'}}
        mock_args = dict(secret=resources.Secret(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Secret.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_secret(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{secret.name=projects/*/secrets/*}' % client.transport._host, args[1])

def test_update_secret_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_secret(service.UpdateSecretRequest(), secret=resources.Secret(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_secret_rest_error():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteSecretRequest, dict])
def test_delete_secret_rest(request_type):
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/secrets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_secret(request)
    assert response is None

def test_delete_secret_rest_required_fields(request_type=service.DeleteSecretRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_secret._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_secret._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_secret(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_secret_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_secret._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_secret_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_delete_secret') as pre:
        pre.assert_not_called()
        pb_message = service.DeleteSecretRequest.pb(service.DeleteSecretRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = service.DeleteSecretRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_secret(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_secret_rest_bad_request(transport: str='rest', request_type=service.DeleteSecretRequest):
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/secrets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_secret(request)

def test_delete_secret_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/secrets/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_secret(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/secrets/*}' % client.transport._host, args[1])

def test_delete_secret_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_secret(service.DeleteSecretRequest(), name='name_value')

def test_delete_secret_rest_error():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListSecretVersionsRequest, dict])
def test_list_secret_versions_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/secrets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListSecretVersionsResponse(next_page_token='next_page_token_value', total_size=1086)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListSecretVersionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_secret_versions(request)
    assert isinstance(response, pagers.ListSecretVersionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_secret_versions_rest_required_fields(request_type=service.ListSecretVersionsRequest):
    if False:
        return 10
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_secret_versions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_secret_versions._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListSecretVersionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListSecretVersionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_secret_versions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_secret_versions_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_secret_versions._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_secret_versions_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'post_list_secret_versions') as post, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_list_secret_versions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListSecretVersionsRequest.pb(service.ListSecretVersionsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListSecretVersionsResponse.to_json(service.ListSecretVersionsResponse())
        request = service.ListSecretVersionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListSecretVersionsResponse()
        client.list_secret_versions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_secret_versions_rest_bad_request(transport: str='rest', request_type=service.ListSecretVersionsRequest):
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/secrets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_secret_versions(request)

def test_list_secret_versions_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListSecretVersionsResponse()
        sample_request = {'parent': 'projects/sample1/secrets/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListSecretVersionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_secret_versions(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/secrets/*}/versions' % client.transport._host, args[1])

def test_list_secret_versions_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_secret_versions(service.ListSecretVersionsRequest(), parent='parent_value')

def test_list_secret_versions_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListSecretVersionsResponse(versions=[resources.SecretVersion(), resources.SecretVersion(), resources.SecretVersion()], next_page_token='abc'), service.ListSecretVersionsResponse(versions=[], next_page_token='def'), service.ListSecretVersionsResponse(versions=[resources.SecretVersion()], next_page_token='ghi'), service.ListSecretVersionsResponse(versions=[resources.SecretVersion(), resources.SecretVersion()]))
        response = response + response
        response = tuple((service.ListSecretVersionsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/secrets/sample2'}
        pager = client.list_secret_versions(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.SecretVersion) for i in results))
        pages = list(client.list_secret_versions(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetSecretVersionRequest, dict])
def test_get_secret_version_rest(request_type):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.SecretVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_secret_version(request)
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

def test_get_secret_version_rest_required_fields(request_type=service.GetSecretVersionRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_secret_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_secret_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.SecretVersion()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.SecretVersion.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_secret_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_secret_version_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_secret_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_secret_version_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'post_get_secret_version') as post, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_get_secret_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetSecretVersionRequest.pb(service.GetSecretVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.SecretVersion.to_json(resources.SecretVersion())
        request = service.GetSecretVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.SecretVersion()
        client.get_secret_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_secret_version_rest_bad_request(transport: str='rest', request_type=service.GetSecretVersionRequest):
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_secret_version(request)

def test_get_secret_version_rest_flattened():
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.SecretVersion()
        sample_request = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.SecretVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_secret_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/secrets/*/versions/*}' % client.transport._host, args[1])

def test_get_secret_version_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_secret_version(service.GetSecretVersionRequest(), name='name_value')

def test_get_secret_version_rest_error():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.AccessSecretVersionRequest, dict])
def test_access_secret_version_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.AccessSecretVersionResponse(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = service.AccessSecretVersionResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.access_secret_version(request)
    assert isinstance(response, service.AccessSecretVersionResponse)
    assert response.name == 'name_value'

def test_access_secret_version_rest_required_fields(request_type=service.AccessSecretVersionRequest):
    if False:
        return 10
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).access_secret_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).access_secret_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.AccessSecretVersionResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.AccessSecretVersionResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.access_secret_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_access_secret_version_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.access_secret_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_access_secret_version_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'post_access_secret_version') as post, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_access_secret_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.AccessSecretVersionRequest.pb(service.AccessSecretVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.AccessSecretVersionResponse.to_json(service.AccessSecretVersionResponse())
        request = service.AccessSecretVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.AccessSecretVersionResponse()
        client.access_secret_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_access_secret_version_rest_bad_request(transport: str='rest', request_type=service.AccessSecretVersionRequest):
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.access_secret_version(request)

def test_access_secret_version_rest_flattened():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.AccessSecretVersionResponse()
        sample_request = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.AccessSecretVersionResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.access_secret_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/secrets/*/versions/*}:access' % client.transport._host, args[1])

def test_access_secret_version_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.access_secret_version(service.AccessSecretVersionRequest(), name='name_value')

def test_access_secret_version_rest_error():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DisableSecretVersionRequest, dict])
def test_disable_secret_version_rest(request_type):
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.SecretVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.disable_secret_version(request)
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

def test_disable_secret_version_rest_required_fields(request_type=service.DisableSecretVersionRequest):
    if False:
        print('Hello World!')
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).disable_secret_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).disable_secret_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.SecretVersion()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.SecretVersion.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.disable_secret_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_disable_secret_version_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.disable_secret_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_disable_secret_version_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'post_disable_secret_version') as post, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_disable_secret_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DisableSecretVersionRequest.pb(service.DisableSecretVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.SecretVersion.to_json(resources.SecretVersion())
        request = service.DisableSecretVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.SecretVersion()
        client.disable_secret_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_disable_secret_version_rest_bad_request(transport: str='rest', request_type=service.DisableSecretVersionRequest):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.disable_secret_version(request)

def test_disable_secret_version_rest_flattened():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.SecretVersion()
        sample_request = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.SecretVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.disable_secret_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/secrets/*/versions/*}:disable' % client.transport._host, args[1])

def test_disable_secret_version_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.disable_secret_version(service.DisableSecretVersionRequest(), name='name_value')

def test_disable_secret_version_rest_error():
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.EnableSecretVersionRequest, dict])
def test_enable_secret_version_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.SecretVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.enable_secret_version(request)
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

def test_enable_secret_version_rest_required_fields(request_type=service.EnableSecretVersionRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).enable_secret_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).enable_secret_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.SecretVersion()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.SecretVersion.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.enable_secret_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_enable_secret_version_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.enable_secret_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_enable_secret_version_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'post_enable_secret_version') as post, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_enable_secret_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.EnableSecretVersionRequest.pb(service.EnableSecretVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.SecretVersion.to_json(resources.SecretVersion())
        request = service.EnableSecretVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.SecretVersion()
        client.enable_secret_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_enable_secret_version_rest_bad_request(transport: str='rest', request_type=service.EnableSecretVersionRequest):
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.enable_secret_version(request)

def test_enable_secret_version_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.SecretVersion()
        sample_request = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.SecretVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.enable_secret_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/secrets/*/versions/*}:enable' % client.transport._host, args[1])

def test_enable_secret_version_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.enable_secret_version(service.EnableSecretVersionRequest(), name='name_value')

def test_enable_secret_version_rest_error():
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DestroySecretVersionRequest, dict])
def test_destroy_secret_version_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.SecretVersion(name='name_value', state=resources.SecretVersion.State.ENABLED)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.SecretVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.destroy_secret_version(request)
    assert isinstance(response, resources.SecretVersion)
    assert response.name == 'name_value'
    assert response.state == resources.SecretVersion.State.ENABLED

def test_destroy_secret_version_rest_required_fields(request_type=service.DestroySecretVersionRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).destroy_secret_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).destroy_secret_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.SecretVersion()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.SecretVersion.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.destroy_secret_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_destroy_secret_version_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.destroy_secret_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_destroy_secret_version_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'post_destroy_secret_version') as post, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_destroy_secret_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DestroySecretVersionRequest.pb(service.DestroySecretVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.SecretVersion.to_json(resources.SecretVersion())
        request = service.DestroySecretVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.SecretVersion()
        client.destroy_secret_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_destroy_secret_version_rest_bad_request(transport: str='rest', request_type=service.DestroySecretVersionRequest):
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.destroy_secret_version(request)

def test_destroy_secret_version_rest_flattened():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.SecretVersion()
        sample_request = {'name': 'projects/sample1/secrets/sample2/versions/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.SecretVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.destroy_secret_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/secrets/*/versions/*}:destroy' % client.transport._host, args[1])

def test_destroy_secret_version_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.destroy_secret_version(service.DestroySecretVersionRequest(), name='name_value')

def test_destroy_secret_version_rest_error():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/secrets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_rest_required_fields(request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.set_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_iam_policy_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'policy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_iam_policy_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'post_set_iam_policy') as post, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_set_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.SetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.SetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.set_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/secrets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

def test_set_iam_policy_rest_error():
    if False:
        i = 10
        return i + 15
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/secrets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_rest_required_fields(request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('options',))
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_iam_policy_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('options',)) & set(('resource',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_iam_policy_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'post_get_iam_policy') as post, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_get_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.GetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.GetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.get_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        print('Hello World!')
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/secrets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

def test_get_iam_policy_rest_error():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/secrets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.test_iam_permissions(request)
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_rest_required_fields(request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.SecretManagerServiceRestTransport
    request_init = {}
    request_init['resource'] = ''
    request_init['permissions'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    jsonified_request['permissions'] = 'permissions_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    assert 'permissions' in jsonified_request
    assert jsonified_request['permissions'] == 'permissions_value'
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = iam_policy_pb2.TestIamPermissionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.test_iam_permissions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_test_iam_permissions_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.test_iam_permissions._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'permissions'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_test_iam_permissions_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.SecretManagerServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecretManagerServiceRestInterceptor())
    client = SecretManagerServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'post_test_iam_permissions') as post, mock.patch.object(transports.SecretManagerServiceRestInterceptor, 'pre_test_iam_permissions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.TestIamPermissionsRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(iam_policy_pb2.TestIamPermissionsResponse())
        request = iam_policy_pb2.TestIamPermissionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_test_iam_permissions_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/secrets/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

def test_test_iam_permissions_rest_error():
    if False:
        while True:
            i = 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.SecretManagerServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.SecretManagerServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SecretManagerServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.SecretManagerServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = SecretManagerServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = SecretManagerServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.SecretManagerServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SecretManagerServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.SecretManagerServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = SecretManagerServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.SecretManagerServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.SecretManagerServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.SecretManagerServiceGrpcTransport, transports.SecretManagerServiceGrpcAsyncIOTransport, transports.SecretManagerServiceRestTransport])
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
        for i in range(10):
            print('nop')
    transport = SecretManagerServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.SecretManagerServiceGrpcTransport)

def test_secret_manager_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.SecretManagerServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_secret_manager_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.secretmanager_v1beta1.services.secret_manager_service.transports.SecretManagerServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.SecretManagerServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_secrets', 'create_secret', 'add_secret_version', 'get_secret', 'update_secret', 'delete_secret', 'list_secret_versions', 'get_secret_version', 'access_secret_version', 'disable_secret_version', 'enable_secret_version', 'destroy_secret_version', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_secret_manager_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.secretmanager_v1beta1.services.secret_manager_service.transports.SecretManagerServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.SecretManagerServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_secret_manager_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.secretmanager_v1beta1.services.secret_manager_service.transports.SecretManagerServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.SecretManagerServiceTransport()
        adc.assert_called_once()

def test_secret_manager_service_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        SecretManagerServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.SecretManagerServiceGrpcTransport, transports.SecretManagerServiceGrpcAsyncIOTransport])
def test_secret_manager_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.SecretManagerServiceGrpcTransport, transports.SecretManagerServiceGrpcAsyncIOTransport, transports.SecretManagerServiceRestTransport])
def test_secret_manager_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.SecretManagerServiceGrpcTransport, grpc_helpers), (transports.SecretManagerServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_secret_manager_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('secretmanager.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='secretmanager.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.SecretManagerServiceGrpcTransport, transports.SecretManagerServiceGrpcAsyncIOTransport])
def test_secret_manager_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_secret_manager_service_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.SecretManagerServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_secret_manager_service_host_no_port(transport_name):
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='secretmanager.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('secretmanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://secretmanager.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_secret_manager_service_host_with_port(transport_name):
    if False:
        return 10
    client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='secretmanager.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('secretmanager.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://secretmanager.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_secret_manager_service_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = SecretManagerServiceClient(credentials=creds1, transport=transport_name)
    client2 = SecretManagerServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_secrets._session
    session2 = client2.transport.list_secrets._session
    assert session1 != session2
    session1 = client1.transport.create_secret._session
    session2 = client2.transport.create_secret._session
    assert session1 != session2
    session1 = client1.transport.add_secret_version._session
    session2 = client2.transport.add_secret_version._session
    assert session1 != session2
    session1 = client1.transport.get_secret._session
    session2 = client2.transport.get_secret._session
    assert session1 != session2
    session1 = client1.transport.update_secret._session
    session2 = client2.transport.update_secret._session
    assert session1 != session2
    session1 = client1.transport.delete_secret._session
    session2 = client2.transport.delete_secret._session
    assert session1 != session2
    session1 = client1.transport.list_secret_versions._session
    session2 = client2.transport.list_secret_versions._session
    assert session1 != session2
    session1 = client1.transport.get_secret_version._session
    session2 = client2.transport.get_secret_version._session
    assert session1 != session2
    session1 = client1.transport.access_secret_version._session
    session2 = client2.transport.access_secret_version._session
    assert session1 != session2
    session1 = client1.transport.disable_secret_version._session
    session2 = client2.transport.disable_secret_version._session
    assert session1 != session2
    session1 = client1.transport.enable_secret_version._session
    session2 = client2.transport.enable_secret_version._session
    assert session1 != session2
    session1 = client1.transport.destroy_secret_version._session
    session2 = client2.transport.destroy_secret_version._session
    assert session1 != session2
    session1 = client1.transport.set_iam_policy._session
    session2 = client2.transport.set_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.get_iam_policy._session
    session2 = client2.transport.get_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.test_iam_permissions._session
    session2 = client2.transport.test_iam_permissions._session
    assert session1 != session2

def test_secret_manager_service_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.SecretManagerServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_secret_manager_service_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.SecretManagerServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.SecretManagerServiceGrpcTransport, transports.SecretManagerServiceGrpcAsyncIOTransport])
def test_secret_manager_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.SecretManagerServiceGrpcTransport, transports.SecretManagerServiceGrpcAsyncIOTransport])
def test_secret_manager_service_transport_channel_mtls_with_adc(transport_class):
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

def test_secret_path():
    if False:
        print('Hello World!')
    project = 'squid'
    secret = 'clam'
    expected = 'projects/{project}/secrets/{secret}'.format(project=project, secret=secret)
    actual = SecretManagerServiceClient.secret_path(project, secret)
    assert expected == actual

def test_parse_secret_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'whelk', 'secret': 'octopus'}
    path = SecretManagerServiceClient.secret_path(**expected)
    actual = SecretManagerServiceClient.parse_secret_path(path)
    assert expected == actual

def test_secret_version_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'oyster'
    secret = 'nudibranch'
    secret_version = 'cuttlefish'
    expected = 'projects/{project}/secrets/{secret}/versions/{secret_version}'.format(project=project, secret=secret, secret_version=secret_version)
    actual = SecretManagerServiceClient.secret_version_path(project, secret, secret_version)
    assert expected == actual

def test_parse_secret_version_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'mussel', 'secret': 'winkle', 'secret_version': 'nautilus'}
    path = SecretManagerServiceClient.secret_version_path(**expected)
    actual = SecretManagerServiceClient.parse_secret_version_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = SecretManagerServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'abalone'}
    path = SecretManagerServiceClient.common_billing_account_path(**expected)
    actual = SecretManagerServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = SecretManagerServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'clam'}
    path = SecretManagerServiceClient.common_folder_path(**expected)
    actual = SecretManagerServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = SecretManagerServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'octopus'}
    path = SecretManagerServiceClient.common_organization_path(**expected)
    actual = SecretManagerServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = SecretManagerServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'nudibranch'}
    path = SecretManagerServiceClient.common_project_path(**expected)
    actual = SecretManagerServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = SecretManagerServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = SecretManagerServiceClient.common_location_path(**expected)
    actual = SecretManagerServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.SecretManagerServiceTransport, '_prep_wrapped_messages') as prep:
        client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.SecretManagerServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = SecretManagerServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = SecretManagerServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = SecretManagerServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(SecretManagerServiceClient, transports.SecretManagerServiceGrpcTransport), (SecretManagerServiceAsyncClient, transports.SecretManagerServiceGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)