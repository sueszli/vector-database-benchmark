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
from google.cloud.location import locations_pb2
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
from google.oauth2 import service_account
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.protobuf import wrappers_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.kms_v1.services.key_management_service import KeyManagementServiceAsyncClient, KeyManagementServiceClient, pagers, transports
from google.cloud.kms_v1.types import resources, service

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
        while True:
            i = 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert KeyManagementServiceClient._get_default_mtls_endpoint(None) is None
    assert KeyManagementServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert KeyManagementServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert KeyManagementServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert KeyManagementServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert KeyManagementServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(KeyManagementServiceClient, 'grpc'), (KeyManagementServiceAsyncClient, 'grpc_asyncio'), (KeyManagementServiceClient, 'rest')])
def test_key_management_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('cloudkms.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudkms.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.KeyManagementServiceGrpcTransport, 'grpc'), (transports.KeyManagementServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.KeyManagementServiceRestTransport, 'rest')])
def test_key_management_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(KeyManagementServiceClient, 'grpc'), (KeyManagementServiceAsyncClient, 'grpc_asyncio'), (KeyManagementServiceClient, 'rest')])
def test_key_management_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('cloudkms.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudkms.googleapis.com')

def test_key_management_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = KeyManagementServiceClient.get_transport_class()
    available_transports = [transports.KeyManagementServiceGrpcTransport, transports.KeyManagementServiceRestTransport]
    assert transport in available_transports
    transport = KeyManagementServiceClient.get_transport_class('grpc')
    assert transport == transports.KeyManagementServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(KeyManagementServiceClient, transports.KeyManagementServiceGrpcTransport, 'grpc'), (KeyManagementServiceAsyncClient, transports.KeyManagementServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (KeyManagementServiceClient, transports.KeyManagementServiceRestTransport, 'rest')])
@mock.patch.object(KeyManagementServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(KeyManagementServiceClient))
@mock.patch.object(KeyManagementServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(KeyManagementServiceAsyncClient))
def test_key_management_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(KeyManagementServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(KeyManagementServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(KeyManagementServiceClient, transports.KeyManagementServiceGrpcTransport, 'grpc', 'true'), (KeyManagementServiceAsyncClient, transports.KeyManagementServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (KeyManagementServiceClient, transports.KeyManagementServiceGrpcTransport, 'grpc', 'false'), (KeyManagementServiceAsyncClient, transports.KeyManagementServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (KeyManagementServiceClient, transports.KeyManagementServiceRestTransport, 'rest', 'true'), (KeyManagementServiceClient, transports.KeyManagementServiceRestTransport, 'rest', 'false')])
@mock.patch.object(KeyManagementServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(KeyManagementServiceClient))
@mock.patch.object(KeyManagementServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(KeyManagementServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_key_management_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [KeyManagementServiceClient, KeyManagementServiceAsyncClient])
@mock.patch.object(KeyManagementServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(KeyManagementServiceClient))
@mock.patch.object(KeyManagementServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(KeyManagementServiceAsyncClient))
def test_key_management_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(KeyManagementServiceClient, transports.KeyManagementServiceGrpcTransport, 'grpc'), (KeyManagementServiceAsyncClient, transports.KeyManagementServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (KeyManagementServiceClient, transports.KeyManagementServiceRestTransport, 'rest')])
def test_key_management_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(KeyManagementServiceClient, transports.KeyManagementServiceGrpcTransport, 'grpc', grpc_helpers), (KeyManagementServiceAsyncClient, transports.KeyManagementServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (KeyManagementServiceClient, transports.KeyManagementServiceRestTransport, 'rest', None)])
def test_key_management_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_key_management_service_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.kms_v1.services.key_management_service.transports.KeyManagementServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = KeyManagementServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(KeyManagementServiceClient, transports.KeyManagementServiceGrpcTransport, 'grpc', grpc_helpers), (KeyManagementServiceAsyncClient, transports.KeyManagementServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_key_management_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('cloudkms.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloudkms'), scopes=None, default_host='cloudkms.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [service.ListKeyRingsRequest, dict])
def test_list_key_rings(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_key_rings), '__call__') as call:
        call.return_value = service.ListKeyRingsResponse(next_page_token='next_page_token_value', total_size=1086)
        response = client.list_key_rings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListKeyRingsRequest()
    assert isinstance(response, pagers.ListKeyRingsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_key_rings_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_key_rings), '__call__') as call:
        client.list_key_rings()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListKeyRingsRequest()

@pytest.mark.asyncio
async def test_list_key_rings_async(transport: str='grpc_asyncio', request_type=service.ListKeyRingsRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_key_rings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListKeyRingsResponse(next_page_token='next_page_token_value', total_size=1086))
        response = await client.list_key_rings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListKeyRingsRequest()
    assert isinstance(response, pagers.ListKeyRingsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

@pytest.mark.asyncio
async def test_list_key_rings_async_from_dict():
    await test_list_key_rings_async(request_type=dict)

def test_list_key_rings_field_headers():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListKeyRingsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_key_rings), '__call__') as call:
        call.return_value = service.ListKeyRingsResponse()
        client.list_key_rings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_key_rings_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListKeyRingsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_key_rings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListKeyRingsResponse())
        await client.list_key_rings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_key_rings_flattened():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_key_rings), '__call__') as call:
        call.return_value = service.ListKeyRingsResponse()
        client.list_key_rings(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_key_rings_flattened_error():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_key_rings(service.ListKeyRingsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_key_rings_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_key_rings), '__call__') as call:
        call.return_value = service.ListKeyRingsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListKeyRingsResponse())
        response = await client.list_key_rings(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_key_rings_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_key_rings(service.ListKeyRingsRequest(), parent='parent_value')

def test_list_key_rings_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_key_rings), '__call__') as call:
        call.side_effect = (service.ListKeyRingsResponse(key_rings=[resources.KeyRing(), resources.KeyRing(), resources.KeyRing()], next_page_token='abc'), service.ListKeyRingsResponse(key_rings=[], next_page_token='def'), service.ListKeyRingsResponse(key_rings=[resources.KeyRing()], next_page_token='ghi'), service.ListKeyRingsResponse(key_rings=[resources.KeyRing(), resources.KeyRing()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_key_rings(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.KeyRing) for i in results))

def test_list_key_rings_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_key_rings), '__call__') as call:
        call.side_effect = (service.ListKeyRingsResponse(key_rings=[resources.KeyRing(), resources.KeyRing(), resources.KeyRing()], next_page_token='abc'), service.ListKeyRingsResponse(key_rings=[], next_page_token='def'), service.ListKeyRingsResponse(key_rings=[resources.KeyRing()], next_page_token='ghi'), service.ListKeyRingsResponse(key_rings=[resources.KeyRing(), resources.KeyRing()]), RuntimeError)
        pages = list(client.list_key_rings(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_key_rings_async_pager():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_key_rings), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListKeyRingsResponse(key_rings=[resources.KeyRing(), resources.KeyRing(), resources.KeyRing()], next_page_token='abc'), service.ListKeyRingsResponse(key_rings=[], next_page_token='def'), service.ListKeyRingsResponse(key_rings=[resources.KeyRing()], next_page_token='ghi'), service.ListKeyRingsResponse(key_rings=[resources.KeyRing(), resources.KeyRing()]), RuntimeError)
        async_pager = await client.list_key_rings(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.KeyRing) for i in responses))

@pytest.mark.asyncio
async def test_list_key_rings_async_pages():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_key_rings), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListKeyRingsResponse(key_rings=[resources.KeyRing(), resources.KeyRing(), resources.KeyRing()], next_page_token='abc'), service.ListKeyRingsResponse(key_rings=[], next_page_token='def'), service.ListKeyRingsResponse(key_rings=[resources.KeyRing()], next_page_token='ghi'), service.ListKeyRingsResponse(key_rings=[resources.KeyRing(), resources.KeyRing()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_key_rings(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListCryptoKeysRequest, dict])
def test_list_crypto_keys(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_crypto_keys), '__call__') as call:
        call.return_value = service.ListCryptoKeysResponse(next_page_token='next_page_token_value', total_size=1086)
        response = client.list_crypto_keys(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCryptoKeysRequest()
    assert isinstance(response, pagers.ListCryptoKeysPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_crypto_keys_empty_call():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_crypto_keys), '__call__') as call:
        client.list_crypto_keys()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCryptoKeysRequest()

@pytest.mark.asyncio
async def test_list_crypto_keys_async(transport: str='grpc_asyncio', request_type=service.ListCryptoKeysRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_crypto_keys), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCryptoKeysResponse(next_page_token='next_page_token_value', total_size=1086))
        response = await client.list_crypto_keys(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCryptoKeysRequest()
    assert isinstance(response, pagers.ListCryptoKeysAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

@pytest.mark.asyncio
async def test_list_crypto_keys_async_from_dict():
    await test_list_crypto_keys_async(request_type=dict)

def test_list_crypto_keys_field_headers():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListCryptoKeysRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_crypto_keys), '__call__') as call:
        call.return_value = service.ListCryptoKeysResponse()
        client.list_crypto_keys(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_crypto_keys_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListCryptoKeysRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_crypto_keys), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCryptoKeysResponse())
        await client.list_crypto_keys(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_crypto_keys_flattened():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_crypto_keys), '__call__') as call:
        call.return_value = service.ListCryptoKeysResponse()
        client.list_crypto_keys(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_crypto_keys_flattened_error():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_crypto_keys(service.ListCryptoKeysRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_crypto_keys_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_crypto_keys), '__call__') as call:
        call.return_value = service.ListCryptoKeysResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCryptoKeysResponse())
        response = await client.list_crypto_keys(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_crypto_keys_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_crypto_keys(service.ListCryptoKeysRequest(), parent='parent_value')

def test_list_crypto_keys_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_crypto_keys), '__call__') as call:
        call.side_effect = (service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey(), resources.CryptoKey(), resources.CryptoKey()], next_page_token='abc'), service.ListCryptoKeysResponse(crypto_keys=[], next_page_token='def'), service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey()], next_page_token='ghi'), service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey(), resources.CryptoKey()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_crypto_keys(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.CryptoKey) for i in results))

def test_list_crypto_keys_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_crypto_keys), '__call__') as call:
        call.side_effect = (service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey(), resources.CryptoKey(), resources.CryptoKey()], next_page_token='abc'), service.ListCryptoKeysResponse(crypto_keys=[], next_page_token='def'), service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey()], next_page_token='ghi'), service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey(), resources.CryptoKey()]), RuntimeError)
        pages = list(client.list_crypto_keys(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_crypto_keys_async_pager():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_crypto_keys), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey(), resources.CryptoKey(), resources.CryptoKey()], next_page_token='abc'), service.ListCryptoKeysResponse(crypto_keys=[], next_page_token='def'), service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey()], next_page_token='ghi'), service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey(), resources.CryptoKey()]), RuntimeError)
        async_pager = await client.list_crypto_keys(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.CryptoKey) for i in responses))

@pytest.mark.asyncio
async def test_list_crypto_keys_async_pages():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_crypto_keys), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey(), resources.CryptoKey(), resources.CryptoKey()], next_page_token='abc'), service.ListCryptoKeysResponse(crypto_keys=[], next_page_token='def'), service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey()], next_page_token='ghi'), service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey(), resources.CryptoKey()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_crypto_keys(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListCryptoKeyVersionsRequest, dict])
def test_list_crypto_key_versions(request_type, transport: str='grpc'):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_crypto_key_versions), '__call__') as call:
        call.return_value = service.ListCryptoKeyVersionsResponse(next_page_token='next_page_token_value', total_size=1086)
        response = client.list_crypto_key_versions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCryptoKeyVersionsRequest()
    assert isinstance(response, pagers.ListCryptoKeyVersionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_crypto_key_versions_empty_call():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_crypto_key_versions), '__call__') as call:
        client.list_crypto_key_versions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCryptoKeyVersionsRequest()

@pytest.mark.asyncio
async def test_list_crypto_key_versions_async(transport: str='grpc_asyncio', request_type=service.ListCryptoKeyVersionsRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_crypto_key_versions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCryptoKeyVersionsResponse(next_page_token='next_page_token_value', total_size=1086))
        response = await client.list_crypto_key_versions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCryptoKeyVersionsRequest()
    assert isinstance(response, pagers.ListCryptoKeyVersionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

@pytest.mark.asyncio
async def test_list_crypto_key_versions_async_from_dict():
    await test_list_crypto_key_versions_async(request_type=dict)

def test_list_crypto_key_versions_field_headers():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListCryptoKeyVersionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_crypto_key_versions), '__call__') as call:
        call.return_value = service.ListCryptoKeyVersionsResponse()
        client.list_crypto_key_versions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_crypto_key_versions_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListCryptoKeyVersionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_crypto_key_versions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCryptoKeyVersionsResponse())
        await client.list_crypto_key_versions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_crypto_key_versions_flattened():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_crypto_key_versions), '__call__') as call:
        call.return_value = service.ListCryptoKeyVersionsResponse()
        client.list_crypto_key_versions(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_crypto_key_versions_flattened_error():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_crypto_key_versions(service.ListCryptoKeyVersionsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_crypto_key_versions_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_crypto_key_versions), '__call__') as call:
        call.return_value = service.ListCryptoKeyVersionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCryptoKeyVersionsResponse())
        response = await client.list_crypto_key_versions(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_crypto_key_versions_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_crypto_key_versions(service.ListCryptoKeyVersionsRequest(), parent='parent_value')

def test_list_crypto_key_versions_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_crypto_key_versions), '__call__') as call:
        call.side_effect = (service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion(), resources.CryptoKeyVersion(), resources.CryptoKeyVersion()], next_page_token='abc'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[], next_page_token='def'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion()], next_page_token='ghi'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion(), resources.CryptoKeyVersion()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_crypto_key_versions(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.CryptoKeyVersion) for i in results))

def test_list_crypto_key_versions_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_crypto_key_versions), '__call__') as call:
        call.side_effect = (service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion(), resources.CryptoKeyVersion(), resources.CryptoKeyVersion()], next_page_token='abc'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[], next_page_token='def'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion()], next_page_token='ghi'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion(), resources.CryptoKeyVersion()]), RuntimeError)
        pages = list(client.list_crypto_key_versions(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_crypto_key_versions_async_pager():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_crypto_key_versions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion(), resources.CryptoKeyVersion(), resources.CryptoKeyVersion()], next_page_token='abc'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[], next_page_token='def'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion()], next_page_token='ghi'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion(), resources.CryptoKeyVersion()]), RuntimeError)
        async_pager = await client.list_crypto_key_versions(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.CryptoKeyVersion) for i in responses))

@pytest.mark.asyncio
async def test_list_crypto_key_versions_async_pages():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_crypto_key_versions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion(), resources.CryptoKeyVersion(), resources.CryptoKeyVersion()], next_page_token='abc'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[], next_page_token='def'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion()], next_page_token='ghi'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion(), resources.CryptoKeyVersion()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_crypto_key_versions(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListImportJobsRequest, dict])
def test_list_import_jobs(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_import_jobs), '__call__') as call:
        call.return_value = service.ListImportJobsResponse(next_page_token='next_page_token_value', total_size=1086)
        response = client.list_import_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListImportJobsRequest()
    assert isinstance(response, pagers.ListImportJobsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_import_jobs_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_import_jobs), '__call__') as call:
        client.list_import_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListImportJobsRequest()

@pytest.mark.asyncio
async def test_list_import_jobs_async(transport: str='grpc_asyncio', request_type=service.ListImportJobsRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_import_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListImportJobsResponse(next_page_token='next_page_token_value', total_size=1086))
        response = await client.list_import_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListImportJobsRequest()
    assert isinstance(response, pagers.ListImportJobsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

@pytest.mark.asyncio
async def test_list_import_jobs_async_from_dict():
    await test_list_import_jobs_async(request_type=dict)

def test_list_import_jobs_field_headers():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListImportJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_import_jobs), '__call__') as call:
        call.return_value = service.ListImportJobsResponse()
        client.list_import_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_import_jobs_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListImportJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_import_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListImportJobsResponse())
        await client.list_import_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_import_jobs_flattened():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_import_jobs), '__call__') as call:
        call.return_value = service.ListImportJobsResponse()
        client.list_import_jobs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_import_jobs_flattened_error():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_import_jobs(service.ListImportJobsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_import_jobs_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_import_jobs), '__call__') as call:
        call.return_value = service.ListImportJobsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListImportJobsResponse())
        response = await client.list_import_jobs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_import_jobs_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_import_jobs(service.ListImportJobsRequest(), parent='parent_value')

def test_list_import_jobs_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_import_jobs), '__call__') as call:
        call.side_effect = (service.ListImportJobsResponse(import_jobs=[resources.ImportJob(), resources.ImportJob(), resources.ImportJob()], next_page_token='abc'), service.ListImportJobsResponse(import_jobs=[], next_page_token='def'), service.ListImportJobsResponse(import_jobs=[resources.ImportJob()], next_page_token='ghi'), service.ListImportJobsResponse(import_jobs=[resources.ImportJob(), resources.ImportJob()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_import_jobs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.ImportJob) for i in results))

def test_list_import_jobs_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_import_jobs), '__call__') as call:
        call.side_effect = (service.ListImportJobsResponse(import_jobs=[resources.ImportJob(), resources.ImportJob(), resources.ImportJob()], next_page_token='abc'), service.ListImportJobsResponse(import_jobs=[], next_page_token='def'), service.ListImportJobsResponse(import_jobs=[resources.ImportJob()], next_page_token='ghi'), service.ListImportJobsResponse(import_jobs=[resources.ImportJob(), resources.ImportJob()]), RuntimeError)
        pages = list(client.list_import_jobs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_import_jobs_async_pager():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_import_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListImportJobsResponse(import_jobs=[resources.ImportJob(), resources.ImportJob(), resources.ImportJob()], next_page_token='abc'), service.ListImportJobsResponse(import_jobs=[], next_page_token='def'), service.ListImportJobsResponse(import_jobs=[resources.ImportJob()], next_page_token='ghi'), service.ListImportJobsResponse(import_jobs=[resources.ImportJob(), resources.ImportJob()]), RuntimeError)
        async_pager = await client.list_import_jobs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.ImportJob) for i in responses))

@pytest.mark.asyncio
async def test_list_import_jobs_async_pages():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_import_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListImportJobsResponse(import_jobs=[resources.ImportJob(), resources.ImportJob(), resources.ImportJob()], next_page_token='abc'), service.ListImportJobsResponse(import_jobs=[], next_page_token='def'), service.ListImportJobsResponse(import_jobs=[resources.ImportJob()], next_page_token='ghi'), service.ListImportJobsResponse(import_jobs=[resources.ImportJob(), resources.ImportJob()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_import_jobs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetKeyRingRequest, dict])
def test_get_key_ring(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_key_ring), '__call__') as call:
        call.return_value = resources.KeyRing(name='name_value')
        response = client.get_key_ring(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetKeyRingRequest()
    assert isinstance(response, resources.KeyRing)
    assert response.name == 'name_value'

def test_get_key_ring_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_key_ring), '__call__') as call:
        client.get_key_ring()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetKeyRingRequest()

@pytest.mark.asyncio
async def test_get_key_ring_async(transport: str='grpc_asyncio', request_type=service.GetKeyRingRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_key_ring), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.KeyRing(name='name_value'))
        response = await client.get_key_ring(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetKeyRingRequest()
    assert isinstance(response, resources.KeyRing)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_key_ring_async_from_dict():
    await test_get_key_ring_async(request_type=dict)

def test_get_key_ring_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetKeyRingRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_key_ring), '__call__') as call:
        call.return_value = resources.KeyRing()
        client.get_key_ring(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_key_ring_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetKeyRingRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_key_ring), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.KeyRing())
        await client.get_key_ring(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_key_ring_flattened():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_key_ring), '__call__') as call:
        call.return_value = resources.KeyRing()
        client.get_key_ring(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_key_ring_flattened_error():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_key_ring(service.GetKeyRingRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_key_ring_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_key_ring), '__call__') as call:
        call.return_value = resources.KeyRing()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.KeyRing())
        response = await client.get_key_ring(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_key_ring_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_key_ring(service.GetKeyRingRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.GetCryptoKeyRequest, dict])
def test_get_crypto_key(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_crypto_key), '__call__') as call:
        call.return_value = resources.CryptoKey(name='name_value', purpose=resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT, import_only=True, crypto_key_backend='crypto_key_backend_value')
        response = client.get_crypto_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCryptoKeyRequest()
    assert isinstance(response, resources.CryptoKey)
    assert response.name == 'name_value'
    assert response.purpose == resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
    assert response.import_only is True
    assert response.crypto_key_backend == 'crypto_key_backend_value'

def test_get_crypto_key_empty_call():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_crypto_key), '__call__') as call:
        client.get_crypto_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCryptoKeyRequest()

@pytest.mark.asyncio
async def test_get_crypto_key_async(transport: str='grpc_asyncio', request_type=service.GetCryptoKeyRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_crypto_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKey(name='name_value', purpose=resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT, import_only=True, crypto_key_backend='crypto_key_backend_value'))
        response = await client.get_crypto_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCryptoKeyRequest()
    assert isinstance(response, resources.CryptoKey)
    assert response.name == 'name_value'
    assert response.purpose == resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
    assert response.import_only is True
    assert response.crypto_key_backend == 'crypto_key_backend_value'

@pytest.mark.asyncio
async def test_get_crypto_key_async_from_dict():
    await test_get_crypto_key_async(request_type=dict)

def test_get_crypto_key_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetCryptoKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_crypto_key), '__call__') as call:
        call.return_value = resources.CryptoKey()
        client.get_crypto_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_crypto_key_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetCryptoKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_crypto_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKey())
        await client.get_crypto_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_crypto_key_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_crypto_key), '__call__') as call:
        call.return_value = resources.CryptoKey()
        client.get_crypto_key(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_crypto_key_flattened_error():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_crypto_key(service.GetCryptoKeyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_crypto_key_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_crypto_key), '__call__') as call:
        call.return_value = resources.CryptoKey()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKey())
        response = await client.get_crypto_key(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_crypto_key_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_crypto_key(service.GetCryptoKeyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.GetCryptoKeyVersionRequest, dict])
def test_get_crypto_key_version(request_type, transport: str='grpc'):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True)
        response = client.get_crypto_key_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCryptoKeyVersionRequest()
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

def test_get_crypto_key_version_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_crypto_key_version), '__call__') as call:
        client.get_crypto_key_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCryptoKeyVersionRequest()

@pytest.mark.asyncio
async def test_get_crypto_key_version_async(transport: str='grpc_asyncio', request_type=service.GetCryptoKeyVersionRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_crypto_key_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True))
        response = await client.get_crypto_key_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCryptoKeyVersionRequest()
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

@pytest.mark.asyncio
async def test_get_crypto_key_version_async_from_dict():
    await test_get_crypto_key_version_async(request_type=dict)

def test_get_crypto_key_version_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetCryptoKeyVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        client.get_crypto_key_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_crypto_key_version_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetCryptoKeyVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_crypto_key_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion())
        await client.get_crypto_key_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_crypto_key_version_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        client.get_crypto_key_version(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_crypto_key_version_flattened_error():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_crypto_key_version(service.GetCryptoKeyVersionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_crypto_key_version_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion())
        response = await client.get_crypto_key_version(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_crypto_key_version_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_crypto_key_version(service.GetCryptoKeyVersionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.GetPublicKeyRequest, dict])
def test_get_public_key(request_type, transport: str='grpc'):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_public_key), '__call__') as call:
        call.return_value = resources.PublicKey(pem='pem_value', algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, name='name_value', protection_level=resources.ProtectionLevel.SOFTWARE)
        response = client.get_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetPublicKeyRequest()
    assert isinstance(response, resources.PublicKey)
    assert response.pem == 'pem_value'
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.name == 'name_value'
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_get_public_key_empty_call():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_public_key), '__call__') as call:
        client.get_public_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetPublicKeyRequest()

@pytest.mark.asyncio
async def test_get_public_key_async(transport: str='grpc_asyncio', request_type=service.GetPublicKeyRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.PublicKey(pem='pem_value', algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, name='name_value', protection_level=resources.ProtectionLevel.SOFTWARE))
        response = await client.get_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetPublicKeyRequest()
    assert isinstance(response, resources.PublicKey)
    assert response.pem == 'pem_value'
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.name == 'name_value'
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

@pytest.mark.asyncio
async def test_get_public_key_async_from_dict():
    await test_get_public_key_async(request_type=dict)

def test_get_public_key_field_headers():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetPublicKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_public_key), '__call__') as call:
        call.return_value = resources.PublicKey()
        client.get_public_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_public_key_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetPublicKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_public_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.PublicKey())
        await client.get_public_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_public_key_flattened():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_public_key), '__call__') as call:
        call.return_value = resources.PublicKey()
        client.get_public_key(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_public_key_flattened_error():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_public_key(service.GetPublicKeyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_public_key_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_public_key), '__call__') as call:
        call.return_value = resources.PublicKey()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.PublicKey())
        response = await client.get_public_key(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_public_key_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_public_key(service.GetPublicKeyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.GetImportJobRequest, dict])
def test_get_import_job(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_import_job), '__call__') as call:
        call.return_value = resources.ImportJob(name='name_value', import_method=resources.ImportJob.ImportMethod.RSA_OAEP_3072_SHA1_AES_256, protection_level=resources.ProtectionLevel.SOFTWARE, state=resources.ImportJob.ImportJobState.PENDING_GENERATION)
        response = client.get_import_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetImportJobRequest()
    assert isinstance(response, resources.ImportJob)
    assert response.name == 'name_value'
    assert response.import_method == resources.ImportJob.ImportMethod.RSA_OAEP_3072_SHA1_AES_256
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.state == resources.ImportJob.ImportJobState.PENDING_GENERATION

def test_get_import_job_empty_call():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_import_job), '__call__') as call:
        client.get_import_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetImportJobRequest()

@pytest.mark.asyncio
async def test_get_import_job_async(transport: str='grpc_asyncio', request_type=service.GetImportJobRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_import_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ImportJob(name='name_value', import_method=resources.ImportJob.ImportMethod.RSA_OAEP_3072_SHA1_AES_256, protection_level=resources.ProtectionLevel.SOFTWARE, state=resources.ImportJob.ImportJobState.PENDING_GENERATION))
        response = await client.get_import_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetImportJobRequest()
    assert isinstance(response, resources.ImportJob)
    assert response.name == 'name_value'
    assert response.import_method == resources.ImportJob.ImportMethod.RSA_OAEP_3072_SHA1_AES_256
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.state == resources.ImportJob.ImportJobState.PENDING_GENERATION

@pytest.mark.asyncio
async def test_get_import_job_async_from_dict():
    await test_get_import_job_async(request_type=dict)

def test_get_import_job_field_headers():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetImportJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_import_job), '__call__') as call:
        call.return_value = resources.ImportJob()
        client.get_import_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_import_job_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetImportJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_import_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ImportJob())
        await client.get_import_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_import_job_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_import_job), '__call__') as call:
        call.return_value = resources.ImportJob()
        client.get_import_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_import_job_flattened_error():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_import_job(service.GetImportJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_import_job_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_import_job), '__call__') as call:
        call.return_value = resources.ImportJob()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ImportJob())
        response = await client.get_import_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_import_job_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_import_job(service.GetImportJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.CreateKeyRingRequest, dict])
def test_create_key_ring(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_key_ring), '__call__') as call:
        call.return_value = resources.KeyRing(name='name_value')
        response = client.create_key_ring(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateKeyRingRequest()
    assert isinstance(response, resources.KeyRing)
    assert response.name == 'name_value'

def test_create_key_ring_empty_call():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_key_ring), '__call__') as call:
        client.create_key_ring()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateKeyRingRequest()

@pytest.mark.asyncio
async def test_create_key_ring_async(transport: str='grpc_asyncio', request_type=service.CreateKeyRingRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_key_ring), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.KeyRing(name='name_value'))
        response = await client.create_key_ring(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateKeyRingRequest()
    assert isinstance(response, resources.KeyRing)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_create_key_ring_async_from_dict():
    await test_create_key_ring_async(request_type=dict)

def test_create_key_ring_field_headers():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateKeyRingRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_key_ring), '__call__') as call:
        call.return_value = resources.KeyRing()
        client.create_key_ring(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_key_ring_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateKeyRingRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_key_ring), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.KeyRing())
        await client.create_key_ring(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_key_ring_flattened():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_key_ring), '__call__') as call:
        call.return_value = resources.KeyRing()
        client.create_key_ring(parent='parent_value', key_ring_id='key_ring_id_value', key_ring=resources.KeyRing(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].key_ring_id
        mock_val = 'key_ring_id_value'
        assert arg == mock_val
        arg = args[0].key_ring
        mock_val = resources.KeyRing(name='name_value')
        assert arg == mock_val

def test_create_key_ring_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_key_ring(service.CreateKeyRingRequest(), parent='parent_value', key_ring_id='key_ring_id_value', key_ring=resources.KeyRing(name='name_value'))

@pytest.mark.asyncio
async def test_create_key_ring_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_key_ring), '__call__') as call:
        call.return_value = resources.KeyRing()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.KeyRing())
        response = await client.create_key_ring(parent='parent_value', key_ring_id='key_ring_id_value', key_ring=resources.KeyRing(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].key_ring_id
        mock_val = 'key_ring_id_value'
        assert arg == mock_val
        arg = args[0].key_ring
        mock_val = resources.KeyRing(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_key_ring_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_key_ring(service.CreateKeyRingRequest(), parent='parent_value', key_ring_id='key_ring_id_value', key_ring=resources.KeyRing(name='name_value'))

@pytest.mark.parametrize('request_type', [service.CreateCryptoKeyRequest, dict])
def test_create_crypto_key(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_crypto_key), '__call__') as call:
        call.return_value = resources.CryptoKey(name='name_value', purpose=resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT, import_only=True, crypto_key_backend='crypto_key_backend_value')
        response = client.create_crypto_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCryptoKeyRequest()
    assert isinstance(response, resources.CryptoKey)
    assert response.name == 'name_value'
    assert response.purpose == resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
    assert response.import_only is True
    assert response.crypto_key_backend == 'crypto_key_backend_value'

def test_create_crypto_key_empty_call():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_crypto_key), '__call__') as call:
        client.create_crypto_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCryptoKeyRequest()

@pytest.mark.asyncio
async def test_create_crypto_key_async(transport: str='grpc_asyncio', request_type=service.CreateCryptoKeyRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_crypto_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKey(name='name_value', purpose=resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT, import_only=True, crypto_key_backend='crypto_key_backend_value'))
        response = await client.create_crypto_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCryptoKeyRequest()
    assert isinstance(response, resources.CryptoKey)
    assert response.name == 'name_value'
    assert response.purpose == resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
    assert response.import_only is True
    assert response.crypto_key_backend == 'crypto_key_backend_value'

@pytest.mark.asyncio
async def test_create_crypto_key_async_from_dict():
    await test_create_crypto_key_async(request_type=dict)

def test_create_crypto_key_field_headers():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateCryptoKeyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_crypto_key), '__call__') as call:
        call.return_value = resources.CryptoKey()
        client.create_crypto_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_crypto_key_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateCryptoKeyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_crypto_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKey())
        await client.create_crypto_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_crypto_key_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_crypto_key), '__call__') as call:
        call.return_value = resources.CryptoKey()
        client.create_crypto_key(parent='parent_value', crypto_key_id='crypto_key_id_value', crypto_key=resources.CryptoKey(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].crypto_key_id
        mock_val = 'crypto_key_id_value'
        assert arg == mock_val
        arg = args[0].crypto_key
        mock_val = resources.CryptoKey(name='name_value')
        assert arg == mock_val

def test_create_crypto_key_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_crypto_key(service.CreateCryptoKeyRequest(), parent='parent_value', crypto_key_id='crypto_key_id_value', crypto_key=resources.CryptoKey(name='name_value'))

@pytest.mark.asyncio
async def test_create_crypto_key_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_crypto_key), '__call__') as call:
        call.return_value = resources.CryptoKey()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKey())
        response = await client.create_crypto_key(parent='parent_value', crypto_key_id='crypto_key_id_value', crypto_key=resources.CryptoKey(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].crypto_key_id
        mock_val = 'crypto_key_id_value'
        assert arg == mock_val
        arg = args[0].crypto_key
        mock_val = resources.CryptoKey(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_crypto_key_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_crypto_key(service.CreateCryptoKeyRequest(), parent='parent_value', crypto_key_id='crypto_key_id_value', crypto_key=resources.CryptoKey(name='name_value'))

@pytest.mark.parametrize('request_type', [service.CreateCryptoKeyVersionRequest, dict])
def test_create_crypto_key_version(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True)
        response = client.create_crypto_key_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCryptoKeyVersionRequest()
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

def test_create_crypto_key_version_empty_call():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_crypto_key_version), '__call__') as call:
        client.create_crypto_key_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCryptoKeyVersionRequest()

@pytest.mark.asyncio
async def test_create_crypto_key_version_async(transport: str='grpc_asyncio', request_type=service.CreateCryptoKeyVersionRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_crypto_key_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True))
        response = await client.create_crypto_key_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCryptoKeyVersionRequest()
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

@pytest.mark.asyncio
async def test_create_crypto_key_version_async_from_dict():
    await test_create_crypto_key_version_async(request_type=dict)

def test_create_crypto_key_version_field_headers():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateCryptoKeyVersionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        client.create_crypto_key_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_crypto_key_version_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateCryptoKeyVersionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_crypto_key_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion())
        await client.create_crypto_key_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_crypto_key_version_flattened():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        client.create_crypto_key_version(parent='parent_value', crypto_key_version=resources.CryptoKeyVersion(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].crypto_key_version
        mock_val = resources.CryptoKeyVersion(name='name_value')
        assert arg == mock_val

def test_create_crypto_key_version_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_crypto_key_version(service.CreateCryptoKeyVersionRequest(), parent='parent_value', crypto_key_version=resources.CryptoKeyVersion(name='name_value'))

@pytest.mark.asyncio
async def test_create_crypto_key_version_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion())
        response = await client.create_crypto_key_version(parent='parent_value', crypto_key_version=resources.CryptoKeyVersion(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].crypto_key_version
        mock_val = resources.CryptoKeyVersion(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_crypto_key_version_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_crypto_key_version(service.CreateCryptoKeyVersionRequest(), parent='parent_value', crypto_key_version=resources.CryptoKeyVersion(name='name_value'))

@pytest.mark.parametrize('request_type', [service.ImportCryptoKeyVersionRequest, dict])
def test_import_crypto_key_version(request_type, transport: str='grpc'):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True)
        response = client.import_crypto_key_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ImportCryptoKeyVersionRequest()
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

def test_import_crypto_key_version_empty_call():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.import_crypto_key_version), '__call__') as call:
        client.import_crypto_key_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ImportCryptoKeyVersionRequest()

@pytest.mark.asyncio
async def test_import_crypto_key_version_async(transport: str='grpc_asyncio', request_type=service.ImportCryptoKeyVersionRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_crypto_key_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True))
        response = await client.import_crypto_key_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ImportCryptoKeyVersionRequest()
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

@pytest.mark.asyncio
async def test_import_crypto_key_version_async_from_dict():
    await test_import_crypto_key_version_async(request_type=dict)

def test_import_crypto_key_version_field_headers():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ImportCryptoKeyVersionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.import_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        client.import_crypto_key_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_import_crypto_key_version_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ImportCryptoKeyVersionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.import_crypto_key_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion())
        await client.import_crypto_key_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.CreateImportJobRequest, dict])
def test_create_import_job(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_import_job), '__call__') as call:
        call.return_value = resources.ImportJob(name='name_value', import_method=resources.ImportJob.ImportMethod.RSA_OAEP_3072_SHA1_AES_256, protection_level=resources.ProtectionLevel.SOFTWARE, state=resources.ImportJob.ImportJobState.PENDING_GENERATION)
        response = client.create_import_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateImportJobRequest()
    assert isinstance(response, resources.ImportJob)
    assert response.name == 'name_value'
    assert response.import_method == resources.ImportJob.ImportMethod.RSA_OAEP_3072_SHA1_AES_256
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.state == resources.ImportJob.ImportJobState.PENDING_GENERATION

def test_create_import_job_empty_call():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_import_job), '__call__') as call:
        client.create_import_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateImportJobRequest()

@pytest.mark.asyncio
async def test_create_import_job_async(transport: str='grpc_asyncio', request_type=service.CreateImportJobRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_import_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ImportJob(name='name_value', import_method=resources.ImportJob.ImportMethod.RSA_OAEP_3072_SHA1_AES_256, protection_level=resources.ProtectionLevel.SOFTWARE, state=resources.ImportJob.ImportJobState.PENDING_GENERATION))
        response = await client.create_import_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateImportJobRequest()
    assert isinstance(response, resources.ImportJob)
    assert response.name == 'name_value'
    assert response.import_method == resources.ImportJob.ImportMethod.RSA_OAEP_3072_SHA1_AES_256
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.state == resources.ImportJob.ImportJobState.PENDING_GENERATION

@pytest.mark.asyncio
async def test_create_import_job_async_from_dict():
    await test_create_import_job_async(request_type=dict)

def test_create_import_job_field_headers():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateImportJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_import_job), '__call__') as call:
        call.return_value = resources.ImportJob()
        client.create_import_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_import_job_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateImportJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_import_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ImportJob())
        await client.create_import_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_import_job_flattened():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_import_job), '__call__') as call:
        call.return_value = resources.ImportJob()
        client.create_import_job(parent='parent_value', import_job_id='import_job_id_value', import_job=resources.ImportJob(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].import_job_id
        mock_val = 'import_job_id_value'
        assert arg == mock_val
        arg = args[0].import_job
        mock_val = resources.ImportJob(name='name_value')
        assert arg == mock_val

def test_create_import_job_flattened_error():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_import_job(service.CreateImportJobRequest(), parent='parent_value', import_job_id='import_job_id_value', import_job=resources.ImportJob(name='name_value'))

@pytest.mark.asyncio
async def test_create_import_job_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_import_job), '__call__') as call:
        call.return_value = resources.ImportJob()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ImportJob())
        response = await client.create_import_job(parent='parent_value', import_job_id='import_job_id_value', import_job=resources.ImportJob(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].import_job_id
        mock_val = 'import_job_id_value'
        assert arg == mock_val
        arg = args[0].import_job
        mock_val = resources.ImportJob(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_import_job_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_import_job(service.CreateImportJobRequest(), parent='parent_value', import_job_id='import_job_id_value', import_job=resources.ImportJob(name='name_value'))

@pytest.mark.parametrize('request_type', [service.UpdateCryptoKeyRequest, dict])
def test_update_crypto_key(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_crypto_key), '__call__') as call:
        call.return_value = resources.CryptoKey(name='name_value', purpose=resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT, import_only=True, crypto_key_backend='crypto_key_backend_value')
        response = client.update_crypto_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCryptoKeyRequest()
    assert isinstance(response, resources.CryptoKey)
    assert response.name == 'name_value'
    assert response.purpose == resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
    assert response.import_only is True
    assert response.crypto_key_backend == 'crypto_key_backend_value'

def test_update_crypto_key_empty_call():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_crypto_key), '__call__') as call:
        client.update_crypto_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCryptoKeyRequest()

@pytest.mark.asyncio
async def test_update_crypto_key_async(transport: str='grpc_asyncio', request_type=service.UpdateCryptoKeyRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_crypto_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKey(name='name_value', purpose=resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT, import_only=True, crypto_key_backend='crypto_key_backend_value'))
        response = await client.update_crypto_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCryptoKeyRequest()
    assert isinstance(response, resources.CryptoKey)
    assert response.name == 'name_value'
    assert response.purpose == resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
    assert response.import_only is True
    assert response.crypto_key_backend == 'crypto_key_backend_value'

@pytest.mark.asyncio
async def test_update_crypto_key_async_from_dict():
    await test_update_crypto_key_async(request_type=dict)

def test_update_crypto_key_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCryptoKeyRequest()
    request.crypto_key.name = 'name_value'
    with mock.patch.object(type(client.transport.update_crypto_key), '__call__') as call:
        call.return_value = resources.CryptoKey()
        client.update_crypto_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'crypto_key.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_crypto_key_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCryptoKeyRequest()
    request.crypto_key.name = 'name_value'
    with mock.patch.object(type(client.transport.update_crypto_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKey())
        await client.update_crypto_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'crypto_key.name=name_value') in kw['metadata']

def test_update_crypto_key_flattened():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_crypto_key), '__call__') as call:
        call.return_value = resources.CryptoKey()
        client.update_crypto_key(crypto_key=resources.CryptoKey(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].crypto_key
        mock_val = resources.CryptoKey(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_crypto_key_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_crypto_key(service.UpdateCryptoKeyRequest(), crypto_key=resources.CryptoKey(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_crypto_key_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_crypto_key), '__call__') as call:
        call.return_value = resources.CryptoKey()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKey())
        response = await client.update_crypto_key(crypto_key=resources.CryptoKey(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].crypto_key
        mock_val = resources.CryptoKey(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_crypto_key_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_crypto_key(service.UpdateCryptoKeyRequest(), crypto_key=resources.CryptoKey(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.UpdateCryptoKeyVersionRequest, dict])
def test_update_crypto_key_version(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True)
        response = client.update_crypto_key_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCryptoKeyVersionRequest()
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

def test_update_crypto_key_version_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_crypto_key_version), '__call__') as call:
        client.update_crypto_key_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCryptoKeyVersionRequest()

@pytest.mark.asyncio
async def test_update_crypto_key_version_async(transport: str='grpc_asyncio', request_type=service.UpdateCryptoKeyVersionRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_crypto_key_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True))
        response = await client.update_crypto_key_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCryptoKeyVersionRequest()
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

@pytest.mark.asyncio
async def test_update_crypto_key_version_async_from_dict():
    await test_update_crypto_key_version_async(request_type=dict)

def test_update_crypto_key_version_field_headers():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCryptoKeyVersionRequest()
    request.crypto_key_version.name = 'name_value'
    with mock.patch.object(type(client.transport.update_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        client.update_crypto_key_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'crypto_key_version.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_crypto_key_version_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCryptoKeyVersionRequest()
    request.crypto_key_version.name = 'name_value'
    with mock.patch.object(type(client.transport.update_crypto_key_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion())
        await client.update_crypto_key_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'crypto_key_version.name=name_value') in kw['metadata']

def test_update_crypto_key_version_flattened():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        client.update_crypto_key_version(crypto_key_version=resources.CryptoKeyVersion(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].crypto_key_version
        mock_val = resources.CryptoKeyVersion(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_crypto_key_version_flattened_error():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_crypto_key_version(service.UpdateCryptoKeyVersionRequest(), crypto_key_version=resources.CryptoKeyVersion(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_crypto_key_version_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion())
        response = await client.update_crypto_key_version(crypto_key_version=resources.CryptoKeyVersion(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].crypto_key_version
        mock_val = resources.CryptoKeyVersion(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_crypto_key_version_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_crypto_key_version(service.UpdateCryptoKeyVersionRequest(), crypto_key_version=resources.CryptoKeyVersion(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.UpdateCryptoKeyPrimaryVersionRequest, dict])
def test_update_crypto_key_primary_version(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_crypto_key_primary_version), '__call__') as call:
        call.return_value = resources.CryptoKey(name='name_value', purpose=resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT, import_only=True, crypto_key_backend='crypto_key_backend_value')
        response = client.update_crypto_key_primary_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCryptoKeyPrimaryVersionRequest()
    assert isinstance(response, resources.CryptoKey)
    assert response.name == 'name_value'
    assert response.purpose == resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
    assert response.import_only is True
    assert response.crypto_key_backend == 'crypto_key_backend_value'

def test_update_crypto_key_primary_version_empty_call():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_crypto_key_primary_version), '__call__') as call:
        client.update_crypto_key_primary_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCryptoKeyPrimaryVersionRequest()

@pytest.mark.asyncio
async def test_update_crypto_key_primary_version_async(transport: str='grpc_asyncio', request_type=service.UpdateCryptoKeyPrimaryVersionRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_crypto_key_primary_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKey(name='name_value', purpose=resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT, import_only=True, crypto_key_backend='crypto_key_backend_value'))
        response = await client.update_crypto_key_primary_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCryptoKeyPrimaryVersionRequest()
    assert isinstance(response, resources.CryptoKey)
    assert response.name == 'name_value'
    assert response.purpose == resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
    assert response.import_only is True
    assert response.crypto_key_backend == 'crypto_key_backend_value'

@pytest.mark.asyncio
async def test_update_crypto_key_primary_version_async_from_dict():
    await test_update_crypto_key_primary_version_async(request_type=dict)

def test_update_crypto_key_primary_version_field_headers():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCryptoKeyPrimaryVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_crypto_key_primary_version), '__call__') as call:
        call.return_value = resources.CryptoKey()
        client.update_crypto_key_primary_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_crypto_key_primary_version_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCryptoKeyPrimaryVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_crypto_key_primary_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKey())
        await client.update_crypto_key_primary_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_crypto_key_primary_version_flattened():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_crypto_key_primary_version), '__call__') as call:
        call.return_value = resources.CryptoKey()
        client.update_crypto_key_primary_version(name='name_value', crypto_key_version_id='crypto_key_version_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].crypto_key_version_id
        mock_val = 'crypto_key_version_id_value'
        assert arg == mock_val

def test_update_crypto_key_primary_version_flattened_error():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_crypto_key_primary_version(service.UpdateCryptoKeyPrimaryVersionRequest(), name='name_value', crypto_key_version_id='crypto_key_version_id_value')

@pytest.mark.asyncio
async def test_update_crypto_key_primary_version_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_crypto_key_primary_version), '__call__') as call:
        call.return_value = resources.CryptoKey()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKey())
        response = await client.update_crypto_key_primary_version(name='name_value', crypto_key_version_id='crypto_key_version_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].crypto_key_version_id
        mock_val = 'crypto_key_version_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_crypto_key_primary_version_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_crypto_key_primary_version(service.UpdateCryptoKeyPrimaryVersionRequest(), name='name_value', crypto_key_version_id='crypto_key_version_id_value')

@pytest.mark.parametrize('request_type', [service.DestroyCryptoKeyVersionRequest, dict])
def test_destroy_crypto_key_version(request_type, transport: str='grpc'):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.destroy_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True)
        response = client.destroy_crypto_key_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DestroyCryptoKeyVersionRequest()
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

def test_destroy_crypto_key_version_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.destroy_crypto_key_version), '__call__') as call:
        client.destroy_crypto_key_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DestroyCryptoKeyVersionRequest()

@pytest.mark.asyncio
async def test_destroy_crypto_key_version_async(transport: str='grpc_asyncio', request_type=service.DestroyCryptoKeyVersionRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.destroy_crypto_key_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True))
        response = await client.destroy_crypto_key_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DestroyCryptoKeyVersionRequest()
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

@pytest.mark.asyncio
async def test_destroy_crypto_key_version_async_from_dict():
    await test_destroy_crypto_key_version_async(request_type=dict)

def test_destroy_crypto_key_version_field_headers():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DestroyCryptoKeyVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.destroy_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        client.destroy_crypto_key_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_destroy_crypto_key_version_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DestroyCryptoKeyVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.destroy_crypto_key_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion())
        await client.destroy_crypto_key_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_destroy_crypto_key_version_flattened():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.destroy_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        client.destroy_crypto_key_version(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_destroy_crypto_key_version_flattened_error():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.destroy_crypto_key_version(service.DestroyCryptoKeyVersionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_destroy_crypto_key_version_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.destroy_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion())
        response = await client.destroy_crypto_key_version(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_destroy_crypto_key_version_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.destroy_crypto_key_version(service.DestroyCryptoKeyVersionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.RestoreCryptoKeyVersionRequest, dict])
def test_restore_crypto_key_version(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.restore_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True)
        response = client.restore_crypto_key_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RestoreCryptoKeyVersionRequest()
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

def test_restore_crypto_key_version_empty_call():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.restore_crypto_key_version), '__call__') as call:
        client.restore_crypto_key_version()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RestoreCryptoKeyVersionRequest()

@pytest.mark.asyncio
async def test_restore_crypto_key_version_async(transport: str='grpc_asyncio', request_type=service.RestoreCryptoKeyVersionRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.restore_crypto_key_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True))
        response = await client.restore_crypto_key_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RestoreCryptoKeyVersionRequest()
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

@pytest.mark.asyncio
async def test_restore_crypto_key_version_async_from_dict():
    await test_restore_crypto_key_version_async(request_type=dict)

def test_restore_crypto_key_version_field_headers():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RestoreCryptoKeyVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.restore_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        client.restore_crypto_key_version(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_restore_crypto_key_version_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RestoreCryptoKeyVersionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.restore_crypto_key_version), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion())
        await client.restore_crypto_key_version(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_restore_crypto_key_version_flattened():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.restore_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        client.restore_crypto_key_version(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_restore_crypto_key_version_flattened_error():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.restore_crypto_key_version(service.RestoreCryptoKeyVersionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_restore_crypto_key_version_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.restore_crypto_key_version), '__call__') as call:
        call.return_value = resources.CryptoKeyVersion()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CryptoKeyVersion())
        response = await client.restore_crypto_key_version(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_restore_crypto_key_version_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.restore_crypto_key_version(service.RestoreCryptoKeyVersionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.EncryptRequest, dict])
def test_encrypt(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.encrypt), '__call__') as call:
        call.return_value = service.EncryptResponse(name='name_value', ciphertext=b'ciphertext_blob', verified_plaintext_crc32c=True, verified_additional_authenticated_data_crc32c=True, protection_level=resources.ProtectionLevel.SOFTWARE)
        response = client.encrypt(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.EncryptRequest()
    assert isinstance(response, service.EncryptResponse)
    assert response.name == 'name_value'
    assert response.ciphertext == b'ciphertext_blob'
    assert response.verified_plaintext_crc32c is True
    assert response.verified_additional_authenticated_data_crc32c is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_encrypt_empty_call():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.encrypt), '__call__') as call:
        client.encrypt()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.EncryptRequest()

@pytest.mark.asyncio
async def test_encrypt_async(transport: str='grpc_asyncio', request_type=service.EncryptRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.encrypt), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.EncryptResponse(name='name_value', ciphertext=b'ciphertext_blob', verified_plaintext_crc32c=True, verified_additional_authenticated_data_crc32c=True, protection_level=resources.ProtectionLevel.SOFTWARE))
        response = await client.encrypt(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.EncryptRequest()
    assert isinstance(response, service.EncryptResponse)
    assert response.name == 'name_value'
    assert response.ciphertext == b'ciphertext_blob'
    assert response.verified_plaintext_crc32c is True
    assert response.verified_additional_authenticated_data_crc32c is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

@pytest.mark.asyncio
async def test_encrypt_async_from_dict():
    await test_encrypt_async(request_type=dict)

def test_encrypt_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.EncryptRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.encrypt), '__call__') as call:
        call.return_value = service.EncryptResponse()
        client.encrypt(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_encrypt_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.EncryptRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.encrypt), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.EncryptResponse())
        await client.encrypt(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_encrypt_flattened():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.encrypt), '__call__') as call:
        call.return_value = service.EncryptResponse()
        client.encrypt(name='name_value', plaintext=b'plaintext_blob')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].plaintext
        mock_val = b'plaintext_blob'
        assert arg == mock_val

def test_encrypt_flattened_error():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.encrypt(service.EncryptRequest(), name='name_value', plaintext=b'plaintext_blob')

@pytest.mark.asyncio
async def test_encrypt_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.encrypt), '__call__') as call:
        call.return_value = service.EncryptResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.EncryptResponse())
        response = await client.encrypt(name='name_value', plaintext=b'plaintext_blob')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].plaintext
        mock_val = b'plaintext_blob'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_encrypt_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.encrypt(service.EncryptRequest(), name='name_value', plaintext=b'plaintext_blob')

@pytest.mark.parametrize('request_type', [service.DecryptRequest, dict])
def test_decrypt(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.decrypt), '__call__') as call:
        call.return_value = service.DecryptResponse(plaintext=b'plaintext_blob', used_primary=True, protection_level=resources.ProtectionLevel.SOFTWARE)
        response = client.decrypt(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DecryptRequest()
    assert isinstance(response, service.DecryptResponse)
    assert response.plaintext == b'plaintext_blob'
    assert response.used_primary is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_decrypt_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.decrypt), '__call__') as call:
        client.decrypt()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DecryptRequest()

@pytest.mark.asyncio
async def test_decrypt_async(transport: str='grpc_asyncio', request_type=service.DecryptRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.decrypt), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.DecryptResponse(plaintext=b'plaintext_blob', used_primary=True, protection_level=resources.ProtectionLevel.SOFTWARE))
        response = await client.decrypt(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DecryptRequest()
    assert isinstance(response, service.DecryptResponse)
    assert response.plaintext == b'plaintext_blob'
    assert response.used_primary is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

@pytest.mark.asyncio
async def test_decrypt_async_from_dict():
    await test_decrypt_async(request_type=dict)

def test_decrypt_field_headers():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DecryptRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.decrypt), '__call__') as call:
        call.return_value = service.DecryptResponse()
        client.decrypt(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_decrypt_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DecryptRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.decrypt), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.DecryptResponse())
        await client.decrypt(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_decrypt_flattened():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.decrypt), '__call__') as call:
        call.return_value = service.DecryptResponse()
        client.decrypt(name='name_value', ciphertext=b'ciphertext_blob')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].ciphertext
        mock_val = b'ciphertext_blob'
        assert arg == mock_val

def test_decrypt_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.decrypt(service.DecryptRequest(), name='name_value', ciphertext=b'ciphertext_blob')

@pytest.mark.asyncio
async def test_decrypt_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.decrypt), '__call__') as call:
        call.return_value = service.DecryptResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.DecryptResponse())
        response = await client.decrypt(name='name_value', ciphertext=b'ciphertext_blob')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].ciphertext
        mock_val = b'ciphertext_blob'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_decrypt_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.decrypt(service.DecryptRequest(), name='name_value', ciphertext=b'ciphertext_blob')

@pytest.mark.parametrize('request_type', [service.RawEncryptRequest, dict])
def test_raw_encrypt(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.raw_encrypt), '__call__') as call:
        call.return_value = service.RawEncryptResponse(ciphertext=b'ciphertext_blob', initialization_vector=b'initialization_vector_blob', tag_length=1053, verified_plaintext_crc32c=True, verified_additional_authenticated_data_crc32c=True, verified_initialization_vector_crc32c=True, name='name_value', protection_level=resources.ProtectionLevel.SOFTWARE)
        response = client.raw_encrypt(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RawEncryptRequest()
    assert isinstance(response, service.RawEncryptResponse)
    assert response.ciphertext == b'ciphertext_blob'
    assert response.initialization_vector == b'initialization_vector_blob'
    assert response.tag_length == 1053
    assert response.verified_plaintext_crc32c is True
    assert response.verified_additional_authenticated_data_crc32c is True
    assert response.verified_initialization_vector_crc32c is True
    assert response.name == 'name_value'
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_raw_encrypt_empty_call():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.raw_encrypt), '__call__') as call:
        client.raw_encrypt()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RawEncryptRequest()

@pytest.mark.asyncio
async def test_raw_encrypt_async(transport: str='grpc_asyncio', request_type=service.RawEncryptRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.raw_encrypt), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.RawEncryptResponse(ciphertext=b'ciphertext_blob', initialization_vector=b'initialization_vector_blob', tag_length=1053, verified_plaintext_crc32c=True, verified_additional_authenticated_data_crc32c=True, verified_initialization_vector_crc32c=True, name='name_value', protection_level=resources.ProtectionLevel.SOFTWARE))
        response = await client.raw_encrypt(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RawEncryptRequest()
    assert isinstance(response, service.RawEncryptResponse)
    assert response.ciphertext == b'ciphertext_blob'
    assert response.initialization_vector == b'initialization_vector_blob'
    assert response.tag_length == 1053
    assert response.verified_plaintext_crc32c is True
    assert response.verified_additional_authenticated_data_crc32c is True
    assert response.verified_initialization_vector_crc32c is True
    assert response.name == 'name_value'
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

@pytest.mark.asyncio
async def test_raw_encrypt_async_from_dict():
    await test_raw_encrypt_async(request_type=dict)

def test_raw_encrypt_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RawEncryptRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.raw_encrypt), '__call__') as call:
        call.return_value = service.RawEncryptResponse()
        client.raw_encrypt(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_raw_encrypt_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RawEncryptRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.raw_encrypt), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.RawEncryptResponse())
        await client.raw_encrypt(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.RawDecryptRequest, dict])
def test_raw_decrypt(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.raw_decrypt), '__call__') as call:
        call.return_value = service.RawDecryptResponse(plaintext=b'plaintext_blob', protection_level=resources.ProtectionLevel.SOFTWARE, verified_ciphertext_crc32c=True, verified_additional_authenticated_data_crc32c=True, verified_initialization_vector_crc32c=True)
        response = client.raw_decrypt(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RawDecryptRequest()
    assert isinstance(response, service.RawDecryptResponse)
    assert response.plaintext == b'plaintext_blob'
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.verified_ciphertext_crc32c is True
    assert response.verified_additional_authenticated_data_crc32c is True
    assert response.verified_initialization_vector_crc32c is True

def test_raw_decrypt_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.raw_decrypt), '__call__') as call:
        client.raw_decrypt()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RawDecryptRequest()

@pytest.mark.asyncio
async def test_raw_decrypt_async(transport: str='grpc_asyncio', request_type=service.RawDecryptRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.raw_decrypt), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.RawDecryptResponse(plaintext=b'plaintext_blob', protection_level=resources.ProtectionLevel.SOFTWARE, verified_ciphertext_crc32c=True, verified_additional_authenticated_data_crc32c=True, verified_initialization_vector_crc32c=True))
        response = await client.raw_decrypt(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RawDecryptRequest()
    assert isinstance(response, service.RawDecryptResponse)
    assert response.plaintext == b'plaintext_blob'
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.verified_ciphertext_crc32c is True
    assert response.verified_additional_authenticated_data_crc32c is True
    assert response.verified_initialization_vector_crc32c is True

@pytest.mark.asyncio
async def test_raw_decrypt_async_from_dict():
    await test_raw_decrypt_async(request_type=dict)

def test_raw_decrypt_field_headers():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RawDecryptRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.raw_decrypt), '__call__') as call:
        call.return_value = service.RawDecryptResponse()
        client.raw_decrypt(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_raw_decrypt_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RawDecryptRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.raw_decrypt), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.RawDecryptResponse())
        await client.raw_decrypt(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.AsymmetricSignRequest, dict])
def test_asymmetric_sign(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.asymmetric_sign), '__call__') as call:
        call.return_value = service.AsymmetricSignResponse(signature=b'signature_blob', verified_digest_crc32c=True, name='name_value', verified_data_crc32c=True, protection_level=resources.ProtectionLevel.SOFTWARE)
        response = client.asymmetric_sign(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.AsymmetricSignRequest()
    assert isinstance(response, service.AsymmetricSignResponse)
    assert response.signature == b'signature_blob'
    assert response.verified_digest_crc32c is True
    assert response.name == 'name_value'
    assert response.verified_data_crc32c is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_asymmetric_sign_empty_call():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.asymmetric_sign), '__call__') as call:
        client.asymmetric_sign()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.AsymmetricSignRequest()

@pytest.mark.asyncio
async def test_asymmetric_sign_async(transport: str='grpc_asyncio', request_type=service.AsymmetricSignRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.asymmetric_sign), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.AsymmetricSignResponse(signature=b'signature_blob', verified_digest_crc32c=True, name='name_value', verified_data_crc32c=True, protection_level=resources.ProtectionLevel.SOFTWARE))
        response = await client.asymmetric_sign(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.AsymmetricSignRequest()
    assert isinstance(response, service.AsymmetricSignResponse)
    assert response.signature == b'signature_blob'
    assert response.verified_digest_crc32c is True
    assert response.name == 'name_value'
    assert response.verified_data_crc32c is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

@pytest.mark.asyncio
async def test_asymmetric_sign_async_from_dict():
    await test_asymmetric_sign_async(request_type=dict)

def test_asymmetric_sign_field_headers():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.AsymmetricSignRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.asymmetric_sign), '__call__') as call:
        call.return_value = service.AsymmetricSignResponse()
        client.asymmetric_sign(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_asymmetric_sign_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.AsymmetricSignRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.asymmetric_sign), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.AsymmetricSignResponse())
        await client.asymmetric_sign(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_asymmetric_sign_flattened():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.asymmetric_sign), '__call__') as call:
        call.return_value = service.AsymmetricSignResponse()
        client.asymmetric_sign(name='name_value', digest=service.Digest(sha256=b'sha256_blob'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].digest
        mock_val = service.Digest(sha256=b'sha256_blob')
        assert arg == mock_val

def test_asymmetric_sign_flattened_error():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.asymmetric_sign(service.AsymmetricSignRequest(), name='name_value', digest=service.Digest(sha256=b'sha256_blob'))

@pytest.mark.asyncio
async def test_asymmetric_sign_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.asymmetric_sign), '__call__') as call:
        call.return_value = service.AsymmetricSignResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.AsymmetricSignResponse())
        response = await client.asymmetric_sign(name='name_value', digest=service.Digest(sha256=b'sha256_blob'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].digest
        mock_val = service.Digest(sha256=b'sha256_blob')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_asymmetric_sign_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.asymmetric_sign(service.AsymmetricSignRequest(), name='name_value', digest=service.Digest(sha256=b'sha256_blob'))

@pytest.mark.parametrize('request_type', [service.AsymmetricDecryptRequest, dict])
def test_asymmetric_decrypt(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.asymmetric_decrypt), '__call__') as call:
        call.return_value = service.AsymmetricDecryptResponse(plaintext=b'plaintext_blob', verified_ciphertext_crc32c=True, protection_level=resources.ProtectionLevel.SOFTWARE)
        response = client.asymmetric_decrypt(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.AsymmetricDecryptRequest()
    assert isinstance(response, service.AsymmetricDecryptResponse)
    assert response.plaintext == b'plaintext_blob'
    assert response.verified_ciphertext_crc32c is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_asymmetric_decrypt_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.asymmetric_decrypt), '__call__') as call:
        client.asymmetric_decrypt()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.AsymmetricDecryptRequest()

@pytest.mark.asyncio
async def test_asymmetric_decrypt_async(transport: str='grpc_asyncio', request_type=service.AsymmetricDecryptRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.asymmetric_decrypt), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.AsymmetricDecryptResponse(plaintext=b'plaintext_blob', verified_ciphertext_crc32c=True, protection_level=resources.ProtectionLevel.SOFTWARE))
        response = await client.asymmetric_decrypt(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.AsymmetricDecryptRequest()
    assert isinstance(response, service.AsymmetricDecryptResponse)
    assert response.plaintext == b'plaintext_blob'
    assert response.verified_ciphertext_crc32c is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

@pytest.mark.asyncio
async def test_asymmetric_decrypt_async_from_dict():
    await test_asymmetric_decrypt_async(request_type=dict)

def test_asymmetric_decrypt_field_headers():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.AsymmetricDecryptRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.asymmetric_decrypt), '__call__') as call:
        call.return_value = service.AsymmetricDecryptResponse()
        client.asymmetric_decrypt(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_asymmetric_decrypt_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.AsymmetricDecryptRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.asymmetric_decrypt), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.AsymmetricDecryptResponse())
        await client.asymmetric_decrypt(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_asymmetric_decrypt_flattened():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.asymmetric_decrypt), '__call__') as call:
        call.return_value = service.AsymmetricDecryptResponse()
        client.asymmetric_decrypt(name='name_value', ciphertext=b'ciphertext_blob')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].ciphertext
        mock_val = b'ciphertext_blob'
        assert arg == mock_val

def test_asymmetric_decrypt_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.asymmetric_decrypt(service.AsymmetricDecryptRequest(), name='name_value', ciphertext=b'ciphertext_blob')

@pytest.mark.asyncio
async def test_asymmetric_decrypt_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.asymmetric_decrypt), '__call__') as call:
        call.return_value = service.AsymmetricDecryptResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.AsymmetricDecryptResponse())
        response = await client.asymmetric_decrypt(name='name_value', ciphertext=b'ciphertext_blob')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].ciphertext
        mock_val = b'ciphertext_blob'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_asymmetric_decrypt_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.asymmetric_decrypt(service.AsymmetricDecryptRequest(), name='name_value', ciphertext=b'ciphertext_blob')

@pytest.mark.parametrize('request_type', [service.MacSignRequest, dict])
def test_mac_sign(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.mac_sign), '__call__') as call:
        call.return_value = service.MacSignResponse(name='name_value', mac=b'mac_blob', verified_data_crc32c=True, protection_level=resources.ProtectionLevel.SOFTWARE)
        response = client.mac_sign(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.MacSignRequest()
    assert isinstance(response, service.MacSignResponse)
    assert response.name == 'name_value'
    assert response.mac == b'mac_blob'
    assert response.verified_data_crc32c is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_mac_sign_empty_call():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.mac_sign), '__call__') as call:
        client.mac_sign()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.MacSignRequest()

@pytest.mark.asyncio
async def test_mac_sign_async(transport: str='grpc_asyncio', request_type=service.MacSignRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.mac_sign), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.MacSignResponse(name='name_value', mac=b'mac_blob', verified_data_crc32c=True, protection_level=resources.ProtectionLevel.SOFTWARE))
        response = await client.mac_sign(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.MacSignRequest()
    assert isinstance(response, service.MacSignResponse)
    assert response.name == 'name_value'
    assert response.mac == b'mac_blob'
    assert response.verified_data_crc32c is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

@pytest.mark.asyncio
async def test_mac_sign_async_from_dict():
    await test_mac_sign_async(request_type=dict)

def test_mac_sign_field_headers():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.MacSignRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.mac_sign), '__call__') as call:
        call.return_value = service.MacSignResponse()
        client.mac_sign(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_mac_sign_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.MacSignRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.mac_sign), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.MacSignResponse())
        await client.mac_sign(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_mac_sign_flattened():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.mac_sign), '__call__') as call:
        call.return_value = service.MacSignResponse()
        client.mac_sign(name='name_value', data=b'data_blob')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].data
        mock_val = b'data_blob'
        assert arg == mock_val

def test_mac_sign_flattened_error():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.mac_sign(service.MacSignRequest(), name='name_value', data=b'data_blob')

@pytest.mark.asyncio
async def test_mac_sign_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.mac_sign), '__call__') as call:
        call.return_value = service.MacSignResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.MacSignResponse())
        response = await client.mac_sign(name='name_value', data=b'data_blob')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].data
        mock_val = b'data_blob'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_mac_sign_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.mac_sign(service.MacSignRequest(), name='name_value', data=b'data_blob')

@pytest.mark.parametrize('request_type', [service.MacVerifyRequest, dict])
def test_mac_verify(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.mac_verify), '__call__') as call:
        call.return_value = service.MacVerifyResponse(name='name_value', success=True, verified_data_crc32c=True, verified_mac_crc32c=True, verified_success_integrity=True, protection_level=resources.ProtectionLevel.SOFTWARE)
        response = client.mac_verify(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.MacVerifyRequest()
    assert isinstance(response, service.MacVerifyResponse)
    assert response.name == 'name_value'
    assert response.success is True
    assert response.verified_data_crc32c is True
    assert response.verified_mac_crc32c is True
    assert response.verified_success_integrity is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_mac_verify_empty_call():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.mac_verify), '__call__') as call:
        client.mac_verify()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.MacVerifyRequest()

@pytest.mark.asyncio
async def test_mac_verify_async(transport: str='grpc_asyncio', request_type=service.MacVerifyRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.mac_verify), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.MacVerifyResponse(name='name_value', success=True, verified_data_crc32c=True, verified_mac_crc32c=True, verified_success_integrity=True, protection_level=resources.ProtectionLevel.SOFTWARE))
        response = await client.mac_verify(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.MacVerifyRequest()
    assert isinstance(response, service.MacVerifyResponse)
    assert response.name == 'name_value'
    assert response.success is True
    assert response.verified_data_crc32c is True
    assert response.verified_mac_crc32c is True
    assert response.verified_success_integrity is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

@pytest.mark.asyncio
async def test_mac_verify_async_from_dict():
    await test_mac_verify_async(request_type=dict)

def test_mac_verify_field_headers():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.MacVerifyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.mac_verify), '__call__') as call:
        call.return_value = service.MacVerifyResponse()
        client.mac_verify(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_mac_verify_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.MacVerifyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.mac_verify), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.MacVerifyResponse())
        await client.mac_verify(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_mac_verify_flattened():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.mac_verify), '__call__') as call:
        call.return_value = service.MacVerifyResponse()
        client.mac_verify(name='name_value', data=b'data_blob', mac=b'mac_blob')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].data
        mock_val = b'data_blob'
        assert arg == mock_val
        arg = args[0].mac
        mock_val = b'mac_blob'
        assert arg == mock_val

def test_mac_verify_flattened_error():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.mac_verify(service.MacVerifyRequest(), name='name_value', data=b'data_blob', mac=b'mac_blob')

@pytest.mark.asyncio
async def test_mac_verify_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.mac_verify), '__call__') as call:
        call.return_value = service.MacVerifyResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.MacVerifyResponse())
        response = await client.mac_verify(name='name_value', data=b'data_blob', mac=b'mac_blob')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].data
        mock_val = b'data_blob'
        assert arg == mock_val
        arg = args[0].mac
        mock_val = b'mac_blob'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_mac_verify_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.mac_verify(service.MacVerifyRequest(), name='name_value', data=b'data_blob', mac=b'mac_blob')

@pytest.mark.parametrize('request_type', [service.GenerateRandomBytesRequest, dict])
def test_generate_random_bytes(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_random_bytes), '__call__') as call:
        call.return_value = service.GenerateRandomBytesResponse(data=b'data_blob')
        response = client.generate_random_bytes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GenerateRandomBytesRequest()
    assert isinstance(response, service.GenerateRandomBytesResponse)
    assert response.data == b'data_blob'

def test_generate_random_bytes_empty_call():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.generate_random_bytes), '__call__') as call:
        client.generate_random_bytes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GenerateRandomBytesRequest()

@pytest.mark.asyncio
async def test_generate_random_bytes_async(transport: str='grpc_asyncio', request_type=service.GenerateRandomBytesRequest):
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_random_bytes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.GenerateRandomBytesResponse(data=b'data_blob'))
        response = await client.generate_random_bytes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GenerateRandomBytesRequest()
    assert isinstance(response, service.GenerateRandomBytesResponse)
    assert response.data == b'data_blob'

@pytest.mark.asyncio
async def test_generate_random_bytes_async_from_dict():
    await test_generate_random_bytes_async(request_type=dict)

def test_generate_random_bytes_field_headers():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GenerateRandomBytesRequest()
    request.location = 'location_value'
    with mock.patch.object(type(client.transport.generate_random_bytes), '__call__') as call:
        call.return_value = service.GenerateRandomBytesResponse()
        client.generate_random_bytes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'location=location_value') in kw['metadata']

@pytest.mark.asyncio
async def test_generate_random_bytes_field_headers_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GenerateRandomBytesRequest()
    request.location = 'location_value'
    with mock.patch.object(type(client.transport.generate_random_bytes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.GenerateRandomBytesResponse())
        await client.generate_random_bytes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'location=location_value') in kw['metadata']

def test_generate_random_bytes_flattened():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.generate_random_bytes), '__call__') as call:
        call.return_value = service.GenerateRandomBytesResponse()
        client.generate_random_bytes(location='location_value', length_bytes=1288, protection_level=resources.ProtectionLevel.SOFTWARE)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].location
        mock_val = 'location_value'
        assert arg == mock_val
        arg = args[0].length_bytes
        mock_val = 1288
        assert arg == mock_val
        arg = args[0].protection_level
        mock_val = resources.ProtectionLevel.SOFTWARE
        assert arg == mock_val

def test_generate_random_bytes_flattened_error():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.generate_random_bytes(service.GenerateRandomBytesRequest(), location='location_value', length_bytes=1288, protection_level=resources.ProtectionLevel.SOFTWARE)

@pytest.mark.asyncio
async def test_generate_random_bytes_flattened_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.generate_random_bytes), '__call__') as call:
        call.return_value = service.GenerateRandomBytesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.GenerateRandomBytesResponse())
        response = await client.generate_random_bytes(location='location_value', length_bytes=1288, protection_level=resources.ProtectionLevel.SOFTWARE)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].location
        mock_val = 'location_value'
        assert arg == mock_val
        arg = args[0].length_bytes
        mock_val = 1288
        assert arg == mock_val
        arg = args[0].protection_level
        mock_val = resources.ProtectionLevel.SOFTWARE
        assert arg == mock_val

@pytest.mark.asyncio
async def test_generate_random_bytes_flattened_error_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.generate_random_bytes(service.GenerateRandomBytesRequest(), location='location_value', length_bytes=1288, protection_level=resources.ProtectionLevel.SOFTWARE)

@pytest.mark.parametrize('request_type', [service.ListKeyRingsRequest, dict])
def test_list_key_rings_rest(request_type):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListKeyRingsResponse(next_page_token='next_page_token_value', total_size=1086)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListKeyRingsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_key_rings(request)
    assert isinstance(response, pagers.ListKeyRingsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_key_rings_rest_required_fields(request_type=service.ListKeyRingsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_key_rings._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_key_rings._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListKeyRingsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListKeyRingsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_key_rings(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_key_rings_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_key_rings._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_key_rings_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_list_key_rings') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_list_key_rings') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListKeyRingsRequest.pb(service.ListKeyRingsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListKeyRingsResponse.to_json(service.ListKeyRingsResponse())
        request = service.ListKeyRingsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListKeyRingsResponse()
        client.list_key_rings(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_key_rings_rest_bad_request(transport: str='rest', request_type=service.ListKeyRingsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_key_rings(request)

def test_list_key_rings_rest_flattened():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListKeyRingsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListKeyRingsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_key_rings(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/keyRings' % client.transport._host, args[1])

def test_list_key_rings_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_key_rings(service.ListKeyRingsRequest(), parent='parent_value')

def test_list_key_rings_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListKeyRingsResponse(key_rings=[resources.KeyRing(), resources.KeyRing(), resources.KeyRing()], next_page_token='abc'), service.ListKeyRingsResponse(key_rings=[], next_page_token='def'), service.ListKeyRingsResponse(key_rings=[resources.KeyRing()], next_page_token='ghi'), service.ListKeyRingsResponse(key_rings=[resources.KeyRing(), resources.KeyRing()]))
        response = response + response
        response = tuple((service.ListKeyRingsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_key_rings(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.KeyRing) for i in results))
        pages = list(client.list_key_rings(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListCryptoKeysRequest, dict])
def test_list_crypto_keys_rest(request_type):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListCryptoKeysResponse(next_page_token='next_page_token_value', total_size=1086)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListCryptoKeysResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_crypto_keys(request)
    assert isinstance(response, pagers.ListCryptoKeysPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_crypto_keys_rest_required_fields(request_type=service.ListCryptoKeysRequest):
    if False:
        return 10
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_crypto_keys._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_crypto_keys._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token', 'version_view'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListCryptoKeysResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListCryptoKeysResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_crypto_keys(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_crypto_keys_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_crypto_keys._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken', 'versionView')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_crypto_keys_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_list_crypto_keys') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_list_crypto_keys') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListCryptoKeysRequest.pb(service.ListCryptoKeysRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListCryptoKeysResponse.to_json(service.ListCryptoKeysResponse())
        request = service.ListCryptoKeysRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListCryptoKeysResponse()
        client.list_crypto_keys(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_crypto_keys_rest_bad_request(transport: str='rest', request_type=service.ListCryptoKeysRequest):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_crypto_keys(request)

def test_list_crypto_keys_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListCryptoKeysResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListCryptoKeysResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_crypto_keys(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/keyRings/*}/cryptoKeys' % client.transport._host, args[1])

def test_list_crypto_keys_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_crypto_keys(service.ListCryptoKeysRequest(), parent='parent_value')

def test_list_crypto_keys_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey(), resources.CryptoKey(), resources.CryptoKey()], next_page_token='abc'), service.ListCryptoKeysResponse(crypto_keys=[], next_page_token='def'), service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey()], next_page_token='ghi'), service.ListCryptoKeysResponse(crypto_keys=[resources.CryptoKey(), resources.CryptoKey()]))
        response = response + response
        response = tuple((service.ListCryptoKeysResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3'}
        pager = client.list_crypto_keys(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.CryptoKey) for i in results))
        pages = list(client.list_crypto_keys(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListCryptoKeyVersionsRequest, dict])
def test_list_crypto_key_versions_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListCryptoKeyVersionsResponse(next_page_token='next_page_token_value', total_size=1086)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListCryptoKeyVersionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_crypto_key_versions(request)
    assert isinstance(response, pagers.ListCryptoKeyVersionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_crypto_key_versions_rest_required_fields(request_type=service.ListCryptoKeyVersionsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_crypto_key_versions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_crypto_key_versions._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token', 'view'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListCryptoKeyVersionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListCryptoKeyVersionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_crypto_key_versions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_crypto_key_versions_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_crypto_key_versions._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken', 'view')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_crypto_key_versions_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_list_crypto_key_versions') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_list_crypto_key_versions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListCryptoKeyVersionsRequest.pb(service.ListCryptoKeyVersionsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListCryptoKeyVersionsResponse.to_json(service.ListCryptoKeyVersionsResponse())
        request = service.ListCryptoKeyVersionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListCryptoKeyVersionsResponse()
        client.list_crypto_key_versions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_crypto_key_versions_rest_bad_request(transport: str='rest', request_type=service.ListCryptoKeyVersionsRequest):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_crypto_key_versions(request)

def test_list_crypto_key_versions_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListCryptoKeyVersionsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListCryptoKeyVersionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_crypto_key_versions(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/keyRings/*/cryptoKeys/*}/cryptoKeyVersions' % client.transport._host, args[1])

def test_list_crypto_key_versions_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_crypto_key_versions(service.ListCryptoKeyVersionsRequest(), parent='parent_value')

def test_list_crypto_key_versions_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion(), resources.CryptoKeyVersion(), resources.CryptoKeyVersion()], next_page_token='abc'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[], next_page_token='def'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion()], next_page_token='ghi'), service.ListCryptoKeyVersionsResponse(crypto_key_versions=[resources.CryptoKeyVersion(), resources.CryptoKeyVersion()]))
        response = response + response
        response = tuple((service.ListCryptoKeyVersionsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
        pager = client.list_crypto_key_versions(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.CryptoKeyVersion) for i in results))
        pages = list(client.list_crypto_key_versions(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListImportJobsRequest, dict])
def test_list_import_jobs_rest(request_type):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListImportJobsResponse(next_page_token='next_page_token_value', total_size=1086)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListImportJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_import_jobs(request)
    assert isinstance(response, pagers.ListImportJobsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_import_jobs_rest_required_fields(request_type=service.ListImportJobsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_import_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_import_jobs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListImportJobsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListImportJobsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_import_jobs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_import_jobs_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_import_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_import_jobs_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_list_import_jobs') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_list_import_jobs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListImportJobsRequest.pb(service.ListImportJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListImportJobsResponse.to_json(service.ListImportJobsResponse())
        request = service.ListImportJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListImportJobsResponse()
        client.list_import_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_import_jobs_rest_bad_request(transport: str='rest', request_type=service.ListImportJobsRequest):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_import_jobs(request)

def test_list_import_jobs_rest_flattened():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListImportJobsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListImportJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_import_jobs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/keyRings/*}/importJobs' % client.transport._host, args[1])

def test_list_import_jobs_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_import_jobs(service.ListImportJobsRequest(), parent='parent_value')

def test_list_import_jobs_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListImportJobsResponse(import_jobs=[resources.ImportJob(), resources.ImportJob(), resources.ImportJob()], next_page_token='abc'), service.ListImportJobsResponse(import_jobs=[], next_page_token='def'), service.ListImportJobsResponse(import_jobs=[resources.ImportJob()], next_page_token='ghi'), service.ListImportJobsResponse(import_jobs=[resources.ImportJob(), resources.ImportJob()]))
        response = response + response
        response = tuple((service.ListImportJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3'}
        pager = client.list_import_jobs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.ImportJob) for i in results))
        pages = list(client.list_import_jobs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetKeyRingRequest, dict])
def test_get_key_ring_rest(request_type):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.KeyRing(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.KeyRing.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_key_ring(request)
    assert isinstance(response, resources.KeyRing)
    assert response.name == 'name_value'

def test_get_key_ring_rest_required_fields(request_type=service.GetKeyRingRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_key_ring._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_key_ring._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.KeyRing()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.KeyRing.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_key_ring(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_key_ring_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_key_ring._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_key_ring_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_get_key_ring') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_get_key_ring') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetKeyRingRequest.pb(service.GetKeyRingRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.KeyRing.to_json(resources.KeyRing())
        request = service.GetKeyRingRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.KeyRing()
        client.get_key_ring(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_key_ring_rest_bad_request(transport: str='rest', request_type=service.GetKeyRingRequest):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_key_ring(request)

def test_get_key_ring_rest_flattened():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.KeyRing()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.KeyRing.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_key_ring(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*}' % client.transport._host, args[1])

def test_get_key_ring_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_key_ring(service.GetKeyRingRequest(), name='name_value')

def test_get_key_ring_rest_error():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetCryptoKeyRequest, dict])
def test_get_crypto_key_rest(request_type):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKey(name='name_value', purpose=resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT, import_only=True, crypto_key_backend='crypto_key_backend_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_crypto_key(request)
    assert isinstance(response, resources.CryptoKey)
    assert response.name == 'name_value'
    assert response.purpose == resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
    assert response.import_only is True
    assert response.crypto_key_backend == 'crypto_key_backend_value'

def test_get_crypto_key_rest_required_fields(request_type=service.GetCryptoKeyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_crypto_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_crypto_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.CryptoKey()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.CryptoKey.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_crypto_key(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_crypto_key_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_crypto_key._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_crypto_key_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_get_crypto_key') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_get_crypto_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetCryptoKeyRequest.pb(service.GetCryptoKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.CryptoKey.to_json(resources.CryptoKey())
        request = service.GetCryptoKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.CryptoKey()
        client.get_crypto_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_crypto_key_rest_bad_request(transport: str='rest', request_type=service.GetCryptoKeyRequest):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_crypto_key(request)

def test_get_crypto_key_rest_flattened():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKey()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_crypto_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*/cryptoKeys/*}' % client.transport._host, args[1])

def test_get_crypto_key_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_crypto_key(service.GetCryptoKeyRequest(), name='name_value')

def test_get_crypto_key_rest_error():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetCryptoKeyVersionRequest, dict])
def test_get_crypto_key_version_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKeyVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_crypto_key_version(request)
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

def test_get_crypto_key_version_rest_required_fields(request_type=service.GetCryptoKeyVersionRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_crypto_key_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_crypto_key_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.CryptoKeyVersion()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.CryptoKeyVersion.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_crypto_key_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_crypto_key_version_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_crypto_key_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_crypto_key_version_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_get_crypto_key_version') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_get_crypto_key_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetCryptoKeyVersionRequest.pb(service.GetCryptoKeyVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.CryptoKeyVersion.to_json(resources.CryptoKeyVersion())
        request = service.GetCryptoKeyVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.CryptoKeyVersion()
        client.get_crypto_key_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_crypto_key_version_rest_bad_request(transport: str='rest', request_type=service.GetCryptoKeyVersionRequest):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_crypto_key_version(request)

def test_get_crypto_key_version_rest_flattened():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKeyVersion()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKeyVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_crypto_key_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*/cryptoKeys/*/cryptoKeyVersions/*}' % client.transport._host, args[1])

def test_get_crypto_key_version_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_crypto_key_version(service.GetCryptoKeyVersionRequest(), name='name_value')

def test_get_crypto_key_version_rest_error():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetPublicKeyRequest, dict])
def test_get_public_key_rest(request_type):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.PublicKey(pem='pem_value', algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, name='name_value', protection_level=resources.ProtectionLevel.SOFTWARE)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.PublicKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_public_key(request)
    assert isinstance(response, resources.PublicKey)
    assert response.pem == 'pem_value'
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.name == 'name_value'
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_get_public_key_rest_required_fields(request_type=service.GetPublicKeyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_public_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_public_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.PublicKey()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.PublicKey.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_public_key(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_public_key_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_public_key._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_public_key_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_get_public_key') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_get_public_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetPublicKeyRequest.pb(service.GetPublicKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.PublicKey.to_json(resources.PublicKey())
        request = service.GetPublicKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.PublicKey()
        client.get_public_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_public_key_rest_bad_request(transport: str='rest', request_type=service.GetPublicKeyRequest):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_public_key(request)

def test_get_public_key_rest_flattened():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.PublicKey()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.PublicKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_public_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*/cryptoKeys/*/cryptoKeyVersions/*}/publicKey' % client.transport._host, args[1])

def test_get_public_key_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_public_key(service.GetPublicKeyRequest(), name='name_value')

def test_get_public_key_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetImportJobRequest, dict])
def test_get_import_job_rest(request_type):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/importJobs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.ImportJob(name='name_value', import_method=resources.ImportJob.ImportMethod.RSA_OAEP_3072_SHA1_AES_256, protection_level=resources.ProtectionLevel.SOFTWARE, state=resources.ImportJob.ImportJobState.PENDING_GENERATION)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.ImportJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_import_job(request)
    assert isinstance(response, resources.ImportJob)
    assert response.name == 'name_value'
    assert response.import_method == resources.ImportJob.ImportMethod.RSA_OAEP_3072_SHA1_AES_256
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.state == resources.ImportJob.ImportJobState.PENDING_GENERATION

def test_get_import_job_rest_required_fields(request_type=service.GetImportJobRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_import_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_import_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.ImportJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.ImportJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_import_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_import_job_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_import_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_import_job_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_get_import_job') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_get_import_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetImportJobRequest.pb(service.GetImportJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.ImportJob.to_json(resources.ImportJob())
        request = service.GetImportJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.ImportJob()
        client.get_import_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_import_job_rest_bad_request(transport: str='rest', request_type=service.GetImportJobRequest):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/importJobs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_import_job(request)

def test_get_import_job_rest_flattened():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.ImportJob()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/importJobs/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.ImportJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_import_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*/importJobs/*}' % client.transport._host, args[1])

def test_get_import_job_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_import_job(service.GetImportJobRequest(), name='name_value')

def test_get_import_job_rest_error():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateKeyRingRequest, dict])
def test_create_key_ring_rest(request_type):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['key_ring'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}}
    test_field = service.CreateKeyRingRequest.meta.fields['key_ring']

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
    for (field, value) in request_init['key_ring'].items():
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
                for i in range(0, len(request_init['key_ring'][field])):
                    del request_init['key_ring'][field][i][subfield]
            else:
                del request_init['key_ring'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.KeyRing(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.KeyRing.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_key_ring(request)
    assert isinstance(response, resources.KeyRing)
    assert response.name == 'name_value'

def test_create_key_ring_rest_required_fields(request_type=service.CreateKeyRingRequest):
    if False:
        return 10
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['key_ring_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'keyRingId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_key_ring._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'keyRingId' in jsonified_request
    assert jsonified_request['keyRingId'] == request_init['key_ring_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['keyRingId'] = 'key_ring_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_key_ring._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('key_ring_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'keyRingId' in jsonified_request
    assert jsonified_request['keyRingId'] == 'key_ring_id_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.KeyRing()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.KeyRing.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_key_ring(request)
            expected_params = [('keyRingId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_key_ring_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_key_ring._get_unset_required_fields({})
    assert set(unset_fields) == set(('keyRingId',)) & set(('parent', 'keyRingId', 'keyRing'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_key_ring_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_create_key_ring') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_create_key_ring') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateKeyRingRequest.pb(service.CreateKeyRingRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.KeyRing.to_json(resources.KeyRing())
        request = service.CreateKeyRingRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.KeyRing()
        client.create_key_ring(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_key_ring_rest_bad_request(transport: str='rest', request_type=service.CreateKeyRingRequest):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_key_ring(request)

def test_create_key_ring_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.KeyRing()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', key_ring_id='key_ring_id_value', key_ring=resources.KeyRing(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.KeyRing.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_key_ring(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/keyRings' % client.transport._host, args[1])

def test_create_key_ring_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_key_ring(service.CreateKeyRingRequest(), parent='parent_value', key_ring_id='key_ring_id_value', key_ring=resources.KeyRing(name='name_value'))

def test_create_key_ring_rest_error():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateCryptoKeyRequest, dict])
def test_create_crypto_key_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3'}
    request_init['crypto_key'] = {'name': 'name_value', 'primary': {'name': 'name_value', 'state': 5, 'protection_level': 1, 'algorithm': 1, 'attestation': {'format': 3, 'content': b'content_blob', 'cert_chains': {'cavium_certs': ['cavium_certs_value1', 'cavium_certs_value2'], 'google_card_certs': ['google_card_certs_value1', 'google_card_certs_value2'], 'google_partition_certs': ['google_partition_certs_value1', 'google_partition_certs_value2']}}, 'create_time': {'seconds': 751, 'nanos': 543}, 'generate_time': {}, 'destroy_time': {}, 'destroy_event_time': {}, 'import_job': 'import_job_value', 'import_time': {}, 'import_failure_reason': 'import_failure_reason_value', 'generation_failure_reason': 'generation_failure_reason_value', 'external_destruction_failure_reason': 'external_destruction_failure_reason_value', 'external_protection_level_options': {'external_key_uri': 'external_key_uri_value', 'ekm_connection_key_path': 'ekm_connection_key_path_value'}, 'reimport_eligible': True}, 'purpose': 1, 'create_time': {}, 'next_rotation_time': {}, 'rotation_period': {'seconds': 751, 'nanos': 543}, 'version_template': {'protection_level': 1, 'algorithm': 1}, 'labels': {}, 'import_only': True, 'destroy_scheduled_duration': {}, 'crypto_key_backend': 'crypto_key_backend_value'}
    test_field = service.CreateCryptoKeyRequest.meta.fields['crypto_key']

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
    for (field, value) in request_init['crypto_key'].items():
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
                for i in range(0, len(request_init['crypto_key'][field])):
                    del request_init['crypto_key'][field][i][subfield]
            else:
                del request_init['crypto_key'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKey(name='name_value', purpose=resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT, import_only=True, crypto_key_backend='crypto_key_backend_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_crypto_key(request)
    assert isinstance(response, resources.CryptoKey)
    assert response.name == 'name_value'
    assert response.purpose == resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
    assert response.import_only is True
    assert response.crypto_key_backend == 'crypto_key_backend_value'

def test_create_crypto_key_rest_required_fields(request_type=service.CreateCryptoKeyRequest):
    if False:
        return 10
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['crypto_key_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'cryptoKeyId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_crypto_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'cryptoKeyId' in jsonified_request
    assert jsonified_request['cryptoKeyId'] == request_init['crypto_key_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['cryptoKeyId'] = 'crypto_key_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_crypto_key._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('crypto_key_id', 'skip_initial_version_creation'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'cryptoKeyId' in jsonified_request
    assert jsonified_request['cryptoKeyId'] == 'crypto_key_id_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.CryptoKey()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.CryptoKey.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_crypto_key(request)
            expected_params = [('cryptoKeyId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_crypto_key_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_crypto_key._get_unset_required_fields({})
    assert set(unset_fields) == set(('cryptoKeyId', 'skipInitialVersionCreation')) & set(('parent', 'cryptoKeyId', 'cryptoKey'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_crypto_key_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_create_crypto_key') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_create_crypto_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateCryptoKeyRequest.pb(service.CreateCryptoKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.CryptoKey.to_json(resources.CryptoKey())
        request = service.CreateCryptoKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.CryptoKey()
        client.create_crypto_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_crypto_key_rest_bad_request(transport: str='rest', request_type=service.CreateCryptoKeyRequest):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_crypto_key(request)

def test_create_crypto_key_rest_flattened():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKey()
        sample_request = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3'}
        mock_args = dict(parent='parent_value', crypto_key_id='crypto_key_id_value', crypto_key=resources.CryptoKey(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_crypto_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/keyRings/*}/cryptoKeys' % client.transport._host, args[1])

def test_create_crypto_key_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_crypto_key(service.CreateCryptoKeyRequest(), parent='parent_value', crypto_key_id='crypto_key_id_value', crypto_key=resources.CryptoKey(name='name_value'))

def test_create_crypto_key_rest_error():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateCryptoKeyVersionRequest, dict])
def test_create_crypto_key_version_rest(request_type):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request_init['crypto_key_version'] = {'name': 'name_value', 'state': 5, 'protection_level': 1, 'algorithm': 1, 'attestation': {'format': 3, 'content': b'content_blob', 'cert_chains': {'cavium_certs': ['cavium_certs_value1', 'cavium_certs_value2'], 'google_card_certs': ['google_card_certs_value1', 'google_card_certs_value2'], 'google_partition_certs': ['google_partition_certs_value1', 'google_partition_certs_value2']}}, 'create_time': {'seconds': 751, 'nanos': 543}, 'generate_time': {}, 'destroy_time': {}, 'destroy_event_time': {}, 'import_job': 'import_job_value', 'import_time': {}, 'import_failure_reason': 'import_failure_reason_value', 'generation_failure_reason': 'generation_failure_reason_value', 'external_destruction_failure_reason': 'external_destruction_failure_reason_value', 'external_protection_level_options': {'external_key_uri': 'external_key_uri_value', 'ekm_connection_key_path': 'ekm_connection_key_path_value'}, 'reimport_eligible': True}
    test_field = service.CreateCryptoKeyVersionRequest.meta.fields['crypto_key_version']

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
    for (field, value) in request_init['crypto_key_version'].items():
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
                for i in range(0, len(request_init['crypto_key_version'][field])):
                    del request_init['crypto_key_version'][field][i][subfield]
            else:
                del request_init['crypto_key_version'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKeyVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_crypto_key_version(request)
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

def test_create_crypto_key_version_rest_required_fields(request_type=service.CreateCryptoKeyVersionRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_crypto_key_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_crypto_key_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.CryptoKeyVersion()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.CryptoKeyVersion.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_crypto_key_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_crypto_key_version_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_crypto_key_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'cryptoKeyVersion'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_crypto_key_version_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_create_crypto_key_version') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_create_crypto_key_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateCryptoKeyVersionRequest.pb(service.CreateCryptoKeyVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.CryptoKeyVersion.to_json(resources.CryptoKeyVersion())
        request = service.CreateCryptoKeyVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.CryptoKeyVersion()
        client.create_crypto_key_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_crypto_key_version_rest_bad_request(transport: str='rest', request_type=service.CreateCryptoKeyVersionRequest):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_crypto_key_version(request)

def test_create_crypto_key_version_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKeyVersion()
        sample_request = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
        mock_args = dict(parent='parent_value', crypto_key_version=resources.CryptoKeyVersion(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKeyVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_crypto_key_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/keyRings/*/cryptoKeys/*}/cryptoKeyVersions' % client.transport._host, args[1])

def test_create_crypto_key_version_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_crypto_key_version(service.CreateCryptoKeyVersionRequest(), parent='parent_value', crypto_key_version=resources.CryptoKeyVersion(name='name_value'))

def test_create_crypto_key_version_rest_error():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ImportCryptoKeyVersionRequest, dict])
def test_import_crypto_key_version_rest(request_type):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKeyVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.import_crypto_key_version(request)
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

def test_import_crypto_key_version_rest_required_fields(request_type=service.ImportCryptoKeyVersionRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['import_job'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).import_crypto_key_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['importJob'] = 'import_job_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).import_crypto_key_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'importJob' in jsonified_request
    assert jsonified_request['importJob'] == 'import_job_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.CryptoKeyVersion()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.CryptoKeyVersion.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.import_crypto_key_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_import_crypto_key_version_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.import_crypto_key_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'algorithm', 'importJob'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_import_crypto_key_version_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_import_crypto_key_version') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_import_crypto_key_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ImportCryptoKeyVersionRequest.pb(service.ImportCryptoKeyVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.CryptoKeyVersion.to_json(resources.CryptoKeyVersion())
        request = service.ImportCryptoKeyVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.CryptoKeyVersion()
        client.import_crypto_key_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_import_crypto_key_version_rest_bad_request(transport: str='rest', request_type=service.ImportCryptoKeyVersionRequest):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.import_crypto_key_version(request)

def test_import_crypto_key_version_rest_error():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateImportJobRequest, dict])
def test_create_import_job_rest(request_type):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3'}
    request_init['import_job'] = {'name': 'name_value', 'import_method': 1, 'protection_level': 1, 'create_time': {'seconds': 751, 'nanos': 543}, 'generate_time': {}, 'expire_time': {}, 'expire_event_time': {}, 'state': 1, 'public_key': {'pem': 'pem_value'}, 'attestation': {'format': 3, 'content': b'content_blob', 'cert_chains': {'cavium_certs': ['cavium_certs_value1', 'cavium_certs_value2'], 'google_card_certs': ['google_card_certs_value1', 'google_card_certs_value2'], 'google_partition_certs': ['google_partition_certs_value1', 'google_partition_certs_value2']}}}
    test_field = service.CreateImportJobRequest.meta.fields['import_job']

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
    for (field, value) in request_init['import_job'].items():
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
                for i in range(0, len(request_init['import_job'][field])):
                    del request_init['import_job'][field][i][subfield]
            else:
                del request_init['import_job'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.ImportJob(name='name_value', import_method=resources.ImportJob.ImportMethod.RSA_OAEP_3072_SHA1_AES_256, protection_level=resources.ProtectionLevel.SOFTWARE, state=resources.ImportJob.ImportJobState.PENDING_GENERATION)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.ImportJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_import_job(request)
    assert isinstance(response, resources.ImportJob)
    assert response.name == 'name_value'
    assert response.import_method == resources.ImportJob.ImportMethod.RSA_OAEP_3072_SHA1_AES_256
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.state == resources.ImportJob.ImportJobState.PENDING_GENERATION

def test_create_import_job_rest_required_fields(request_type=service.CreateImportJobRequest):
    if False:
        print('Hello World!')
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['import_job_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'importJobId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_import_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'importJobId' in jsonified_request
    assert jsonified_request['importJobId'] == request_init['import_job_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['importJobId'] = 'import_job_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_import_job._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('import_job_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'importJobId' in jsonified_request
    assert jsonified_request['importJobId'] == 'import_job_id_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.ImportJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.ImportJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_import_job(request)
            expected_params = [('importJobId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_import_job_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_import_job._get_unset_required_fields({})
    assert set(unset_fields) == set(('importJobId',)) & set(('parent', 'importJobId', 'importJob'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_import_job_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_create_import_job') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_create_import_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateImportJobRequest.pb(service.CreateImportJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.ImportJob.to_json(resources.ImportJob())
        request = service.CreateImportJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.ImportJob()
        client.create_import_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_import_job_rest_bad_request(transport: str='rest', request_type=service.CreateImportJobRequest):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_import_job(request)

def test_create_import_job_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.ImportJob()
        sample_request = {'parent': 'projects/sample1/locations/sample2/keyRings/sample3'}
        mock_args = dict(parent='parent_value', import_job_id='import_job_id_value', import_job=resources.ImportJob(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.ImportJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_import_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/keyRings/*}/importJobs' % client.transport._host, args[1])

def test_create_import_job_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_import_job(service.CreateImportJobRequest(), parent='parent_value', import_job_id='import_job_id_value', import_job=resources.ImportJob(name='name_value'))

def test_create_import_job_rest_error():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateCryptoKeyRequest, dict])
def test_update_crypto_key_rest(request_type):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'crypto_key': {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}}
    request_init['crypto_key'] = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4', 'primary': {'name': 'name_value', 'state': 5, 'protection_level': 1, 'algorithm': 1, 'attestation': {'format': 3, 'content': b'content_blob', 'cert_chains': {'cavium_certs': ['cavium_certs_value1', 'cavium_certs_value2'], 'google_card_certs': ['google_card_certs_value1', 'google_card_certs_value2'], 'google_partition_certs': ['google_partition_certs_value1', 'google_partition_certs_value2']}}, 'create_time': {'seconds': 751, 'nanos': 543}, 'generate_time': {}, 'destroy_time': {}, 'destroy_event_time': {}, 'import_job': 'import_job_value', 'import_time': {}, 'import_failure_reason': 'import_failure_reason_value', 'generation_failure_reason': 'generation_failure_reason_value', 'external_destruction_failure_reason': 'external_destruction_failure_reason_value', 'external_protection_level_options': {'external_key_uri': 'external_key_uri_value', 'ekm_connection_key_path': 'ekm_connection_key_path_value'}, 'reimport_eligible': True}, 'purpose': 1, 'create_time': {}, 'next_rotation_time': {}, 'rotation_period': {'seconds': 751, 'nanos': 543}, 'version_template': {'protection_level': 1, 'algorithm': 1}, 'labels': {}, 'import_only': True, 'destroy_scheduled_duration': {}, 'crypto_key_backend': 'crypto_key_backend_value'}
    test_field = service.UpdateCryptoKeyRequest.meta.fields['crypto_key']

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
    for (field, value) in request_init['crypto_key'].items():
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
                for i in range(0, len(request_init['crypto_key'][field])):
                    del request_init['crypto_key'][field][i][subfield]
            else:
                del request_init['crypto_key'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKey(name='name_value', purpose=resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT, import_only=True, crypto_key_backend='crypto_key_backend_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_crypto_key(request)
    assert isinstance(response, resources.CryptoKey)
    assert response.name == 'name_value'
    assert response.purpose == resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
    assert response.import_only is True
    assert response.crypto_key_backend == 'crypto_key_backend_value'

def test_update_crypto_key_rest_required_fields(request_type=service.UpdateCryptoKeyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_crypto_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_crypto_key._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.CryptoKey()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.CryptoKey.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_crypto_key(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_crypto_key_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_crypto_key._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('cryptoKey', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_crypto_key_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_update_crypto_key') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_update_crypto_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateCryptoKeyRequest.pb(service.UpdateCryptoKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.CryptoKey.to_json(resources.CryptoKey())
        request = service.UpdateCryptoKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.CryptoKey()
        client.update_crypto_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_crypto_key_rest_bad_request(transport: str='rest', request_type=service.UpdateCryptoKeyRequest):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'crypto_key': {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_crypto_key(request)

def test_update_crypto_key_rest_flattened():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKey()
        sample_request = {'crypto_key': {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}}
        mock_args = dict(crypto_key=resources.CryptoKey(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_crypto_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{crypto_key.name=projects/*/locations/*/keyRings/*/cryptoKeys/*}' % client.transport._host, args[1])

def test_update_crypto_key_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_crypto_key(service.UpdateCryptoKeyRequest(), crypto_key=resources.CryptoKey(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_crypto_key_rest_error():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateCryptoKeyVersionRequest, dict])
def test_update_crypto_key_version_rest(request_type):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'crypto_key_version': {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}}
    request_init['crypto_key_version'] = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5', 'state': 5, 'protection_level': 1, 'algorithm': 1, 'attestation': {'format': 3, 'content': b'content_blob', 'cert_chains': {'cavium_certs': ['cavium_certs_value1', 'cavium_certs_value2'], 'google_card_certs': ['google_card_certs_value1', 'google_card_certs_value2'], 'google_partition_certs': ['google_partition_certs_value1', 'google_partition_certs_value2']}}, 'create_time': {'seconds': 751, 'nanos': 543}, 'generate_time': {}, 'destroy_time': {}, 'destroy_event_time': {}, 'import_job': 'import_job_value', 'import_time': {}, 'import_failure_reason': 'import_failure_reason_value', 'generation_failure_reason': 'generation_failure_reason_value', 'external_destruction_failure_reason': 'external_destruction_failure_reason_value', 'external_protection_level_options': {'external_key_uri': 'external_key_uri_value', 'ekm_connection_key_path': 'ekm_connection_key_path_value'}, 'reimport_eligible': True}
    test_field = service.UpdateCryptoKeyVersionRequest.meta.fields['crypto_key_version']

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
    for (field, value) in request_init['crypto_key_version'].items():
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
                for i in range(0, len(request_init['crypto_key_version'][field])):
                    del request_init['crypto_key_version'][field][i][subfield]
            else:
                del request_init['crypto_key_version'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKeyVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_crypto_key_version(request)
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

def test_update_crypto_key_version_rest_required_fields(request_type=service.UpdateCryptoKeyVersionRequest):
    if False:
        return 10
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_crypto_key_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_crypto_key_version._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.CryptoKeyVersion()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.CryptoKeyVersion.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_crypto_key_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_crypto_key_version_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_crypto_key_version._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('cryptoKeyVersion', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_crypto_key_version_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_update_crypto_key_version') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_update_crypto_key_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateCryptoKeyVersionRequest.pb(service.UpdateCryptoKeyVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.CryptoKeyVersion.to_json(resources.CryptoKeyVersion())
        request = service.UpdateCryptoKeyVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.CryptoKeyVersion()
        client.update_crypto_key_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_crypto_key_version_rest_bad_request(transport: str='rest', request_type=service.UpdateCryptoKeyVersionRequest):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'crypto_key_version': {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_crypto_key_version(request)

def test_update_crypto_key_version_rest_flattened():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKeyVersion()
        sample_request = {'crypto_key_version': {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}}
        mock_args = dict(crypto_key_version=resources.CryptoKeyVersion(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKeyVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_crypto_key_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{crypto_key_version.name=projects/*/locations/*/keyRings/*/cryptoKeys/*/cryptoKeyVersions/*}' % client.transport._host, args[1])

def test_update_crypto_key_version_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_crypto_key_version(service.UpdateCryptoKeyVersionRequest(), crypto_key_version=resources.CryptoKeyVersion(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_crypto_key_version_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateCryptoKeyPrimaryVersionRequest, dict])
def test_update_crypto_key_primary_version_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKey(name='name_value', purpose=resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT, import_only=True, crypto_key_backend='crypto_key_backend_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_crypto_key_primary_version(request)
    assert isinstance(response, resources.CryptoKey)
    assert response.name == 'name_value'
    assert response.purpose == resources.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
    assert response.import_only is True
    assert response.crypto_key_backend == 'crypto_key_backend_value'

def test_update_crypto_key_primary_version_rest_required_fields(request_type=service.UpdateCryptoKeyPrimaryVersionRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['crypto_key_version_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_crypto_key_primary_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['cryptoKeyVersionId'] = 'crypto_key_version_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_crypto_key_primary_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'cryptoKeyVersionId' in jsonified_request
    assert jsonified_request['cryptoKeyVersionId'] == 'crypto_key_version_id_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.CryptoKey()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.CryptoKey.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_crypto_key_primary_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_crypto_key_primary_version_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_crypto_key_primary_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'cryptoKeyVersionId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_crypto_key_primary_version_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_update_crypto_key_primary_version') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_update_crypto_key_primary_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateCryptoKeyPrimaryVersionRequest.pb(service.UpdateCryptoKeyPrimaryVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.CryptoKey.to_json(resources.CryptoKey())
        request = service.UpdateCryptoKeyPrimaryVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.CryptoKey()
        client.update_crypto_key_primary_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_crypto_key_primary_version_rest_bad_request(transport: str='rest', request_type=service.UpdateCryptoKeyPrimaryVersionRequest):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_crypto_key_primary_version(request)

def test_update_crypto_key_primary_version_rest_flattened():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKey()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
        mock_args = dict(name='name_value', crypto_key_version_id='crypto_key_version_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_crypto_key_primary_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*/cryptoKeys/*}:updatePrimaryVersion' % client.transport._host, args[1])

def test_update_crypto_key_primary_version_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_crypto_key_primary_version(service.UpdateCryptoKeyPrimaryVersionRequest(), name='name_value', crypto_key_version_id='crypto_key_version_id_value')

def test_update_crypto_key_primary_version_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DestroyCryptoKeyVersionRequest, dict])
def test_destroy_crypto_key_version_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKeyVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.destroy_crypto_key_version(request)
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

def test_destroy_crypto_key_version_rest_required_fields(request_type=service.DestroyCryptoKeyVersionRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).destroy_crypto_key_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).destroy_crypto_key_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.CryptoKeyVersion()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.CryptoKeyVersion.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.destroy_crypto_key_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_destroy_crypto_key_version_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.destroy_crypto_key_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_destroy_crypto_key_version_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_destroy_crypto_key_version') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_destroy_crypto_key_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DestroyCryptoKeyVersionRequest.pb(service.DestroyCryptoKeyVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.CryptoKeyVersion.to_json(resources.CryptoKeyVersion())
        request = service.DestroyCryptoKeyVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.CryptoKeyVersion()
        client.destroy_crypto_key_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_destroy_crypto_key_version_rest_bad_request(transport: str='rest', request_type=service.DestroyCryptoKeyVersionRequest):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.destroy_crypto_key_version(request)

def test_destroy_crypto_key_version_rest_flattened():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKeyVersion()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKeyVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.destroy_crypto_key_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*/cryptoKeys/*/cryptoKeyVersions/*}:destroy' % client.transport._host, args[1])

def test_destroy_crypto_key_version_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.destroy_crypto_key_version(service.DestroyCryptoKeyVersionRequest(), name='name_value')

def test_destroy_crypto_key_version_rest_error():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.RestoreCryptoKeyVersionRequest, dict])
def test_restore_crypto_key_version_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKeyVersion(name='name_value', state=resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION, protection_level=resources.ProtectionLevel.SOFTWARE, algorithm=resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION, import_job='import_job_value', import_failure_reason='import_failure_reason_value', generation_failure_reason='generation_failure_reason_value', external_destruction_failure_reason='external_destruction_failure_reason_value', reimport_eligible=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKeyVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.restore_crypto_key_version(request)
    assert isinstance(response, resources.CryptoKeyVersion)
    assert response.name == 'name_value'
    assert response.state == resources.CryptoKeyVersion.CryptoKeyVersionState.PENDING_GENERATION
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.algorithm == resources.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    assert response.import_job == 'import_job_value'
    assert response.import_failure_reason == 'import_failure_reason_value'
    assert response.generation_failure_reason == 'generation_failure_reason_value'
    assert response.external_destruction_failure_reason == 'external_destruction_failure_reason_value'
    assert response.reimport_eligible is True

def test_restore_crypto_key_version_rest_required_fields(request_type=service.RestoreCryptoKeyVersionRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).restore_crypto_key_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).restore_crypto_key_version._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.CryptoKeyVersion()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.CryptoKeyVersion.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.restore_crypto_key_version(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_restore_crypto_key_version_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.restore_crypto_key_version._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_restore_crypto_key_version_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_restore_crypto_key_version') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_restore_crypto_key_version') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.RestoreCryptoKeyVersionRequest.pb(service.RestoreCryptoKeyVersionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.CryptoKeyVersion.to_json(resources.CryptoKeyVersion())
        request = service.RestoreCryptoKeyVersionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.CryptoKeyVersion()
        client.restore_crypto_key_version(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_restore_crypto_key_version_rest_bad_request(transport: str='rest', request_type=service.RestoreCryptoKeyVersionRequest):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.restore_crypto_key_version(request)

def test_restore_crypto_key_version_rest_flattened():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CryptoKeyVersion()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CryptoKeyVersion.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.restore_crypto_key_version(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*/cryptoKeys/*/cryptoKeyVersions/*}:restore' % client.transport._host, args[1])

def test_restore_crypto_key_version_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.restore_crypto_key_version(service.RestoreCryptoKeyVersionRequest(), name='name_value')

def test_restore_crypto_key_version_rest_error():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.EncryptRequest, dict])
def test_encrypt_rest(request_type):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.EncryptResponse(name='name_value', ciphertext=b'ciphertext_blob', verified_plaintext_crc32c=True, verified_additional_authenticated_data_crc32c=True, protection_level=resources.ProtectionLevel.SOFTWARE)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.EncryptResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.encrypt(request)
    assert isinstance(response, service.EncryptResponse)
    assert response.name == 'name_value'
    assert response.ciphertext == b'ciphertext_blob'
    assert response.verified_plaintext_crc32c is True
    assert response.verified_additional_authenticated_data_crc32c is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_encrypt_rest_required_fields(request_type=service.EncryptRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['plaintext'] = b''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).encrypt._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['plaintext'] = b'plaintext_blob'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).encrypt._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'plaintext' in jsonified_request
    assert jsonified_request['plaintext'] == b'plaintext_blob'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.EncryptResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.EncryptResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.encrypt(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_encrypt_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.encrypt._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'plaintext'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_encrypt_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_encrypt') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_encrypt') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.EncryptRequest.pb(service.EncryptRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.EncryptResponse.to_json(service.EncryptResponse())
        request = service.EncryptRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.EncryptResponse()
        client.encrypt(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_encrypt_rest_bad_request(transport: str='rest', request_type=service.EncryptRequest):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.encrypt(request)

def test_encrypt_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.EncryptResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
        mock_args = dict(name='name_value', plaintext=b'plaintext_blob')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.EncryptResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.encrypt(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*/cryptoKeys/**}:encrypt' % client.transport._host, args[1])

def test_encrypt_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.encrypt(service.EncryptRequest(), name='name_value', plaintext=b'plaintext_blob')

def test_encrypt_rest_error():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DecryptRequest, dict])
def test_decrypt_rest(request_type):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.DecryptResponse(plaintext=b'plaintext_blob', used_primary=True, protection_level=resources.ProtectionLevel.SOFTWARE)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.DecryptResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.decrypt(request)
    assert isinstance(response, service.DecryptResponse)
    assert response.plaintext == b'plaintext_blob'
    assert response.used_primary is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_decrypt_rest_required_fields(request_type=service.DecryptRequest):
    if False:
        return 10
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['ciphertext'] = b''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).decrypt._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['ciphertext'] = b'ciphertext_blob'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).decrypt._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'ciphertext' in jsonified_request
    assert jsonified_request['ciphertext'] == b'ciphertext_blob'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.DecryptResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.DecryptResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.decrypt(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_decrypt_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.decrypt._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'ciphertext'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_decrypt_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_decrypt') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_decrypt') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DecryptRequest.pb(service.DecryptRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.DecryptResponse.to_json(service.DecryptResponse())
        request = service.DecryptRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.DecryptResponse()
        client.decrypt(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_decrypt_rest_bad_request(transport: str='rest', request_type=service.DecryptRequest):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.decrypt(request)

def test_decrypt_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.DecryptResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
        mock_args = dict(name='name_value', ciphertext=b'ciphertext_blob')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.DecryptResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.decrypt(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*/cryptoKeys/*}:decrypt' % client.transport._host, args[1])

def test_decrypt_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.decrypt(service.DecryptRequest(), name='name_value', ciphertext=b'ciphertext_blob')

def test_decrypt_rest_error():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.RawEncryptRequest, dict])
def test_raw_encrypt_rest(request_type):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.RawEncryptResponse(ciphertext=b'ciphertext_blob', initialization_vector=b'initialization_vector_blob', tag_length=1053, verified_plaintext_crc32c=True, verified_additional_authenticated_data_crc32c=True, verified_initialization_vector_crc32c=True, name='name_value', protection_level=resources.ProtectionLevel.SOFTWARE)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.RawEncryptResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.raw_encrypt(request)
    assert isinstance(response, service.RawEncryptResponse)
    assert response.ciphertext == b'ciphertext_blob'
    assert response.initialization_vector == b'initialization_vector_blob'
    assert response.tag_length == 1053
    assert response.verified_plaintext_crc32c is True
    assert response.verified_additional_authenticated_data_crc32c is True
    assert response.verified_initialization_vector_crc32c is True
    assert response.name == 'name_value'
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_raw_encrypt_rest_required_fields(request_type=service.RawEncryptRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['plaintext'] = b''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).raw_encrypt._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['plaintext'] = b'plaintext_blob'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).raw_encrypt._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'plaintext' in jsonified_request
    assert jsonified_request['plaintext'] == b'plaintext_blob'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.RawEncryptResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.RawEncryptResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.raw_encrypt(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_raw_encrypt_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.raw_encrypt._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'plaintext'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_raw_encrypt_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_raw_encrypt') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_raw_encrypt') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.RawEncryptRequest.pb(service.RawEncryptRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.RawEncryptResponse.to_json(service.RawEncryptResponse())
        request = service.RawEncryptRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.RawEncryptResponse()
        client.raw_encrypt(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_raw_encrypt_rest_bad_request(transport: str='rest', request_type=service.RawEncryptRequest):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.raw_encrypt(request)

def test_raw_encrypt_rest_error():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.RawDecryptRequest, dict])
def test_raw_decrypt_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.RawDecryptResponse(plaintext=b'plaintext_blob', protection_level=resources.ProtectionLevel.SOFTWARE, verified_ciphertext_crc32c=True, verified_additional_authenticated_data_crc32c=True, verified_initialization_vector_crc32c=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.RawDecryptResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.raw_decrypt(request)
    assert isinstance(response, service.RawDecryptResponse)
    assert response.plaintext == b'plaintext_blob'
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE
    assert response.verified_ciphertext_crc32c is True
    assert response.verified_additional_authenticated_data_crc32c is True
    assert response.verified_initialization_vector_crc32c is True

def test_raw_decrypt_rest_required_fields(request_type=service.RawDecryptRequest):
    if False:
        return 10
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['ciphertext'] = b''
    request_init['initialization_vector'] = b''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).raw_decrypt._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['ciphertext'] = b'ciphertext_blob'
    jsonified_request['initializationVector'] = b'initialization_vector_blob'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).raw_decrypt._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'ciphertext' in jsonified_request
    assert jsonified_request['ciphertext'] == b'ciphertext_blob'
    assert 'initializationVector' in jsonified_request
    assert jsonified_request['initializationVector'] == b'initialization_vector_blob'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.RawDecryptResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.RawDecryptResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.raw_decrypt(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_raw_decrypt_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.raw_decrypt._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'ciphertext', 'initializationVector'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_raw_decrypt_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_raw_decrypt') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_raw_decrypt') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.RawDecryptRequest.pb(service.RawDecryptRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.RawDecryptResponse.to_json(service.RawDecryptResponse())
        request = service.RawDecryptRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.RawDecryptResponse()
        client.raw_decrypt(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_raw_decrypt_rest_bad_request(transport: str='rest', request_type=service.RawDecryptRequest):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.raw_decrypt(request)

def test_raw_decrypt_rest_error():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.AsymmetricSignRequest, dict])
def test_asymmetric_sign_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.AsymmetricSignResponse(signature=b'signature_blob', verified_digest_crc32c=True, name='name_value', verified_data_crc32c=True, protection_level=resources.ProtectionLevel.SOFTWARE)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.AsymmetricSignResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.asymmetric_sign(request)
    assert isinstance(response, service.AsymmetricSignResponse)
    assert response.signature == b'signature_blob'
    assert response.verified_digest_crc32c is True
    assert response.name == 'name_value'
    assert response.verified_data_crc32c is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_asymmetric_sign_rest_required_fields(request_type=service.AsymmetricSignRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).asymmetric_sign._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).asymmetric_sign._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.AsymmetricSignResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.AsymmetricSignResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.asymmetric_sign(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_asymmetric_sign_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.asymmetric_sign._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_asymmetric_sign_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_asymmetric_sign') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_asymmetric_sign') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.AsymmetricSignRequest.pb(service.AsymmetricSignRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.AsymmetricSignResponse.to_json(service.AsymmetricSignResponse())
        request = service.AsymmetricSignRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.AsymmetricSignResponse()
        client.asymmetric_sign(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_asymmetric_sign_rest_bad_request(transport: str='rest', request_type=service.AsymmetricSignRequest):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.asymmetric_sign(request)

def test_asymmetric_sign_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.AsymmetricSignResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
        mock_args = dict(name='name_value', digest=service.Digest(sha256=b'sha256_blob'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.AsymmetricSignResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.asymmetric_sign(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*/cryptoKeys/*/cryptoKeyVersions/*}:asymmetricSign' % client.transport._host, args[1])

def test_asymmetric_sign_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.asymmetric_sign(service.AsymmetricSignRequest(), name='name_value', digest=service.Digest(sha256=b'sha256_blob'))

def test_asymmetric_sign_rest_error():
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.AsymmetricDecryptRequest, dict])
def test_asymmetric_decrypt_rest(request_type):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.AsymmetricDecryptResponse(plaintext=b'plaintext_blob', verified_ciphertext_crc32c=True, protection_level=resources.ProtectionLevel.SOFTWARE)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.AsymmetricDecryptResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.asymmetric_decrypt(request)
    assert isinstance(response, service.AsymmetricDecryptResponse)
    assert response.plaintext == b'plaintext_blob'
    assert response.verified_ciphertext_crc32c is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_asymmetric_decrypt_rest_required_fields(request_type=service.AsymmetricDecryptRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['ciphertext'] = b''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).asymmetric_decrypt._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['ciphertext'] = b'ciphertext_blob'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).asymmetric_decrypt._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'ciphertext' in jsonified_request
    assert jsonified_request['ciphertext'] == b'ciphertext_blob'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.AsymmetricDecryptResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.AsymmetricDecryptResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.asymmetric_decrypt(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_asymmetric_decrypt_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.asymmetric_decrypt._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'ciphertext'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_asymmetric_decrypt_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_asymmetric_decrypt') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_asymmetric_decrypt') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.AsymmetricDecryptRequest.pb(service.AsymmetricDecryptRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.AsymmetricDecryptResponse.to_json(service.AsymmetricDecryptResponse())
        request = service.AsymmetricDecryptRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.AsymmetricDecryptResponse()
        client.asymmetric_decrypt(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_asymmetric_decrypt_rest_bad_request(transport: str='rest', request_type=service.AsymmetricDecryptRequest):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.asymmetric_decrypt(request)

def test_asymmetric_decrypt_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.AsymmetricDecryptResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
        mock_args = dict(name='name_value', ciphertext=b'ciphertext_blob')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.AsymmetricDecryptResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.asymmetric_decrypt(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*/cryptoKeys/*/cryptoKeyVersions/*}:asymmetricDecrypt' % client.transport._host, args[1])

def test_asymmetric_decrypt_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.asymmetric_decrypt(service.AsymmetricDecryptRequest(), name='name_value', ciphertext=b'ciphertext_blob')

def test_asymmetric_decrypt_rest_error():
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.MacSignRequest, dict])
def test_mac_sign_rest(request_type):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.MacSignResponse(name='name_value', mac=b'mac_blob', verified_data_crc32c=True, protection_level=resources.ProtectionLevel.SOFTWARE)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.MacSignResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.mac_sign(request)
    assert isinstance(response, service.MacSignResponse)
    assert response.name == 'name_value'
    assert response.mac == b'mac_blob'
    assert response.verified_data_crc32c is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_mac_sign_rest_required_fields(request_type=service.MacSignRequest):
    if False:
        print('Hello World!')
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['data'] = b''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).mac_sign._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['data'] = b'data_blob'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).mac_sign._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'data' in jsonified_request
    assert jsonified_request['data'] == b'data_blob'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.MacSignResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.MacSignResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.mac_sign(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_mac_sign_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.mac_sign._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'data'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_mac_sign_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_mac_sign') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_mac_sign') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.MacSignRequest.pb(service.MacSignRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.MacSignResponse.to_json(service.MacSignResponse())
        request = service.MacSignRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.MacSignResponse()
        client.mac_sign(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_mac_sign_rest_bad_request(transport: str='rest', request_type=service.MacSignRequest):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.mac_sign(request)

def test_mac_sign_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.MacSignResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
        mock_args = dict(name='name_value', data=b'data_blob')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.MacSignResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.mac_sign(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*/cryptoKeys/*/cryptoKeyVersions/*}:macSign' % client.transport._host, args[1])

def test_mac_sign_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.mac_sign(service.MacSignRequest(), name='name_value', data=b'data_blob')

def test_mac_sign_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.MacVerifyRequest, dict])
def test_mac_verify_rest(request_type):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.MacVerifyResponse(name='name_value', success=True, verified_data_crc32c=True, verified_mac_crc32c=True, verified_success_integrity=True, protection_level=resources.ProtectionLevel.SOFTWARE)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.MacVerifyResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.mac_verify(request)
    assert isinstance(response, service.MacVerifyResponse)
    assert response.name == 'name_value'
    assert response.success is True
    assert response.verified_data_crc32c is True
    assert response.verified_mac_crc32c is True
    assert response.verified_success_integrity is True
    assert response.protection_level == resources.ProtectionLevel.SOFTWARE

def test_mac_verify_rest_required_fields(request_type=service.MacVerifyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.KeyManagementServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['data'] = b''
    request_init['mac'] = b''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).mac_verify._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['data'] = b'data_blob'
    jsonified_request['mac'] = b'mac_blob'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).mac_verify._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'data' in jsonified_request
    assert jsonified_request['data'] == b'data_blob'
    assert 'mac' in jsonified_request
    assert jsonified_request['mac'] == b'mac_blob'
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.MacVerifyResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.MacVerifyResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.mac_verify(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_mac_verify_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.mac_verify._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'data', 'mac'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_mac_verify_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_mac_verify') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_mac_verify') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.MacVerifyRequest.pb(service.MacVerifyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.MacVerifyResponse.to_json(service.MacVerifyResponse())
        request = service.MacVerifyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.MacVerifyResponse()
        client.mac_verify(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_mac_verify_rest_bad_request(transport: str='rest', request_type=service.MacVerifyRequest):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.mac_verify(request)

def test_mac_verify_rest_flattened():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.MacVerifyResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4/cryptoKeyVersions/sample5'}
        mock_args = dict(name='name_value', data=b'data_blob', mac=b'mac_blob')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.MacVerifyResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.mac_verify(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*/cryptoKeys/*/cryptoKeyVersions/*}:macVerify' % client.transport._host, args[1])

def test_mac_verify_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.mac_verify(service.MacVerifyRequest(), name='name_value', data=b'data_blob', mac=b'mac_blob')

def test_mac_verify_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GenerateRandomBytesRequest, dict])
def test_generate_random_bytes_rest(request_type):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'location': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.GenerateRandomBytesResponse(data=b'data_blob')
        response_value = Response()
        response_value.status_code = 200
        return_value = service.GenerateRandomBytesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.generate_random_bytes(request)
    assert isinstance(response, service.GenerateRandomBytesResponse)
    assert response.data == b'data_blob'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_generate_random_bytes_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.KeyManagementServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyManagementServiceRestInterceptor())
    client = KeyManagementServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'post_generate_random_bytes') as post, mock.patch.object(transports.KeyManagementServiceRestInterceptor, 'pre_generate_random_bytes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GenerateRandomBytesRequest.pb(service.GenerateRandomBytesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.GenerateRandomBytesResponse.to_json(service.GenerateRandomBytesResponse())
        request = service.GenerateRandomBytesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.GenerateRandomBytesResponse()
        client.generate_random_bytes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_generate_random_bytes_rest_bad_request(transport: str='rest', request_type=service.GenerateRandomBytesRequest):
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'location': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.generate_random_bytes(request)

def test_generate_random_bytes_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.GenerateRandomBytesResponse()
        sample_request = {'location': 'projects/sample1/locations/sample2'}
        mock_args = dict(location='location_value', length_bytes=1288, protection_level=resources.ProtectionLevel.SOFTWARE)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.GenerateRandomBytesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.generate_random_bytes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{location=projects/*/locations/*}:generateRandomBytes' % client.transport._host, args[1])

def test_generate_random_bytes_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.generate_random_bytes(service.GenerateRandomBytesRequest(), location='location_value', length_bytes=1288, protection_level=resources.ProtectionLevel.SOFTWARE)

def test_generate_random_bytes_rest_error():
    if False:
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.KeyManagementServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.KeyManagementServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = KeyManagementServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.KeyManagementServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = KeyManagementServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = KeyManagementServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.KeyManagementServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = KeyManagementServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.KeyManagementServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = KeyManagementServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.KeyManagementServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.KeyManagementServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.KeyManagementServiceGrpcTransport, transports.KeyManagementServiceGrpcAsyncIOTransport, transports.KeyManagementServiceRestTransport])
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
    transport = KeyManagementServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.KeyManagementServiceGrpcTransport)

def test_key_management_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.KeyManagementServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_key_management_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.kms_v1.services.key_management_service.transports.KeyManagementServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.KeyManagementServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_key_rings', 'list_crypto_keys', 'list_crypto_key_versions', 'list_import_jobs', 'get_key_ring', 'get_crypto_key', 'get_crypto_key_version', 'get_public_key', 'get_import_job', 'create_key_ring', 'create_crypto_key', 'create_crypto_key_version', 'import_crypto_key_version', 'create_import_job', 'update_crypto_key', 'update_crypto_key_version', 'update_crypto_key_primary_version', 'destroy_crypto_key_version', 'restore_crypto_key_version', 'encrypt', 'decrypt', 'raw_encrypt', 'raw_decrypt', 'asymmetric_sign', 'asymmetric_decrypt', 'mac_sign', 'mac_verify', 'generate_random_bytes', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_location', 'list_locations')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_key_management_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.kms_v1.services.key_management_service.transports.KeyManagementServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.KeyManagementServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloudkms'), quota_project_id='octopus')

def test_key_management_service_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.kms_v1.services.key_management_service.transports.KeyManagementServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.KeyManagementServiceTransport()
        adc.assert_called_once()

def test_key_management_service_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        KeyManagementServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloudkms'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.KeyManagementServiceGrpcTransport, transports.KeyManagementServiceGrpcAsyncIOTransport])
def test_key_management_service_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloudkms'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.KeyManagementServiceGrpcTransport, transports.KeyManagementServiceGrpcAsyncIOTransport, transports.KeyManagementServiceRestTransport])
def test_key_management_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.KeyManagementServiceGrpcTransport, grpc_helpers), (transports.KeyManagementServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_key_management_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudkms.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloudkms'), scopes=['1', '2'], default_host='cloudkms.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.KeyManagementServiceGrpcTransport, transports.KeyManagementServiceGrpcAsyncIOTransport])
def test_key_management_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_key_management_service_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.KeyManagementServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_key_management_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudkms.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudkms.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudkms.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_key_management_service_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudkms.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudkms.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudkms.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_key_management_service_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = KeyManagementServiceClient(credentials=creds1, transport=transport_name)
    client2 = KeyManagementServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_key_rings._session
    session2 = client2.transport.list_key_rings._session
    assert session1 != session2
    session1 = client1.transport.list_crypto_keys._session
    session2 = client2.transport.list_crypto_keys._session
    assert session1 != session2
    session1 = client1.transport.list_crypto_key_versions._session
    session2 = client2.transport.list_crypto_key_versions._session
    assert session1 != session2
    session1 = client1.transport.list_import_jobs._session
    session2 = client2.transport.list_import_jobs._session
    assert session1 != session2
    session1 = client1.transport.get_key_ring._session
    session2 = client2.transport.get_key_ring._session
    assert session1 != session2
    session1 = client1.transport.get_crypto_key._session
    session2 = client2.transport.get_crypto_key._session
    assert session1 != session2
    session1 = client1.transport.get_crypto_key_version._session
    session2 = client2.transport.get_crypto_key_version._session
    assert session1 != session2
    session1 = client1.transport.get_public_key._session
    session2 = client2.transport.get_public_key._session
    assert session1 != session2
    session1 = client1.transport.get_import_job._session
    session2 = client2.transport.get_import_job._session
    assert session1 != session2
    session1 = client1.transport.create_key_ring._session
    session2 = client2.transport.create_key_ring._session
    assert session1 != session2
    session1 = client1.transport.create_crypto_key._session
    session2 = client2.transport.create_crypto_key._session
    assert session1 != session2
    session1 = client1.transport.create_crypto_key_version._session
    session2 = client2.transport.create_crypto_key_version._session
    assert session1 != session2
    session1 = client1.transport.import_crypto_key_version._session
    session2 = client2.transport.import_crypto_key_version._session
    assert session1 != session2
    session1 = client1.transport.create_import_job._session
    session2 = client2.transport.create_import_job._session
    assert session1 != session2
    session1 = client1.transport.update_crypto_key._session
    session2 = client2.transport.update_crypto_key._session
    assert session1 != session2
    session1 = client1.transport.update_crypto_key_version._session
    session2 = client2.transport.update_crypto_key_version._session
    assert session1 != session2
    session1 = client1.transport.update_crypto_key_primary_version._session
    session2 = client2.transport.update_crypto_key_primary_version._session
    assert session1 != session2
    session1 = client1.transport.destroy_crypto_key_version._session
    session2 = client2.transport.destroy_crypto_key_version._session
    assert session1 != session2
    session1 = client1.transport.restore_crypto_key_version._session
    session2 = client2.transport.restore_crypto_key_version._session
    assert session1 != session2
    session1 = client1.transport.encrypt._session
    session2 = client2.transport.encrypt._session
    assert session1 != session2
    session1 = client1.transport.decrypt._session
    session2 = client2.transport.decrypt._session
    assert session1 != session2
    session1 = client1.transport.raw_encrypt._session
    session2 = client2.transport.raw_encrypt._session
    assert session1 != session2
    session1 = client1.transport.raw_decrypt._session
    session2 = client2.transport.raw_decrypt._session
    assert session1 != session2
    session1 = client1.transport.asymmetric_sign._session
    session2 = client2.transport.asymmetric_sign._session
    assert session1 != session2
    session1 = client1.transport.asymmetric_decrypt._session
    session2 = client2.transport.asymmetric_decrypt._session
    assert session1 != session2
    session1 = client1.transport.mac_sign._session
    session2 = client2.transport.mac_sign._session
    assert session1 != session2
    session1 = client1.transport.mac_verify._session
    session2 = client2.transport.mac_verify._session
    assert session1 != session2
    session1 = client1.transport.generate_random_bytes._session
    session2 = client2.transport.generate_random_bytes._session
    assert session1 != session2

def test_key_management_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.KeyManagementServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_key_management_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.KeyManagementServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.KeyManagementServiceGrpcTransport, transports.KeyManagementServiceGrpcAsyncIOTransport])
def test_key_management_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.KeyManagementServiceGrpcTransport, transports.KeyManagementServiceGrpcAsyncIOTransport])
def test_key_management_service_transport_channel_mtls_with_adc(transport_class):
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

def test_crypto_key_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    key_ring = 'whelk'
    crypto_key = 'octopus'
    expected = 'projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}'.format(project=project, location=location, key_ring=key_ring, crypto_key=crypto_key)
    actual = KeyManagementServiceClient.crypto_key_path(project, location, key_ring, crypto_key)
    assert expected == actual

def test_parse_crypto_key_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'oyster', 'location': 'nudibranch', 'key_ring': 'cuttlefish', 'crypto_key': 'mussel'}
    path = KeyManagementServiceClient.crypto_key_path(**expected)
    actual = KeyManagementServiceClient.parse_crypto_key_path(path)
    assert expected == actual

def test_crypto_key_version_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    key_ring = 'scallop'
    crypto_key = 'abalone'
    crypto_key_version = 'squid'
    expected = 'projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}/cryptoKeyVersions/{crypto_key_version}'.format(project=project, location=location, key_ring=key_ring, crypto_key=crypto_key, crypto_key_version=crypto_key_version)
    actual = KeyManagementServiceClient.crypto_key_version_path(project, location, key_ring, crypto_key, crypto_key_version)
    assert expected == actual

def test_parse_crypto_key_version_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'clam', 'location': 'whelk', 'key_ring': 'octopus', 'crypto_key': 'oyster', 'crypto_key_version': 'nudibranch'}
    path = KeyManagementServiceClient.crypto_key_version_path(**expected)
    actual = KeyManagementServiceClient.parse_crypto_key_version_path(path)
    assert expected == actual

def test_import_job_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    location = 'mussel'
    key_ring = 'winkle'
    import_job = 'nautilus'
    expected = 'projects/{project}/locations/{location}/keyRings/{key_ring}/importJobs/{import_job}'.format(project=project, location=location, key_ring=key_ring, import_job=import_job)
    actual = KeyManagementServiceClient.import_job_path(project, location, key_ring, import_job)
    assert expected == actual

def test_parse_import_job_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'scallop', 'location': 'abalone', 'key_ring': 'squid', 'import_job': 'clam'}
    path = KeyManagementServiceClient.import_job_path(**expected)
    actual = KeyManagementServiceClient.parse_import_job_path(path)
    assert expected == actual

def test_key_ring_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    key_ring = 'oyster'
    expected = 'projects/{project}/locations/{location}/keyRings/{key_ring}'.format(project=project, location=location, key_ring=key_ring)
    actual = KeyManagementServiceClient.key_ring_path(project, location, key_ring)
    assert expected == actual

def test_parse_key_ring_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'key_ring': 'mussel'}
    path = KeyManagementServiceClient.key_ring_path(**expected)
    actual = KeyManagementServiceClient.parse_key_ring_path(path)
    assert expected == actual

def test_public_key_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    location = 'nautilus'
    key_ring = 'scallop'
    crypto_key = 'abalone'
    crypto_key_version = 'squid'
    expected = 'projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}/cryptoKeyVersions/{crypto_key_version}/publicKey'.format(project=project, location=location, key_ring=key_ring, crypto_key=crypto_key, crypto_key_version=crypto_key_version)
    actual = KeyManagementServiceClient.public_key_path(project, location, key_ring, crypto_key, crypto_key_version)
    assert expected == actual

def test_parse_public_key_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'clam', 'location': 'whelk', 'key_ring': 'octopus', 'crypto_key': 'oyster', 'crypto_key_version': 'nudibranch'}
    path = KeyManagementServiceClient.public_key_path(**expected)
    actual = KeyManagementServiceClient.parse_public_key_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = KeyManagementServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'mussel'}
    path = KeyManagementServiceClient.common_billing_account_path(**expected)
    actual = KeyManagementServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = KeyManagementServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'nautilus'}
    path = KeyManagementServiceClient.common_folder_path(**expected)
    actual = KeyManagementServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = KeyManagementServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'abalone'}
    path = KeyManagementServiceClient.common_organization_path(**expected)
    actual = KeyManagementServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = KeyManagementServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'clam'}
    path = KeyManagementServiceClient.common_project_path(**expected)
    actual = KeyManagementServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = KeyManagementServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = KeyManagementServiceClient.common_location_path(**expected)
    actual = KeyManagementServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.KeyManagementServiceTransport, '_prep_wrapped_messages') as prep:
        client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.KeyManagementServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = KeyManagementServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/keyRings/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/keyRings/sample3'}
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
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/keyRings/sample3'}, request)
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
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/keyRings/sample3'}
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
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/keyRings/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/keyRings/sample3'}
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

def test_list_locations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = KeyManagementServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = KeyManagementServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(KeyManagementServiceClient, transports.KeyManagementServiceGrpcTransport), (KeyManagementServiceAsyncClient, transports.KeyManagementServiceGrpcAsyncIOTransport)])
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