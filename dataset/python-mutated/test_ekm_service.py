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
from google.cloud.kms_v1.services.ekm_service import EkmServiceAsyncClient, EkmServiceClient, pagers, transports
from google.cloud.kms_v1.types import ekm_service

def client_cert_source_callback():
    if False:
        for i in range(10):
            print('nop')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        return 10
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
    assert EkmServiceClient._get_default_mtls_endpoint(None) is None
    assert EkmServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert EkmServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert EkmServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert EkmServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert EkmServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(EkmServiceClient, 'grpc'), (EkmServiceAsyncClient, 'grpc_asyncio'), (EkmServiceClient, 'rest')])
def test_ekm_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('cloudkms.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudkms.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.EkmServiceGrpcTransport, 'grpc'), (transports.EkmServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.EkmServiceRestTransport, 'rest')])
def test_ekm_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(EkmServiceClient, 'grpc'), (EkmServiceAsyncClient, 'grpc_asyncio'), (EkmServiceClient, 'rest')])
def test_ekm_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('cloudkms.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudkms.googleapis.com')

def test_ekm_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = EkmServiceClient.get_transport_class()
    available_transports = [transports.EkmServiceGrpcTransport, transports.EkmServiceRestTransport]
    assert transport in available_transports
    transport = EkmServiceClient.get_transport_class('grpc')
    assert transport == transports.EkmServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(EkmServiceClient, transports.EkmServiceGrpcTransport, 'grpc'), (EkmServiceAsyncClient, transports.EkmServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (EkmServiceClient, transports.EkmServiceRestTransport, 'rest')])
@mock.patch.object(EkmServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EkmServiceClient))
@mock.patch.object(EkmServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EkmServiceAsyncClient))
def test_ekm_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(EkmServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(EkmServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(EkmServiceClient, transports.EkmServiceGrpcTransport, 'grpc', 'true'), (EkmServiceAsyncClient, transports.EkmServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (EkmServiceClient, transports.EkmServiceGrpcTransport, 'grpc', 'false'), (EkmServiceAsyncClient, transports.EkmServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (EkmServiceClient, transports.EkmServiceRestTransport, 'rest', 'true'), (EkmServiceClient, transports.EkmServiceRestTransport, 'rest', 'false')])
@mock.patch.object(EkmServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EkmServiceClient))
@mock.patch.object(EkmServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EkmServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_ekm_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [EkmServiceClient, EkmServiceAsyncClient])
@mock.patch.object(EkmServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EkmServiceClient))
@mock.patch.object(EkmServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EkmServiceAsyncClient))
def test_ekm_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(EkmServiceClient, transports.EkmServiceGrpcTransport, 'grpc'), (EkmServiceAsyncClient, transports.EkmServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (EkmServiceClient, transports.EkmServiceRestTransport, 'rest')])
def test_ekm_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(EkmServiceClient, transports.EkmServiceGrpcTransport, 'grpc', grpc_helpers), (EkmServiceAsyncClient, transports.EkmServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (EkmServiceClient, transports.EkmServiceRestTransport, 'rest', None)])
def test_ekm_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_ekm_service_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.kms_v1.services.ekm_service.transports.EkmServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = EkmServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(EkmServiceClient, transports.EkmServiceGrpcTransport, 'grpc', grpc_helpers), (EkmServiceAsyncClient, transports.EkmServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_ekm_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('cloudkms.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloudkms'), scopes=None, default_host='cloudkms.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [ekm_service.ListEkmConnectionsRequest, dict])
def test_list_ekm_connections(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_ekm_connections), '__call__') as call:
        call.return_value = ekm_service.ListEkmConnectionsResponse(next_page_token='next_page_token_value', total_size=1086)
        response = client.list_ekm_connections(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.ListEkmConnectionsRequest()
    assert isinstance(response, pagers.ListEkmConnectionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_ekm_connections_empty_call():
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_ekm_connections), '__call__') as call:
        client.list_ekm_connections()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.ListEkmConnectionsRequest()

@pytest.mark.asyncio
async def test_list_ekm_connections_async(transport: str='grpc_asyncio', request_type=ekm_service.ListEkmConnectionsRequest):
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_ekm_connections), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.ListEkmConnectionsResponse(next_page_token='next_page_token_value', total_size=1086))
        response = await client.list_ekm_connections(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.ListEkmConnectionsRequest()
    assert isinstance(response, pagers.ListEkmConnectionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

@pytest.mark.asyncio
async def test_list_ekm_connections_async_from_dict():
    await test_list_ekm_connections_async(request_type=dict)

def test_list_ekm_connections_field_headers():
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = ekm_service.ListEkmConnectionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_ekm_connections), '__call__') as call:
        call.return_value = ekm_service.ListEkmConnectionsResponse()
        client.list_ekm_connections(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_ekm_connections_field_headers_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = ekm_service.ListEkmConnectionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_ekm_connections), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.ListEkmConnectionsResponse())
        await client.list_ekm_connections(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_ekm_connections_flattened():
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_ekm_connections), '__call__') as call:
        call.return_value = ekm_service.ListEkmConnectionsResponse()
        client.list_ekm_connections(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_ekm_connections_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_ekm_connections(ekm_service.ListEkmConnectionsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_ekm_connections_flattened_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_ekm_connections), '__call__') as call:
        call.return_value = ekm_service.ListEkmConnectionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.ListEkmConnectionsResponse())
        response = await client.list_ekm_connections(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_ekm_connections_flattened_error_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_ekm_connections(ekm_service.ListEkmConnectionsRequest(), parent='parent_value')

def test_list_ekm_connections_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_ekm_connections), '__call__') as call:
        call.side_effect = (ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection(), ekm_service.EkmConnection(), ekm_service.EkmConnection()], next_page_token='abc'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[], next_page_token='def'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection()], next_page_token='ghi'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection(), ekm_service.EkmConnection()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_ekm_connections(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, ekm_service.EkmConnection) for i in results))

def test_list_ekm_connections_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_ekm_connections), '__call__') as call:
        call.side_effect = (ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection(), ekm_service.EkmConnection(), ekm_service.EkmConnection()], next_page_token='abc'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[], next_page_token='def'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection()], next_page_token='ghi'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection(), ekm_service.EkmConnection()]), RuntimeError)
        pages = list(client.list_ekm_connections(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_ekm_connections_async_pager():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_ekm_connections), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection(), ekm_service.EkmConnection(), ekm_service.EkmConnection()], next_page_token='abc'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[], next_page_token='def'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection()], next_page_token='ghi'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection(), ekm_service.EkmConnection()]), RuntimeError)
        async_pager = await client.list_ekm_connections(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, ekm_service.EkmConnection) for i in responses))

@pytest.mark.asyncio
async def test_list_ekm_connections_async_pages():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_ekm_connections), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection(), ekm_service.EkmConnection(), ekm_service.EkmConnection()], next_page_token='abc'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[], next_page_token='def'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection()], next_page_token='ghi'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection(), ekm_service.EkmConnection()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_ekm_connections(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [ekm_service.GetEkmConnectionRequest, dict])
def test_get_ekm_connection(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_ekm_connection), '__call__') as call:
        call.return_value = ekm_service.EkmConnection(name='name_value', etag='etag_value', key_management_mode=ekm_service.EkmConnection.KeyManagementMode.MANUAL, crypto_space_path='crypto_space_path_value')
        response = client.get_ekm_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.GetEkmConnectionRequest()
    assert isinstance(response, ekm_service.EkmConnection)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'
    assert response.key_management_mode == ekm_service.EkmConnection.KeyManagementMode.MANUAL
    assert response.crypto_space_path == 'crypto_space_path_value'

def test_get_ekm_connection_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_ekm_connection), '__call__') as call:
        client.get_ekm_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.GetEkmConnectionRequest()

@pytest.mark.asyncio
async def test_get_ekm_connection_async(transport: str='grpc_asyncio', request_type=ekm_service.GetEkmConnectionRequest):
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_ekm_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConnection(name='name_value', etag='etag_value', key_management_mode=ekm_service.EkmConnection.KeyManagementMode.MANUAL, crypto_space_path='crypto_space_path_value'))
        response = await client.get_ekm_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.GetEkmConnectionRequest()
    assert isinstance(response, ekm_service.EkmConnection)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'
    assert response.key_management_mode == ekm_service.EkmConnection.KeyManagementMode.MANUAL
    assert response.crypto_space_path == 'crypto_space_path_value'

@pytest.mark.asyncio
async def test_get_ekm_connection_async_from_dict():
    await test_get_ekm_connection_async(request_type=dict)

def test_get_ekm_connection_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = ekm_service.GetEkmConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_ekm_connection), '__call__') as call:
        call.return_value = ekm_service.EkmConnection()
        client.get_ekm_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_ekm_connection_field_headers_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = ekm_service.GetEkmConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_ekm_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConnection())
        await client.get_ekm_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_ekm_connection_flattened():
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_ekm_connection), '__call__') as call:
        call.return_value = ekm_service.EkmConnection()
        client.get_ekm_connection(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_ekm_connection_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_ekm_connection(ekm_service.GetEkmConnectionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_ekm_connection_flattened_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_ekm_connection), '__call__') as call:
        call.return_value = ekm_service.EkmConnection()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConnection())
        response = await client.get_ekm_connection(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_ekm_connection_flattened_error_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_ekm_connection(ekm_service.GetEkmConnectionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [ekm_service.CreateEkmConnectionRequest, dict])
def test_create_ekm_connection(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_ekm_connection), '__call__') as call:
        call.return_value = ekm_service.EkmConnection(name='name_value', etag='etag_value', key_management_mode=ekm_service.EkmConnection.KeyManagementMode.MANUAL, crypto_space_path='crypto_space_path_value')
        response = client.create_ekm_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.CreateEkmConnectionRequest()
    assert isinstance(response, ekm_service.EkmConnection)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'
    assert response.key_management_mode == ekm_service.EkmConnection.KeyManagementMode.MANUAL
    assert response.crypto_space_path == 'crypto_space_path_value'

def test_create_ekm_connection_empty_call():
    if False:
        i = 10
        return i + 15
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_ekm_connection), '__call__') as call:
        client.create_ekm_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.CreateEkmConnectionRequest()

@pytest.mark.asyncio
async def test_create_ekm_connection_async(transport: str='grpc_asyncio', request_type=ekm_service.CreateEkmConnectionRequest):
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_ekm_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConnection(name='name_value', etag='etag_value', key_management_mode=ekm_service.EkmConnection.KeyManagementMode.MANUAL, crypto_space_path='crypto_space_path_value'))
        response = await client.create_ekm_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.CreateEkmConnectionRequest()
    assert isinstance(response, ekm_service.EkmConnection)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'
    assert response.key_management_mode == ekm_service.EkmConnection.KeyManagementMode.MANUAL
    assert response.crypto_space_path == 'crypto_space_path_value'

@pytest.mark.asyncio
async def test_create_ekm_connection_async_from_dict():
    await test_create_ekm_connection_async(request_type=dict)

def test_create_ekm_connection_field_headers():
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = ekm_service.CreateEkmConnectionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_ekm_connection), '__call__') as call:
        call.return_value = ekm_service.EkmConnection()
        client.create_ekm_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_ekm_connection_field_headers_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = ekm_service.CreateEkmConnectionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_ekm_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConnection())
        await client.create_ekm_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_ekm_connection_flattened():
    if False:
        i = 10
        return i + 15
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_ekm_connection), '__call__') as call:
        call.return_value = ekm_service.EkmConnection()
        client.create_ekm_connection(parent='parent_value', ekm_connection_id='ekm_connection_id_value', ekm_connection=ekm_service.EkmConnection(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].ekm_connection_id
        mock_val = 'ekm_connection_id_value'
        assert arg == mock_val
        arg = args[0].ekm_connection
        mock_val = ekm_service.EkmConnection(name='name_value')
        assert arg == mock_val

def test_create_ekm_connection_flattened_error():
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_ekm_connection(ekm_service.CreateEkmConnectionRequest(), parent='parent_value', ekm_connection_id='ekm_connection_id_value', ekm_connection=ekm_service.EkmConnection(name='name_value'))

@pytest.mark.asyncio
async def test_create_ekm_connection_flattened_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_ekm_connection), '__call__') as call:
        call.return_value = ekm_service.EkmConnection()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConnection())
        response = await client.create_ekm_connection(parent='parent_value', ekm_connection_id='ekm_connection_id_value', ekm_connection=ekm_service.EkmConnection(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].ekm_connection_id
        mock_val = 'ekm_connection_id_value'
        assert arg == mock_val
        arg = args[0].ekm_connection
        mock_val = ekm_service.EkmConnection(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_ekm_connection_flattened_error_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_ekm_connection(ekm_service.CreateEkmConnectionRequest(), parent='parent_value', ekm_connection_id='ekm_connection_id_value', ekm_connection=ekm_service.EkmConnection(name='name_value'))

@pytest.mark.parametrize('request_type', [ekm_service.UpdateEkmConnectionRequest, dict])
def test_update_ekm_connection(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_ekm_connection), '__call__') as call:
        call.return_value = ekm_service.EkmConnection(name='name_value', etag='etag_value', key_management_mode=ekm_service.EkmConnection.KeyManagementMode.MANUAL, crypto_space_path='crypto_space_path_value')
        response = client.update_ekm_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.UpdateEkmConnectionRequest()
    assert isinstance(response, ekm_service.EkmConnection)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'
    assert response.key_management_mode == ekm_service.EkmConnection.KeyManagementMode.MANUAL
    assert response.crypto_space_path == 'crypto_space_path_value'

def test_update_ekm_connection_empty_call():
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_ekm_connection), '__call__') as call:
        client.update_ekm_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.UpdateEkmConnectionRequest()

@pytest.mark.asyncio
async def test_update_ekm_connection_async(transport: str='grpc_asyncio', request_type=ekm_service.UpdateEkmConnectionRequest):
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_ekm_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConnection(name='name_value', etag='etag_value', key_management_mode=ekm_service.EkmConnection.KeyManagementMode.MANUAL, crypto_space_path='crypto_space_path_value'))
        response = await client.update_ekm_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.UpdateEkmConnectionRequest()
    assert isinstance(response, ekm_service.EkmConnection)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'
    assert response.key_management_mode == ekm_service.EkmConnection.KeyManagementMode.MANUAL
    assert response.crypto_space_path == 'crypto_space_path_value'

@pytest.mark.asyncio
async def test_update_ekm_connection_async_from_dict():
    await test_update_ekm_connection_async(request_type=dict)

def test_update_ekm_connection_field_headers():
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = ekm_service.UpdateEkmConnectionRequest()
    request.ekm_connection.name = 'name_value'
    with mock.patch.object(type(client.transport.update_ekm_connection), '__call__') as call:
        call.return_value = ekm_service.EkmConnection()
        client.update_ekm_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'ekm_connection.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_ekm_connection_field_headers_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = ekm_service.UpdateEkmConnectionRequest()
    request.ekm_connection.name = 'name_value'
    with mock.patch.object(type(client.transport.update_ekm_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConnection())
        await client.update_ekm_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'ekm_connection.name=name_value') in kw['metadata']

def test_update_ekm_connection_flattened():
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_ekm_connection), '__call__') as call:
        call.return_value = ekm_service.EkmConnection()
        client.update_ekm_connection(ekm_connection=ekm_service.EkmConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].ekm_connection
        mock_val = ekm_service.EkmConnection(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_ekm_connection_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_ekm_connection(ekm_service.UpdateEkmConnectionRequest(), ekm_connection=ekm_service.EkmConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_ekm_connection_flattened_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_ekm_connection), '__call__') as call:
        call.return_value = ekm_service.EkmConnection()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConnection())
        response = await client.update_ekm_connection(ekm_connection=ekm_service.EkmConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].ekm_connection
        mock_val = ekm_service.EkmConnection(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_ekm_connection_flattened_error_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_ekm_connection(ekm_service.UpdateEkmConnectionRequest(), ekm_connection=ekm_service.EkmConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [ekm_service.GetEkmConfigRequest, dict])
def test_get_ekm_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_ekm_config), '__call__') as call:
        call.return_value = ekm_service.EkmConfig(name='name_value', default_ekm_connection='default_ekm_connection_value')
        response = client.get_ekm_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.GetEkmConfigRequest()
    assert isinstance(response, ekm_service.EkmConfig)
    assert response.name == 'name_value'
    assert response.default_ekm_connection == 'default_ekm_connection_value'

def test_get_ekm_config_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_ekm_config), '__call__') as call:
        client.get_ekm_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.GetEkmConfigRequest()

@pytest.mark.asyncio
async def test_get_ekm_config_async(transport: str='grpc_asyncio', request_type=ekm_service.GetEkmConfigRequest):
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_ekm_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConfig(name='name_value', default_ekm_connection='default_ekm_connection_value'))
        response = await client.get_ekm_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.GetEkmConfigRequest()
    assert isinstance(response, ekm_service.EkmConfig)
    assert response.name == 'name_value'
    assert response.default_ekm_connection == 'default_ekm_connection_value'

@pytest.mark.asyncio
async def test_get_ekm_config_async_from_dict():
    await test_get_ekm_config_async(request_type=dict)

def test_get_ekm_config_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = ekm_service.GetEkmConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_ekm_config), '__call__') as call:
        call.return_value = ekm_service.EkmConfig()
        client.get_ekm_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_ekm_config_field_headers_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = ekm_service.GetEkmConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_ekm_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConfig())
        await client.get_ekm_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_ekm_config_flattened():
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_ekm_config), '__call__') as call:
        call.return_value = ekm_service.EkmConfig()
        client.get_ekm_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_ekm_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_ekm_config(ekm_service.GetEkmConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_ekm_config_flattened_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_ekm_config), '__call__') as call:
        call.return_value = ekm_service.EkmConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConfig())
        response = await client.get_ekm_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_ekm_config_flattened_error_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_ekm_config(ekm_service.GetEkmConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [ekm_service.UpdateEkmConfigRequest, dict])
def test_update_ekm_config(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_ekm_config), '__call__') as call:
        call.return_value = ekm_service.EkmConfig(name='name_value', default_ekm_connection='default_ekm_connection_value')
        response = client.update_ekm_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.UpdateEkmConfigRequest()
    assert isinstance(response, ekm_service.EkmConfig)
    assert response.name == 'name_value'
    assert response.default_ekm_connection == 'default_ekm_connection_value'

def test_update_ekm_config_empty_call():
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_ekm_config), '__call__') as call:
        client.update_ekm_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.UpdateEkmConfigRequest()

@pytest.mark.asyncio
async def test_update_ekm_config_async(transport: str='grpc_asyncio', request_type=ekm_service.UpdateEkmConfigRequest):
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_ekm_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConfig(name='name_value', default_ekm_connection='default_ekm_connection_value'))
        response = await client.update_ekm_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.UpdateEkmConfigRequest()
    assert isinstance(response, ekm_service.EkmConfig)
    assert response.name == 'name_value'
    assert response.default_ekm_connection == 'default_ekm_connection_value'

@pytest.mark.asyncio
async def test_update_ekm_config_async_from_dict():
    await test_update_ekm_config_async(request_type=dict)

def test_update_ekm_config_field_headers():
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = ekm_service.UpdateEkmConfigRequest()
    request.ekm_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_ekm_config), '__call__') as call:
        call.return_value = ekm_service.EkmConfig()
        client.update_ekm_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'ekm_config.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_ekm_config_field_headers_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = ekm_service.UpdateEkmConfigRequest()
    request.ekm_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_ekm_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConfig())
        await client.update_ekm_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'ekm_config.name=name_value') in kw['metadata']

def test_update_ekm_config_flattened():
    if False:
        i = 10
        return i + 15
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_ekm_config), '__call__') as call:
        call.return_value = ekm_service.EkmConfig()
        client.update_ekm_config(ekm_config=ekm_service.EkmConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].ekm_config
        mock_val = ekm_service.EkmConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_ekm_config_flattened_error():
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_ekm_config(ekm_service.UpdateEkmConfigRequest(), ekm_config=ekm_service.EkmConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_ekm_config_flattened_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_ekm_config), '__call__') as call:
        call.return_value = ekm_service.EkmConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.EkmConfig())
        response = await client.update_ekm_config(ekm_config=ekm_service.EkmConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].ekm_config
        mock_val = ekm_service.EkmConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_ekm_config_flattened_error_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_ekm_config(ekm_service.UpdateEkmConfigRequest(), ekm_config=ekm_service.EkmConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [ekm_service.VerifyConnectivityRequest, dict])
def test_verify_connectivity(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.verify_connectivity), '__call__') as call:
        call.return_value = ekm_service.VerifyConnectivityResponse()
        response = client.verify_connectivity(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.VerifyConnectivityRequest()
    assert isinstance(response, ekm_service.VerifyConnectivityResponse)

def test_verify_connectivity_empty_call():
    if False:
        i = 10
        return i + 15
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.verify_connectivity), '__call__') as call:
        client.verify_connectivity()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.VerifyConnectivityRequest()

@pytest.mark.asyncio
async def test_verify_connectivity_async(transport: str='grpc_asyncio', request_type=ekm_service.VerifyConnectivityRequest):
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.verify_connectivity), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.VerifyConnectivityResponse())
        response = await client.verify_connectivity(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == ekm_service.VerifyConnectivityRequest()
    assert isinstance(response, ekm_service.VerifyConnectivityResponse)

@pytest.mark.asyncio
async def test_verify_connectivity_async_from_dict():
    await test_verify_connectivity_async(request_type=dict)

def test_verify_connectivity_field_headers():
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = ekm_service.VerifyConnectivityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.verify_connectivity), '__call__') as call:
        call.return_value = ekm_service.VerifyConnectivityResponse()
        client.verify_connectivity(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_verify_connectivity_field_headers_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = ekm_service.VerifyConnectivityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.verify_connectivity), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.VerifyConnectivityResponse())
        await client.verify_connectivity(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_verify_connectivity_flattened():
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.verify_connectivity), '__call__') as call:
        call.return_value = ekm_service.VerifyConnectivityResponse()
        client.verify_connectivity(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_verify_connectivity_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.verify_connectivity(ekm_service.VerifyConnectivityRequest(), name='name_value')

@pytest.mark.asyncio
async def test_verify_connectivity_flattened_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.verify_connectivity), '__call__') as call:
        call.return_value = ekm_service.VerifyConnectivityResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ekm_service.VerifyConnectivityResponse())
        response = await client.verify_connectivity(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_verify_connectivity_flattened_error_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.verify_connectivity(ekm_service.VerifyConnectivityRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [ekm_service.ListEkmConnectionsRequest, dict])
def test_list_ekm_connections_rest(request_type):
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ekm_service.ListEkmConnectionsResponse(next_page_token='next_page_token_value', total_size=1086)
        response_value = Response()
        response_value.status_code = 200
        return_value = ekm_service.ListEkmConnectionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_ekm_connections(request)
    assert isinstance(response, pagers.ListEkmConnectionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_ekm_connections_rest_required_fields(request_type=ekm_service.ListEkmConnectionsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EkmServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_ekm_connections._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_ekm_connections._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = ekm_service.ListEkmConnectionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = ekm_service.ListEkmConnectionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_ekm_connections(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_ekm_connections_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EkmServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_ekm_connections._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_ekm_connections_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EkmServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EkmServiceRestInterceptor())
    client = EkmServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EkmServiceRestInterceptor, 'post_list_ekm_connections') as post, mock.patch.object(transports.EkmServiceRestInterceptor, 'pre_list_ekm_connections') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = ekm_service.ListEkmConnectionsRequest.pb(ekm_service.ListEkmConnectionsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = ekm_service.ListEkmConnectionsResponse.to_json(ekm_service.ListEkmConnectionsResponse())
        request = ekm_service.ListEkmConnectionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = ekm_service.ListEkmConnectionsResponse()
        client.list_ekm_connections(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_ekm_connections_rest_bad_request(transport: str='rest', request_type=ekm_service.ListEkmConnectionsRequest):
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_ekm_connections(request)

def test_list_ekm_connections_rest_flattened():
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ekm_service.ListEkmConnectionsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = ekm_service.ListEkmConnectionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_ekm_connections(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/ekmConnections' % client.transport._host, args[1])

def test_list_ekm_connections_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_ekm_connections(ekm_service.ListEkmConnectionsRequest(), parent='parent_value')

def test_list_ekm_connections_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection(), ekm_service.EkmConnection(), ekm_service.EkmConnection()], next_page_token='abc'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[], next_page_token='def'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection()], next_page_token='ghi'), ekm_service.ListEkmConnectionsResponse(ekm_connections=[ekm_service.EkmConnection(), ekm_service.EkmConnection()]))
        response = response + response
        response = tuple((ekm_service.ListEkmConnectionsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_ekm_connections(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, ekm_service.EkmConnection) for i in results))
        pages = list(client.list_ekm_connections(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [ekm_service.GetEkmConnectionRequest, dict])
def test_get_ekm_connection_rest(request_type):
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/ekmConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ekm_service.EkmConnection(name='name_value', etag='etag_value', key_management_mode=ekm_service.EkmConnection.KeyManagementMode.MANUAL, crypto_space_path='crypto_space_path_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = ekm_service.EkmConnection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_ekm_connection(request)
    assert isinstance(response, ekm_service.EkmConnection)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'
    assert response.key_management_mode == ekm_service.EkmConnection.KeyManagementMode.MANUAL
    assert response.crypto_space_path == 'crypto_space_path_value'

def test_get_ekm_connection_rest_required_fields(request_type=ekm_service.GetEkmConnectionRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EkmServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_ekm_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_ekm_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = ekm_service.EkmConnection()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = ekm_service.EkmConnection.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_ekm_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_ekm_connection_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EkmServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_ekm_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_ekm_connection_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EkmServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EkmServiceRestInterceptor())
    client = EkmServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EkmServiceRestInterceptor, 'post_get_ekm_connection') as post, mock.patch.object(transports.EkmServiceRestInterceptor, 'pre_get_ekm_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = ekm_service.GetEkmConnectionRequest.pb(ekm_service.GetEkmConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = ekm_service.EkmConnection.to_json(ekm_service.EkmConnection())
        request = ekm_service.GetEkmConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = ekm_service.EkmConnection()
        client.get_ekm_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_ekm_connection_rest_bad_request(transport: str='rest', request_type=ekm_service.GetEkmConnectionRequest):
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/ekmConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_ekm_connection(request)

def test_get_ekm_connection_rest_flattened():
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ekm_service.EkmConnection()
        sample_request = {'name': 'projects/sample1/locations/sample2/ekmConnections/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = ekm_service.EkmConnection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_ekm_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/ekmConnections/*}' % client.transport._host, args[1])

def test_get_ekm_connection_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_ekm_connection(ekm_service.GetEkmConnectionRequest(), name='name_value')

def test_get_ekm_connection_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [ekm_service.CreateEkmConnectionRequest, dict])
def test_create_ekm_connection_rest(request_type):
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['ekm_connection'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'service_resolvers': [{'service_directory_service': 'service_directory_service_value', 'endpoint_filter': 'endpoint_filter_value', 'hostname': 'hostname_value', 'server_certificates': [{'raw_der': b'raw_der_blob', 'parsed': True, 'issuer': 'issuer_value', 'subject': 'subject_value', 'subject_alternative_dns_names': ['subject_alternative_dns_names_value1', 'subject_alternative_dns_names_value2'], 'not_before_time': {}, 'not_after_time': {}, 'serial_number': 'serial_number_value', 'sha256_fingerprint': 'sha256_fingerprint_value'}]}], 'etag': 'etag_value', 'key_management_mode': 1, 'crypto_space_path': 'crypto_space_path_value'}
    test_field = ekm_service.CreateEkmConnectionRequest.meta.fields['ekm_connection']

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
    for (field, value) in request_init['ekm_connection'].items():
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
                for i in range(0, len(request_init['ekm_connection'][field])):
                    del request_init['ekm_connection'][field][i][subfield]
            else:
                del request_init['ekm_connection'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ekm_service.EkmConnection(name='name_value', etag='etag_value', key_management_mode=ekm_service.EkmConnection.KeyManagementMode.MANUAL, crypto_space_path='crypto_space_path_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = ekm_service.EkmConnection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_ekm_connection(request)
    assert isinstance(response, ekm_service.EkmConnection)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'
    assert response.key_management_mode == ekm_service.EkmConnection.KeyManagementMode.MANUAL
    assert response.crypto_space_path == 'crypto_space_path_value'

def test_create_ekm_connection_rest_required_fields(request_type=ekm_service.CreateEkmConnectionRequest):
    if False:
        return 10
    transport_class = transports.EkmServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['ekm_connection_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'ekmConnectionId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_ekm_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'ekmConnectionId' in jsonified_request
    assert jsonified_request['ekmConnectionId'] == request_init['ekm_connection_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['ekmConnectionId'] = 'ekm_connection_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_ekm_connection._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('ekm_connection_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'ekmConnectionId' in jsonified_request
    assert jsonified_request['ekmConnectionId'] == 'ekm_connection_id_value'
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = ekm_service.EkmConnection()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = ekm_service.EkmConnection.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_ekm_connection(request)
            expected_params = [('ekmConnectionId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_ekm_connection_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EkmServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_ekm_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(('ekmConnectionId',)) & set(('parent', 'ekmConnectionId', 'ekmConnection'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_ekm_connection_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EkmServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EkmServiceRestInterceptor())
    client = EkmServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EkmServiceRestInterceptor, 'post_create_ekm_connection') as post, mock.patch.object(transports.EkmServiceRestInterceptor, 'pre_create_ekm_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = ekm_service.CreateEkmConnectionRequest.pb(ekm_service.CreateEkmConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = ekm_service.EkmConnection.to_json(ekm_service.EkmConnection())
        request = ekm_service.CreateEkmConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = ekm_service.EkmConnection()
        client.create_ekm_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_ekm_connection_rest_bad_request(transport: str='rest', request_type=ekm_service.CreateEkmConnectionRequest):
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_ekm_connection(request)

def test_create_ekm_connection_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ekm_service.EkmConnection()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', ekm_connection_id='ekm_connection_id_value', ekm_connection=ekm_service.EkmConnection(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = ekm_service.EkmConnection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_ekm_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/ekmConnections' % client.transport._host, args[1])

def test_create_ekm_connection_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_ekm_connection(ekm_service.CreateEkmConnectionRequest(), parent='parent_value', ekm_connection_id='ekm_connection_id_value', ekm_connection=ekm_service.EkmConnection(name='name_value'))

def test_create_ekm_connection_rest_error():
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [ekm_service.UpdateEkmConnectionRequest, dict])
def test_update_ekm_connection_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'ekm_connection': {'name': 'projects/sample1/locations/sample2/ekmConnections/sample3'}}
    request_init['ekm_connection'] = {'name': 'projects/sample1/locations/sample2/ekmConnections/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'service_resolvers': [{'service_directory_service': 'service_directory_service_value', 'endpoint_filter': 'endpoint_filter_value', 'hostname': 'hostname_value', 'server_certificates': [{'raw_der': b'raw_der_blob', 'parsed': True, 'issuer': 'issuer_value', 'subject': 'subject_value', 'subject_alternative_dns_names': ['subject_alternative_dns_names_value1', 'subject_alternative_dns_names_value2'], 'not_before_time': {}, 'not_after_time': {}, 'serial_number': 'serial_number_value', 'sha256_fingerprint': 'sha256_fingerprint_value'}]}], 'etag': 'etag_value', 'key_management_mode': 1, 'crypto_space_path': 'crypto_space_path_value'}
    test_field = ekm_service.UpdateEkmConnectionRequest.meta.fields['ekm_connection']

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
    for (field, value) in request_init['ekm_connection'].items():
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
                for i in range(0, len(request_init['ekm_connection'][field])):
                    del request_init['ekm_connection'][field][i][subfield]
            else:
                del request_init['ekm_connection'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ekm_service.EkmConnection(name='name_value', etag='etag_value', key_management_mode=ekm_service.EkmConnection.KeyManagementMode.MANUAL, crypto_space_path='crypto_space_path_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = ekm_service.EkmConnection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_ekm_connection(request)
    assert isinstance(response, ekm_service.EkmConnection)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'
    assert response.key_management_mode == ekm_service.EkmConnection.KeyManagementMode.MANUAL
    assert response.crypto_space_path == 'crypto_space_path_value'

def test_update_ekm_connection_rest_required_fields(request_type=ekm_service.UpdateEkmConnectionRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EkmServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_ekm_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_ekm_connection._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = ekm_service.EkmConnection()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = ekm_service.EkmConnection.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_ekm_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_ekm_connection_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EkmServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_ekm_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('ekmConnection', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_ekm_connection_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EkmServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EkmServiceRestInterceptor())
    client = EkmServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EkmServiceRestInterceptor, 'post_update_ekm_connection') as post, mock.patch.object(transports.EkmServiceRestInterceptor, 'pre_update_ekm_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = ekm_service.UpdateEkmConnectionRequest.pb(ekm_service.UpdateEkmConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = ekm_service.EkmConnection.to_json(ekm_service.EkmConnection())
        request = ekm_service.UpdateEkmConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = ekm_service.EkmConnection()
        client.update_ekm_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_ekm_connection_rest_bad_request(transport: str='rest', request_type=ekm_service.UpdateEkmConnectionRequest):
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'ekm_connection': {'name': 'projects/sample1/locations/sample2/ekmConnections/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_ekm_connection(request)

def test_update_ekm_connection_rest_flattened():
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ekm_service.EkmConnection()
        sample_request = {'ekm_connection': {'name': 'projects/sample1/locations/sample2/ekmConnections/sample3'}}
        mock_args = dict(ekm_connection=ekm_service.EkmConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = ekm_service.EkmConnection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_ekm_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{ekm_connection.name=projects/*/locations/*/ekmConnections/*}' % client.transport._host, args[1])

def test_update_ekm_connection_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_ekm_connection(ekm_service.UpdateEkmConnectionRequest(), ekm_connection=ekm_service.EkmConnection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_ekm_connection_rest_error():
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [ekm_service.GetEkmConfigRequest, dict])
def test_get_ekm_config_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/ekmConfig'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ekm_service.EkmConfig(name='name_value', default_ekm_connection='default_ekm_connection_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = ekm_service.EkmConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_ekm_config(request)
    assert isinstance(response, ekm_service.EkmConfig)
    assert response.name == 'name_value'
    assert response.default_ekm_connection == 'default_ekm_connection_value'

def test_get_ekm_config_rest_required_fields(request_type=ekm_service.GetEkmConfigRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EkmServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_ekm_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_ekm_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = ekm_service.EkmConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = ekm_service.EkmConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_ekm_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_ekm_config_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EkmServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_ekm_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_ekm_config_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EkmServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EkmServiceRestInterceptor())
    client = EkmServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EkmServiceRestInterceptor, 'post_get_ekm_config') as post, mock.patch.object(transports.EkmServiceRestInterceptor, 'pre_get_ekm_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = ekm_service.GetEkmConfigRequest.pb(ekm_service.GetEkmConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = ekm_service.EkmConfig.to_json(ekm_service.EkmConfig())
        request = ekm_service.GetEkmConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = ekm_service.EkmConfig()
        client.get_ekm_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_ekm_config_rest_bad_request(transport: str='rest', request_type=ekm_service.GetEkmConfigRequest):
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/ekmConfig'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_ekm_config(request)

def test_get_ekm_config_rest_flattened():
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ekm_service.EkmConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/ekmConfig'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = ekm_service.EkmConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_ekm_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/ekmConfig}' % client.transport._host, args[1])

def test_get_ekm_config_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_ekm_config(ekm_service.GetEkmConfigRequest(), name='name_value')

def test_get_ekm_config_rest_error():
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [ekm_service.UpdateEkmConfigRequest, dict])
def test_update_ekm_config_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'ekm_config': {'name': 'projects/sample1/locations/sample2/ekmConfig'}}
    request_init['ekm_config'] = {'name': 'projects/sample1/locations/sample2/ekmConfig', 'default_ekm_connection': 'default_ekm_connection_value'}
    test_field = ekm_service.UpdateEkmConfigRequest.meta.fields['ekm_config']

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
    for (field, value) in request_init['ekm_config'].items():
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
                for i in range(0, len(request_init['ekm_config'][field])):
                    del request_init['ekm_config'][field][i][subfield]
            else:
                del request_init['ekm_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ekm_service.EkmConfig(name='name_value', default_ekm_connection='default_ekm_connection_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = ekm_service.EkmConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_ekm_config(request)
    assert isinstance(response, ekm_service.EkmConfig)
    assert response.name == 'name_value'
    assert response.default_ekm_connection == 'default_ekm_connection_value'

def test_update_ekm_config_rest_required_fields(request_type=ekm_service.UpdateEkmConfigRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EkmServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_ekm_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_ekm_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = ekm_service.EkmConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = ekm_service.EkmConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_ekm_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_ekm_config_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EkmServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_ekm_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('ekmConfig', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_ekm_config_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EkmServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EkmServiceRestInterceptor())
    client = EkmServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EkmServiceRestInterceptor, 'post_update_ekm_config') as post, mock.patch.object(transports.EkmServiceRestInterceptor, 'pre_update_ekm_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = ekm_service.UpdateEkmConfigRequest.pb(ekm_service.UpdateEkmConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = ekm_service.EkmConfig.to_json(ekm_service.EkmConfig())
        request = ekm_service.UpdateEkmConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = ekm_service.EkmConfig()
        client.update_ekm_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_ekm_config_rest_bad_request(transport: str='rest', request_type=ekm_service.UpdateEkmConfigRequest):
    if False:
        i = 10
        return i + 15
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'ekm_config': {'name': 'projects/sample1/locations/sample2/ekmConfig'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_ekm_config(request)

def test_update_ekm_config_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ekm_service.EkmConfig()
        sample_request = {'ekm_config': {'name': 'projects/sample1/locations/sample2/ekmConfig'}}
        mock_args = dict(ekm_config=ekm_service.EkmConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = ekm_service.EkmConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_ekm_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{ekm_config.name=projects/*/locations/*/ekmConfig}' % client.transport._host, args[1])

def test_update_ekm_config_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_ekm_config(ekm_service.UpdateEkmConfigRequest(), ekm_config=ekm_service.EkmConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_ekm_config_rest_error():
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [ekm_service.VerifyConnectivityRequest, dict])
def test_verify_connectivity_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/ekmConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ekm_service.VerifyConnectivityResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = ekm_service.VerifyConnectivityResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.verify_connectivity(request)
    assert isinstance(response, ekm_service.VerifyConnectivityResponse)

def test_verify_connectivity_rest_required_fields(request_type=ekm_service.VerifyConnectivityRequest):
    if False:
        return 10
    transport_class = transports.EkmServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).verify_connectivity._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).verify_connectivity._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = ekm_service.VerifyConnectivityResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = ekm_service.VerifyConnectivityResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.verify_connectivity(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_verify_connectivity_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EkmServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.verify_connectivity._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_verify_connectivity_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EkmServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EkmServiceRestInterceptor())
    client = EkmServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EkmServiceRestInterceptor, 'post_verify_connectivity') as post, mock.patch.object(transports.EkmServiceRestInterceptor, 'pre_verify_connectivity') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = ekm_service.VerifyConnectivityRequest.pb(ekm_service.VerifyConnectivityRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = ekm_service.VerifyConnectivityResponse.to_json(ekm_service.VerifyConnectivityResponse())
        request = ekm_service.VerifyConnectivityRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = ekm_service.VerifyConnectivityResponse()
        client.verify_connectivity(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_verify_connectivity_rest_bad_request(transport: str='rest', request_type=ekm_service.VerifyConnectivityRequest):
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/ekmConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.verify_connectivity(request)

def test_verify_connectivity_rest_flattened():
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = ekm_service.VerifyConnectivityResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/ekmConnections/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = ekm_service.VerifyConnectivityResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.verify_connectivity(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/ekmConnections/*}:verifyConnectivity' % client.transport._host, args[1])

def test_verify_connectivity_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.verify_connectivity(ekm_service.VerifyConnectivityRequest(), name='name_value')

def test_verify_connectivity_rest_error():
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.EkmServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.EkmServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EkmServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.EkmServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = EkmServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = EkmServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.EkmServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EkmServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EkmServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = EkmServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.EkmServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.EkmServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.EkmServiceGrpcTransport, transports.EkmServiceGrpcAsyncIOTransport, transports.EkmServiceRestTransport])
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
    transport = EkmServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.EkmServiceGrpcTransport)

def test_ekm_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.EkmServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_ekm_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.kms_v1.services.ekm_service.transports.EkmServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.EkmServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_ekm_connections', 'get_ekm_connection', 'create_ekm_connection', 'update_ekm_connection', 'get_ekm_config', 'update_ekm_config', 'verify_connectivity', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_location', 'list_locations')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_ekm_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.kms_v1.services.ekm_service.transports.EkmServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.EkmServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloudkms'), quota_project_id='octopus')

def test_ekm_service_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.kms_v1.services.ekm_service.transports.EkmServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.EkmServiceTransport()
        adc.assert_called_once()

def test_ekm_service_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        EkmServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloudkms'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.EkmServiceGrpcTransport, transports.EkmServiceGrpcAsyncIOTransport])
def test_ekm_service_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloudkms'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.EkmServiceGrpcTransport, transports.EkmServiceGrpcAsyncIOTransport, transports.EkmServiceRestTransport])
def test_ekm_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.EkmServiceGrpcTransport, grpc_helpers), (transports.EkmServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_ekm_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudkms.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloudkms'), scopes=['1', '2'], default_host='cloudkms.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.EkmServiceGrpcTransport, transports.EkmServiceGrpcAsyncIOTransport])
def test_ekm_service_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        i = 10
        return i + 15
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

def test_ekm_service_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.EkmServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_ekm_service_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudkms.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudkms.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudkms.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_ekm_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudkms.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudkms.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudkms.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_ekm_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = EkmServiceClient(credentials=creds1, transport=transport_name)
    client2 = EkmServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_ekm_connections._session
    session2 = client2.transport.list_ekm_connections._session
    assert session1 != session2
    session1 = client1.transport.get_ekm_connection._session
    session2 = client2.transport.get_ekm_connection._session
    assert session1 != session2
    session1 = client1.transport.create_ekm_connection._session
    session2 = client2.transport.create_ekm_connection._session
    assert session1 != session2
    session1 = client1.transport.update_ekm_connection._session
    session2 = client2.transport.update_ekm_connection._session
    assert session1 != session2
    session1 = client1.transport.get_ekm_config._session
    session2 = client2.transport.get_ekm_config._session
    assert session1 != session2
    session1 = client1.transport.update_ekm_config._session
    session2 = client2.transport.update_ekm_config._session
    assert session1 != session2
    session1 = client1.transport.verify_connectivity._session
    session2 = client2.transport.verify_connectivity._session
    assert session1 != session2

def test_ekm_service_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.EkmServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_ekm_service_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.EkmServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.EkmServiceGrpcTransport, transports.EkmServiceGrpcAsyncIOTransport])
def test_ekm_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.EkmServiceGrpcTransport, transports.EkmServiceGrpcAsyncIOTransport])
def test_ekm_service_transport_channel_mtls_with_adc(transport_class):
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

def test_ekm_config_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}/ekmConfig'.format(project=project, location=location)
    actual = EkmServiceClient.ekm_config_path(project, location)
    assert expected == actual

def test_parse_ekm_config_path():
    if False:
        return 10
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = EkmServiceClient.ekm_config_path(**expected)
    actual = EkmServiceClient.parse_ekm_config_path(path)
    assert expected == actual

def test_ekm_connection_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    location = 'nudibranch'
    ekm_connection = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/ekmConnections/{ekm_connection}'.format(project=project, location=location, ekm_connection=ekm_connection)
    actual = EkmServiceClient.ekm_connection_path(project, location, ekm_connection)
    assert expected == actual

def test_parse_ekm_connection_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'mussel', 'location': 'winkle', 'ekm_connection': 'nautilus'}
    path = EkmServiceClient.ekm_connection_path(**expected)
    actual = EkmServiceClient.parse_ekm_connection_path(path)
    assert expected == actual

def test_service_path():
    if False:
        return 10
    project = 'scallop'
    location = 'abalone'
    namespace = 'squid'
    service = 'clam'
    expected = 'projects/{project}/locations/{location}/namespaces/{namespace}/services/{service}'.format(project=project, location=location, namespace=namespace, service=service)
    actual = EkmServiceClient.service_path(project, location, namespace, service)
    assert expected == actual

def test_parse_service_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'whelk', 'location': 'octopus', 'namespace': 'oyster', 'service': 'nudibranch'}
    path = EkmServiceClient.service_path(**expected)
    actual = EkmServiceClient.parse_service_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = EkmServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'mussel'}
    path = EkmServiceClient.common_billing_account_path(**expected)
    actual = EkmServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = EkmServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'nautilus'}
    path = EkmServiceClient.common_folder_path(**expected)
    actual = EkmServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = EkmServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'abalone'}
    path = EkmServiceClient.common_organization_path(**expected)
    actual = EkmServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = EkmServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'clam'}
    path = EkmServiceClient.common_project_path(**expected)
    actual = EkmServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = EkmServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = EkmServiceClient.common_location_path(**expected)
    actual = EkmServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.EkmServiceTransport, '_prep_wrapped_messages') as prep:
        client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.EkmServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = EkmServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = EkmServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = EkmServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(EkmServiceClient, transports.EkmServiceGrpcTransport), (EkmServiceAsyncClient, transports.EkmServiceGrpcAsyncIOTransport)])
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