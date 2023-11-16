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
from google.cloud.gkehub_v1.services.gke_hub import GkeHubAsyncClient, GkeHubClient, pagers, transports
from google.cloud.gkehub_v1.types import feature, membership, service

def client_cert_source_callback():
    if False:
        while True:
            i = 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        return 10
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert GkeHubClient._get_default_mtls_endpoint(None) is None
    assert GkeHubClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert GkeHubClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert GkeHubClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert GkeHubClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert GkeHubClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(GkeHubClient, 'grpc'), (GkeHubAsyncClient, 'grpc_asyncio'), (GkeHubClient, 'rest')])
def test_gke_hub_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('gkehub.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gkehub.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.GkeHubGrpcTransport, 'grpc'), (transports.GkeHubGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.GkeHubRestTransport, 'rest')])
def test_gke_hub_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(GkeHubClient, 'grpc'), (GkeHubAsyncClient, 'grpc_asyncio'), (GkeHubClient, 'rest')])
def test_gke_hub_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('gkehub.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gkehub.googleapis.com')

def test_gke_hub_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = GkeHubClient.get_transport_class()
    available_transports = [transports.GkeHubGrpcTransport, transports.GkeHubRestTransport]
    assert transport in available_transports
    transport = GkeHubClient.get_transport_class('grpc')
    assert transport == transports.GkeHubGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(GkeHubClient, transports.GkeHubGrpcTransport, 'grpc'), (GkeHubAsyncClient, transports.GkeHubGrpcAsyncIOTransport, 'grpc_asyncio'), (GkeHubClient, transports.GkeHubRestTransport, 'rest')])
@mock.patch.object(GkeHubClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GkeHubClient))
@mock.patch.object(GkeHubAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GkeHubAsyncClient))
def test_gke_hub_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(GkeHubClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(GkeHubClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(GkeHubClient, transports.GkeHubGrpcTransport, 'grpc', 'true'), (GkeHubAsyncClient, transports.GkeHubGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (GkeHubClient, transports.GkeHubGrpcTransport, 'grpc', 'false'), (GkeHubAsyncClient, transports.GkeHubGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (GkeHubClient, transports.GkeHubRestTransport, 'rest', 'true'), (GkeHubClient, transports.GkeHubRestTransport, 'rest', 'false')])
@mock.patch.object(GkeHubClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GkeHubClient))
@mock.patch.object(GkeHubAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GkeHubAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_gke_hub_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [GkeHubClient, GkeHubAsyncClient])
@mock.patch.object(GkeHubClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GkeHubClient))
@mock.patch.object(GkeHubAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GkeHubAsyncClient))
def test_gke_hub_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        while True:
            i = 10
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(GkeHubClient, transports.GkeHubGrpcTransport, 'grpc'), (GkeHubAsyncClient, transports.GkeHubGrpcAsyncIOTransport, 'grpc_asyncio'), (GkeHubClient, transports.GkeHubRestTransport, 'rest')])
def test_gke_hub_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(GkeHubClient, transports.GkeHubGrpcTransport, 'grpc', grpc_helpers), (GkeHubAsyncClient, transports.GkeHubGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (GkeHubClient, transports.GkeHubRestTransport, 'rest', None)])
def test_gke_hub_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_gke_hub_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.gkehub_v1.services.gke_hub.transports.GkeHubGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = GkeHubClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(GkeHubClient, transports.GkeHubGrpcTransport, 'grpc', grpc_helpers), (GkeHubAsyncClient, transports.GkeHubGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_gke_hub_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('gkehub.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='gkehub.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [service.ListMembershipsRequest, dict])
def test_list_memberships(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.return_value = service.ListMembershipsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_memberships(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListMembershipsRequest()
    assert isinstance(response, pagers.ListMembershipsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_memberships_empty_call():
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        client.list_memberships()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListMembershipsRequest()

@pytest.mark.asyncio
async def test_list_memberships_async(transport: str='grpc_asyncio', request_type=service.ListMembershipsRequest):
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListMembershipsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_memberships(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListMembershipsRequest()
    assert isinstance(response, pagers.ListMembershipsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_memberships_async_from_dict():
    await test_list_memberships_async(request_type=dict)

def test_list_memberships_field_headers():
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListMembershipsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.return_value = service.ListMembershipsResponse()
        client.list_memberships(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_memberships_field_headers_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListMembershipsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListMembershipsResponse())
        await client.list_memberships(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_memberships_flattened():
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.return_value = service.ListMembershipsResponse()
        client.list_memberships(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_memberships_flattened_error():
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_memberships(service.ListMembershipsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_memberships_flattened_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.return_value = service.ListMembershipsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListMembershipsResponse())
        response = await client.list_memberships(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_memberships_flattened_error_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_memberships(service.ListMembershipsRequest(), parent='parent_value')

def test_list_memberships_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.side_effect = (service.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership(), membership.Membership()], next_page_token='abc'), service.ListMembershipsResponse(resources=[], next_page_token='def'), service.ListMembershipsResponse(resources=[membership.Membership()], next_page_token='ghi'), service.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_memberships(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, membership.Membership) for i in results))

def test_list_memberships_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.side_effect = (service.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership(), membership.Membership()], next_page_token='abc'), service.ListMembershipsResponse(resources=[], next_page_token='def'), service.ListMembershipsResponse(resources=[membership.Membership()], next_page_token='ghi'), service.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership()]), RuntimeError)
        pages = list(client.list_memberships(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_memberships_async_pager():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_memberships), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership(), membership.Membership()], next_page_token='abc'), service.ListMembershipsResponse(resources=[], next_page_token='def'), service.ListMembershipsResponse(resources=[membership.Membership()], next_page_token='ghi'), service.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership()]), RuntimeError)
        async_pager = await client.list_memberships(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, membership.Membership) for i in responses))

@pytest.mark.asyncio
async def test_list_memberships_async_pages():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_memberships), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership(), membership.Membership()], next_page_token='abc'), service.ListMembershipsResponse(resources=[], next_page_token='def'), service.ListMembershipsResponse(resources=[membership.Membership()], next_page_token='ghi'), service.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_memberships(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListFeaturesRequest, dict])
def test_list_features(request_type, transport: str='grpc'):
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_features), '__call__') as call:
        call.return_value = service.ListFeaturesResponse(next_page_token='next_page_token_value')
        response = client.list_features(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListFeaturesRequest()
    assert isinstance(response, pagers.ListFeaturesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_features_empty_call():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_features), '__call__') as call:
        client.list_features()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListFeaturesRequest()

@pytest.mark.asyncio
async def test_list_features_async(transport: str='grpc_asyncio', request_type=service.ListFeaturesRequest):
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_features), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListFeaturesResponse(next_page_token='next_page_token_value'))
        response = await client.list_features(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListFeaturesRequest()
    assert isinstance(response, pagers.ListFeaturesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_features_async_from_dict():
    await test_list_features_async(request_type=dict)

def test_list_features_field_headers():
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListFeaturesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_features), '__call__') as call:
        call.return_value = service.ListFeaturesResponse()
        client.list_features(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_features_field_headers_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListFeaturesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_features), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListFeaturesResponse())
        await client.list_features(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_features_flattened():
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_features), '__call__') as call:
        call.return_value = service.ListFeaturesResponse()
        client.list_features(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_features_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_features(service.ListFeaturesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_features_flattened_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_features), '__call__') as call:
        call.return_value = service.ListFeaturesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListFeaturesResponse())
        response = await client.list_features(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_features_flattened_error_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_features(service.ListFeaturesRequest(), parent='parent_value')

def test_list_features_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_features), '__call__') as call:
        call.side_effect = (service.ListFeaturesResponse(resources=[feature.Feature(), feature.Feature(), feature.Feature()], next_page_token='abc'), service.ListFeaturesResponse(resources=[], next_page_token='def'), service.ListFeaturesResponse(resources=[feature.Feature()], next_page_token='ghi'), service.ListFeaturesResponse(resources=[feature.Feature(), feature.Feature()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_features(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, feature.Feature) for i in results))

def test_list_features_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_features), '__call__') as call:
        call.side_effect = (service.ListFeaturesResponse(resources=[feature.Feature(), feature.Feature(), feature.Feature()], next_page_token='abc'), service.ListFeaturesResponse(resources=[], next_page_token='def'), service.ListFeaturesResponse(resources=[feature.Feature()], next_page_token='ghi'), service.ListFeaturesResponse(resources=[feature.Feature(), feature.Feature()]), RuntimeError)
        pages = list(client.list_features(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_features_async_pager():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_features), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListFeaturesResponse(resources=[feature.Feature(), feature.Feature(), feature.Feature()], next_page_token='abc'), service.ListFeaturesResponse(resources=[], next_page_token='def'), service.ListFeaturesResponse(resources=[feature.Feature()], next_page_token='ghi'), service.ListFeaturesResponse(resources=[feature.Feature(), feature.Feature()]), RuntimeError)
        async_pager = await client.list_features(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, feature.Feature) for i in responses))

@pytest.mark.asyncio
async def test_list_features_async_pages():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_features), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListFeaturesResponse(resources=[feature.Feature(), feature.Feature(), feature.Feature()], next_page_token='abc'), service.ListFeaturesResponse(resources=[], next_page_token='def'), service.ListFeaturesResponse(resources=[feature.Feature()], next_page_token='ghi'), service.ListFeaturesResponse(resources=[feature.Feature(), feature.Feature()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_features(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetMembershipRequest, dict])
def test_get_membership(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_membership), '__call__') as call:
        call.return_value = membership.Membership(name='name_value', description='description_value', external_id='external_id_value', unique_id='unique_id_value')
        response = client.get_membership(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetMembershipRequest()
    assert isinstance(response, membership.Membership)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.external_id == 'external_id_value'
    assert response.unique_id == 'unique_id_value'

def test_get_membership_empty_call():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_membership), '__call__') as call:
        client.get_membership()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetMembershipRequest()

@pytest.mark.asyncio
async def test_get_membership_async(transport: str='grpc_asyncio', request_type=service.GetMembershipRequest):
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_membership), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(membership.Membership(name='name_value', description='description_value', external_id='external_id_value', unique_id='unique_id_value'))
        response = await client.get_membership(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetMembershipRequest()
    assert isinstance(response, membership.Membership)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.external_id == 'external_id_value'
    assert response.unique_id == 'unique_id_value'

@pytest.mark.asyncio
async def test_get_membership_async_from_dict():
    await test_get_membership_async(request_type=dict)

def test_get_membership_field_headers():
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetMembershipRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_membership), '__call__') as call:
        call.return_value = membership.Membership()
        client.get_membership(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_membership_field_headers_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetMembershipRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_membership), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(membership.Membership())
        await client.get_membership(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_membership_flattened():
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_membership), '__call__') as call:
        call.return_value = membership.Membership()
        client.get_membership(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_membership_flattened_error():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_membership(service.GetMembershipRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_membership_flattened_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_membership), '__call__') as call:
        call.return_value = membership.Membership()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(membership.Membership())
        response = await client.get_membership(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_membership_flattened_error_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_membership(service.GetMembershipRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.GetFeatureRequest, dict])
def test_get_feature(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_feature), '__call__') as call:
        call.return_value = feature.Feature(name='name_value')
        response = client.get_feature(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetFeatureRequest()
    assert isinstance(response, feature.Feature)
    assert response.name == 'name_value'

def test_get_feature_empty_call():
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_feature), '__call__') as call:
        client.get_feature()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetFeatureRequest()

@pytest.mark.asyncio
async def test_get_feature_async(transport: str='grpc_asyncio', request_type=service.GetFeatureRequest):
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_feature), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(feature.Feature(name='name_value'))
        response = await client.get_feature(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetFeatureRequest()
    assert isinstance(response, feature.Feature)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_feature_async_from_dict():
    await test_get_feature_async(request_type=dict)

def test_get_feature_field_headers():
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetFeatureRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_feature), '__call__') as call:
        call.return_value = feature.Feature()
        client.get_feature(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_feature_field_headers_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetFeatureRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_feature), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(feature.Feature())
        await client.get_feature(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_feature_flattened():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_feature), '__call__') as call:
        call.return_value = feature.Feature()
        client.get_feature(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_feature_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_feature(service.GetFeatureRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_feature_flattened_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_feature), '__call__') as call:
        call.return_value = feature.Feature()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(feature.Feature())
        response = await client.get_feature(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_feature_flattened_error_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_feature(service.GetFeatureRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.CreateMembershipRequest, dict])
def test_create_membership(request_type, transport: str='grpc'):
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_membership(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateMembershipRequest()
    assert isinstance(response, future.Future)

def test_create_membership_empty_call():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_membership), '__call__') as call:
        client.create_membership()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateMembershipRequest()

@pytest.mark.asyncio
async def test_create_membership_async(transport: str='grpc_asyncio', request_type=service.CreateMembershipRequest):
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_membership), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_membership(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateMembershipRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_membership_async_from_dict():
    await test_create_membership_async(request_type=dict)

def test_create_membership_field_headers():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateMembershipRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_membership(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_membership_field_headers_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateMembershipRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_membership), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_membership(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_membership_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_membership(parent='parent_value', resource=membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value'))), membership_id='membership_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].resource
        mock_val = membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value')))
        assert arg == mock_val
        arg = args[0].membership_id
        mock_val = 'membership_id_value'
        assert arg == mock_val

def test_create_membership_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_membership(service.CreateMembershipRequest(), parent='parent_value', resource=membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value'))), membership_id='membership_id_value')

@pytest.mark.asyncio
async def test_create_membership_flattened_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_membership(parent='parent_value', resource=membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value'))), membership_id='membership_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].resource
        mock_val = membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value')))
        assert arg == mock_val
        arg = args[0].membership_id
        mock_val = 'membership_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_membership_flattened_error_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_membership(service.CreateMembershipRequest(), parent='parent_value', resource=membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value'))), membership_id='membership_id_value')

@pytest.mark.parametrize('request_type', [service.CreateFeatureRequest, dict])
def test_create_feature(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_feature), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_feature(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateFeatureRequest()
    assert isinstance(response, future.Future)

def test_create_feature_empty_call():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_feature), '__call__') as call:
        client.create_feature()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateFeatureRequest()

@pytest.mark.asyncio
async def test_create_feature_async(transport: str='grpc_asyncio', request_type=service.CreateFeatureRequest):
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_feature), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_feature(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateFeatureRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_feature_async_from_dict():
    await test_create_feature_async(request_type=dict)

def test_create_feature_field_headers():
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateFeatureRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_feature), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_feature(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_feature_field_headers_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateFeatureRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_feature), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_feature(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_feature_flattened():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_feature), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_feature(parent='parent_value', resource=feature.Feature(name='name_value'), feature_id='feature_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].resource
        mock_val = feature.Feature(name='name_value')
        assert arg == mock_val
        arg = args[0].feature_id
        mock_val = 'feature_id_value'
        assert arg == mock_val

def test_create_feature_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_feature(service.CreateFeatureRequest(), parent='parent_value', resource=feature.Feature(name='name_value'), feature_id='feature_id_value')

@pytest.mark.asyncio
async def test_create_feature_flattened_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_feature), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_feature(parent='parent_value', resource=feature.Feature(name='name_value'), feature_id='feature_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].resource
        mock_val = feature.Feature(name='name_value')
        assert arg == mock_val
        arg = args[0].feature_id
        mock_val = 'feature_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_feature_flattened_error_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_feature(service.CreateFeatureRequest(), parent='parent_value', resource=feature.Feature(name='name_value'), feature_id='feature_id_value')

@pytest.mark.parametrize('request_type', [service.DeleteMembershipRequest, dict])
def test_delete_membership(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_membership(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteMembershipRequest()
    assert isinstance(response, future.Future)

def test_delete_membership_empty_call():
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_membership), '__call__') as call:
        client.delete_membership()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteMembershipRequest()

@pytest.mark.asyncio
async def test_delete_membership_async(transport: str='grpc_asyncio', request_type=service.DeleteMembershipRequest):
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_membership), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_membership(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteMembershipRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_membership_async_from_dict():
    await test_delete_membership_async(request_type=dict)

def test_delete_membership_field_headers():
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteMembershipRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_membership(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_membership_field_headers_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteMembershipRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_membership), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_membership(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_membership_flattened():
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_membership(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_membership_flattened_error():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_membership(service.DeleteMembershipRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_membership_flattened_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_membership(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_membership_flattened_error_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_membership(service.DeleteMembershipRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.DeleteFeatureRequest, dict])
def test_delete_feature(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_feature), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_feature(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteFeatureRequest()
    assert isinstance(response, future.Future)

def test_delete_feature_empty_call():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_feature), '__call__') as call:
        client.delete_feature()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteFeatureRequest()

@pytest.mark.asyncio
async def test_delete_feature_async(transport: str='grpc_asyncio', request_type=service.DeleteFeatureRequest):
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_feature), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_feature(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteFeatureRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_feature_async_from_dict():
    await test_delete_feature_async(request_type=dict)

def test_delete_feature_field_headers():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteFeatureRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_feature), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_feature(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_feature_field_headers_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteFeatureRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_feature), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_feature(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_feature_flattened():
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_feature), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_feature(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_feature_flattened_error():
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_feature(service.DeleteFeatureRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_feature_flattened_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_feature), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_feature(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_feature_flattened_error_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_feature(service.DeleteFeatureRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.UpdateMembershipRequest, dict])
def test_update_membership(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_membership(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateMembershipRequest()
    assert isinstance(response, future.Future)

def test_update_membership_empty_call():
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_membership), '__call__') as call:
        client.update_membership()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateMembershipRequest()

@pytest.mark.asyncio
async def test_update_membership_async(transport: str='grpc_asyncio', request_type=service.UpdateMembershipRequest):
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_membership), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_membership(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateMembershipRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_membership_async_from_dict():
    await test_update_membership_async(request_type=dict)

def test_update_membership_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateMembershipRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_membership(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_membership_field_headers_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateMembershipRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_membership), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_membership(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_membership_flattened():
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_membership(name='name_value', resource=membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value'))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].resource
        mock_val = membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value')))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_membership_flattened_error():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_membership(service.UpdateMembershipRequest(), name='name_value', resource=membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value'))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_membership_flattened_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_membership(name='name_value', resource=membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value'))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].resource
        mock_val = membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value')))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_membership_flattened_error_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_membership(service.UpdateMembershipRequest(), name='name_value', resource=membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value'))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.UpdateFeatureRequest, dict])
def test_update_feature(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_feature), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_feature(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateFeatureRequest()
    assert isinstance(response, future.Future)

def test_update_feature_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_feature), '__call__') as call:
        client.update_feature()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateFeatureRequest()

@pytest.mark.asyncio
async def test_update_feature_async(transport: str='grpc_asyncio', request_type=service.UpdateFeatureRequest):
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_feature), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_feature(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateFeatureRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_feature_async_from_dict():
    await test_update_feature_async(request_type=dict)

def test_update_feature_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateFeatureRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_feature), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_feature(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_feature_field_headers_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateFeatureRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_feature), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_feature(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_feature_flattened():
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_feature), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_feature(name='name_value', resource=feature.Feature(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].resource
        mock_val = feature.Feature(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_feature_flattened_error():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_feature(service.UpdateFeatureRequest(), name='name_value', resource=feature.Feature(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_feature_flattened_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_feature), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_feature(name='name_value', resource=feature.Feature(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].resource
        mock_val = feature.Feature(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_feature_flattened_error_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_feature(service.UpdateFeatureRequest(), name='name_value', resource=feature.Feature(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.GenerateConnectManifestRequest, dict])
def test_generate_connect_manifest(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_connect_manifest), '__call__') as call:
        call.return_value = service.GenerateConnectManifestResponse()
        response = client.generate_connect_manifest(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GenerateConnectManifestRequest()
    assert isinstance(response, service.GenerateConnectManifestResponse)

def test_generate_connect_manifest_empty_call():
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.generate_connect_manifest), '__call__') as call:
        client.generate_connect_manifest()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GenerateConnectManifestRequest()

@pytest.mark.asyncio
async def test_generate_connect_manifest_async(transport: str='grpc_asyncio', request_type=service.GenerateConnectManifestRequest):
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_connect_manifest), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.GenerateConnectManifestResponse())
        response = await client.generate_connect_manifest(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GenerateConnectManifestRequest()
    assert isinstance(response, service.GenerateConnectManifestResponse)

@pytest.mark.asyncio
async def test_generate_connect_manifest_async_from_dict():
    await test_generate_connect_manifest_async(request_type=dict)

def test_generate_connect_manifest_field_headers():
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GenerateConnectManifestRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.generate_connect_manifest), '__call__') as call:
        call.return_value = service.GenerateConnectManifestResponse()
        client.generate_connect_manifest(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_generate_connect_manifest_field_headers_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GenerateConnectManifestRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.generate_connect_manifest), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.GenerateConnectManifestResponse())
        await client.generate_connect_manifest(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.ListMembershipsRequest, dict])
def test_list_memberships_rest(request_type):
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListMembershipsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListMembershipsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_memberships(request)
    assert isinstance(response, pagers.ListMembershipsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_memberships_rest_required_fields(request_type=service.ListMembershipsRequest):
    if False:
        return 10
    transport_class = transports.GkeHubRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_memberships._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_memberships._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListMembershipsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListMembershipsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_memberships(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_memberships_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_memberships._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_memberships_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubRestInterceptor())
    client = GkeHubClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GkeHubRestInterceptor, 'post_list_memberships') as post, mock.patch.object(transports.GkeHubRestInterceptor, 'pre_list_memberships') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListMembershipsRequest.pb(service.ListMembershipsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListMembershipsResponse.to_json(service.ListMembershipsResponse())
        request = service.ListMembershipsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListMembershipsResponse()
        client.list_memberships(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_memberships_rest_bad_request(transport: str='rest', request_type=service.ListMembershipsRequest):
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_memberships(request)

def test_list_memberships_rest_flattened():
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListMembershipsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListMembershipsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_memberships(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/memberships' % client.transport._host, args[1])

def test_list_memberships_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_memberships(service.ListMembershipsRequest(), parent='parent_value')

def test_list_memberships_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership(), membership.Membership()], next_page_token='abc'), service.ListMembershipsResponse(resources=[], next_page_token='def'), service.ListMembershipsResponse(resources=[membership.Membership()], next_page_token='ghi'), service.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership()]))
        response = response + response
        response = tuple((service.ListMembershipsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_memberships(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, membership.Membership) for i in results))
        pages = list(client.list_memberships(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.ListFeaturesRequest, dict])
def test_list_features_rest(request_type):
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListFeaturesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListFeaturesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_features(request)
    assert isinstance(response, pagers.ListFeaturesPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_features_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubRestInterceptor())
    client = GkeHubClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GkeHubRestInterceptor, 'post_list_features') as post, mock.patch.object(transports.GkeHubRestInterceptor, 'pre_list_features') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListFeaturesRequest.pb(service.ListFeaturesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListFeaturesResponse.to_json(service.ListFeaturesResponse())
        request = service.ListFeaturesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListFeaturesResponse()
        client.list_features(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_features_rest_bad_request(transport: str='rest', request_type=service.ListFeaturesRequest):
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_features(request)

def test_list_features_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListFeaturesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListFeaturesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_features(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/features' % client.transport._host, args[1])

def test_list_features_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_features(service.ListFeaturesRequest(), parent='parent_value')

def test_list_features_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListFeaturesResponse(resources=[feature.Feature(), feature.Feature(), feature.Feature()], next_page_token='abc'), service.ListFeaturesResponse(resources=[], next_page_token='def'), service.ListFeaturesResponse(resources=[feature.Feature()], next_page_token='ghi'), service.ListFeaturesResponse(resources=[feature.Feature(), feature.Feature()]))
        response = response + response
        response = tuple((service.ListFeaturesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_features(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, feature.Feature) for i in results))
        pages = list(client.list_features(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetMembershipRequest, dict])
def test_get_membership_rest(request_type):
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = membership.Membership(name='name_value', description='description_value', external_id='external_id_value', unique_id='unique_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = membership.Membership.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_membership(request)
    assert isinstance(response, membership.Membership)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.external_id == 'external_id_value'
    assert response.unique_id == 'unique_id_value'

def test_get_membership_rest_required_fields(request_type=service.GetMembershipRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.GkeHubRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_membership._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_membership._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = membership.Membership()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = membership.Membership.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_membership(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_membership_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_membership._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_membership_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubRestInterceptor())
    client = GkeHubClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GkeHubRestInterceptor, 'post_get_membership') as post, mock.patch.object(transports.GkeHubRestInterceptor, 'pre_get_membership') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetMembershipRequest.pb(service.GetMembershipRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = membership.Membership.to_json(membership.Membership())
        request = service.GetMembershipRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = membership.Membership()
        client.get_membership(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_membership_rest_bad_request(transport: str='rest', request_type=service.GetMembershipRequest):
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_membership(request)

def test_get_membership_rest_flattened():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = membership.Membership()
        sample_request = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = membership.Membership.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_membership(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/memberships/*}' % client.transport._host, args[1])

def test_get_membership_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_membership(service.GetMembershipRequest(), name='name_value')

def test_get_membership_rest_error():
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetFeatureRequest, dict])
def test_get_feature_rest(request_type):
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/features/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = feature.Feature(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = feature.Feature.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_feature(request)
    assert isinstance(response, feature.Feature)
    assert response.name == 'name_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_feature_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubRestInterceptor())
    client = GkeHubClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GkeHubRestInterceptor, 'post_get_feature') as post, mock.patch.object(transports.GkeHubRestInterceptor, 'pre_get_feature') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetFeatureRequest.pb(service.GetFeatureRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = feature.Feature.to_json(feature.Feature())
        request = service.GetFeatureRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = feature.Feature()
        client.get_feature(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_feature_rest_bad_request(transport: str='rest', request_type=service.GetFeatureRequest):
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/features/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_feature(request)

def test_get_feature_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = feature.Feature()
        sample_request = {'name': 'projects/sample1/locations/sample2/features/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = feature.Feature.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_feature(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/features/*}' % client.transport._host, args[1])

def test_get_feature_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_feature(service.GetFeatureRequest(), name='name_value')

def test_get_feature_rest_error():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateMembershipRequest, dict])
def test_create_membership_rest(request_type):
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['resource'] = {'endpoint': {'gke_cluster': {'resource_link': 'resource_link_value', 'cluster_missing': True}, 'kubernetes_metadata': {'kubernetes_api_server_version': 'kubernetes_api_server_version_value', 'node_provider_id': 'node_provider_id_value', 'node_count': 1070, 'vcpu_count': 1094, 'memory_mb': 967, 'update_time': {'seconds': 751, 'nanos': 543}}, 'kubernetes_resource': {'membership_cr_manifest': 'membership_cr_manifest_value', 'membership_resources': [{'manifest': 'manifest_value', 'cluster_scoped': True}], 'connect_resources': {}, 'resource_options': {'connect_version': 'connect_version_value', 'v1beta1_crd': True, 'k8s_version': 'k8s_version_value'}}, 'google_managed': True}, 'name': 'name_value', 'labels': {}, 'description': 'description_value', 'state': {'code': 1}, 'create_time': {}, 'update_time': {}, 'delete_time': {}, 'external_id': 'external_id_value', 'last_connection_time': {}, 'unique_id': 'unique_id_value', 'authority': {'issuer': 'issuer_value', 'workload_identity_pool': 'workload_identity_pool_value', 'identity_provider': 'identity_provider_value', 'oidc_jwks': b'oidc_jwks_blob'}, 'monitoring_config': {'project_id': 'project_id_value', 'location': 'location_value', 'cluster': 'cluster_value', 'kubernetes_metrics_prefix': 'kubernetes_metrics_prefix_value', 'cluster_hash': 'cluster_hash_value'}}
    test_field = service.CreateMembershipRequest.meta.fields['resource']

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
    for (field, value) in request_init['resource'].items():
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
                for i in range(0, len(request_init['resource'][field])):
                    del request_init['resource'][field][i][subfield]
            else:
                del request_init['resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_membership(request)
    assert response.operation.name == 'operations/spam'

def test_create_membership_rest_required_fields(request_type=service.CreateMembershipRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.GkeHubRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['membership_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'membershipId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_membership._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'membershipId' in jsonified_request
    assert jsonified_request['membershipId'] == request_init['membership_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['membershipId'] = 'membership_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_membership._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('membership_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'membershipId' in jsonified_request
    assert jsonified_request['membershipId'] == 'membership_id_value'
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_membership(request)
            expected_params = [('membershipId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_membership_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_membership._get_unset_required_fields({})
    assert set(unset_fields) == set(('membershipId', 'requestId')) & set(('parent', 'membershipId', 'resource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_membership_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubRestInterceptor())
    client = GkeHubClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.GkeHubRestInterceptor, 'post_create_membership') as post, mock.patch.object(transports.GkeHubRestInterceptor, 'pre_create_membership') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateMembershipRequest.pb(service.CreateMembershipRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateMembershipRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_membership(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_membership_rest_bad_request(transport: str='rest', request_type=service.CreateMembershipRequest):
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_membership(request)

def test_create_membership_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', resource=membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value'))), membership_id='membership_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_membership(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/memberships' % client.transport._host, args[1])

def test_create_membership_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_membership(service.CreateMembershipRequest(), parent='parent_value', resource=membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value'))), membership_id='membership_id_value')

def test_create_membership_rest_error():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateFeatureRequest, dict])
def test_create_feature_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['resource'] = {'name': 'name_value', 'labels': {}, 'resource_state': {'state': 1}, 'spec': {'multiclusteringress': {'config_membership': 'config_membership_value'}}, 'membership_specs': {}, 'state': {'state': {'code': 1, 'description': 'description_value', 'update_time': {'seconds': 751, 'nanos': 543}}}, 'membership_states': {}, 'create_time': {}, 'update_time': {}, 'delete_time': {}}
    test_field = service.CreateFeatureRequest.meta.fields['resource']

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
    for (field, value) in request_init['resource'].items():
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
                for i in range(0, len(request_init['resource'][field])):
                    del request_init['resource'][field][i][subfield]
            else:
                del request_init['resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_feature(request)
    assert response.operation.name == 'operations/spam'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_feature_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubRestInterceptor())
    client = GkeHubClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.GkeHubRestInterceptor, 'post_create_feature') as post, mock.patch.object(transports.GkeHubRestInterceptor, 'pre_create_feature') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateFeatureRequest.pb(service.CreateFeatureRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateFeatureRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_feature(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_feature_rest_bad_request(transport: str='rest', request_type=service.CreateFeatureRequest):
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_feature(request)

def test_create_feature_rest_flattened():
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', resource=feature.Feature(name='name_value'), feature_id='feature_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_feature(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/features' % client.transport._host, args[1])

def test_create_feature_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_feature(service.CreateFeatureRequest(), parent='parent_value', resource=feature.Feature(name='name_value'), feature_id='feature_id_value')

def test_create_feature_rest_error():
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteMembershipRequest, dict])
def test_delete_membership_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_membership(request)
    assert response.operation.name == 'operations/spam'

def test_delete_membership_rest_required_fields(request_type=service.DeleteMembershipRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.GkeHubRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_membership._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_membership._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('force', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_membership(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_membership_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_membership._get_unset_required_fields({})
    assert set(unset_fields) == set(('force', 'requestId')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_membership_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubRestInterceptor())
    client = GkeHubClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.GkeHubRestInterceptor, 'post_delete_membership') as post, mock.patch.object(transports.GkeHubRestInterceptor, 'pre_delete_membership') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DeleteMembershipRequest.pb(service.DeleteMembershipRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.DeleteMembershipRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_membership(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_membership_rest_bad_request(transport: str='rest', request_type=service.DeleteMembershipRequest):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_membership(request)

def test_delete_membership_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_membership(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/memberships/*}' % client.transport._host, args[1])

def test_delete_membership_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_membership(service.DeleteMembershipRequest(), name='name_value')

def test_delete_membership_rest_error():
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteFeatureRequest, dict])
def test_delete_feature_rest(request_type):
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/features/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_feature(request)
    assert response.operation.name == 'operations/spam'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_feature_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubRestInterceptor())
    client = GkeHubClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.GkeHubRestInterceptor, 'post_delete_feature') as post, mock.patch.object(transports.GkeHubRestInterceptor, 'pre_delete_feature') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DeleteFeatureRequest.pb(service.DeleteFeatureRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.DeleteFeatureRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_feature(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_feature_rest_bad_request(transport: str='rest', request_type=service.DeleteFeatureRequest):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/features/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_feature(request)

def test_delete_feature_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/features/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_feature(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/features/*}' % client.transport._host, args[1])

def test_delete_feature_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_feature(service.DeleteFeatureRequest(), name='name_value')

def test_delete_feature_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateMembershipRequest, dict])
def test_update_membership_rest(request_type):
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
    request_init['resource'] = {'endpoint': {'gke_cluster': {'resource_link': 'resource_link_value', 'cluster_missing': True}, 'kubernetes_metadata': {'kubernetes_api_server_version': 'kubernetes_api_server_version_value', 'node_provider_id': 'node_provider_id_value', 'node_count': 1070, 'vcpu_count': 1094, 'memory_mb': 967, 'update_time': {'seconds': 751, 'nanos': 543}}, 'kubernetes_resource': {'membership_cr_manifest': 'membership_cr_manifest_value', 'membership_resources': [{'manifest': 'manifest_value', 'cluster_scoped': True}], 'connect_resources': {}, 'resource_options': {'connect_version': 'connect_version_value', 'v1beta1_crd': True, 'k8s_version': 'k8s_version_value'}}, 'google_managed': True}, 'name': 'name_value', 'labels': {}, 'description': 'description_value', 'state': {'code': 1}, 'create_time': {}, 'update_time': {}, 'delete_time': {}, 'external_id': 'external_id_value', 'last_connection_time': {}, 'unique_id': 'unique_id_value', 'authority': {'issuer': 'issuer_value', 'workload_identity_pool': 'workload_identity_pool_value', 'identity_provider': 'identity_provider_value', 'oidc_jwks': b'oidc_jwks_blob'}, 'monitoring_config': {'project_id': 'project_id_value', 'location': 'location_value', 'cluster': 'cluster_value', 'kubernetes_metrics_prefix': 'kubernetes_metrics_prefix_value', 'cluster_hash': 'cluster_hash_value'}}
    test_field = service.UpdateMembershipRequest.meta.fields['resource']

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
    for (field, value) in request_init['resource'].items():
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
                for i in range(0, len(request_init['resource'][field])):
                    del request_init['resource'][field][i][subfield]
            else:
                del request_init['resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_membership(request)
    assert response.operation.name == 'operations/spam'

def test_update_membership_rest_required_fields(request_type=service.UpdateMembershipRequest):
    if False:
        print('Hello World!')
    transport_class = transports.GkeHubRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_membership._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_membership._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_membership(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_membership_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_membership._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('name', 'updateMask', 'resource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_membership_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubRestInterceptor())
    client = GkeHubClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.GkeHubRestInterceptor, 'post_update_membership') as post, mock.patch.object(transports.GkeHubRestInterceptor, 'pre_update_membership') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateMembershipRequest.pb(service.UpdateMembershipRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.UpdateMembershipRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_membership(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_membership_rest_bad_request(transport: str='rest', request_type=service.UpdateMembershipRequest):
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_membership(request)

def test_update_membership_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
        mock_args = dict(name='name_value', resource=membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value'))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_membership(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/memberships/*}' % client.transport._host, args[1])

def test_update_membership_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_membership(service.UpdateMembershipRequest(), name='name_value', resource=membership.Membership(endpoint=membership.MembershipEndpoint(gke_cluster=membership.GkeCluster(resource_link='resource_link_value'))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_membership_rest_error():
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateFeatureRequest, dict])
def test_update_feature_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/features/sample3'}
    request_init['resource'] = {'name': 'name_value', 'labels': {}, 'resource_state': {'state': 1}, 'spec': {'multiclusteringress': {'config_membership': 'config_membership_value'}}, 'membership_specs': {}, 'state': {'state': {'code': 1, 'description': 'description_value', 'update_time': {'seconds': 751, 'nanos': 543}}}, 'membership_states': {}, 'create_time': {}, 'update_time': {}, 'delete_time': {}}
    test_field = service.UpdateFeatureRequest.meta.fields['resource']

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
    for (field, value) in request_init['resource'].items():
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
                for i in range(0, len(request_init['resource'][field])):
                    del request_init['resource'][field][i][subfield]
            else:
                del request_init['resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_feature(request)
    assert response.operation.name == 'operations/spam'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_feature_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubRestInterceptor())
    client = GkeHubClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.GkeHubRestInterceptor, 'post_update_feature') as post, mock.patch.object(transports.GkeHubRestInterceptor, 'pre_update_feature') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateFeatureRequest.pb(service.UpdateFeatureRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.UpdateFeatureRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_feature(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_feature_rest_bad_request(transport: str='rest', request_type=service.UpdateFeatureRequest):
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/features/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_feature(request)

def test_update_feature_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/features/sample3'}
        mock_args = dict(name='name_value', resource=feature.Feature(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_feature(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/features/*}' % client.transport._host, args[1])

def test_update_feature_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_feature(service.UpdateFeatureRequest(), name='name_value', resource=feature.Feature(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_feature_rest_error():
    if False:
        while True:
            i = 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GenerateConnectManifestRequest, dict])
def test_generate_connect_manifest_rest(request_type):
    if False:
        print('Hello World!')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.GenerateConnectManifestResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = service.GenerateConnectManifestResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.generate_connect_manifest(request)
    assert isinstance(response, service.GenerateConnectManifestResponse)

def test_generate_connect_manifest_rest_required_fields(request_type=service.GenerateConnectManifestRequest):
    if False:
        print('Hello World!')
    transport_class = transports.GkeHubRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_connect_manifest._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_connect_manifest._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('image_pull_secret_content', 'is_upgrade', 'namespace', 'proxy', 'registry', 'version'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.GenerateConnectManifestResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.GenerateConnectManifestResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.generate_connect_manifest(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_generate_connect_manifest_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.generate_connect_manifest._get_unset_required_fields({})
    assert set(unset_fields) == set(('imagePullSecretContent', 'isUpgrade', 'namespace', 'proxy', 'registry', 'version')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_generate_connect_manifest_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.GkeHubRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubRestInterceptor())
    client = GkeHubClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GkeHubRestInterceptor, 'post_generate_connect_manifest') as post, mock.patch.object(transports.GkeHubRestInterceptor, 'pre_generate_connect_manifest') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GenerateConnectManifestRequest.pb(service.GenerateConnectManifestRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.GenerateConnectManifestResponse.to_json(service.GenerateConnectManifestResponse())
        request = service.GenerateConnectManifestRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.GenerateConnectManifestResponse()
        client.generate_connect_manifest(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_generate_connect_manifest_rest_bad_request(transport: str='rest', request_type=service.GenerateConnectManifestRequest):
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.generate_connect_manifest(request)

def test_generate_connect_manifest_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.GkeHubGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.GkeHubGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = GkeHubClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.GkeHubGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = GkeHubClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = GkeHubClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.GkeHubGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = GkeHubClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.GkeHubGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = GkeHubClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.GkeHubGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.GkeHubGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.GkeHubGrpcTransport, transports.GkeHubGrpcAsyncIOTransport, transports.GkeHubRestTransport])
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
        print('Hello World!')
    transport = GkeHubClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.GkeHubGrpcTransport)

def test_gke_hub_base_transport_error():
    if False:
        return 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.GkeHubTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_gke_hub_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.gkehub_v1.services.gke_hub.transports.GkeHubTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.GkeHubTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_memberships', 'list_features', 'get_membership', 'get_feature', 'create_membership', 'create_feature', 'delete_membership', 'delete_feature', 'update_membership', 'update_feature', 'generate_connect_manifest')
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

def test_gke_hub_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.gkehub_v1.services.gke_hub.transports.GkeHubTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.GkeHubTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_gke_hub_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.gkehub_v1.services.gke_hub.transports.GkeHubTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.GkeHubTransport()
        adc.assert_called_once()

def test_gke_hub_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        GkeHubClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.GkeHubGrpcTransport, transports.GkeHubGrpcAsyncIOTransport])
def test_gke_hub_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.GkeHubGrpcTransport, transports.GkeHubGrpcAsyncIOTransport, transports.GkeHubRestTransport])
def test_gke_hub_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.GkeHubGrpcTransport, grpc_helpers), (transports.GkeHubGrpcAsyncIOTransport, grpc_helpers_async)])
def test_gke_hub_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('gkehub.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='gkehub.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.GkeHubGrpcTransport, transports.GkeHubGrpcAsyncIOTransport])
def test_gke_hub_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_gke_hub_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.GkeHubRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_gke_hub_rest_lro_client():
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_gke_hub_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='gkehub.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('gkehub.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gkehub.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_gke_hub_host_with_port(transport_name):
    if False:
        return 10
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='gkehub.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('gkehub.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gkehub.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_gke_hub_client_transport_session_collision(transport_name):
    if False:
        while True:
            i = 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = GkeHubClient(credentials=creds1, transport=transport_name)
    client2 = GkeHubClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_memberships._session
    session2 = client2.transport.list_memberships._session
    assert session1 != session2
    session1 = client1.transport.list_features._session
    session2 = client2.transport.list_features._session
    assert session1 != session2
    session1 = client1.transport.get_membership._session
    session2 = client2.transport.get_membership._session
    assert session1 != session2
    session1 = client1.transport.get_feature._session
    session2 = client2.transport.get_feature._session
    assert session1 != session2
    session1 = client1.transport.create_membership._session
    session2 = client2.transport.create_membership._session
    assert session1 != session2
    session1 = client1.transport.create_feature._session
    session2 = client2.transport.create_feature._session
    assert session1 != session2
    session1 = client1.transport.delete_membership._session
    session2 = client2.transport.delete_membership._session
    assert session1 != session2
    session1 = client1.transport.delete_feature._session
    session2 = client2.transport.delete_feature._session
    assert session1 != session2
    session1 = client1.transport.update_membership._session
    session2 = client2.transport.update_membership._session
    assert session1 != session2
    session1 = client1.transport.update_feature._session
    session2 = client2.transport.update_feature._session
    assert session1 != session2
    session1 = client1.transport.generate_connect_manifest._session
    session2 = client2.transport.generate_connect_manifest._session
    assert session1 != session2

def test_gke_hub_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.GkeHubGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_gke_hub_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.GkeHubGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.GkeHubGrpcTransport, transports.GkeHubGrpcAsyncIOTransport])
def test_gke_hub_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.GkeHubGrpcTransport, transports.GkeHubGrpcAsyncIOTransport])
def test_gke_hub_transport_channel_mtls_with_adc(transport_class):
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

def test_gke_hub_grpc_lro_client():
    if False:
        i = 10
        return i + 15
    client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_gke_hub_grpc_lro_async_client():
    if False:
        while True:
            i = 10
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_feature_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    feature = 'whelk'
    expected = 'projects/{project}/locations/{location}/features/{feature}'.format(project=project, location=location, feature=feature)
    actual = GkeHubClient.feature_path(project, location, feature)
    assert expected == actual

def test_parse_feature_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'location': 'oyster', 'feature': 'nudibranch'}
    path = GkeHubClient.feature_path(**expected)
    actual = GkeHubClient.parse_feature_path(path)
    assert expected == actual

def test_membership_path():
    if False:
        return 10
    project = 'cuttlefish'
    location = 'mussel'
    membership = 'winkle'
    expected = 'projects/{project}/locations/{location}/memberships/{membership}'.format(project=project, location=location, membership=membership)
    actual = GkeHubClient.membership_path(project, location, membership)
    assert expected == actual

def test_parse_membership_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nautilus', 'location': 'scallop', 'membership': 'abalone'}
    path = GkeHubClient.membership_path(**expected)
    actual = GkeHubClient.parse_membership_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = GkeHubClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'clam'}
    path = GkeHubClient.common_billing_account_path(**expected)
    actual = GkeHubClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = GkeHubClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'octopus'}
    path = GkeHubClient.common_folder_path(**expected)
    actual = GkeHubClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = GkeHubClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'nudibranch'}
    path = GkeHubClient.common_organization_path(**expected)
    actual = GkeHubClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = GkeHubClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'mussel'}
    path = GkeHubClient.common_project_path(**expected)
    actual = GkeHubClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = GkeHubClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = GkeHubClient.common_location_path(**expected)
    actual = GkeHubClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.GkeHubTransport, '_prep_wrapped_messages') as prep:
        client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.GkeHubTransport, '_prep_wrapped_messages') as prep:
        transport_class = GkeHubClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = GkeHubAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = GkeHubClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(GkeHubClient, transports.GkeHubGrpcTransport), (GkeHubAsyncClient, transports.GkeHubGrpcAsyncIOTransport)])
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