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
from google.rpc import status_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.gkehub_v1beta1.services.gke_hub_membership_service import GkeHubMembershipServiceAsyncClient, GkeHubMembershipServiceClient, pagers, transports
from google.cloud.gkehub_v1beta1.types import membership

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        print('Hello World!')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        i = 10
        return i + 15
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert GkeHubMembershipServiceClient._get_default_mtls_endpoint(None) is None
    assert GkeHubMembershipServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert GkeHubMembershipServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert GkeHubMembershipServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert GkeHubMembershipServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert GkeHubMembershipServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(GkeHubMembershipServiceClient, 'grpc'), (GkeHubMembershipServiceAsyncClient, 'grpc_asyncio'), (GkeHubMembershipServiceClient, 'rest')])
def test_gke_hub_membership_service_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('gkehub.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gkehub.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.GkeHubMembershipServiceGrpcTransport, 'grpc'), (transports.GkeHubMembershipServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.GkeHubMembershipServiceRestTransport, 'rest')])
def test_gke_hub_membership_service_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(GkeHubMembershipServiceClient, 'grpc'), (GkeHubMembershipServiceAsyncClient, 'grpc_asyncio'), (GkeHubMembershipServiceClient, 'rest')])
def test_gke_hub_membership_service_client_from_service_account_file(client_class, transport_name):
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

def test_gke_hub_membership_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = GkeHubMembershipServiceClient.get_transport_class()
    available_transports = [transports.GkeHubMembershipServiceGrpcTransport, transports.GkeHubMembershipServiceRestTransport]
    assert transport in available_transports
    transport = GkeHubMembershipServiceClient.get_transport_class('grpc')
    assert transport == transports.GkeHubMembershipServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(GkeHubMembershipServiceClient, transports.GkeHubMembershipServiceGrpcTransport, 'grpc'), (GkeHubMembershipServiceAsyncClient, transports.GkeHubMembershipServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (GkeHubMembershipServiceClient, transports.GkeHubMembershipServiceRestTransport, 'rest')])
@mock.patch.object(GkeHubMembershipServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GkeHubMembershipServiceClient))
@mock.patch.object(GkeHubMembershipServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GkeHubMembershipServiceAsyncClient))
def test_gke_hub_membership_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(GkeHubMembershipServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(GkeHubMembershipServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(GkeHubMembershipServiceClient, transports.GkeHubMembershipServiceGrpcTransport, 'grpc', 'true'), (GkeHubMembershipServiceAsyncClient, transports.GkeHubMembershipServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (GkeHubMembershipServiceClient, transports.GkeHubMembershipServiceGrpcTransport, 'grpc', 'false'), (GkeHubMembershipServiceAsyncClient, transports.GkeHubMembershipServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (GkeHubMembershipServiceClient, transports.GkeHubMembershipServiceRestTransport, 'rest', 'true'), (GkeHubMembershipServiceClient, transports.GkeHubMembershipServiceRestTransport, 'rest', 'false')])
@mock.patch.object(GkeHubMembershipServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GkeHubMembershipServiceClient))
@mock.patch.object(GkeHubMembershipServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GkeHubMembershipServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_gke_hub_membership_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [GkeHubMembershipServiceClient, GkeHubMembershipServiceAsyncClient])
@mock.patch.object(GkeHubMembershipServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GkeHubMembershipServiceClient))
@mock.patch.object(GkeHubMembershipServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GkeHubMembershipServiceAsyncClient))
def test_gke_hub_membership_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(GkeHubMembershipServiceClient, transports.GkeHubMembershipServiceGrpcTransport, 'grpc'), (GkeHubMembershipServiceAsyncClient, transports.GkeHubMembershipServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (GkeHubMembershipServiceClient, transports.GkeHubMembershipServiceRestTransport, 'rest')])
def test_gke_hub_membership_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(GkeHubMembershipServiceClient, transports.GkeHubMembershipServiceGrpcTransport, 'grpc', grpc_helpers), (GkeHubMembershipServiceAsyncClient, transports.GkeHubMembershipServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (GkeHubMembershipServiceClient, transports.GkeHubMembershipServiceRestTransport, 'rest', None)])
def test_gke_hub_membership_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_gke_hub_membership_service_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.gkehub_v1beta1.services.gke_hub_membership_service.transports.GkeHubMembershipServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = GkeHubMembershipServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(GkeHubMembershipServiceClient, transports.GkeHubMembershipServiceGrpcTransport, 'grpc', grpc_helpers), (GkeHubMembershipServiceAsyncClient, transports.GkeHubMembershipServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_gke_hub_membership_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('gkehub.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='gkehub.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [membership.ListMembershipsRequest, dict])
def test_list_memberships(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.return_value = membership.ListMembershipsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_memberships(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.ListMembershipsRequest()
    assert isinstance(response, pagers.ListMembershipsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_memberships_empty_call():
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        client.list_memberships()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.ListMembershipsRequest()

@pytest.mark.asyncio
async def test_list_memberships_async(transport: str='grpc_asyncio', request_type=membership.ListMembershipsRequest):
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(membership.ListMembershipsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_memberships(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.ListMembershipsRequest()
    assert isinstance(response, pagers.ListMembershipsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_memberships_async_from_dict():
    await test_list_memberships_async(request_type=dict)

def test_list_memberships_field_headers():
    if False:
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.ListMembershipsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.return_value = membership.ListMembershipsResponse()
        client.list_memberships(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_memberships_field_headers_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.ListMembershipsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(membership.ListMembershipsResponse())
        await client.list_memberships(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_memberships_flattened():
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.return_value = membership.ListMembershipsResponse()
        client.list_memberships(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_memberships_flattened_error():
    if False:
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_memberships(membership.ListMembershipsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_memberships_flattened_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.return_value = membership.ListMembershipsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(membership.ListMembershipsResponse())
        response = await client.list_memberships(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_memberships_flattened_error_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_memberships(membership.ListMembershipsRequest(), parent='parent_value')

def test_list_memberships_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.side_effect = (membership.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership(), membership.Membership()], next_page_token='abc'), membership.ListMembershipsResponse(resources=[], next_page_token='def'), membership.ListMembershipsResponse(resources=[membership.Membership()], next_page_token='ghi'), membership.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_memberships(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, membership.Membership) for i in results))

def test_list_memberships_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_memberships), '__call__') as call:
        call.side_effect = (membership.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership(), membership.Membership()], next_page_token='abc'), membership.ListMembershipsResponse(resources=[], next_page_token='def'), membership.ListMembershipsResponse(resources=[membership.Membership()], next_page_token='ghi'), membership.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership()]), RuntimeError)
        pages = list(client.list_memberships(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_memberships_async_pager():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_memberships), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (membership.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership(), membership.Membership()], next_page_token='abc'), membership.ListMembershipsResponse(resources=[], next_page_token='def'), membership.ListMembershipsResponse(resources=[membership.Membership()], next_page_token='ghi'), membership.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership()]), RuntimeError)
        async_pager = await client.list_memberships(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, membership.Membership) for i in responses))

@pytest.mark.asyncio
async def test_list_memberships_async_pages():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_memberships), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (membership.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership(), membership.Membership()], next_page_token='abc'), membership.ListMembershipsResponse(resources=[], next_page_token='def'), membership.ListMembershipsResponse(resources=[membership.Membership()], next_page_token='ghi'), membership.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_memberships(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [membership.GetMembershipRequest, dict])
def test_get_membership(request_type, transport: str='grpc'):
    if False:
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_membership), '__call__') as call:
        call.return_value = membership.Membership(name='name_value', description='description_value', external_id='external_id_value', unique_id='unique_id_value', infrastructure_type=membership.Membership.InfrastructureType.ON_PREM)
        response = client.get_membership(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.GetMembershipRequest()
    assert isinstance(response, membership.Membership)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.external_id == 'external_id_value'
    assert response.unique_id == 'unique_id_value'
    assert response.infrastructure_type == membership.Membership.InfrastructureType.ON_PREM

def test_get_membership_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_membership), '__call__') as call:
        client.get_membership()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.GetMembershipRequest()

@pytest.mark.asyncio
async def test_get_membership_async(transport: str='grpc_asyncio', request_type=membership.GetMembershipRequest):
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_membership), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(membership.Membership(name='name_value', description='description_value', external_id='external_id_value', unique_id='unique_id_value', infrastructure_type=membership.Membership.InfrastructureType.ON_PREM))
        response = await client.get_membership(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.GetMembershipRequest()
    assert isinstance(response, membership.Membership)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.external_id == 'external_id_value'
    assert response.unique_id == 'unique_id_value'
    assert response.infrastructure_type == membership.Membership.InfrastructureType.ON_PREM

@pytest.mark.asyncio
async def test_get_membership_async_from_dict():
    await test_get_membership_async(request_type=dict)

def test_get_membership_field_headers():
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.GetMembershipRequest()
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.GetMembershipRequest()
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
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_membership(membership.GetMembershipRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_membership_flattened_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_membership(membership.GetMembershipRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [membership.CreateMembershipRequest, dict])
def test_create_membership(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_membership(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.CreateMembershipRequest()
    assert isinstance(response, future.Future)

def test_create_membership_empty_call():
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_membership), '__call__') as call:
        client.create_membership()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.CreateMembershipRequest()

@pytest.mark.asyncio
async def test_create_membership_async(transport: str='grpc_asyncio', request_type=membership.CreateMembershipRequest):
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_membership), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_membership(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.CreateMembershipRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_membership_async_from_dict():
    await test_create_membership_async(request_type=dict)

def test_create_membership_field_headers():
    if False:
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.CreateMembershipRequest()
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.CreateMembershipRequest()
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
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_membership(parent='parent_value', resource=membership.Membership(name='name_value'), membership_id='membership_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].resource
        mock_val = membership.Membership(name='name_value')
        assert arg == mock_val
        arg = args[0].membership_id
        mock_val = 'membership_id_value'
        assert arg == mock_val

def test_create_membership_flattened_error():
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_membership(membership.CreateMembershipRequest(), parent='parent_value', resource=membership.Membership(name='name_value'), membership_id='membership_id_value')

@pytest.mark.asyncio
async def test_create_membership_flattened_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_membership(parent='parent_value', resource=membership.Membership(name='name_value'), membership_id='membership_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].resource
        mock_val = membership.Membership(name='name_value')
        assert arg == mock_val
        arg = args[0].membership_id
        mock_val = 'membership_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_membership_flattened_error_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_membership(membership.CreateMembershipRequest(), parent='parent_value', resource=membership.Membership(name='name_value'), membership_id='membership_id_value')

@pytest.mark.parametrize('request_type', [membership.DeleteMembershipRequest, dict])
def test_delete_membership(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_membership(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.DeleteMembershipRequest()
    assert isinstance(response, future.Future)

def test_delete_membership_empty_call():
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_membership), '__call__') as call:
        client.delete_membership()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.DeleteMembershipRequest()

@pytest.mark.asyncio
async def test_delete_membership_async(transport: str='grpc_asyncio', request_type=membership.DeleteMembershipRequest):
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_membership), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_membership(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.DeleteMembershipRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_membership_async_from_dict():
    await test_delete_membership_async(request_type=dict)

def test_delete_membership_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.DeleteMembershipRequest()
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.DeleteMembershipRequest()
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
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_membership(membership.DeleteMembershipRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_membership_flattened_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_membership(membership.DeleteMembershipRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [membership.UpdateMembershipRequest, dict])
def test_update_membership(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_membership(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.UpdateMembershipRequest()
    assert isinstance(response, future.Future)

def test_update_membership_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_membership), '__call__') as call:
        client.update_membership()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.UpdateMembershipRequest()

@pytest.mark.asyncio
async def test_update_membership_async(transport: str='grpc_asyncio', request_type=membership.UpdateMembershipRequest):
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_membership), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_membership(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.UpdateMembershipRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_membership_async_from_dict():
    await test_update_membership_async(request_type=dict)

def test_update_membership_field_headers():
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.UpdateMembershipRequest()
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.UpdateMembershipRequest()
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
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_membership(name='name_value', resource=membership.Membership(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].resource
        mock_val = membership.Membership(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_membership_flattened_error():
    if False:
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_membership(membership.UpdateMembershipRequest(), name='name_value', resource=membership.Membership(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_membership_flattened_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_membership), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_membership(name='name_value', resource=membership.Membership(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].resource
        mock_val = membership.Membership(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_membership_flattened_error_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_membership(membership.UpdateMembershipRequest(), name='name_value', resource=membership.Membership(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [membership.GenerateConnectManifestRequest, dict])
def test_generate_connect_manifest(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_connect_manifest), '__call__') as call:
        call.return_value = membership.GenerateConnectManifestResponse()
        response = client.generate_connect_manifest(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.GenerateConnectManifestRequest()
    assert isinstance(response, membership.GenerateConnectManifestResponse)

def test_generate_connect_manifest_empty_call():
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.generate_connect_manifest), '__call__') as call:
        client.generate_connect_manifest()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.GenerateConnectManifestRequest()

@pytest.mark.asyncio
async def test_generate_connect_manifest_async(transport: str='grpc_asyncio', request_type=membership.GenerateConnectManifestRequest):
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_connect_manifest), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(membership.GenerateConnectManifestResponse())
        response = await client.generate_connect_manifest(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.GenerateConnectManifestRequest()
    assert isinstance(response, membership.GenerateConnectManifestResponse)

@pytest.mark.asyncio
async def test_generate_connect_manifest_async_from_dict():
    await test_generate_connect_manifest_async(request_type=dict)

def test_generate_connect_manifest_field_headers():
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.GenerateConnectManifestRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.generate_connect_manifest), '__call__') as call:
        call.return_value = membership.GenerateConnectManifestResponse()
        client.generate_connect_manifest(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_generate_connect_manifest_field_headers_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.GenerateConnectManifestRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.generate_connect_manifest), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(membership.GenerateConnectManifestResponse())
        await client.generate_connect_manifest(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [membership.ValidateExclusivityRequest, dict])
def test_validate_exclusivity(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.validate_exclusivity), '__call__') as call:
        call.return_value = membership.ValidateExclusivityResponse()
        response = client.validate_exclusivity(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.ValidateExclusivityRequest()
    assert isinstance(response, membership.ValidateExclusivityResponse)

def test_validate_exclusivity_empty_call():
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.validate_exclusivity), '__call__') as call:
        client.validate_exclusivity()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.ValidateExclusivityRequest()

@pytest.mark.asyncio
async def test_validate_exclusivity_async(transport: str='grpc_asyncio', request_type=membership.ValidateExclusivityRequest):
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.validate_exclusivity), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(membership.ValidateExclusivityResponse())
        response = await client.validate_exclusivity(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.ValidateExclusivityRequest()
    assert isinstance(response, membership.ValidateExclusivityResponse)

@pytest.mark.asyncio
async def test_validate_exclusivity_async_from_dict():
    await test_validate_exclusivity_async(request_type=dict)

def test_validate_exclusivity_field_headers():
    if False:
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.ValidateExclusivityRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.validate_exclusivity), '__call__') as call:
        call.return_value = membership.ValidateExclusivityResponse()
        client.validate_exclusivity(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_validate_exclusivity_field_headers_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.ValidateExclusivityRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.validate_exclusivity), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(membership.ValidateExclusivityResponse())
        await client.validate_exclusivity(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [membership.GenerateExclusivityManifestRequest, dict])
def test_generate_exclusivity_manifest(request_type, transport: str='grpc'):
    if False:
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_exclusivity_manifest), '__call__') as call:
        call.return_value = membership.GenerateExclusivityManifestResponse(crd_manifest='crd_manifest_value', cr_manifest='cr_manifest_value')
        response = client.generate_exclusivity_manifest(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.GenerateExclusivityManifestRequest()
    assert isinstance(response, membership.GenerateExclusivityManifestResponse)
    assert response.crd_manifest == 'crd_manifest_value'
    assert response.cr_manifest == 'cr_manifest_value'

def test_generate_exclusivity_manifest_empty_call():
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.generate_exclusivity_manifest), '__call__') as call:
        client.generate_exclusivity_manifest()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.GenerateExclusivityManifestRequest()

@pytest.mark.asyncio
async def test_generate_exclusivity_manifest_async(transport: str='grpc_asyncio', request_type=membership.GenerateExclusivityManifestRequest):
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_exclusivity_manifest), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(membership.GenerateExclusivityManifestResponse(crd_manifest='crd_manifest_value', cr_manifest='cr_manifest_value'))
        response = await client.generate_exclusivity_manifest(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == membership.GenerateExclusivityManifestRequest()
    assert isinstance(response, membership.GenerateExclusivityManifestResponse)
    assert response.crd_manifest == 'crd_manifest_value'
    assert response.cr_manifest == 'cr_manifest_value'

@pytest.mark.asyncio
async def test_generate_exclusivity_manifest_async_from_dict():
    await test_generate_exclusivity_manifest_async(request_type=dict)

def test_generate_exclusivity_manifest_field_headers():
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.GenerateExclusivityManifestRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.generate_exclusivity_manifest), '__call__') as call:
        call.return_value = membership.GenerateExclusivityManifestResponse()
        client.generate_exclusivity_manifest(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_generate_exclusivity_manifest_field_headers_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = membership.GenerateExclusivityManifestRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.generate_exclusivity_manifest), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(membership.GenerateExclusivityManifestResponse())
        await client.generate_exclusivity_manifest(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [membership.ListMembershipsRequest, dict])
def test_list_memberships_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = membership.ListMembershipsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = membership.ListMembershipsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_memberships(request)
    assert isinstance(response, pagers.ListMembershipsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_memberships_rest_required_fields(request_type=membership.ListMembershipsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.GkeHubMembershipServiceRestTransport
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = membership.ListMembershipsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = membership.ListMembershipsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_memberships(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_memberships_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_memberships._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_memberships_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubMembershipServiceRestInterceptor())
    client = GkeHubMembershipServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'post_list_memberships') as post, mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'pre_list_memberships') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = membership.ListMembershipsRequest.pb(membership.ListMembershipsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = membership.ListMembershipsResponse.to_json(membership.ListMembershipsResponse())
        request = membership.ListMembershipsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = membership.ListMembershipsResponse()
        client.list_memberships(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_memberships_rest_bad_request(transport: str='rest', request_type=membership.ListMembershipsRequest):
    if False:
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = membership.ListMembershipsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = membership.ListMembershipsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_memberships(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*}/memberships' % client.transport._host, args[1])

def test_list_memberships_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_memberships(membership.ListMembershipsRequest(), parent='parent_value')

def test_list_memberships_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (membership.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership(), membership.Membership()], next_page_token='abc'), membership.ListMembershipsResponse(resources=[], next_page_token='def'), membership.ListMembershipsResponse(resources=[membership.Membership()], next_page_token='ghi'), membership.ListMembershipsResponse(resources=[membership.Membership(), membership.Membership()]))
        response = response + response
        response = tuple((membership.ListMembershipsResponse.to_json(x) for x in response))
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

@pytest.mark.parametrize('request_type', [membership.GetMembershipRequest, dict])
def test_get_membership_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = membership.Membership(name='name_value', description='description_value', external_id='external_id_value', unique_id='unique_id_value', infrastructure_type=membership.Membership.InfrastructureType.ON_PREM)
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
    assert response.infrastructure_type == membership.Membership.InfrastructureType.ON_PREM

def test_get_membership_rest_required_fields(request_type=membership.GetMembershipRequest):
    if False:
        return 10
    transport_class = transports.GkeHubMembershipServiceRestTransport
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_membership._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_membership_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubMembershipServiceRestInterceptor())
    client = GkeHubMembershipServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'post_get_membership') as post, mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'pre_get_membership') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = membership.GetMembershipRequest.pb(membership.GetMembershipRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = membership.Membership.to_json(membership.Membership())
        request = membership.GetMembershipRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = membership.Membership()
        client.get_membership(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_membership_rest_bad_request(transport: str='rest', request_type=membership.GetMembershipRequest):
    if False:
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/memberships/*}' % client.transport._host, args[1])

def test_get_membership_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_membership(membership.GetMembershipRequest(), name='name_value')

def test_get_membership_rest_error():
    if False:
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [membership.CreateMembershipRequest, dict])
def test_create_membership_rest(request_type):
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['resource'] = {'name': 'name_value', 'labels': {}, 'description': 'description_value', 'endpoint': {'gke_cluster': {'resource_link': 'resource_link_value', 'cluster_missing': True}, 'on_prem_cluster': {'resource_link': 'resource_link_value', 'cluster_missing': True, 'admin_cluster': True, 'cluster_type': 1}, 'multi_cloud_cluster': {'resource_link': 'resource_link_value', 'cluster_missing': True}, 'edge_cluster': {'resource_link': 'resource_link_value'}, 'appliance_cluster': {'resource_link': 'resource_link_value'}, 'kubernetes_metadata': {'kubernetes_api_server_version': 'kubernetes_api_server_version_value', 'node_provider_id': 'node_provider_id_value', 'node_count': 1070, 'vcpu_count': 1094, 'memory_mb': 967, 'update_time': {'seconds': 751, 'nanos': 543}}, 'kubernetes_resource': {'membership_cr_manifest': 'membership_cr_manifest_value', 'membership_resources': [{'manifest': 'manifest_value', 'cluster_scoped': True}], 'connect_resources': {}, 'resource_options': {'connect_version': 'connect_version_value', 'v1beta1_crd': True, 'k8s_version': 'k8s_version_value'}}}, 'state': {'code': 1, 'description': 'description_value', 'update_time': {}}, 'authority': {'issuer': 'issuer_value', 'workload_identity_pool': 'workload_identity_pool_value', 'identity_provider': 'identity_provider_value', 'oidc_jwks': b'oidc_jwks_blob'}, 'create_time': {}, 'update_time': {}, 'delete_time': {}, 'external_id': 'external_id_value', 'last_connection_time': {}, 'unique_id': 'unique_id_value', 'infrastructure_type': 1, 'monitoring_config': {'project_id': 'project_id_value', 'location': 'location_value', 'cluster': 'cluster_value', 'kubernetes_metrics_prefix': 'kubernetes_metrics_prefix_value', 'cluster_hash': 'cluster_hash_value'}}
    test_field = membership.CreateMembershipRequest.meta.fields['resource']

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

def test_create_membership_rest_required_fields(request_type=membership.CreateMembershipRequest):
    if False:
        return 10
    transport_class = transports.GkeHubMembershipServiceRestTransport
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_membership._get_unset_required_fields({})
    assert set(unset_fields) == set(('membershipId', 'requestId')) & set(('parent', 'membershipId', 'resource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_membership_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubMembershipServiceRestInterceptor())
    client = GkeHubMembershipServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'post_create_membership') as post, mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'pre_create_membership') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = membership.CreateMembershipRequest.pb(membership.CreateMembershipRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = membership.CreateMembershipRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_membership(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_membership_rest_bad_request(transport: str='rest', request_type=membership.CreateMembershipRequest):
    if False:
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', resource=membership.Membership(name='name_value'), membership_id='membership_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_membership(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*}/memberships' % client.transport._host, args[1])

def test_create_membership_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_membership(membership.CreateMembershipRequest(), parent='parent_value', resource=membership.Membership(name='name_value'), membership_id='membership_id_value')

def test_create_membership_rest_error():
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [membership.DeleteMembershipRequest, dict])
def test_delete_membership_rest(request_type):
    if False:
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_delete_membership_rest_required_fields(request_type=membership.DeleteMembershipRequest):
    if False:
        print('Hello World!')
    transport_class = transports.GkeHubMembershipServiceRestTransport
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_membership._get_unset_required_fields({})
    assert set(unset_fields) == set(('force', 'requestId')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_membership_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubMembershipServiceRestInterceptor())
    client = GkeHubMembershipServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'post_delete_membership') as post, mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'pre_delete_membership') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = membership.DeleteMembershipRequest.pb(membership.DeleteMembershipRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = membership.DeleteMembershipRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_membership(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_membership_rest_bad_request(transport: str='rest', request_type=membership.DeleteMembershipRequest):
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/memberships/*}' % client.transport._host, args[1])

def test_delete_membership_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_membership(membership.DeleteMembershipRequest(), name='name_value')

def test_delete_membership_rest_error():
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [membership.UpdateMembershipRequest, dict])
def test_update_membership_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
    request_init['resource'] = {'name': 'name_value', 'labels': {}, 'description': 'description_value', 'endpoint': {'gke_cluster': {'resource_link': 'resource_link_value', 'cluster_missing': True}, 'on_prem_cluster': {'resource_link': 'resource_link_value', 'cluster_missing': True, 'admin_cluster': True, 'cluster_type': 1}, 'multi_cloud_cluster': {'resource_link': 'resource_link_value', 'cluster_missing': True}, 'edge_cluster': {'resource_link': 'resource_link_value'}, 'appliance_cluster': {'resource_link': 'resource_link_value'}, 'kubernetes_metadata': {'kubernetes_api_server_version': 'kubernetes_api_server_version_value', 'node_provider_id': 'node_provider_id_value', 'node_count': 1070, 'vcpu_count': 1094, 'memory_mb': 967, 'update_time': {'seconds': 751, 'nanos': 543}}, 'kubernetes_resource': {'membership_cr_manifest': 'membership_cr_manifest_value', 'membership_resources': [{'manifest': 'manifest_value', 'cluster_scoped': True}], 'connect_resources': {}, 'resource_options': {'connect_version': 'connect_version_value', 'v1beta1_crd': True, 'k8s_version': 'k8s_version_value'}}}, 'state': {'code': 1, 'description': 'description_value', 'update_time': {}}, 'authority': {'issuer': 'issuer_value', 'workload_identity_pool': 'workload_identity_pool_value', 'identity_provider': 'identity_provider_value', 'oidc_jwks': b'oidc_jwks_blob'}, 'create_time': {}, 'update_time': {}, 'delete_time': {}, 'external_id': 'external_id_value', 'last_connection_time': {}, 'unique_id': 'unique_id_value', 'infrastructure_type': 1, 'monitoring_config': {'project_id': 'project_id_value', 'location': 'location_value', 'cluster': 'cluster_value', 'kubernetes_metrics_prefix': 'kubernetes_metrics_prefix_value', 'cluster_hash': 'cluster_hash_value'}}
    test_field = membership.UpdateMembershipRequest.meta.fields['resource']

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

def test_update_membership_rest_required_fields(request_type=membership.UpdateMembershipRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.GkeHubMembershipServiceRestTransport
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_membership._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('name', 'updateMask', 'resource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_membership_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubMembershipServiceRestInterceptor())
    client = GkeHubMembershipServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'post_update_membership') as post, mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'pre_update_membership') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = membership.UpdateMembershipRequest.pb(membership.UpdateMembershipRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = membership.UpdateMembershipRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_membership(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_membership_rest_bad_request(transport: str='rest', request_type=membership.UpdateMembershipRequest):
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
        mock_args = dict(name='name_value', resource=membership.Membership(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_membership(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/memberships/*}' % client.transport._host, args[1])

def test_update_membership_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_membership(membership.UpdateMembershipRequest(), name='name_value', resource=membership.Membership(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_membership_rest_error():
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [membership.GenerateConnectManifestRequest, dict])
def test_generate_connect_manifest_rest(request_type):
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = membership.GenerateConnectManifestResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = membership.GenerateConnectManifestResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.generate_connect_manifest(request)
    assert isinstance(response, membership.GenerateConnectManifestResponse)

def test_generate_connect_manifest_rest_required_fields(request_type=membership.GenerateConnectManifestRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.GkeHubMembershipServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_connect_manifest._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_connect_manifest._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('connect_agent', 'image_pull_secret_content', 'is_upgrade', 'registry', 'version'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = membership.GenerateConnectManifestResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = membership.GenerateConnectManifestResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.generate_connect_manifest(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_generate_connect_manifest_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.generate_connect_manifest._get_unset_required_fields({})
    assert set(unset_fields) == set(('connectAgent', 'imagePullSecretContent', 'isUpgrade', 'registry', 'version')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_generate_connect_manifest_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubMembershipServiceRestInterceptor())
    client = GkeHubMembershipServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'post_generate_connect_manifest') as post, mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'pre_generate_connect_manifest') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = membership.GenerateConnectManifestRequest.pb(membership.GenerateConnectManifestRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = membership.GenerateConnectManifestResponse.to_json(membership.GenerateConnectManifestResponse())
        request = membership.GenerateConnectManifestRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = membership.GenerateConnectManifestResponse()
        client.generate_connect_manifest(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_generate_connect_manifest_rest_bad_request(transport: str='rest', request_type=membership.GenerateConnectManifestRequest):
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [membership.ValidateExclusivityRequest, dict])
def test_validate_exclusivity_rest(request_type):
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = membership.ValidateExclusivityResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = membership.ValidateExclusivityResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.validate_exclusivity(request)
    assert isinstance(response, membership.ValidateExclusivityResponse)

def test_validate_exclusivity_rest_required_fields(request_type=membership.ValidateExclusivityRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.GkeHubMembershipServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['intended_membership'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'intendedMembership' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).validate_exclusivity._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'intendedMembership' in jsonified_request
    assert jsonified_request['intendedMembership'] == request_init['intended_membership']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['intendedMembership'] = 'intended_membership_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).validate_exclusivity._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('cr_manifest', 'intended_membership'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'intendedMembership' in jsonified_request
    assert jsonified_request['intendedMembership'] == 'intended_membership_value'
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = membership.ValidateExclusivityResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = membership.ValidateExclusivityResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.validate_exclusivity(request)
            expected_params = [('intendedMembership', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_validate_exclusivity_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.validate_exclusivity._get_unset_required_fields({})
    assert set(unset_fields) == set(('crManifest', 'intendedMembership')) & set(('parent', 'intendedMembership'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_validate_exclusivity_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubMembershipServiceRestInterceptor())
    client = GkeHubMembershipServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'post_validate_exclusivity') as post, mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'pre_validate_exclusivity') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = membership.ValidateExclusivityRequest.pb(membership.ValidateExclusivityRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = membership.ValidateExclusivityResponse.to_json(membership.ValidateExclusivityResponse())
        request = membership.ValidateExclusivityRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = membership.ValidateExclusivityResponse()
        client.validate_exclusivity(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_validate_exclusivity_rest_bad_request(transport: str='rest', request_type=membership.ValidateExclusivityRequest):
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.validate_exclusivity(request)

def test_validate_exclusivity_rest_error():
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [membership.GenerateExclusivityManifestRequest, dict])
def test_generate_exclusivity_manifest_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = membership.GenerateExclusivityManifestResponse(crd_manifest='crd_manifest_value', cr_manifest='cr_manifest_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = membership.GenerateExclusivityManifestResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.generate_exclusivity_manifest(request)
    assert isinstance(response, membership.GenerateExclusivityManifestResponse)
    assert response.crd_manifest == 'crd_manifest_value'
    assert response.cr_manifest == 'cr_manifest_value'

def test_generate_exclusivity_manifest_rest_required_fields(request_type=membership.GenerateExclusivityManifestRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.GkeHubMembershipServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_exclusivity_manifest._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_exclusivity_manifest._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('cr_manifest', 'crd_manifest'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = membership.GenerateExclusivityManifestResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = membership.GenerateExclusivityManifestResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.generate_exclusivity_manifest(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_generate_exclusivity_manifest_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.generate_exclusivity_manifest._get_unset_required_fields({})
    assert set(unset_fields) == set(('crManifest', 'crdManifest')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_generate_exclusivity_manifest_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.GkeHubMembershipServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GkeHubMembershipServiceRestInterceptor())
    client = GkeHubMembershipServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'post_generate_exclusivity_manifest') as post, mock.patch.object(transports.GkeHubMembershipServiceRestInterceptor, 'pre_generate_exclusivity_manifest') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = membership.GenerateExclusivityManifestRequest.pb(membership.GenerateExclusivityManifestRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = membership.GenerateExclusivityManifestResponse.to_json(membership.GenerateExclusivityManifestResponse())
        request = membership.GenerateExclusivityManifestRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = membership.GenerateExclusivityManifestResponse()
        client.generate_exclusivity_manifest(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_generate_exclusivity_manifest_rest_bad_request(transport: str='rest', request_type=membership.GenerateExclusivityManifestRequest):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/memberships/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.generate_exclusivity_manifest(request)

def test_generate_exclusivity_manifest_rest_error():
    if False:
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.GkeHubMembershipServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.GkeHubMembershipServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = GkeHubMembershipServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.GkeHubMembershipServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = GkeHubMembershipServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = GkeHubMembershipServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.GkeHubMembershipServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = GkeHubMembershipServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.GkeHubMembershipServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = GkeHubMembershipServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.GkeHubMembershipServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.GkeHubMembershipServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.GkeHubMembershipServiceGrpcTransport, transports.GkeHubMembershipServiceGrpcAsyncIOTransport, transports.GkeHubMembershipServiceRestTransport])
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
        return 10
    transport = GkeHubMembershipServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.GkeHubMembershipServiceGrpcTransport)

def test_gke_hub_membership_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.GkeHubMembershipServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_gke_hub_membership_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.gkehub_v1beta1.services.gke_hub_membership_service.transports.GkeHubMembershipServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.GkeHubMembershipServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_memberships', 'get_membership', 'create_membership', 'delete_membership', 'update_membership', 'generate_connect_manifest', 'validate_exclusivity', 'generate_exclusivity_manifest', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_gke_hub_membership_service_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.gkehub_v1beta1.services.gke_hub_membership_service.transports.GkeHubMembershipServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.GkeHubMembershipServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_gke_hub_membership_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.gkehub_v1beta1.services.gke_hub_membership_service.transports.GkeHubMembershipServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.GkeHubMembershipServiceTransport()
        adc.assert_called_once()

def test_gke_hub_membership_service_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        GkeHubMembershipServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.GkeHubMembershipServiceGrpcTransport, transports.GkeHubMembershipServiceGrpcAsyncIOTransport])
def test_gke_hub_membership_service_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.GkeHubMembershipServiceGrpcTransport, transports.GkeHubMembershipServiceGrpcAsyncIOTransport, transports.GkeHubMembershipServiceRestTransport])
def test_gke_hub_membership_service_transport_auth_gdch_credentials(transport_class):
    if False:
        while True:
            i = 10
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.GkeHubMembershipServiceGrpcTransport, grpc_helpers), (transports.GkeHubMembershipServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_gke_hub_membership_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('gkehub.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='gkehub.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.GkeHubMembershipServiceGrpcTransport, transports.GkeHubMembershipServiceGrpcAsyncIOTransport])
def test_gke_hub_membership_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_gke_hub_membership_service_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.GkeHubMembershipServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_gke_hub_membership_service_rest_lro_client():
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_gke_hub_membership_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='gkehub.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('gkehub.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gkehub.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_gke_hub_membership_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='gkehub.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('gkehub.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gkehub.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_gke_hub_membership_service_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = GkeHubMembershipServiceClient(credentials=creds1, transport=transport_name)
    client2 = GkeHubMembershipServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_memberships._session
    session2 = client2.transport.list_memberships._session
    assert session1 != session2
    session1 = client1.transport.get_membership._session
    session2 = client2.transport.get_membership._session
    assert session1 != session2
    session1 = client1.transport.create_membership._session
    session2 = client2.transport.create_membership._session
    assert session1 != session2
    session1 = client1.transport.delete_membership._session
    session2 = client2.transport.delete_membership._session
    assert session1 != session2
    session1 = client1.transport.update_membership._session
    session2 = client2.transport.update_membership._session
    assert session1 != session2
    session1 = client1.transport.generate_connect_manifest._session
    session2 = client2.transport.generate_connect_manifest._session
    assert session1 != session2
    session1 = client1.transport.validate_exclusivity._session
    session2 = client2.transport.validate_exclusivity._session
    assert session1 != session2
    session1 = client1.transport.generate_exclusivity_manifest._session
    session2 = client2.transport.generate_exclusivity_manifest._session
    assert session1 != session2

def test_gke_hub_membership_service_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.GkeHubMembershipServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_gke_hub_membership_service_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.GkeHubMembershipServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.GkeHubMembershipServiceGrpcTransport, transports.GkeHubMembershipServiceGrpcAsyncIOTransport])
def test_gke_hub_membership_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.GkeHubMembershipServiceGrpcTransport, transports.GkeHubMembershipServiceGrpcAsyncIOTransport])
def test_gke_hub_membership_service_transport_channel_mtls_with_adc(transport_class):
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

def test_gke_hub_membership_service_grpc_lro_client():
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_gke_hub_membership_service_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_membership_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    membership = 'whelk'
    expected = 'projects/{project}/locations/{location}/memberships/{membership}'.format(project=project, location=location, membership=membership)
    actual = GkeHubMembershipServiceClient.membership_path(project, location, membership)
    assert expected == actual

def test_parse_membership_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus', 'location': 'oyster', 'membership': 'nudibranch'}
    path = GkeHubMembershipServiceClient.membership_path(**expected)
    actual = GkeHubMembershipServiceClient.parse_membership_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = GkeHubMembershipServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'mussel'}
    path = GkeHubMembershipServiceClient.common_billing_account_path(**expected)
    actual = GkeHubMembershipServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = GkeHubMembershipServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'nautilus'}
    path = GkeHubMembershipServiceClient.common_folder_path(**expected)
    actual = GkeHubMembershipServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = GkeHubMembershipServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'abalone'}
    path = GkeHubMembershipServiceClient.common_organization_path(**expected)
    actual = GkeHubMembershipServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = GkeHubMembershipServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'clam'}
    path = GkeHubMembershipServiceClient.common_project_path(**expected)
    actual = GkeHubMembershipServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = GkeHubMembershipServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = GkeHubMembershipServiceClient.common_location_path(**expected)
    actual = GkeHubMembershipServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.GkeHubMembershipServiceTransport, '_prep_wrapped_messages') as prep:
        client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.GkeHubMembershipServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = GkeHubMembershipServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/memberships/sample3'}, request)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/memberships/sample3'}
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/memberships/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/memberships/sample3'}
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
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/memberships/sample3'}, request)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/memberships/sample3'}
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
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = GkeHubMembershipServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = GkeHubMembershipServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(GkeHubMembershipServiceClient, transports.GkeHubMembershipServiceGrpcTransport), (GkeHubMembershipServiceAsyncClient, transports.GkeHubMembershipServiceGrpcAsyncIOTransport)])
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