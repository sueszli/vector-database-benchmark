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
from google.protobuf import duration_pb2
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
from google.cloud.network_services_v1.services.network_services import NetworkServicesAsyncClient, NetworkServicesClient, pagers, transports
from google.cloud.network_services_v1.types import endpoint_policy as gcn_endpoint_policy
from google.cloud.network_services_v1.types import service_binding as gcn_service_binding
from google.cloud.network_services_v1.types import common
from google.cloud.network_services_v1.types import endpoint_policy
from google.cloud.network_services_v1.types import gateway
from google.cloud.network_services_v1.types import gateway as gcn_gateway
from google.cloud.network_services_v1.types import grpc_route
from google.cloud.network_services_v1.types import grpc_route as gcn_grpc_route
from google.cloud.network_services_v1.types import http_route
from google.cloud.network_services_v1.types import http_route as gcn_http_route
from google.cloud.network_services_v1.types import mesh
from google.cloud.network_services_v1.types import mesh as gcn_mesh
from google.cloud.network_services_v1.types import service_binding
from google.cloud.network_services_v1.types import tcp_route
from google.cloud.network_services_v1.types import tcp_route as gcn_tcp_route
from google.cloud.network_services_v1.types import tls_route
from google.cloud.network_services_v1.types import tls_route as gcn_tls_route

def client_cert_source_callback():
    if False:
        print('Hello World!')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        for i in range(10):
            print('nop')
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
    assert NetworkServicesClient._get_default_mtls_endpoint(None) is None
    assert NetworkServicesClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert NetworkServicesClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert NetworkServicesClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert NetworkServicesClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert NetworkServicesClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(NetworkServicesClient, 'grpc'), (NetworkServicesAsyncClient, 'grpc_asyncio'), (NetworkServicesClient, 'rest')])
def test_network_services_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('networkservices.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://networkservices.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.NetworkServicesGrpcTransport, 'grpc'), (transports.NetworkServicesGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.NetworkServicesRestTransport, 'rest')])
def test_network_services_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(NetworkServicesClient, 'grpc'), (NetworkServicesAsyncClient, 'grpc_asyncio'), (NetworkServicesClient, 'rest')])
def test_network_services_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('networkservices.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://networkservices.googleapis.com')

def test_network_services_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = NetworkServicesClient.get_transport_class()
    available_transports = [transports.NetworkServicesGrpcTransport, transports.NetworkServicesRestTransport]
    assert transport in available_transports
    transport = NetworkServicesClient.get_transport_class('grpc')
    assert transport == transports.NetworkServicesGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(NetworkServicesClient, transports.NetworkServicesGrpcTransport, 'grpc'), (NetworkServicesAsyncClient, transports.NetworkServicesGrpcAsyncIOTransport, 'grpc_asyncio'), (NetworkServicesClient, transports.NetworkServicesRestTransport, 'rest')])
@mock.patch.object(NetworkServicesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetworkServicesClient))
@mock.patch.object(NetworkServicesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetworkServicesAsyncClient))
def test_network_services_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(NetworkServicesClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(NetworkServicesClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(NetworkServicesClient, transports.NetworkServicesGrpcTransport, 'grpc', 'true'), (NetworkServicesAsyncClient, transports.NetworkServicesGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (NetworkServicesClient, transports.NetworkServicesGrpcTransport, 'grpc', 'false'), (NetworkServicesAsyncClient, transports.NetworkServicesGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (NetworkServicesClient, transports.NetworkServicesRestTransport, 'rest', 'true'), (NetworkServicesClient, transports.NetworkServicesRestTransport, 'rest', 'false')])
@mock.patch.object(NetworkServicesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetworkServicesClient))
@mock.patch.object(NetworkServicesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetworkServicesAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_network_services_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [NetworkServicesClient, NetworkServicesAsyncClient])
@mock.patch.object(NetworkServicesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetworkServicesClient))
@mock.patch.object(NetworkServicesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetworkServicesAsyncClient))
def test_network_services_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(NetworkServicesClient, transports.NetworkServicesGrpcTransport, 'grpc'), (NetworkServicesAsyncClient, transports.NetworkServicesGrpcAsyncIOTransport, 'grpc_asyncio'), (NetworkServicesClient, transports.NetworkServicesRestTransport, 'rest')])
def test_network_services_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(NetworkServicesClient, transports.NetworkServicesGrpcTransport, 'grpc', grpc_helpers), (NetworkServicesAsyncClient, transports.NetworkServicesGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (NetworkServicesClient, transports.NetworkServicesRestTransport, 'rest', None)])
def test_network_services_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_network_services_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.network_services_v1.services.network_services.transports.NetworkServicesGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = NetworkServicesClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(NetworkServicesClient, transports.NetworkServicesGrpcTransport, 'grpc', grpc_helpers), (NetworkServicesAsyncClient, transports.NetworkServicesGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_network_services_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('networkservices.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='networkservices.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [endpoint_policy.ListEndpointPoliciesRequest, dict])
def test_list_endpoint_policies(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_endpoint_policies), '__call__') as call:
        call.return_value = endpoint_policy.ListEndpointPoliciesResponse(next_page_token='next_page_token_value')
        response = client.list_endpoint_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == endpoint_policy.ListEndpointPoliciesRequest()
    assert isinstance(response, pagers.ListEndpointPoliciesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_endpoint_policies_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_endpoint_policies), '__call__') as call:
        client.list_endpoint_policies()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == endpoint_policy.ListEndpointPoliciesRequest()

@pytest.mark.asyncio
async def test_list_endpoint_policies_async(transport: str='grpc_asyncio', request_type=endpoint_policy.ListEndpointPoliciesRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_endpoint_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(endpoint_policy.ListEndpointPoliciesResponse(next_page_token='next_page_token_value'))
        response = await client.list_endpoint_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == endpoint_policy.ListEndpointPoliciesRequest()
    assert isinstance(response, pagers.ListEndpointPoliciesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_endpoint_policies_async_from_dict():
    await test_list_endpoint_policies_async(request_type=dict)

def test_list_endpoint_policies_field_headers():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = endpoint_policy.ListEndpointPoliciesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_endpoint_policies), '__call__') as call:
        call.return_value = endpoint_policy.ListEndpointPoliciesResponse()
        client.list_endpoint_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_endpoint_policies_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = endpoint_policy.ListEndpointPoliciesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_endpoint_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(endpoint_policy.ListEndpointPoliciesResponse())
        await client.list_endpoint_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_endpoint_policies_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_endpoint_policies), '__call__') as call:
        call.return_value = endpoint_policy.ListEndpointPoliciesResponse()
        client.list_endpoint_policies(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_endpoint_policies_flattened_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_endpoint_policies(endpoint_policy.ListEndpointPoliciesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_endpoint_policies_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_endpoint_policies), '__call__') as call:
        call.return_value = endpoint_policy.ListEndpointPoliciesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(endpoint_policy.ListEndpointPoliciesResponse())
        response = await client.list_endpoint_policies(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_endpoint_policies_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_endpoint_policies(endpoint_policy.ListEndpointPoliciesRequest(), parent='parent_value')

def test_list_endpoint_policies_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_endpoint_policies), '__call__') as call:
        call.side_effect = (endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy()], next_page_token='abc'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[], next_page_token='def'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy()], next_page_token='ghi'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_endpoint_policies(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, endpoint_policy.EndpointPolicy) for i in results))

def test_list_endpoint_policies_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_endpoint_policies), '__call__') as call:
        call.side_effect = (endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy()], next_page_token='abc'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[], next_page_token='def'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy()], next_page_token='ghi'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy()]), RuntimeError)
        pages = list(client.list_endpoint_policies(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_endpoint_policies_async_pager():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_endpoint_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy()], next_page_token='abc'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[], next_page_token='def'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy()], next_page_token='ghi'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy()]), RuntimeError)
        async_pager = await client.list_endpoint_policies(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, endpoint_policy.EndpointPolicy) for i in responses))

@pytest.mark.asyncio
async def test_list_endpoint_policies_async_pages():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_endpoint_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy()], next_page_token='abc'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[], next_page_token='def'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy()], next_page_token='ghi'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_endpoint_policies(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [endpoint_policy.GetEndpointPolicyRequest, dict])
def test_get_endpoint_policy(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_endpoint_policy), '__call__') as call:
        call.return_value = endpoint_policy.EndpointPolicy(name='name_value', type_=endpoint_policy.EndpointPolicy.EndpointPolicyType.SIDECAR_PROXY, authorization_policy='authorization_policy_value', description='description_value', server_tls_policy='server_tls_policy_value', client_tls_policy='client_tls_policy_value')
        response = client.get_endpoint_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == endpoint_policy.GetEndpointPolicyRequest()
    assert isinstance(response, endpoint_policy.EndpointPolicy)
    assert response.name == 'name_value'
    assert response.type_ == endpoint_policy.EndpointPolicy.EndpointPolicyType.SIDECAR_PROXY
    assert response.authorization_policy == 'authorization_policy_value'
    assert response.description == 'description_value'
    assert response.server_tls_policy == 'server_tls_policy_value'
    assert response.client_tls_policy == 'client_tls_policy_value'

def test_get_endpoint_policy_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_endpoint_policy), '__call__') as call:
        client.get_endpoint_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == endpoint_policy.GetEndpointPolicyRequest()

@pytest.mark.asyncio
async def test_get_endpoint_policy_async(transport: str='grpc_asyncio', request_type=endpoint_policy.GetEndpointPolicyRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_endpoint_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(endpoint_policy.EndpointPolicy(name='name_value', type_=endpoint_policy.EndpointPolicy.EndpointPolicyType.SIDECAR_PROXY, authorization_policy='authorization_policy_value', description='description_value', server_tls_policy='server_tls_policy_value', client_tls_policy='client_tls_policy_value'))
        response = await client.get_endpoint_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == endpoint_policy.GetEndpointPolicyRequest()
    assert isinstance(response, endpoint_policy.EndpointPolicy)
    assert response.name == 'name_value'
    assert response.type_ == endpoint_policy.EndpointPolicy.EndpointPolicyType.SIDECAR_PROXY
    assert response.authorization_policy == 'authorization_policy_value'
    assert response.description == 'description_value'
    assert response.server_tls_policy == 'server_tls_policy_value'
    assert response.client_tls_policy == 'client_tls_policy_value'

@pytest.mark.asyncio
async def test_get_endpoint_policy_async_from_dict():
    await test_get_endpoint_policy_async(request_type=dict)

def test_get_endpoint_policy_field_headers():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = endpoint_policy.GetEndpointPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_endpoint_policy), '__call__') as call:
        call.return_value = endpoint_policy.EndpointPolicy()
        client.get_endpoint_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_endpoint_policy_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = endpoint_policy.GetEndpointPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_endpoint_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(endpoint_policy.EndpointPolicy())
        await client.get_endpoint_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_endpoint_policy_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_endpoint_policy), '__call__') as call:
        call.return_value = endpoint_policy.EndpointPolicy()
        client.get_endpoint_policy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_endpoint_policy_flattened_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_endpoint_policy(endpoint_policy.GetEndpointPolicyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_endpoint_policy_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_endpoint_policy), '__call__') as call:
        call.return_value = endpoint_policy.EndpointPolicy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(endpoint_policy.EndpointPolicy())
        response = await client.get_endpoint_policy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_endpoint_policy_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_endpoint_policy(endpoint_policy.GetEndpointPolicyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_endpoint_policy.CreateEndpointPolicyRequest, dict])
def test_create_endpoint_policy(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_endpoint_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_endpoint_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_endpoint_policy.CreateEndpointPolicyRequest()
    assert isinstance(response, future.Future)

def test_create_endpoint_policy_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_endpoint_policy), '__call__') as call:
        client.create_endpoint_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_endpoint_policy.CreateEndpointPolicyRequest()

@pytest.mark.asyncio
async def test_create_endpoint_policy_async(transport: str='grpc_asyncio', request_type=gcn_endpoint_policy.CreateEndpointPolicyRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_endpoint_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_endpoint_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_endpoint_policy.CreateEndpointPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_endpoint_policy_async_from_dict():
    await test_create_endpoint_policy_async(request_type=dict)

def test_create_endpoint_policy_field_headers():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_endpoint_policy.CreateEndpointPolicyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_endpoint_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_endpoint_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_endpoint_policy_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_endpoint_policy.CreateEndpointPolicyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_endpoint_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_endpoint_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_endpoint_policy_flattened():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_endpoint_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_endpoint_policy(parent='parent_value', endpoint_policy=gcn_endpoint_policy.EndpointPolicy(name='name_value'), endpoint_policy_id='endpoint_policy_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].endpoint_policy
        mock_val = gcn_endpoint_policy.EndpointPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].endpoint_policy_id
        mock_val = 'endpoint_policy_id_value'
        assert arg == mock_val

def test_create_endpoint_policy_flattened_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_endpoint_policy(gcn_endpoint_policy.CreateEndpointPolicyRequest(), parent='parent_value', endpoint_policy=gcn_endpoint_policy.EndpointPolicy(name='name_value'), endpoint_policy_id='endpoint_policy_id_value')

@pytest.mark.asyncio
async def test_create_endpoint_policy_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_endpoint_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_endpoint_policy(parent='parent_value', endpoint_policy=gcn_endpoint_policy.EndpointPolicy(name='name_value'), endpoint_policy_id='endpoint_policy_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].endpoint_policy
        mock_val = gcn_endpoint_policy.EndpointPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].endpoint_policy_id
        mock_val = 'endpoint_policy_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_endpoint_policy_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_endpoint_policy(gcn_endpoint_policy.CreateEndpointPolicyRequest(), parent='parent_value', endpoint_policy=gcn_endpoint_policy.EndpointPolicy(name='name_value'), endpoint_policy_id='endpoint_policy_id_value')

@pytest.mark.parametrize('request_type', [gcn_endpoint_policy.UpdateEndpointPolicyRequest, dict])
def test_update_endpoint_policy(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_endpoint_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_endpoint_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_endpoint_policy.UpdateEndpointPolicyRequest()
    assert isinstance(response, future.Future)

def test_update_endpoint_policy_empty_call():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_endpoint_policy), '__call__') as call:
        client.update_endpoint_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_endpoint_policy.UpdateEndpointPolicyRequest()

@pytest.mark.asyncio
async def test_update_endpoint_policy_async(transport: str='grpc_asyncio', request_type=gcn_endpoint_policy.UpdateEndpointPolicyRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_endpoint_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_endpoint_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_endpoint_policy.UpdateEndpointPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_endpoint_policy_async_from_dict():
    await test_update_endpoint_policy_async(request_type=dict)

def test_update_endpoint_policy_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_endpoint_policy.UpdateEndpointPolicyRequest()
    request.endpoint_policy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_endpoint_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_endpoint_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'endpoint_policy.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_endpoint_policy_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_endpoint_policy.UpdateEndpointPolicyRequest()
    request.endpoint_policy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_endpoint_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_endpoint_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'endpoint_policy.name=name_value') in kw['metadata']

def test_update_endpoint_policy_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_endpoint_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_endpoint_policy(endpoint_policy=gcn_endpoint_policy.EndpointPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].endpoint_policy
        mock_val = gcn_endpoint_policy.EndpointPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_endpoint_policy_flattened_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_endpoint_policy(gcn_endpoint_policy.UpdateEndpointPolicyRequest(), endpoint_policy=gcn_endpoint_policy.EndpointPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_endpoint_policy_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_endpoint_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_endpoint_policy(endpoint_policy=gcn_endpoint_policy.EndpointPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].endpoint_policy
        mock_val = gcn_endpoint_policy.EndpointPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_endpoint_policy_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_endpoint_policy(gcn_endpoint_policy.UpdateEndpointPolicyRequest(), endpoint_policy=gcn_endpoint_policy.EndpointPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [endpoint_policy.DeleteEndpointPolicyRequest, dict])
def test_delete_endpoint_policy(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_endpoint_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_endpoint_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == endpoint_policy.DeleteEndpointPolicyRequest()
    assert isinstance(response, future.Future)

def test_delete_endpoint_policy_empty_call():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_endpoint_policy), '__call__') as call:
        client.delete_endpoint_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == endpoint_policy.DeleteEndpointPolicyRequest()

@pytest.mark.asyncio
async def test_delete_endpoint_policy_async(transport: str='grpc_asyncio', request_type=endpoint_policy.DeleteEndpointPolicyRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_endpoint_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_endpoint_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == endpoint_policy.DeleteEndpointPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_endpoint_policy_async_from_dict():
    await test_delete_endpoint_policy_async(request_type=dict)

def test_delete_endpoint_policy_field_headers():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = endpoint_policy.DeleteEndpointPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_endpoint_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_endpoint_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_endpoint_policy_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = endpoint_policy.DeleteEndpointPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_endpoint_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_endpoint_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_endpoint_policy_flattened():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_endpoint_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_endpoint_policy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_endpoint_policy_flattened_error():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_endpoint_policy(endpoint_policy.DeleteEndpointPolicyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_endpoint_policy_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_endpoint_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_endpoint_policy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_endpoint_policy_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_endpoint_policy(endpoint_policy.DeleteEndpointPolicyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gateway.ListGatewaysRequest, dict])
def test_list_gateways(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.return_value = gateway.ListGatewaysResponse(next_page_token='next_page_token_value')
        response = client.list_gateways(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gateway.ListGatewaysRequest()
    assert isinstance(response, pagers.ListGatewaysPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_gateways_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        client.list_gateways()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gateway.ListGatewaysRequest()

@pytest.mark.asyncio
async def test_list_gateways_async(transport: str='grpc_asyncio', request_type=gateway.ListGatewaysRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gateway.ListGatewaysResponse(next_page_token='next_page_token_value'))
        response = await client.list_gateways(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gateway.ListGatewaysRequest()
    assert isinstance(response, pagers.ListGatewaysAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_gateways_async_from_dict():
    await test_list_gateways_async(request_type=dict)

def test_list_gateways_field_headers():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gateway.ListGatewaysRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.return_value = gateway.ListGatewaysResponse()
        client.list_gateways(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_gateways_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gateway.ListGatewaysRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gateway.ListGatewaysResponse())
        await client.list_gateways(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_gateways_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.return_value = gateway.ListGatewaysResponse()
        client.list_gateways(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_gateways_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_gateways(gateway.ListGatewaysRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_gateways_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.return_value = gateway.ListGatewaysResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gateway.ListGatewaysResponse())
        response = await client.list_gateways(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_gateways_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_gateways(gateway.ListGatewaysRequest(), parent='parent_value')

def test_list_gateways_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.side_effect = (gateway.ListGatewaysResponse(gateways=[gateway.Gateway(), gateway.Gateway(), gateway.Gateway()], next_page_token='abc'), gateway.ListGatewaysResponse(gateways=[], next_page_token='def'), gateway.ListGatewaysResponse(gateways=[gateway.Gateway()], next_page_token='ghi'), gateway.ListGatewaysResponse(gateways=[gateway.Gateway(), gateway.Gateway()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_gateways(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, gateway.Gateway) for i in results))

def test_list_gateways_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_gateways), '__call__') as call:
        call.side_effect = (gateway.ListGatewaysResponse(gateways=[gateway.Gateway(), gateway.Gateway(), gateway.Gateway()], next_page_token='abc'), gateway.ListGatewaysResponse(gateways=[], next_page_token='def'), gateway.ListGatewaysResponse(gateways=[gateway.Gateway()], next_page_token='ghi'), gateway.ListGatewaysResponse(gateways=[gateway.Gateway(), gateway.Gateway()]), RuntimeError)
        pages = list(client.list_gateways(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_gateways_async_pager():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_gateways), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (gateway.ListGatewaysResponse(gateways=[gateway.Gateway(), gateway.Gateway(), gateway.Gateway()], next_page_token='abc'), gateway.ListGatewaysResponse(gateways=[], next_page_token='def'), gateway.ListGatewaysResponse(gateways=[gateway.Gateway()], next_page_token='ghi'), gateway.ListGatewaysResponse(gateways=[gateway.Gateway(), gateway.Gateway()]), RuntimeError)
        async_pager = await client.list_gateways(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, gateway.Gateway) for i in responses))

@pytest.mark.asyncio
async def test_list_gateways_async_pages():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_gateways), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (gateway.ListGatewaysResponse(gateways=[gateway.Gateway(), gateway.Gateway(), gateway.Gateway()], next_page_token='abc'), gateway.ListGatewaysResponse(gateways=[], next_page_token='def'), gateway.ListGatewaysResponse(gateways=[gateway.Gateway()], next_page_token='ghi'), gateway.ListGatewaysResponse(gateways=[gateway.Gateway(), gateway.Gateway()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_gateways(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [gateway.GetGatewayRequest, dict])
def test_get_gateway(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_gateway), '__call__') as call:
        call.return_value = gateway.Gateway(name='name_value', self_link='self_link_value', description='description_value', type_=gateway.Gateway.Type.OPEN_MESH, ports=[568], scope='scope_value', server_tls_policy='server_tls_policy_value')
        response = client.get_gateway(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gateway.GetGatewayRequest()
    assert isinstance(response, gateway.Gateway)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.type_ == gateway.Gateway.Type.OPEN_MESH
    assert response.ports == [568]
    assert response.scope == 'scope_value'
    assert response.server_tls_policy == 'server_tls_policy_value'

def test_get_gateway_empty_call():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_gateway), '__call__') as call:
        client.get_gateway()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gateway.GetGatewayRequest()

@pytest.mark.asyncio
async def test_get_gateway_async(transport: str='grpc_asyncio', request_type=gateway.GetGatewayRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_gateway), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gateway.Gateway(name='name_value', self_link='self_link_value', description='description_value', type_=gateway.Gateway.Type.OPEN_MESH, ports=[568], scope='scope_value', server_tls_policy='server_tls_policy_value'))
        response = await client.get_gateway(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gateway.GetGatewayRequest()
    assert isinstance(response, gateway.Gateway)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.type_ == gateway.Gateway.Type.OPEN_MESH
    assert response.ports == [568]
    assert response.scope == 'scope_value'
    assert response.server_tls_policy == 'server_tls_policy_value'

@pytest.mark.asyncio
async def test_get_gateway_async_from_dict():
    await test_get_gateway_async(request_type=dict)

def test_get_gateway_field_headers():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gateway.GetGatewayRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_gateway), '__call__') as call:
        call.return_value = gateway.Gateway()
        client.get_gateway(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_gateway_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gateway.GetGatewayRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_gateway), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gateway.Gateway())
        await client.get_gateway(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_gateway_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_gateway), '__call__') as call:
        call.return_value = gateway.Gateway()
        client.get_gateway(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_gateway_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_gateway(gateway.GetGatewayRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_gateway_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_gateway), '__call__') as call:
        call.return_value = gateway.Gateway()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gateway.Gateway())
        response = await client.get_gateway(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_gateway_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_gateway(gateway.GetGatewayRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_gateway.CreateGatewayRequest, dict])
def test_create_gateway(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_gateway(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_gateway.CreateGatewayRequest()
    assert isinstance(response, future.Future)

def test_create_gateway_empty_call():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_gateway), '__call__') as call:
        client.create_gateway()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_gateway.CreateGatewayRequest()

@pytest.mark.asyncio
async def test_create_gateway_async(transport: str='grpc_asyncio', request_type=gcn_gateway.CreateGatewayRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_gateway), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_gateway(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_gateway.CreateGatewayRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_gateway_async_from_dict():
    await test_create_gateway_async(request_type=dict)

def test_create_gateway_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_gateway.CreateGatewayRequest()
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_gateway.CreateGatewayRequest()
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_gateway(parent='parent_value', gateway=gcn_gateway.Gateway(name='name_value'), gateway_id='gateway_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].gateway
        mock_val = gcn_gateway.Gateway(name='name_value')
        assert arg == mock_val
        arg = args[0].gateway_id
        mock_val = 'gateway_id_value'
        assert arg == mock_val

def test_create_gateway_flattened_error():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_gateway(gcn_gateway.CreateGatewayRequest(), parent='parent_value', gateway=gcn_gateway.Gateway(name='name_value'), gateway_id='gateway_id_value')

@pytest.mark.asyncio
async def test_create_gateway_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_gateway(parent='parent_value', gateway=gcn_gateway.Gateway(name='name_value'), gateway_id='gateway_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].gateway
        mock_val = gcn_gateway.Gateway(name='name_value')
        assert arg == mock_val
        arg = args[0].gateway_id
        mock_val = 'gateway_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_gateway_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_gateway(gcn_gateway.CreateGatewayRequest(), parent='parent_value', gateway=gcn_gateway.Gateway(name='name_value'), gateway_id='gateway_id_value')

@pytest.mark.parametrize('request_type', [gcn_gateway.UpdateGatewayRequest, dict])
def test_update_gateway(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_gateway(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_gateway.UpdateGatewayRequest()
    assert isinstance(response, future.Future)

def test_update_gateway_empty_call():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_gateway), '__call__') as call:
        client.update_gateway()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_gateway.UpdateGatewayRequest()

@pytest.mark.asyncio
async def test_update_gateway_async(transport: str='grpc_asyncio', request_type=gcn_gateway.UpdateGatewayRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_gateway), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_gateway(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_gateway.UpdateGatewayRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_gateway_async_from_dict():
    await test_update_gateway_async(request_type=dict)

def test_update_gateway_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_gateway.UpdateGatewayRequest()
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_gateway.UpdateGatewayRequest()
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
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_gateway(gateway=gcn_gateway.Gateway(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].gateway
        mock_val = gcn_gateway.Gateway(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_gateway_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_gateway(gcn_gateway.UpdateGatewayRequest(), gateway=gcn_gateway.Gateway(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_gateway_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_gateway(gateway=gcn_gateway.Gateway(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].gateway
        mock_val = gcn_gateway.Gateway(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_gateway_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_gateway(gcn_gateway.UpdateGatewayRequest(), gateway=gcn_gateway.Gateway(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [gateway.DeleteGatewayRequest, dict])
def test_delete_gateway(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_gateway), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_gateway(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gateway.DeleteGatewayRequest()
    assert isinstance(response, future.Future)

def test_delete_gateway_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_gateway), '__call__') as call:
        client.delete_gateway()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gateway.DeleteGatewayRequest()

@pytest.mark.asyncio
async def test_delete_gateway_async(transport: str='grpc_asyncio', request_type=gateway.DeleteGatewayRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_gateway), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_gateway(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gateway.DeleteGatewayRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_gateway_async_from_dict():
    await test_delete_gateway_async(request_type=dict)

def test_delete_gateway_field_headers():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gateway.DeleteGatewayRequest()
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gateway.DeleteGatewayRequest()
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
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_gateway(gateway.DeleteGatewayRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_gateway_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_gateway(gateway.DeleteGatewayRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [grpc_route.ListGrpcRoutesRequest, dict])
def test_list_grpc_routes(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_grpc_routes), '__call__') as call:
        call.return_value = grpc_route.ListGrpcRoutesResponse(next_page_token='next_page_token_value')
        response = client.list_grpc_routes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grpc_route.ListGrpcRoutesRequest()
    assert isinstance(response, pagers.ListGrpcRoutesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_grpc_routes_empty_call():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_grpc_routes), '__call__') as call:
        client.list_grpc_routes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grpc_route.ListGrpcRoutesRequest()

@pytest.mark.asyncio
async def test_list_grpc_routes_async(transport: str='grpc_asyncio', request_type=grpc_route.ListGrpcRoutesRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_grpc_routes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grpc_route.ListGrpcRoutesResponse(next_page_token='next_page_token_value'))
        response = await client.list_grpc_routes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grpc_route.ListGrpcRoutesRequest()
    assert isinstance(response, pagers.ListGrpcRoutesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_grpc_routes_async_from_dict():
    await test_list_grpc_routes_async(request_type=dict)

def test_list_grpc_routes_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = grpc_route.ListGrpcRoutesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_grpc_routes), '__call__') as call:
        call.return_value = grpc_route.ListGrpcRoutesResponse()
        client.list_grpc_routes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_grpc_routes_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grpc_route.ListGrpcRoutesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_grpc_routes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grpc_route.ListGrpcRoutesResponse())
        await client.list_grpc_routes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_grpc_routes_flattened():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_grpc_routes), '__call__') as call:
        call.return_value = grpc_route.ListGrpcRoutesResponse()
        client.list_grpc_routes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_grpc_routes_flattened_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_grpc_routes(grpc_route.ListGrpcRoutesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_grpc_routes_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_grpc_routes), '__call__') as call:
        call.return_value = grpc_route.ListGrpcRoutesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grpc_route.ListGrpcRoutesResponse())
        response = await client.list_grpc_routes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_grpc_routes_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_grpc_routes(grpc_route.ListGrpcRoutesRequest(), parent='parent_value')

def test_list_grpc_routes_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_grpc_routes), '__call__') as call:
        call.side_effect = (grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute(), grpc_route.GrpcRoute(), grpc_route.GrpcRoute()], next_page_token='abc'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[], next_page_token='def'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute()], next_page_token='ghi'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute(), grpc_route.GrpcRoute()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_grpc_routes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, grpc_route.GrpcRoute) for i in results))

def test_list_grpc_routes_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_grpc_routes), '__call__') as call:
        call.side_effect = (grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute(), grpc_route.GrpcRoute(), grpc_route.GrpcRoute()], next_page_token='abc'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[], next_page_token='def'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute()], next_page_token='ghi'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute(), grpc_route.GrpcRoute()]), RuntimeError)
        pages = list(client.list_grpc_routes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_grpc_routes_async_pager():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_grpc_routes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute(), grpc_route.GrpcRoute(), grpc_route.GrpcRoute()], next_page_token='abc'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[], next_page_token='def'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute()], next_page_token='ghi'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute(), grpc_route.GrpcRoute()]), RuntimeError)
        async_pager = await client.list_grpc_routes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, grpc_route.GrpcRoute) for i in responses))

@pytest.mark.asyncio
async def test_list_grpc_routes_async_pages():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_grpc_routes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute(), grpc_route.GrpcRoute(), grpc_route.GrpcRoute()], next_page_token='abc'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[], next_page_token='def'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute()], next_page_token='ghi'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute(), grpc_route.GrpcRoute()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_grpc_routes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [grpc_route.GetGrpcRouteRequest, dict])
def test_get_grpc_route(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_grpc_route), '__call__') as call:
        call.return_value = grpc_route.GrpcRoute(name='name_value', self_link='self_link_value', description='description_value', hostnames=['hostnames_value'], meshes=['meshes_value'], gateways=['gateways_value'])
        response = client.get_grpc_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grpc_route.GetGrpcRouteRequest()
    assert isinstance(response, grpc_route.GrpcRoute)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.hostnames == ['hostnames_value']
    assert response.meshes == ['meshes_value']
    assert response.gateways == ['gateways_value']

def test_get_grpc_route_empty_call():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_grpc_route), '__call__') as call:
        client.get_grpc_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grpc_route.GetGrpcRouteRequest()

@pytest.mark.asyncio
async def test_get_grpc_route_async(transport: str='grpc_asyncio', request_type=grpc_route.GetGrpcRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_grpc_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grpc_route.GrpcRoute(name='name_value', self_link='self_link_value', description='description_value', hostnames=['hostnames_value'], meshes=['meshes_value'], gateways=['gateways_value']))
        response = await client.get_grpc_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grpc_route.GetGrpcRouteRequest()
    assert isinstance(response, grpc_route.GrpcRoute)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.hostnames == ['hostnames_value']
    assert response.meshes == ['meshes_value']
    assert response.gateways == ['gateways_value']

@pytest.mark.asyncio
async def test_get_grpc_route_async_from_dict():
    await test_get_grpc_route_async(request_type=dict)

def test_get_grpc_route_field_headers():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = grpc_route.GetGrpcRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_grpc_route), '__call__') as call:
        call.return_value = grpc_route.GrpcRoute()
        client.get_grpc_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_grpc_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grpc_route.GetGrpcRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_grpc_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grpc_route.GrpcRoute())
        await client.get_grpc_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_grpc_route_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_grpc_route), '__call__') as call:
        call.return_value = grpc_route.GrpcRoute()
        client.get_grpc_route(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_grpc_route_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_grpc_route(grpc_route.GetGrpcRouteRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_grpc_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_grpc_route), '__call__') as call:
        call.return_value = grpc_route.GrpcRoute()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grpc_route.GrpcRoute())
        response = await client.get_grpc_route(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_grpc_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_grpc_route(grpc_route.GetGrpcRouteRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_grpc_route.CreateGrpcRouteRequest, dict])
def test_create_grpc_route(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_grpc_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_grpc_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_grpc_route.CreateGrpcRouteRequest()
    assert isinstance(response, future.Future)

def test_create_grpc_route_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_grpc_route), '__call__') as call:
        client.create_grpc_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_grpc_route.CreateGrpcRouteRequest()

@pytest.mark.asyncio
async def test_create_grpc_route_async(transport: str='grpc_asyncio', request_type=gcn_grpc_route.CreateGrpcRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_grpc_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_grpc_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_grpc_route.CreateGrpcRouteRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_grpc_route_async_from_dict():
    await test_create_grpc_route_async(request_type=dict)

def test_create_grpc_route_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_grpc_route.CreateGrpcRouteRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_grpc_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_grpc_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_grpc_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_grpc_route.CreateGrpcRouteRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_grpc_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_grpc_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_grpc_route_flattened():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_grpc_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_grpc_route(parent='parent_value', grpc_route=gcn_grpc_route.GrpcRoute(name='name_value'), grpc_route_id='grpc_route_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].grpc_route
        mock_val = gcn_grpc_route.GrpcRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].grpc_route_id
        mock_val = 'grpc_route_id_value'
        assert arg == mock_val

def test_create_grpc_route_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_grpc_route(gcn_grpc_route.CreateGrpcRouteRequest(), parent='parent_value', grpc_route=gcn_grpc_route.GrpcRoute(name='name_value'), grpc_route_id='grpc_route_id_value')

@pytest.mark.asyncio
async def test_create_grpc_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_grpc_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_grpc_route(parent='parent_value', grpc_route=gcn_grpc_route.GrpcRoute(name='name_value'), grpc_route_id='grpc_route_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].grpc_route
        mock_val = gcn_grpc_route.GrpcRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].grpc_route_id
        mock_val = 'grpc_route_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_grpc_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_grpc_route(gcn_grpc_route.CreateGrpcRouteRequest(), parent='parent_value', grpc_route=gcn_grpc_route.GrpcRoute(name='name_value'), grpc_route_id='grpc_route_id_value')

@pytest.mark.parametrize('request_type', [gcn_grpc_route.UpdateGrpcRouteRequest, dict])
def test_update_grpc_route(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_grpc_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_grpc_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_grpc_route.UpdateGrpcRouteRequest()
    assert isinstance(response, future.Future)

def test_update_grpc_route_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_grpc_route), '__call__') as call:
        client.update_grpc_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_grpc_route.UpdateGrpcRouteRequest()

@pytest.mark.asyncio
async def test_update_grpc_route_async(transport: str='grpc_asyncio', request_type=gcn_grpc_route.UpdateGrpcRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_grpc_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_grpc_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_grpc_route.UpdateGrpcRouteRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_grpc_route_async_from_dict():
    await test_update_grpc_route_async(request_type=dict)

def test_update_grpc_route_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_grpc_route.UpdateGrpcRouteRequest()
    request.grpc_route.name = 'name_value'
    with mock.patch.object(type(client.transport.update_grpc_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_grpc_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'grpc_route.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_grpc_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_grpc_route.UpdateGrpcRouteRequest()
    request.grpc_route.name = 'name_value'
    with mock.patch.object(type(client.transport.update_grpc_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_grpc_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'grpc_route.name=name_value') in kw['metadata']

def test_update_grpc_route_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_grpc_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_grpc_route(grpc_route=gcn_grpc_route.GrpcRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].grpc_route
        mock_val = gcn_grpc_route.GrpcRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_grpc_route_flattened_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_grpc_route(gcn_grpc_route.UpdateGrpcRouteRequest(), grpc_route=gcn_grpc_route.GrpcRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_grpc_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_grpc_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_grpc_route(grpc_route=gcn_grpc_route.GrpcRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].grpc_route
        mock_val = gcn_grpc_route.GrpcRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_grpc_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_grpc_route(gcn_grpc_route.UpdateGrpcRouteRequest(), grpc_route=gcn_grpc_route.GrpcRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [grpc_route.DeleteGrpcRouteRequest, dict])
def test_delete_grpc_route(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_grpc_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_grpc_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grpc_route.DeleteGrpcRouteRequest()
    assert isinstance(response, future.Future)

def test_delete_grpc_route_empty_call():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_grpc_route), '__call__') as call:
        client.delete_grpc_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grpc_route.DeleteGrpcRouteRequest()

@pytest.mark.asyncio
async def test_delete_grpc_route_async(transport: str='grpc_asyncio', request_type=grpc_route.DeleteGrpcRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_grpc_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_grpc_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grpc_route.DeleteGrpcRouteRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_grpc_route_async_from_dict():
    await test_delete_grpc_route_async(request_type=dict)

def test_delete_grpc_route_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = grpc_route.DeleteGrpcRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_grpc_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_grpc_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_grpc_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grpc_route.DeleteGrpcRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_grpc_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_grpc_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_grpc_route_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_grpc_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_grpc_route(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_grpc_route_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_grpc_route(grpc_route.DeleteGrpcRouteRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_grpc_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_grpc_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_grpc_route(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_grpc_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_grpc_route(grpc_route.DeleteGrpcRouteRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [http_route.ListHttpRoutesRequest, dict])
def test_list_http_routes(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_http_routes), '__call__') as call:
        call.return_value = http_route.ListHttpRoutesResponse(next_page_token='next_page_token_value')
        response = client.list_http_routes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == http_route.ListHttpRoutesRequest()
    assert isinstance(response, pagers.ListHttpRoutesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_http_routes_empty_call():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_http_routes), '__call__') as call:
        client.list_http_routes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == http_route.ListHttpRoutesRequest()

@pytest.mark.asyncio
async def test_list_http_routes_async(transport: str='grpc_asyncio', request_type=http_route.ListHttpRoutesRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_http_routes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(http_route.ListHttpRoutesResponse(next_page_token='next_page_token_value'))
        response = await client.list_http_routes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == http_route.ListHttpRoutesRequest()
    assert isinstance(response, pagers.ListHttpRoutesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_http_routes_async_from_dict():
    await test_list_http_routes_async(request_type=dict)

def test_list_http_routes_field_headers():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = http_route.ListHttpRoutesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_http_routes), '__call__') as call:
        call.return_value = http_route.ListHttpRoutesResponse()
        client.list_http_routes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_http_routes_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = http_route.ListHttpRoutesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_http_routes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(http_route.ListHttpRoutesResponse())
        await client.list_http_routes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_http_routes_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_http_routes), '__call__') as call:
        call.return_value = http_route.ListHttpRoutesResponse()
        client.list_http_routes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_http_routes_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_http_routes(http_route.ListHttpRoutesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_http_routes_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_http_routes), '__call__') as call:
        call.return_value = http_route.ListHttpRoutesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(http_route.ListHttpRoutesResponse())
        response = await client.list_http_routes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_http_routes_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_http_routes(http_route.ListHttpRoutesRequest(), parent='parent_value')

def test_list_http_routes_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_http_routes), '__call__') as call:
        call.side_effect = (http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute(), http_route.HttpRoute(), http_route.HttpRoute()], next_page_token='abc'), http_route.ListHttpRoutesResponse(http_routes=[], next_page_token='def'), http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute()], next_page_token='ghi'), http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute(), http_route.HttpRoute()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_http_routes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, http_route.HttpRoute) for i in results))

def test_list_http_routes_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_http_routes), '__call__') as call:
        call.side_effect = (http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute(), http_route.HttpRoute(), http_route.HttpRoute()], next_page_token='abc'), http_route.ListHttpRoutesResponse(http_routes=[], next_page_token='def'), http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute()], next_page_token='ghi'), http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute(), http_route.HttpRoute()]), RuntimeError)
        pages = list(client.list_http_routes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_http_routes_async_pager():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_http_routes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute(), http_route.HttpRoute(), http_route.HttpRoute()], next_page_token='abc'), http_route.ListHttpRoutesResponse(http_routes=[], next_page_token='def'), http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute()], next_page_token='ghi'), http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute(), http_route.HttpRoute()]), RuntimeError)
        async_pager = await client.list_http_routes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, http_route.HttpRoute) for i in responses))

@pytest.mark.asyncio
async def test_list_http_routes_async_pages():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_http_routes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute(), http_route.HttpRoute(), http_route.HttpRoute()], next_page_token='abc'), http_route.ListHttpRoutesResponse(http_routes=[], next_page_token='def'), http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute()], next_page_token='ghi'), http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute(), http_route.HttpRoute()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_http_routes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [http_route.GetHttpRouteRequest, dict])
def test_get_http_route(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_http_route), '__call__') as call:
        call.return_value = http_route.HttpRoute(name='name_value', self_link='self_link_value', description='description_value', hostnames=['hostnames_value'], meshes=['meshes_value'], gateways=['gateways_value'])
        response = client.get_http_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == http_route.GetHttpRouteRequest()
    assert isinstance(response, http_route.HttpRoute)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.hostnames == ['hostnames_value']
    assert response.meshes == ['meshes_value']
    assert response.gateways == ['gateways_value']

def test_get_http_route_empty_call():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_http_route), '__call__') as call:
        client.get_http_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == http_route.GetHttpRouteRequest()

@pytest.mark.asyncio
async def test_get_http_route_async(transport: str='grpc_asyncio', request_type=http_route.GetHttpRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_http_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(http_route.HttpRoute(name='name_value', self_link='self_link_value', description='description_value', hostnames=['hostnames_value'], meshes=['meshes_value'], gateways=['gateways_value']))
        response = await client.get_http_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == http_route.GetHttpRouteRequest()
    assert isinstance(response, http_route.HttpRoute)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.hostnames == ['hostnames_value']
    assert response.meshes == ['meshes_value']
    assert response.gateways == ['gateways_value']

@pytest.mark.asyncio
async def test_get_http_route_async_from_dict():
    await test_get_http_route_async(request_type=dict)

def test_get_http_route_field_headers():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = http_route.GetHttpRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_http_route), '__call__') as call:
        call.return_value = http_route.HttpRoute()
        client.get_http_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_http_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = http_route.GetHttpRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_http_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(http_route.HttpRoute())
        await client.get_http_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_http_route_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_http_route), '__call__') as call:
        call.return_value = http_route.HttpRoute()
        client.get_http_route(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_http_route_flattened_error():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_http_route(http_route.GetHttpRouteRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_http_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_http_route), '__call__') as call:
        call.return_value = http_route.HttpRoute()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(http_route.HttpRoute())
        response = await client.get_http_route(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_http_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_http_route(http_route.GetHttpRouteRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_http_route.CreateHttpRouteRequest, dict])
def test_create_http_route(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_http_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_http_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_http_route.CreateHttpRouteRequest()
    assert isinstance(response, future.Future)

def test_create_http_route_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_http_route), '__call__') as call:
        client.create_http_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_http_route.CreateHttpRouteRequest()

@pytest.mark.asyncio
async def test_create_http_route_async(transport: str='grpc_asyncio', request_type=gcn_http_route.CreateHttpRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_http_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_http_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_http_route.CreateHttpRouteRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_http_route_async_from_dict():
    await test_create_http_route_async(request_type=dict)

def test_create_http_route_field_headers():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_http_route.CreateHttpRouteRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_http_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_http_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_http_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_http_route.CreateHttpRouteRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_http_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_http_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_http_route_flattened():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_http_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_http_route(parent='parent_value', http_route=gcn_http_route.HttpRoute(name='name_value'), http_route_id='http_route_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].http_route
        mock_val = gcn_http_route.HttpRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].http_route_id
        mock_val = 'http_route_id_value'
        assert arg == mock_val

def test_create_http_route_flattened_error():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_http_route(gcn_http_route.CreateHttpRouteRequest(), parent='parent_value', http_route=gcn_http_route.HttpRoute(name='name_value'), http_route_id='http_route_id_value')

@pytest.mark.asyncio
async def test_create_http_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_http_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_http_route(parent='parent_value', http_route=gcn_http_route.HttpRoute(name='name_value'), http_route_id='http_route_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].http_route
        mock_val = gcn_http_route.HttpRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].http_route_id
        mock_val = 'http_route_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_http_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_http_route(gcn_http_route.CreateHttpRouteRequest(), parent='parent_value', http_route=gcn_http_route.HttpRoute(name='name_value'), http_route_id='http_route_id_value')

@pytest.mark.parametrize('request_type', [gcn_http_route.UpdateHttpRouteRequest, dict])
def test_update_http_route(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_http_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_http_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_http_route.UpdateHttpRouteRequest()
    assert isinstance(response, future.Future)

def test_update_http_route_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_http_route), '__call__') as call:
        client.update_http_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_http_route.UpdateHttpRouteRequest()

@pytest.mark.asyncio
async def test_update_http_route_async(transport: str='grpc_asyncio', request_type=gcn_http_route.UpdateHttpRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_http_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_http_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_http_route.UpdateHttpRouteRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_http_route_async_from_dict():
    await test_update_http_route_async(request_type=dict)

def test_update_http_route_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_http_route.UpdateHttpRouteRequest()
    request.http_route.name = 'name_value'
    with mock.patch.object(type(client.transport.update_http_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_http_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'http_route.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_http_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_http_route.UpdateHttpRouteRequest()
    request.http_route.name = 'name_value'
    with mock.patch.object(type(client.transport.update_http_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_http_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'http_route.name=name_value') in kw['metadata']

def test_update_http_route_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_http_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_http_route(http_route=gcn_http_route.HttpRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].http_route
        mock_val = gcn_http_route.HttpRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_http_route_flattened_error():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_http_route(gcn_http_route.UpdateHttpRouteRequest(), http_route=gcn_http_route.HttpRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_http_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_http_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_http_route(http_route=gcn_http_route.HttpRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].http_route
        mock_val = gcn_http_route.HttpRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_http_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_http_route(gcn_http_route.UpdateHttpRouteRequest(), http_route=gcn_http_route.HttpRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [http_route.DeleteHttpRouteRequest, dict])
def test_delete_http_route(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_http_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_http_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == http_route.DeleteHttpRouteRequest()
    assert isinstance(response, future.Future)

def test_delete_http_route_empty_call():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_http_route), '__call__') as call:
        client.delete_http_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == http_route.DeleteHttpRouteRequest()

@pytest.mark.asyncio
async def test_delete_http_route_async(transport: str='grpc_asyncio', request_type=http_route.DeleteHttpRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_http_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_http_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == http_route.DeleteHttpRouteRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_http_route_async_from_dict():
    await test_delete_http_route_async(request_type=dict)

def test_delete_http_route_field_headers():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = http_route.DeleteHttpRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_http_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_http_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_http_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = http_route.DeleteHttpRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_http_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_http_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_http_route_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_http_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_http_route(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_http_route_flattened_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_http_route(http_route.DeleteHttpRouteRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_http_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_http_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_http_route(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_http_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_http_route(http_route.DeleteHttpRouteRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [tcp_route.ListTcpRoutesRequest, dict])
def test_list_tcp_routes(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tcp_routes), '__call__') as call:
        call.return_value = tcp_route.ListTcpRoutesResponse(next_page_token='next_page_token_value')
        response = client.list_tcp_routes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tcp_route.ListTcpRoutesRequest()
    assert isinstance(response, pagers.ListTcpRoutesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tcp_routes_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_tcp_routes), '__call__') as call:
        client.list_tcp_routes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tcp_route.ListTcpRoutesRequest()

@pytest.mark.asyncio
async def test_list_tcp_routes_async(transport: str='grpc_asyncio', request_type=tcp_route.ListTcpRoutesRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tcp_routes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tcp_route.ListTcpRoutesResponse(next_page_token='next_page_token_value'))
        response = await client.list_tcp_routes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tcp_route.ListTcpRoutesRequest()
    assert isinstance(response, pagers.ListTcpRoutesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_tcp_routes_async_from_dict():
    await test_list_tcp_routes_async(request_type=dict)

def test_list_tcp_routes_field_headers():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = tcp_route.ListTcpRoutesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_tcp_routes), '__call__') as call:
        call.return_value = tcp_route.ListTcpRoutesResponse()
        client.list_tcp_routes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_tcp_routes_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tcp_route.ListTcpRoutesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_tcp_routes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tcp_route.ListTcpRoutesResponse())
        await client.list_tcp_routes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_tcp_routes_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tcp_routes), '__call__') as call:
        call.return_value = tcp_route.ListTcpRoutesResponse()
        client.list_tcp_routes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_tcp_routes_flattened_error():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_tcp_routes(tcp_route.ListTcpRoutesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_tcp_routes_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tcp_routes), '__call__') as call:
        call.return_value = tcp_route.ListTcpRoutesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tcp_route.ListTcpRoutesResponse())
        response = await client.list_tcp_routes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_tcp_routes_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_tcp_routes(tcp_route.ListTcpRoutesRequest(), parent='parent_value')

def test_list_tcp_routes_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tcp_routes), '__call__') as call:
        call.side_effect = (tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute(), tcp_route.TcpRoute(), tcp_route.TcpRoute()], next_page_token='abc'), tcp_route.ListTcpRoutesResponse(tcp_routes=[], next_page_token='def'), tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute()], next_page_token='ghi'), tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute(), tcp_route.TcpRoute()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_tcp_routes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tcp_route.TcpRoute) for i in results))

def test_list_tcp_routes_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tcp_routes), '__call__') as call:
        call.side_effect = (tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute(), tcp_route.TcpRoute(), tcp_route.TcpRoute()], next_page_token='abc'), tcp_route.ListTcpRoutesResponse(tcp_routes=[], next_page_token='def'), tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute()], next_page_token='ghi'), tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute(), tcp_route.TcpRoute()]), RuntimeError)
        pages = list(client.list_tcp_routes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_tcp_routes_async_pager():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tcp_routes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute(), tcp_route.TcpRoute(), tcp_route.TcpRoute()], next_page_token='abc'), tcp_route.ListTcpRoutesResponse(tcp_routes=[], next_page_token='def'), tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute()], next_page_token='ghi'), tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute(), tcp_route.TcpRoute()]), RuntimeError)
        async_pager = await client.list_tcp_routes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, tcp_route.TcpRoute) for i in responses))

@pytest.mark.asyncio
async def test_list_tcp_routes_async_pages():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tcp_routes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute(), tcp_route.TcpRoute(), tcp_route.TcpRoute()], next_page_token='abc'), tcp_route.ListTcpRoutesResponse(tcp_routes=[], next_page_token='def'), tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute()], next_page_token='ghi'), tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute(), tcp_route.TcpRoute()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_tcp_routes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tcp_route.GetTcpRouteRequest, dict])
def test_get_tcp_route(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_tcp_route), '__call__') as call:
        call.return_value = tcp_route.TcpRoute(name='name_value', self_link='self_link_value', description='description_value', meshes=['meshes_value'], gateways=['gateways_value'])
        response = client.get_tcp_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tcp_route.GetTcpRouteRequest()
    assert isinstance(response, tcp_route.TcpRoute)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.meshes == ['meshes_value']
    assert response.gateways == ['gateways_value']

def test_get_tcp_route_empty_call():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_tcp_route), '__call__') as call:
        client.get_tcp_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tcp_route.GetTcpRouteRequest()

@pytest.mark.asyncio
async def test_get_tcp_route_async(transport: str='grpc_asyncio', request_type=tcp_route.GetTcpRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_tcp_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tcp_route.TcpRoute(name='name_value', self_link='self_link_value', description='description_value', meshes=['meshes_value'], gateways=['gateways_value']))
        response = await client.get_tcp_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tcp_route.GetTcpRouteRequest()
    assert isinstance(response, tcp_route.TcpRoute)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.meshes == ['meshes_value']
    assert response.gateways == ['gateways_value']

@pytest.mark.asyncio
async def test_get_tcp_route_async_from_dict():
    await test_get_tcp_route_async(request_type=dict)

def test_get_tcp_route_field_headers():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = tcp_route.GetTcpRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_tcp_route), '__call__') as call:
        call.return_value = tcp_route.TcpRoute()
        client.get_tcp_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_tcp_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tcp_route.GetTcpRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_tcp_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tcp_route.TcpRoute())
        await client.get_tcp_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_tcp_route_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_tcp_route), '__call__') as call:
        call.return_value = tcp_route.TcpRoute()
        client.get_tcp_route(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_tcp_route_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_tcp_route(tcp_route.GetTcpRouteRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_tcp_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_tcp_route), '__call__') as call:
        call.return_value = tcp_route.TcpRoute()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tcp_route.TcpRoute())
        response = await client.get_tcp_route(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_tcp_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_tcp_route(tcp_route.GetTcpRouteRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_tcp_route.CreateTcpRouteRequest, dict])
def test_create_tcp_route(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_tcp_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_tcp_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_tcp_route.CreateTcpRouteRequest()
    assert isinstance(response, future.Future)

def test_create_tcp_route_empty_call():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_tcp_route), '__call__') as call:
        client.create_tcp_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_tcp_route.CreateTcpRouteRequest()

@pytest.mark.asyncio
async def test_create_tcp_route_async(transport: str='grpc_asyncio', request_type=gcn_tcp_route.CreateTcpRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_tcp_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_tcp_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_tcp_route.CreateTcpRouteRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_tcp_route_async_from_dict():
    await test_create_tcp_route_async(request_type=dict)

def test_create_tcp_route_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_tcp_route.CreateTcpRouteRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_tcp_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_tcp_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_tcp_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_tcp_route.CreateTcpRouteRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_tcp_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_tcp_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_tcp_route_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_tcp_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_tcp_route(parent='parent_value', tcp_route=gcn_tcp_route.TcpRoute(name='name_value'), tcp_route_id='tcp_route_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].tcp_route
        mock_val = gcn_tcp_route.TcpRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].tcp_route_id
        mock_val = 'tcp_route_id_value'
        assert arg == mock_val

def test_create_tcp_route_flattened_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_tcp_route(gcn_tcp_route.CreateTcpRouteRequest(), parent='parent_value', tcp_route=gcn_tcp_route.TcpRoute(name='name_value'), tcp_route_id='tcp_route_id_value')

@pytest.mark.asyncio
async def test_create_tcp_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_tcp_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_tcp_route(parent='parent_value', tcp_route=gcn_tcp_route.TcpRoute(name='name_value'), tcp_route_id='tcp_route_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].tcp_route
        mock_val = gcn_tcp_route.TcpRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].tcp_route_id
        mock_val = 'tcp_route_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_tcp_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_tcp_route(gcn_tcp_route.CreateTcpRouteRequest(), parent='parent_value', tcp_route=gcn_tcp_route.TcpRoute(name='name_value'), tcp_route_id='tcp_route_id_value')

@pytest.mark.parametrize('request_type', [gcn_tcp_route.UpdateTcpRouteRequest, dict])
def test_update_tcp_route(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_tcp_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_tcp_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_tcp_route.UpdateTcpRouteRequest()
    assert isinstance(response, future.Future)

def test_update_tcp_route_empty_call():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_tcp_route), '__call__') as call:
        client.update_tcp_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_tcp_route.UpdateTcpRouteRequest()

@pytest.mark.asyncio
async def test_update_tcp_route_async(transport: str='grpc_asyncio', request_type=gcn_tcp_route.UpdateTcpRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_tcp_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_tcp_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_tcp_route.UpdateTcpRouteRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_tcp_route_async_from_dict():
    await test_update_tcp_route_async(request_type=dict)

def test_update_tcp_route_field_headers():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_tcp_route.UpdateTcpRouteRequest()
    request.tcp_route.name = 'name_value'
    with mock.patch.object(type(client.transport.update_tcp_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_tcp_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'tcp_route.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_tcp_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_tcp_route.UpdateTcpRouteRequest()
    request.tcp_route.name = 'name_value'
    with mock.patch.object(type(client.transport.update_tcp_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_tcp_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'tcp_route.name=name_value') in kw['metadata']

def test_update_tcp_route_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_tcp_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_tcp_route(tcp_route=gcn_tcp_route.TcpRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].tcp_route
        mock_val = gcn_tcp_route.TcpRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_tcp_route_flattened_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_tcp_route(gcn_tcp_route.UpdateTcpRouteRequest(), tcp_route=gcn_tcp_route.TcpRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_tcp_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_tcp_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_tcp_route(tcp_route=gcn_tcp_route.TcpRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].tcp_route
        mock_val = gcn_tcp_route.TcpRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_tcp_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_tcp_route(gcn_tcp_route.UpdateTcpRouteRequest(), tcp_route=gcn_tcp_route.TcpRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [tcp_route.DeleteTcpRouteRequest, dict])
def test_delete_tcp_route(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_tcp_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_tcp_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tcp_route.DeleteTcpRouteRequest()
    assert isinstance(response, future.Future)

def test_delete_tcp_route_empty_call():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_tcp_route), '__call__') as call:
        client.delete_tcp_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tcp_route.DeleteTcpRouteRequest()

@pytest.mark.asyncio
async def test_delete_tcp_route_async(transport: str='grpc_asyncio', request_type=tcp_route.DeleteTcpRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_tcp_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_tcp_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tcp_route.DeleteTcpRouteRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_tcp_route_async_from_dict():
    await test_delete_tcp_route_async(request_type=dict)

def test_delete_tcp_route_field_headers():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = tcp_route.DeleteTcpRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_tcp_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_tcp_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_tcp_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tcp_route.DeleteTcpRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_tcp_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_tcp_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_tcp_route_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_tcp_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_tcp_route(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_tcp_route_flattened_error():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_tcp_route(tcp_route.DeleteTcpRouteRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_tcp_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_tcp_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_tcp_route(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_tcp_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_tcp_route(tcp_route.DeleteTcpRouteRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [tls_route.ListTlsRoutesRequest, dict])
def test_list_tls_routes(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tls_routes), '__call__') as call:
        call.return_value = tls_route.ListTlsRoutesResponse(next_page_token='next_page_token_value')
        response = client.list_tls_routes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tls_route.ListTlsRoutesRequest()
    assert isinstance(response, pagers.ListTlsRoutesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tls_routes_empty_call():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_tls_routes), '__call__') as call:
        client.list_tls_routes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tls_route.ListTlsRoutesRequest()

@pytest.mark.asyncio
async def test_list_tls_routes_async(transport: str='grpc_asyncio', request_type=tls_route.ListTlsRoutesRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tls_routes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tls_route.ListTlsRoutesResponse(next_page_token='next_page_token_value'))
        response = await client.list_tls_routes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tls_route.ListTlsRoutesRequest()
    assert isinstance(response, pagers.ListTlsRoutesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_tls_routes_async_from_dict():
    await test_list_tls_routes_async(request_type=dict)

def test_list_tls_routes_field_headers():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = tls_route.ListTlsRoutesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_tls_routes), '__call__') as call:
        call.return_value = tls_route.ListTlsRoutesResponse()
        client.list_tls_routes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_tls_routes_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tls_route.ListTlsRoutesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_tls_routes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tls_route.ListTlsRoutesResponse())
        await client.list_tls_routes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_tls_routes_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tls_routes), '__call__') as call:
        call.return_value = tls_route.ListTlsRoutesResponse()
        client.list_tls_routes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_tls_routes_flattened_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_tls_routes(tls_route.ListTlsRoutesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_tls_routes_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tls_routes), '__call__') as call:
        call.return_value = tls_route.ListTlsRoutesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tls_route.ListTlsRoutesResponse())
        response = await client.list_tls_routes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_tls_routes_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_tls_routes(tls_route.ListTlsRoutesRequest(), parent='parent_value')

def test_list_tls_routes_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tls_routes), '__call__') as call:
        call.side_effect = (tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute(), tls_route.TlsRoute(), tls_route.TlsRoute()], next_page_token='abc'), tls_route.ListTlsRoutesResponse(tls_routes=[], next_page_token='def'), tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute()], next_page_token='ghi'), tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute(), tls_route.TlsRoute()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_tls_routes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tls_route.TlsRoute) for i in results))

def test_list_tls_routes_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tls_routes), '__call__') as call:
        call.side_effect = (tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute(), tls_route.TlsRoute(), tls_route.TlsRoute()], next_page_token='abc'), tls_route.ListTlsRoutesResponse(tls_routes=[], next_page_token='def'), tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute()], next_page_token='ghi'), tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute(), tls_route.TlsRoute()]), RuntimeError)
        pages = list(client.list_tls_routes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_tls_routes_async_pager():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tls_routes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute(), tls_route.TlsRoute(), tls_route.TlsRoute()], next_page_token='abc'), tls_route.ListTlsRoutesResponse(tls_routes=[], next_page_token='def'), tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute()], next_page_token='ghi'), tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute(), tls_route.TlsRoute()]), RuntimeError)
        async_pager = await client.list_tls_routes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, tls_route.TlsRoute) for i in responses))

@pytest.mark.asyncio
async def test_list_tls_routes_async_pages():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tls_routes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute(), tls_route.TlsRoute(), tls_route.TlsRoute()], next_page_token='abc'), tls_route.ListTlsRoutesResponse(tls_routes=[], next_page_token='def'), tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute()], next_page_token='ghi'), tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute(), tls_route.TlsRoute()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_tls_routes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tls_route.GetTlsRouteRequest, dict])
def test_get_tls_route(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_tls_route), '__call__') as call:
        call.return_value = tls_route.TlsRoute(name='name_value', self_link='self_link_value', description='description_value', meshes=['meshes_value'], gateways=['gateways_value'])
        response = client.get_tls_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tls_route.GetTlsRouteRequest()
    assert isinstance(response, tls_route.TlsRoute)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.meshes == ['meshes_value']
    assert response.gateways == ['gateways_value']

def test_get_tls_route_empty_call():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_tls_route), '__call__') as call:
        client.get_tls_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tls_route.GetTlsRouteRequest()

@pytest.mark.asyncio
async def test_get_tls_route_async(transport: str='grpc_asyncio', request_type=tls_route.GetTlsRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_tls_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tls_route.TlsRoute(name='name_value', self_link='self_link_value', description='description_value', meshes=['meshes_value'], gateways=['gateways_value']))
        response = await client.get_tls_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tls_route.GetTlsRouteRequest()
    assert isinstance(response, tls_route.TlsRoute)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.meshes == ['meshes_value']
    assert response.gateways == ['gateways_value']

@pytest.mark.asyncio
async def test_get_tls_route_async_from_dict():
    await test_get_tls_route_async(request_type=dict)

def test_get_tls_route_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = tls_route.GetTlsRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_tls_route), '__call__') as call:
        call.return_value = tls_route.TlsRoute()
        client.get_tls_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_tls_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tls_route.GetTlsRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_tls_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tls_route.TlsRoute())
        await client.get_tls_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_tls_route_flattened():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_tls_route), '__call__') as call:
        call.return_value = tls_route.TlsRoute()
        client.get_tls_route(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_tls_route_flattened_error():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_tls_route(tls_route.GetTlsRouteRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_tls_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_tls_route), '__call__') as call:
        call.return_value = tls_route.TlsRoute()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tls_route.TlsRoute())
        response = await client.get_tls_route(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_tls_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_tls_route(tls_route.GetTlsRouteRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_tls_route.CreateTlsRouteRequest, dict])
def test_create_tls_route(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_tls_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_tls_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_tls_route.CreateTlsRouteRequest()
    assert isinstance(response, future.Future)

def test_create_tls_route_empty_call():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_tls_route), '__call__') as call:
        client.create_tls_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_tls_route.CreateTlsRouteRequest()

@pytest.mark.asyncio
async def test_create_tls_route_async(transport: str='grpc_asyncio', request_type=gcn_tls_route.CreateTlsRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_tls_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_tls_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_tls_route.CreateTlsRouteRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_tls_route_async_from_dict():
    await test_create_tls_route_async(request_type=dict)

def test_create_tls_route_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_tls_route.CreateTlsRouteRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_tls_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_tls_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_tls_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_tls_route.CreateTlsRouteRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_tls_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_tls_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_tls_route_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_tls_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_tls_route(parent='parent_value', tls_route=gcn_tls_route.TlsRoute(name='name_value'), tls_route_id='tls_route_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].tls_route
        mock_val = gcn_tls_route.TlsRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].tls_route_id
        mock_val = 'tls_route_id_value'
        assert arg == mock_val

def test_create_tls_route_flattened_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_tls_route(gcn_tls_route.CreateTlsRouteRequest(), parent='parent_value', tls_route=gcn_tls_route.TlsRoute(name='name_value'), tls_route_id='tls_route_id_value')

@pytest.mark.asyncio
async def test_create_tls_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_tls_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_tls_route(parent='parent_value', tls_route=gcn_tls_route.TlsRoute(name='name_value'), tls_route_id='tls_route_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].tls_route
        mock_val = gcn_tls_route.TlsRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].tls_route_id
        mock_val = 'tls_route_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_tls_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_tls_route(gcn_tls_route.CreateTlsRouteRequest(), parent='parent_value', tls_route=gcn_tls_route.TlsRoute(name='name_value'), tls_route_id='tls_route_id_value')

@pytest.mark.parametrize('request_type', [gcn_tls_route.UpdateTlsRouteRequest, dict])
def test_update_tls_route(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_tls_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_tls_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_tls_route.UpdateTlsRouteRequest()
    assert isinstance(response, future.Future)

def test_update_tls_route_empty_call():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_tls_route), '__call__') as call:
        client.update_tls_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_tls_route.UpdateTlsRouteRequest()

@pytest.mark.asyncio
async def test_update_tls_route_async(transport: str='grpc_asyncio', request_type=gcn_tls_route.UpdateTlsRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_tls_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_tls_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_tls_route.UpdateTlsRouteRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_tls_route_async_from_dict():
    await test_update_tls_route_async(request_type=dict)

def test_update_tls_route_field_headers():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_tls_route.UpdateTlsRouteRequest()
    request.tls_route.name = 'name_value'
    with mock.patch.object(type(client.transport.update_tls_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_tls_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'tls_route.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_tls_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_tls_route.UpdateTlsRouteRequest()
    request.tls_route.name = 'name_value'
    with mock.patch.object(type(client.transport.update_tls_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_tls_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'tls_route.name=name_value') in kw['metadata']

def test_update_tls_route_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_tls_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_tls_route(tls_route=gcn_tls_route.TlsRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].tls_route
        mock_val = gcn_tls_route.TlsRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_tls_route_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_tls_route(gcn_tls_route.UpdateTlsRouteRequest(), tls_route=gcn_tls_route.TlsRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_tls_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_tls_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_tls_route(tls_route=gcn_tls_route.TlsRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].tls_route
        mock_val = gcn_tls_route.TlsRoute(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_tls_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_tls_route(gcn_tls_route.UpdateTlsRouteRequest(), tls_route=gcn_tls_route.TlsRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [tls_route.DeleteTlsRouteRequest, dict])
def test_delete_tls_route(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_tls_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_tls_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tls_route.DeleteTlsRouteRequest()
    assert isinstance(response, future.Future)

def test_delete_tls_route_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_tls_route), '__call__') as call:
        client.delete_tls_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tls_route.DeleteTlsRouteRequest()

@pytest.mark.asyncio
async def test_delete_tls_route_async(transport: str='grpc_asyncio', request_type=tls_route.DeleteTlsRouteRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_tls_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_tls_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tls_route.DeleteTlsRouteRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_tls_route_async_from_dict():
    await test_delete_tls_route_async(request_type=dict)

def test_delete_tls_route_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = tls_route.DeleteTlsRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_tls_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_tls_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_tls_route_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tls_route.DeleteTlsRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_tls_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_tls_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_tls_route_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_tls_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_tls_route(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_tls_route_flattened_error():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_tls_route(tls_route.DeleteTlsRouteRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_tls_route_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_tls_route), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_tls_route(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_tls_route_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_tls_route(tls_route.DeleteTlsRouteRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service_binding.ListServiceBindingsRequest, dict])
def test_list_service_bindings(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_service_bindings), '__call__') as call:
        call.return_value = service_binding.ListServiceBindingsResponse(next_page_token='next_page_token_value')
        response = client.list_service_bindings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_binding.ListServiceBindingsRequest()
    assert isinstance(response, pagers.ListServiceBindingsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_service_bindings_empty_call():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_service_bindings), '__call__') as call:
        client.list_service_bindings()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_binding.ListServiceBindingsRequest()

@pytest.mark.asyncio
async def test_list_service_bindings_async(transport: str='grpc_asyncio', request_type=service_binding.ListServiceBindingsRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_service_bindings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_binding.ListServiceBindingsResponse(next_page_token='next_page_token_value'))
        response = await client.list_service_bindings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_binding.ListServiceBindingsRequest()
    assert isinstance(response, pagers.ListServiceBindingsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_service_bindings_async_from_dict():
    await test_list_service_bindings_async(request_type=dict)

def test_list_service_bindings_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_binding.ListServiceBindingsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_service_bindings), '__call__') as call:
        call.return_value = service_binding.ListServiceBindingsResponse()
        client.list_service_bindings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_service_bindings_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_binding.ListServiceBindingsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_service_bindings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_binding.ListServiceBindingsResponse())
        await client.list_service_bindings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_service_bindings_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_service_bindings), '__call__') as call:
        call.return_value = service_binding.ListServiceBindingsResponse()
        client.list_service_bindings(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_service_bindings_flattened_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_service_bindings(service_binding.ListServiceBindingsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_service_bindings_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_service_bindings), '__call__') as call:
        call.return_value = service_binding.ListServiceBindingsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_binding.ListServiceBindingsResponse())
        response = await client.list_service_bindings(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_service_bindings_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_service_bindings(service_binding.ListServiceBindingsRequest(), parent='parent_value')

def test_list_service_bindings_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_service_bindings), '__call__') as call:
        call.side_effect = (service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding(), service_binding.ServiceBinding(), service_binding.ServiceBinding()], next_page_token='abc'), service_binding.ListServiceBindingsResponse(service_bindings=[], next_page_token='def'), service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding()], next_page_token='ghi'), service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding(), service_binding.ServiceBinding()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_service_bindings(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service_binding.ServiceBinding) for i in results))

def test_list_service_bindings_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_service_bindings), '__call__') as call:
        call.side_effect = (service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding(), service_binding.ServiceBinding(), service_binding.ServiceBinding()], next_page_token='abc'), service_binding.ListServiceBindingsResponse(service_bindings=[], next_page_token='def'), service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding()], next_page_token='ghi'), service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding(), service_binding.ServiceBinding()]), RuntimeError)
        pages = list(client.list_service_bindings(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_service_bindings_async_pager():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_service_bindings), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding(), service_binding.ServiceBinding(), service_binding.ServiceBinding()], next_page_token='abc'), service_binding.ListServiceBindingsResponse(service_bindings=[], next_page_token='def'), service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding()], next_page_token='ghi'), service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding(), service_binding.ServiceBinding()]), RuntimeError)
        async_pager = await client.list_service_bindings(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, service_binding.ServiceBinding) for i in responses))

@pytest.mark.asyncio
async def test_list_service_bindings_async_pages():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_service_bindings), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding(), service_binding.ServiceBinding(), service_binding.ServiceBinding()], next_page_token='abc'), service_binding.ListServiceBindingsResponse(service_bindings=[], next_page_token='def'), service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding()], next_page_token='ghi'), service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding(), service_binding.ServiceBinding()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_service_bindings(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service_binding.GetServiceBindingRequest, dict])
def test_get_service_binding(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service_binding), '__call__') as call:
        call.return_value = service_binding.ServiceBinding(name='name_value', description='description_value', service='service_value')
        response = client.get_service_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_binding.GetServiceBindingRequest()
    assert isinstance(response, service_binding.ServiceBinding)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.service == 'service_value'

def test_get_service_binding_empty_call():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_service_binding), '__call__') as call:
        client.get_service_binding()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_binding.GetServiceBindingRequest()

@pytest.mark.asyncio
async def test_get_service_binding_async(transport: str='grpc_asyncio', request_type=service_binding.GetServiceBindingRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_binding.ServiceBinding(name='name_value', description='description_value', service='service_value'))
        response = await client.get_service_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_binding.GetServiceBindingRequest()
    assert isinstance(response, service_binding.ServiceBinding)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.service == 'service_value'

@pytest.mark.asyncio
async def test_get_service_binding_async_from_dict():
    await test_get_service_binding_async(request_type=dict)

def test_get_service_binding_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_binding.GetServiceBindingRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_service_binding), '__call__') as call:
        call.return_value = service_binding.ServiceBinding()
        client.get_service_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_service_binding_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_binding.GetServiceBindingRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_service_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_binding.ServiceBinding())
        await client.get_service_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_service_binding_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_service_binding), '__call__') as call:
        call.return_value = service_binding.ServiceBinding()
        client.get_service_binding(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_service_binding_flattened_error():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_service_binding(service_binding.GetServiceBindingRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_service_binding_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_service_binding), '__call__') as call:
        call.return_value = service_binding.ServiceBinding()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_binding.ServiceBinding())
        response = await client.get_service_binding(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_service_binding_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_service_binding(service_binding.GetServiceBindingRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_service_binding.CreateServiceBindingRequest, dict])
def test_create_service_binding(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_service_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_service_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_service_binding.CreateServiceBindingRequest()
    assert isinstance(response, future.Future)

def test_create_service_binding_empty_call():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_service_binding), '__call__') as call:
        client.create_service_binding()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_service_binding.CreateServiceBindingRequest()

@pytest.mark.asyncio
async def test_create_service_binding_async(transport: str='grpc_asyncio', request_type=gcn_service_binding.CreateServiceBindingRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_service_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_service_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_service_binding.CreateServiceBindingRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_service_binding_async_from_dict():
    await test_create_service_binding_async(request_type=dict)

def test_create_service_binding_field_headers():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_service_binding.CreateServiceBindingRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_service_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_service_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_service_binding_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_service_binding.CreateServiceBindingRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_service_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_service_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_service_binding_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_service_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_service_binding(parent='parent_value', service_binding=gcn_service_binding.ServiceBinding(name='name_value'), service_binding_id='service_binding_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].service_binding
        mock_val = gcn_service_binding.ServiceBinding(name='name_value')
        assert arg == mock_val
        arg = args[0].service_binding_id
        mock_val = 'service_binding_id_value'
        assert arg == mock_val

def test_create_service_binding_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_service_binding(gcn_service_binding.CreateServiceBindingRequest(), parent='parent_value', service_binding=gcn_service_binding.ServiceBinding(name='name_value'), service_binding_id='service_binding_id_value')

@pytest.mark.asyncio
async def test_create_service_binding_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_service_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_service_binding(parent='parent_value', service_binding=gcn_service_binding.ServiceBinding(name='name_value'), service_binding_id='service_binding_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].service_binding
        mock_val = gcn_service_binding.ServiceBinding(name='name_value')
        assert arg == mock_val
        arg = args[0].service_binding_id
        mock_val = 'service_binding_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_service_binding_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_service_binding(gcn_service_binding.CreateServiceBindingRequest(), parent='parent_value', service_binding=gcn_service_binding.ServiceBinding(name='name_value'), service_binding_id='service_binding_id_value')

@pytest.mark.parametrize('request_type', [service_binding.DeleteServiceBindingRequest, dict])
def test_delete_service_binding(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_service_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_service_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_binding.DeleteServiceBindingRequest()
    assert isinstance(response, future.Future)

def test_delete_service_binding_empty_call():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_service_binding), '__call__') as call:
        client.delete_service_binding()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_binding.DeleteServiceBindingRequest()

@pytest.mark.asyncio
async def test_delete_service_binding_async(transport: str='grpc_asyncio', request_type=service_binding.DeleteServiceBindingRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_service_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_service_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_binding.DeleteServiceBindingRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_service_binding_async_from_dict():
    await test_delete_service_binding_async(request_type=dict)

def test_delete_service_binding_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_binding.DeleteServiceBindingRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_service_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_service_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_service_binding_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_binding.DeleteServiceBindingRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_service_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_service_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_service_binding_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_service_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_service_binding(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_service_binding_flattened_error():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_service_binding(service_binding.DeleteServiceBindingRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_service_binding_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_service_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_service_binding(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_service_binding_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_service_binding(service_binding.DeleteServiceBindingRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [mesh.ListMeshesRequest, dict])
def test_list_meshes(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_meshes), '__call__') as call:
        call.return_value = mesh.ListMeshesResponse(next_page_token='next_page_token_value')
        response = client.list_meshes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == mesh.ListMeshesRequest()
    assert isinstance(response, pagers.ListMeshesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_meshes_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_meshes), '__call__') as call:
        client.list_meshes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == mesh.ListMeshesRequest()

@pytest.mark.asyncio
async def test_list_meshes_async(transport: str='grpc_asyncio', request_type=mesh.ListMeshesRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_meshes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(mesh.ListMeshesResponse(next_page_token='next_page_token_value'))
        response = await client.list_meshes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == mesh.ListMeshesRequest()
    assert isinstance(response, pagers.ListMeshesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_meshes_async_from_dict():
    await test_list_meshes_async(request_type=dict)

def test_list_meshes_field_headers():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = mesh.ListMeshesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_meshes), '__call__') as call:
        call.return_value = mesh.ListMeshesResponse()
        client.list_meshes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_meshes_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = mesh.ListMeshesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_meshes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(mesh.ListMeshesResponse())
        await client.list_meshes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_meshes_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_meshes), '__call__') as call:
        call.return_value = mesh.ListMeshesResponse()
        client.list_meshes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_meshes_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_meshes(mesh.ListMeshesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_meshes_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_meshes), '__call__') as call:
        call.return_value = mesh.ListMeshesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(mesh.ListMeshesResponse())
        response = await client.list_meshes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_meshes_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_meshes(mesh.ListMeshesRequest(), parent='parent_value')

def test_list_meshes_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_meshes), '__call__') as call:
        call.side_effect = (mesh.ListMeshesResponse(meshes=[mesh.Mesh(), mesh.Mesh(), mesh.Mesh()], next_page_token='abc'), mesh.ListMeshesResponse(meshes=[], next_page_token='def'), mesh.ListMeshesResponse(meshes=[mesh.Mesh()], next_page_token='ghi'), mesh.ListMeshesResponse(meshes=[mesh.Mesh(), mesh.Mesh()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_meshes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, mesh.Mesh) for i in results))

def test_list_meshes_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_meshes), '__call__') as call:
        call.side_effect = (mesh.ListMeshesResponse(meshes=[mesh.Mesh(), mesh.Mesh(), mesh.Mesh()], next_page_token='abc'), mesh.ListMeshesResponse(meshes=[], next_page_token='def'), mesh.ListMeshesResponse(meshes=[mesh.Mesh()], next_page_token='ghi'), mesh.ListMeshesResponse(meshes=[mesh.Mesh(), mesh.Mesh()]), RuntimeError)
        pages = list(client.list_meshes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_meshes_async_pager():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_meshes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (mesh.ListMeshesResponse(meshes=[mesh.Mesh(), mesh.Mesh(), mesh.Mesh()], next_page_token='abc'), mesh.ListMeshesResponse(meshes=[], next_page_token='def'), mesh.ListMeshesResponse(meshes=[mesh.Mesh()], next_page_token='ghi'), mesh.ListMeshesResponse(meshes=[mesh.Mesh(), mesh.Mesh()]), RuntimeError)
        async_pager = await client.list_meshes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, mesh.Mesh) for i in responses))

@pytest.mark.asyncio
async def test_list_meshes_async_pages():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_meshes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (mesh.ListMeshesResponse(meshes=[mesh.Mesh(), mesh.Mesh(), mesh.Mesh()], next_page_token='abc'), mesh.ListMeshesResponse(meshes=[], next_page_token='def'), mesh.ListMeshesResponse(meshes=[mesh.Mesh()], next_page_token='ghi'), mesh.ListMeshesResponse(meshes=[mesh.Mesh(), mesh.Mesh()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_meshes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [mesh.GetMeshRequest, dict])
def test_get_mesh(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_mesh), '__call__') as call:
        call.return_value = mesh.Mesh(name='name_value', self_link='self_link_value', description='description_value', interception_port=1848)
        response = client.get_mesh(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == mesh.GetMeshRequest()
    assert isinstance(response, mesh.Mesh)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.interception_port == 1848

def test_get_mesh_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_mesh), '__call__') as call:
        client.get_mesh()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == mesh.GetMeshRequest()

@pytest.mark.asyncio
async def test_get_mesh_async(transport: str='grpc_asyncio', request_type=mesh.GetMeshRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_mesh), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(mesh.Mesh(name='name_value', self_link='self_link_value', description='description_value', interception_port=1848))
        response = await client.get_mesh(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == mesh.GetMeshRequest()
    assert isinstance(response, mesh.Mesh)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.interception_port == 1848

@pytest.mark.asyncio
async def test_get_mesh_async_from_dict():
    await test_get_mesh_async(request_type=dict)

def test_get_mesh_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = mesh.GetMeshRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_mesh), '__call__') as call:
        call.return_value = mesh.Mesh()
        client.get_mesh(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_mesh_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = mesh.GetMeshRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_mesh), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(mesh.Mesh())
        await client.get_mesh(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_mesh_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_mesh), '__call__') as call:
        call.return_value = mesh.Mesh()
        client.get_mesh(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_mesh_flattened_error():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_mesh(mesh.GetMeshRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_mesh_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_mesh), '__call__') as call:
        call.return_value = mesh.Mesh()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(mesh.Mesh())
        response = await client.get_mesh(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_mesh_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_mesh(mesh.GetMeshRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_mesh.CreateMeshRequest, dict])
def test_create_mesh(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_mesh), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_mesh(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_mesh.CreateMeshRequest()
    assert isinstance(response, future.Future)

def test_create_mesh_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_mesh), '__call__') as call:
        client.create_mesh()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_mesh.CreateMeshRequest()

@pytest.mark.asyncio
async def test_create_mesh_async(transport: str='grpc_asyncio', request_type=gcn_mesh.CreateMeshRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_mesh), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_mesh(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_mesh.CreateMeshRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_mesh_async_from_dict():
    await test_create_mesh_async(request_type=dict)

def test_create_mesh_field_headers():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_mesh.CreateMeshRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_mesh), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_mesh(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_mesh_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_mesh.CreateMeshRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_mesh), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_mesh(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_mesh_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_mesh), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_mesh(parent='parent_value', mesh=gcn_mesh.Mesh(name='name_value'), mesh_id='mesh_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].mesh
        mock_val = gcn_mesh.Mesh(name='name_value')
        assert arg == mock_val
        arg = args[0].mesh_id
        mock_val = 'mesh_id_value'
        assert arg == mock_val

def test_create_mesh_flattened_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_mesh(gcn_mesh.CreateMeshRequest(), parent='parent_value', mesh=gcn_mesh.Mesh(name='name_value'), mesh_id='mesh_id_value')

@pytest.mark.asyncio
async def test_create_mesh_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_mesh), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_mesh(parent='parent_value', mesh=gcn_mesh.Mesh(name='name_value'), mesh_id='mesh_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].mesh
        mock_val = gcn_mesh.Mesh(name='name_value')
        assert arg == mock_val
        arg = args[0].mesh_id
        mock_val = 'mesh_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_mesh_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_mesh(gcn_mesh.CreateMeshRequest(), parent='parent_value', mesh=gcn_mesh.Mesh(name='name_value'), mesh_id='mesh_id_value')

@pytest.mark.parametrize('request_type', [gcn_mesh.UpdateMeshRequest, dict])
def test_update_mesh(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_mesh), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_mesh(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_mesh.UpdateMeshRequest()
    assert isinstance(response, future.Future)

def test_update_mesh_empty_call():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_mesh), '__call__') as call:
        client.update_mesh()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_mesh.UpdateMeshRequest()

@pytest.mark.asyncio
async def test_update_mesh_async(transport: str='grpc_asyncio', request_type=gcn_mesh.UpdateMeshRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_mesh), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_mesh(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_mesh.UpdateMeshRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_mesh_async_from_dict():
    await test_update_mesh_async(request_type=dict)

def test_update_mesh_field_headers():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_mesh.UpdateMeshRequest()
    request.mesh.name = 'name_value'
    with mock.patch.object(type(client.transport.update_mesh), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_mesh(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'mesh.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_mesh_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_mesh.UpdateMeshRequest()
    request.mesh.name = 'name_value'
    with mock.patch.object(type(client.transport.update_mesh), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_mesh(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'mesh.name=name_value') in kw['metadata']

def test_update_mesh_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_mesh), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_mesh(mesh=gcn_mesh.Mesh(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].mesh
        mock_val = gcn_mesh.Mesh(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_mesh_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_mesh(gcn_mesh.UpdateMeshRequest(), mesh=gcn_mesh.Mesh(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_mesh_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_mesh), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_mesh(mesh=gcn_mesh.Mesh(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].mesh
        mock_val = gcn_mesh.Mesh(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_mesh_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_mesh(gcn_mesh.UpdateMeshRequest(), mesh=gcn_mesh.Mesh(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [mesh.DeleteMeshRequest, dict])
def test_delete_mesh(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_mesh), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_mesh(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == mesh.DeleteMeshRequest()
    assert isinstance(response, future.Future)

def test_delete_mesh_empty_call():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_mesh), '__call__') as call:
        client.delete_mesh()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == mesh.DeleteMeshRequest()

@pytest.mark.asyncio
async def test_delete_mesh_async(transport: str='grpc_asyncio', request_type=mesh.DeleteMeshRequest):
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_mesh), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_mesh(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == mesh.DeleteMeshRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_mesh_async_from_dict():
    await test_delete_mesh_async(request_type=dict)

def test_delete_mesh_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    request = mesh.DeleteMeshRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_mesh), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_mesh(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_mesh_field_headers_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = mesh.DeleteMeshRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_mesh), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_mesh(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_mesh_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_mesh), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_mesh(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_mesh_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_mesh(mesh.DeleteMeshRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_mesh_flattened_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_mesh), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_mesh(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_mesh_flattened_error_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_mesh(mesh.DeleteMeshRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [endpoint_policy.ListEndpointPoliciesRequest, dict])
def test_list_endpoint_policies_rest(request_type):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = endpoint_policy.ListEndpointPoliciesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = endpoint_policy.ListEndpointPoliciesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_endpoint_policies(request)
    assert isinstance(response, pagers.ListEndpointPoliciesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_endpoint_policies_rest_required_fields(request_type=endpoint_policy.ListEndpointPoliciesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_endpoint_policies._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_endpoint_policies._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = endpoint_policy.ListEndpointPoliciesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = endpoint_policy.ListEndpointPoliciesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_endpoint_policies(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_endpoint_policies_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_endpoint_policies._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_endpoint_policies_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_list_endpoint_policies') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_list_endpoint_policies') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = endpoint_policy.ListEndpointPoliciesRequest.pb(endpoint_policy.ListEndpointPoliciesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = endpoint_policy.ListEndpointPoliciesResponse.to_json(endpoint_policy.ListEndpointPoliciesResponse())
        request = endpoint_policy.ListEndpointPoliciesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = endpoint_policy.ListEndpointPoliciesResponse()
        client.list_endpoint_policies(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_endpoint_policies_rest_bad_request(transport: str='rest', request_type=endpoint_policy.ListEndpointPoliciesRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_endpoint_policies(request)

def test_list_endpoint_policies_rest_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = endpoint_policy.ListEndpointPoliciesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = endpoint_policy.ListEndpointPoliciesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_endpoint_policies(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/endpointPolicies' % client.transport._host, args[1])

def test_list_endpoint_policies_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_endpoint_policies(endpoint_policy.ListEndpointPoliciesRequest(), parent='parent_value')

def test_list_endpoint_policies_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy()], next_page_token='abc'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[], next_page_token='def'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy()], next_page_token='ghi'), endpoint_policy.ListEndpointPoliciesResponse(endpoint_policies=[endpoint_policy.EndpointPolicy(), endpoint_policy.EndpointPolicy()]))
        response = response + response
        response = tuple((endpoint_policy.ListEndpointPoliciesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_endpoint_policies(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, endpoint_policy.EndpointPolicy) for i in results))
        pages = list(client.list_endpoint_policies(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [endpoint_policy.GetEndpointPolicyRequest, dict])
def test_get_endpoint_policy_rest(request_type):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/endpointPolicies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = endpoint_policy.EndpointPolicy(name='name_value', type_=endpoint_policy.EndpointPolicy.EndpointPolicyType.SIDECAR_PROXY, authorization_policy='authorization_policy_value', description='description_value', server_tls_policy='server_tls_policy_value', client_tls_policy='client_tls_policy_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = endpoint_policy.EndpointPolicy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_endpoint_policy(request)
    assert isinstance(response, endpoint_policy.EndpointPolicy)
    assert response.name == 'name_value'
    assert response.type_ == endpoint_policy.EndpointPolicy.EndpointPolicyType.SIDECAR_PROXY
    assert response.authorization_policy == 'authorization_policy_value'
    assert response.description == 'description_value'
    assert response.server_tls_policy == 'server_tls_policy_value'
    assert response.client_tls_policy == 'client_tls_policy_value'

def test_get_endpoint_policy_rest_required_fields(request_type=endpoint_policy.GetEndpointPolicyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_endpoint_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_endpoint_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = endpoint_policy.EndpointPolicy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = endpoint_policy.EndpointPolicy.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_endpoint_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_endpoint_policy_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_endpoint_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_endpoint_policy_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_get_endpoint_policy') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_get_endpoint_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = endpoint_policy.GetEndpointPolicyRequest.pb(endpoint_policy.GetEndpointPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = endpoint_policy.EndpointPolicy.to_json(endpoint_policy.EndpointPolicy())
        request = endpoint_policy.GetEndpointPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = endpoint_policy.EndpointPolicy()
        client.get_endpoint_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_endpoint_policy_rest_bad_request(transport: str='rest', request_type=endpoint_policy.GetEndpointPolicyRequest):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/endpointPolicies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_endpoint_policy(request)

def test_get_endpoint_policy_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = endpoint_policy.EndpointPolicy()
        sample_request = {'name': 'projects/sample1/locations/sample2/endpointPolicies/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = endpoint_policy.EndpointPolicy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_endpoint_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/endpointPolicies/*}' % client.transport._host, args[1])

def test_get_endpoint_policy_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_endpoint_policy(endpoint_policy.GetEndpointPolicyRequest(), name='name_value')

def test_get_endpoint_policy_rest_error():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_endpoint_policy.CreateEndpointPolicyRequest, dict])
def test_create_endpoint_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['endpoint_policy'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'type_': 1, 'authorization_policy': 'authorization_policy_value', 'endpoint_matcher': {'metadata_label_matcher': {'metadata_label_match_criteria': 1, 'metadata_labels': [{'label_name': 'label_name_value', 'label_value': 'label_value_value'}]}}, 'traffic_port_selector': {'ports': ['ports_value1', 'ports_value2']}, 'description': 'description_value', 'server_tls_policy': 'server_tls_policy_value', 'client_tls_policy': 'client_tls_policy_value'}
    test_field = gcn_endpoint_policy.CreateEndpointPolicyRequest.meta.fields['endpoint_policy']

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
    for (field, value) in request_init['endpoint_policy'].items():
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
                for i in range(0, len(request_init['endpoint_policy'][field])):
                    del request_init['endpoint_policy'][field][i][subfield]
            else:
                del request_init['endpoint_policy'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_endpoint_policy(request)
    assert response.operation.name == 'operations/spam'

def test_create_endpoint_policy_rest_required_fields(request_type=gcn_endpoint_policy.CreateEndpointPolicyRequest):
    if False:
        return 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['endpoint_policy_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'endpointPolicyId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_endpoint_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'endpointPolicyId' in jsonified_request
    assert jsonified_request['endpointPolicyId'] == request_init['endpoint_policy_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['endpointPolicyId'] = 'endpoint_policy_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_endpoint_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('endpoint_policy_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'endpointPolicyId' in jsonified_request
    assert jsonified_request['endpointPolicyId'] == 'endpoint_policy_id_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_endpoint_policy(request)
            expected_params = [('endpointPolicyId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_endpoint_policy_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_endpoint_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('endpointPolicyId',)) & set(('parent', 'endpointPolicyId', 'endpointPolicy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_endpoint_policy_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_create_endpoint_policy') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_create_endpoint_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_endpoint_policy.CreateEndpointPolicyRequest.pb(gcn_endpoint_policy.CreateEndpointPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_endpoint_policy.CreateEndpointPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_endpoint_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_endpoint_policy_rest_bad_request(transport: str='rest', request_type=gcn_endpoint_policy.CreateEndpointPolicyRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_endpoint_policy(request)

def test_create_endpoint_policy_rest_flattened():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', endpoint_policy=gcn_endpoint_policy.EndpointPolicy(name='name_value'), endpoint_policy_id='endpoint_policy_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_endpoint_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/endpointPolicies' % client.transport._host, args[1])

def test_create_endpoint_policy_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_endpoint_policy(gcn_endpoint_policy.CreateEndpointPolicyRequest(), parent='parent_value', endpoint_policy=gcn_endpoint_policy.EndpointPolicy(name='name_value'), endpoint_policy_id='endpoint_policy_id_value')

def test_create_endpoint_policy_rest_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_endpoint_policy.UpdateEndpointPolicyRequest, dict])
def test_update_endpoint_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'endpoint_policy': {'name': 'projects/sample1/locations/sample2/endpointPolicies/sample3'}}
    request_init['endpoint_policy'] = {'name': 'projects/sample1/locations/sample2/endpointPolicies/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'type_': 1, 'authorization_policy': 'authorization_policy_value', 'endpoint_matcher': {'metadata_label_matcher': {'metadata_label_match_criteria': 1, 'metadata_labels': [{'label_name': 'label_name_value', 'label_value': 'label_value_value'}]}}, 'traffic_port_selector': {'ports': ['ports_value1', 'ports_value2']}, 'description': 'description_value', 'server_tls_policy': 'server_tls_policy_value', 'client_tls_policy': 'client_tls_policy_value'}
    test_field = gcn_endpoint_policy.UpdateEndpointPolicyRequest.meta.fields['endpoint_policy']

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
    for (field, value) in request_init['endpoint_policy'].items():
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
                for i in range(0, len(request_init['endpoint_policy'][field])):
                    del request_init['endpoint_policy'][field][i][subfield]
            else:
                del request_init['endpoint_policy'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_endpoint_policy(request)
    assert response.operation.name == 'operations/spam'

def test_update_endpoint_policy_rest_required_fields(request_type=gcn_endpoint_policy.UpdateEndpointPolicyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_endpoint_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_endpoint_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_endpoint_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_endpoint_policy_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_endpoint_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('endpointPolicy',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_endpoint_policy_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_update_endpoint_policy') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_update_endpoint_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_endpoint_policy.UpdateEndpointPolicyRequest.pb(gcn_endpoint_policy.UpdateEndpointPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_endpoint_policy.UpdateEndpointPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_endpoint_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_endpoint_policy_rest_bad_request(transport: str='rest', request_type=gcn_endpoint_policy.UpdateEndpointPolicyRequest):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'endpoint_policy': {'name': 'projects/sample1/locations/sample2/endpointPolicies/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_endpoint_policy(request)

def test_update_endpoint_policy_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'endpoint_policy': {'name': 'projects/sample1/locations/sample2/endpointPolicies/sample3'}}
        mock_args = dict(endpoint_policy=gcn_endpoint_policy.EndpointPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_endpoint_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{endpoint_policy.name=projects/*/locations/*/endpointPolicies/*}' % client.transport._host, args[1])

def test_update_endpoint_policy_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_endpoint_policy(gcn_endpoint_policy.UpdateEndpointPolicyRequest(), endpoint_policy=gcn_endpoint_policy.EndpointPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_endpoint_policy_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [endpoint_policy.DeleteEndpointPolicyRequest, dict])
def test_delete_endpoint_policy_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/endpointPolicies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_endpoint_policy(request)
    assert response.operation.name == 'operations/spam'

def test_delete_endpoint_policy_rest_required_fields(request_type=endpoint_policy.DeleteEndpointPolicyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_endpoint_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_endpoint_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_endpoint_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_endpoint_policy_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_endpoint_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_endpoint_policy_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_delete_endpoint_policy') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_delete_endpoint_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = endpoint_policy.DeleteEndpointPolicyRequest.pb(endpoint_policy.DeleteEndpointPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = endpoint_policy.DeleteEndpointPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_endpoint_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_endpoint_policy_rest_bad_request(transport: str='rest', request_type=endpoint_policy.DeleteEndpointPolicyRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/endpointPolicies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_endpoint_policy(request)

def test_delete_endpoint_policy_rest_flattened():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/endpointPolicies/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_endpoint_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/endpointPolicies/*}' % client.transport._host, args[1])

def test_delete_endpoint_policy_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_endpoint_policy(endpoint_policy.DeleteEndpointPolicyRequest(), name='name_value')

def test_delete_endpoint_policy_rest_error():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gateway.ListGatewaysRequest, dict])
def test_list_gateways_rest(request_type):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gateway.ListGatewaysResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gateway.ListGatewaysResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_gateways(request)
    assert isinstance(response, pagers.ListGatewaysPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_gateways_rest_required_fields(request_type=gateway.ListGatewaysRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_gateways._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_gateways._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gateway.ListGatewaysResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gateway.ListGatewaysResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_gateways(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_gateways_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_gateways._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_gateways_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_list_gateways') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_list_gateways') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gateway.ListGatewaysRequest.pb(gateway.ListGatewaysRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gateway.ListGatewaysResponse.to_json(gateway.ListGatewaysResponse())
        request = gateway.ListGatewaysRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gateway.ListGatewaysResponse()
        client.list_gateways(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_gateways_rest_bad_request(transport: str='rest', request_type=gateway.ListGatewaysRequest):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gateway.ListGatewaysResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gateway.ListGatewaysResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_gateways(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/gateways' % client.transport._host, args[1])

def test_list_gateways_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_gateways(gateway.ListGatewaysRequest(), parent='parent_value')

def test_list_gateways_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (gateway.ListGatewaysResponse(gateways=[gateway.Gateway(), gateway.Gateway(), gateway.Gateway()], next_page_token='abc'), gateway.ListGatewaysResponse(gateways=[], next_page_token='def'), gateway.ListGatewaysResponse(gateways=[gateway.Gateway()], next_page_token='ghi'), gateway.ListGatewaysResponse(gateways=[gateway.Gateway(), gateway.Gateway()]))
        response = response + response
        response = tuple((gateway.ListGatewaysResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_gateways(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, gateway.Gateway) for i in results))
        pages = list(client.list_gateways(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [gateway.GetGatewayRequest, dict])
def test_get_gateway_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/gateways/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gateway.Gateway(name='name_value', self_link='self_link_value', description='description_value', type_=gateway.Gateway.Type.OPEN_MESH, ports=[568], scope='scope_value', server_tls_policy='server_tls_policy_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gateway.Gateway.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_gateway(request)
    assert isinstance(response, gateway.Gateway)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.type_ == gateway.Gateway.Type.OPEN_MESH
    assert response.ports == [568]
    assert response.scope == 'scope_value'
    assert response.server_tls_policy == 'server_tls_policy_value'

def test_get_gateway_rest_required_fields(request_type=gateway.GetGatewayRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetworkServicesRestTransport
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gateway.Gateway()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gateway.Gateway.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_gateway(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_gateway_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_gateway._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_gateway_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_get_gateway') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_get_gateway') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gateway.GetGatewayRequest.pb(gateway.GetGatewayRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gateway.Gateway.to_json(gateway.Gateway())
        request = gateway.GetGatewayRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gateway.Gateway()
        client.get_gateway(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_gateway_rest_bad_request(transport: str='rest', request_type=gateway.GetGatewayRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gateway.Gateway()
        sample_request = {'name': 'projects/sample1/locations/sample2/gateways/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gateway.Gateway.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_gateway(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/gateways/*}' % client.transport._host, args[1])

def test_get_gateway_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_gateway(gateway.GetGatewayRequest(), name='name_value')

def test_get_gateway_rest_error():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_gateway.CreateGatewayRequest, dict])
def test_create_gateway_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['gateway'] = {'name': 'name_value', 'self_link': 'self_link_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'type_': 1, 'ports': [569, 570], 'scope': 'scope_value', 'server_tls_policy': 'server_tls_policy_value'}
    test_field = gcn_gateway.CreateGatewayRequest.meta.fields['gateway']

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

def test_create_gateway_rest_required_fields(request_type=gcn_gateway.CreateGatewayRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetworkServicesRestTransport
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_gateway._get_unset_required_fields({})
    assert set(unset_fields) == set(('gatewayId',)) & set(('parent', 'gatewayId', 'gateway'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_gateway_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_create_gateway') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_create_gateway') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_gateway.CreateGatewayRequest.pb(gcn_gateway.CreateGatewayRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_gateway.CreateGatewayRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_gateway(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_gateway_rest_bad_request(transport: str='rest', request_type=gcn_gateway.CreateGatewayRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', gateway=gcn_gateway.Gateway(name='name_value'), gateway_id='gateway_id_value')
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
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_gateway(gcn_gateway.CreateGatewayRequest(), parent='parent_value', gateway=gcn_gateway.Gateway(name='name_value'), gateway_id='gateway_id_value')

def test_create_gateway_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_gateway.UpdateGatewayRequest, dict])
def test_update_gateway_rest(request_type):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'gateway': {'name': 'projects/sample1/locations/sample2/gateways/sample3'}}
    request_init['gateway'] = {'name': 'projects/sample1/locations/sample2/gateways/sample3', 'self_link': 'self_link_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'type_': 1, 'ports': [569, 570], 'scope': 'scope_value', 'server_tls_policy': 'server_tls_policy_value'}
    test_field = gcn_gateway.UpdateGatewayRequest.meta.fields['gateway']

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

def test_update_gateway_rest_required_fields(request_type=gcn_gateway.UpdateGatewayRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_gateway._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_gateway._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_gateway._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('gateway',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_gateway_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_update_gateway') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_update_gateway') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_gateway.UpdateGatewayRequest.pb(gcn_gateway.UpdateGatewayRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_gateway.UpdateGatewayRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_gateway(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_gateway_rest_bad_request(transport: str='rest', request_type=gcn_gateway.UpdateGatewayRequest):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'gateway': {'name': 'projects/sample1/locations/sample2/gateways/sample3'}}
        mock_args = dict(gateway=gcn_gateway.Gateway(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
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
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_gateway(gcn_gateway.UpdateGatewayRequest(), gateway=gcn_gateway.Gateway(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_gateway_rest_error():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gateway.DeleteGatewayRequest, dict])
def test_delete_gateway_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_delete_gateway_rest_required_fields(request_type=gateway.DeleteGatewayRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NetworkServicesRestTransport
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_gateway._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_gateway_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_delete_gateway') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_delete_gateway') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gateway.DeleteGatewayRequest.pb(gateway.DeleteGatewayRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gateway.DeleteGatewayRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_gateway(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_gateway_rest_bad_request(transport: str='rest', request_type=gateway.DeleteGatewayRequest):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_gateway(gateway.DeleteGatewayRequest(), name='name_value')

def test_delete_gateway_rest_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [grpc_route.ListGrpcRoutesRequest, dict])
def test_list_grpc_routes_rest(request_type):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grpc_route.ListGrpcRoutesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = grpc_route.ListGrpcRoutesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_grpc_routes(request)
    assert isinstance(response, pagers.ListGrpcRoutesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_grpc_routes_rest_required_fields(request_type=grpc_route.ListGrpcRoutesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_grpc_routes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_grpc_routes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = grpc_route.ListGrpcRoutesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = grpc_route.ListGrpcRoutesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_grpc_routes(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_grpc_routes_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_grpc_routes._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_grpc_routes_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_list_grpc_routes') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_list_grpc_routes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grpc_route.ListGrpcRoutesRequest.pb(grpc_route.ListGrpcRoutesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = grpc_route.ListGrpcRoutesResponse.to_json(grpc_route.ListGrpcRoutesResponse())
        request = grpc_route.ListGrpcRoutesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = grpc_route.ListGrpcRoutesResponse()
        client.list_grpc_routes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_grpc_routes_rest_bad_request(transport: str='rest', request_type=grpc_route.ListGrpcRoutesRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_grpc_routes(request)

def test_list_grpc_routes_rest_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grpc_route.ListGrpcRoutesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = grpc_route.ListGrpcRoutesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_grpc_routes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/grpcRoutes' % client.transport._host, args[1])

def test_list_grpc_routes_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_grpc_routes(grpc_route.ListGrpcRoutesRequest(), parent='parent_value')

def test_list_grpc_routes_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute(), grpc_route.GrpcRoute(), grpc_route.GrpcRoute()], next_page_token='abc'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[], next_page_token='def'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute()], next_page_token='ghi'), grpc_route.ListGrpcRoutesResponse(grpc_routes=[grpc_route.GrpcRoute(), grpc_route.GrpcRoute()]))
        response = response + response
        response = tuple((grpc_route.ListGrpcRoutesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_grpc_routes(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, grpc_route.GrpcRoute) for i in results))
        pages = list(client.list_grpc_routes(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [grpc_route.GetGrpcRouteRequest, dict])
def test_get_grpc_route_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/grpcRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grpc_route.GrpcRoute(name='name_value', self_link='self_link_value', description='description_value', hostnames=['hostnames_value'], meshes=['meshes_value'], gateways=['gateways_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = grpc_route.GrpcRoute.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_grpc_route(request)
    assert isinstance(response, grpc_route.GrpcRoute)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.hostnames == ['hostnames_value']
    assert response.meshes == ['meshes_value']
    assert response.gateways == ['gateways_value']

def test_get_grpc_route_rest_required_fields(request_type=grpc_route.GetGrpcRouteRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_grpc_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_grpc_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = grpc_route.GrpcRoute()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = grpc_route.GrpcRoute.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_grpc_route(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_grpc_route_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_grpc_route._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_grpc_route_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_get_grpc_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_get_grpc_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grpc_route.GetGrpcRouteRequest.pb(grpc_route.GetGrpcRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = grpc_route.GrpcRoute.to_json(grpc_route.GrpcRoute())
        request = grpc_route.GetGrpcRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = grpc_route.GrpcRoute()
        client.get_grpc_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_grpc_route_rest_bad_request(transport: str='rest', request_type=grpc_route.GetGrpcRouteRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/grpcRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_grpc_route(request)

def test_get_grpc_route_rest_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grpc_route.GrpcRoute()
        sample_request = {'name': 'projects/sample1/locations/sample2/grpcRoutes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = grpc_route.GrpcRoute.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_grpc_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/grpcRoutes/*}' % client.transport._host, args[1])

def test_get_grpc_route_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_grpc_route(grpc_route.GetGrpcRouteRequest(), name='name_value')

def test_get_grpc_route_rest_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_grpc_route.CreateGrpcRouteRequest, dict])
def test_create_grpc_route_rest(request_type):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['grpc_route'] = {'name': 'name_value', 'self_link': 'self_link_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'hostnames': ['hostnames_value1', 'hostnames_value2'], 'meshes': ['meshes_value1', 'meshes_value2'], 'gateways': ['gateways_value1', 'gateways_value2'], 'rules': [{'matches': [{'method': {'type_': 1, 'grpc_service': 'grpc_service_value', 'grpc_method': 'grpc_method_value', 'case_sensitive': True}, 'headers': [{'type_': 1, 'key': 'key_value', 'value': 'value_value'}]}], 'action': {'destinations': [{'service_name': 'service_name_value', 'weight': 648}], 'fault_injection_policy': {'delay': {'fixed_delay': {'seconds': 751, 'nanos': 543}, 'percentage': 1054}, 'abort': {'http_status': 1219, 'percentage': 1054}}, 'timeout': {}, 'retry_policy': {'retry_conditions': ['retry_conditions_value1', 'retry_conditions_value2'], 'num_retries': 1197}}}]}
    test_field = gcn_grpc_route.CreateGrpcRouteRequest.meta.fields['grpc_route']

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
    for (field, value) in request_init['grpc_route'].items():
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
                for i in range(0, len(request_init['grpc_route'][field])):
                    del request_init['grpc_route'][field][i][subfield]
            else:
                del request_init['grpc_route'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_grpc_route(request)
    assert response.operation.name == 'operations/spam'

def test_create_grpc_route_rest_required_fields(request_type=gcn_grpc_route.CreateGrpcRouteRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['grpc_route_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'grpcRouteId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_grpc_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'grpcRouteId' in jsonified_request
    assert jsonified_request['grpcRouteId'] == request_init['grpc_route_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['grpcRouteId'] = 'grpc_route_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_grpc_route._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('grpc_route_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'grpcRouteId' in jsonified_request
    assert jsonified_request['grpcRouteId'] == 'grpc_route_id_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_grpc_route(request)
            expected_params = [('grpcRouteId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_grpc_route_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_grpc_route._get_unset_required_fields({})
    assert set(unset_fields) == set(('grpcRouteId',)) & set(('parent', 'grpcRouteId', 'grpcRoute'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_grpc_route_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_create_grpc_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_create_grpc_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_grpc_route.CreateGrpcRouteRequest.pb(gcn_grpc_route.CreateGrpcRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_grpc_route.CreateGrpcRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_grpc_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_grpc_route_rest_bad_request(transport: str='rest', request_type=gcn_grpc_route.CreateGrpcRouteRequest):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_grpc_route(request)

def test_create_grpc_route_rest_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', grpc_route=gcn_grpc_route.GrpcRoute(name='name_value'), grpc_route_id='grpc_route_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_grpc_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/grpcRoutes' % client.transport._host, args[1])

def test_create_grpc_route_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_grpc_route(gcn_grpc_route.CreateGrpcRouteRequest(), parent='parent_value', grpc_route=gcn_grpc_route.GrpcRoute(name='name_value'), grpc_route_id='grpc_route_id_value')

def test_create_grpc_route_rest_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_grpc_route.UpdateGrpcRouteRequest, dict])
def test_update_grpc_route_rest(request_type):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'grpc_route': {'name': 'projects/sample1/locations/sample2/grpcRoutes/sample3'}}
    request_init['grpc_route'] = {'name': 'projects/sample1/locations/sample2/grpcRoutes/sample3', 'self_link': 'self_link_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'hostnames': ['hostnames_value1', 'hostnames_value2'], 'meshes': ['meshes_value1', 'meshes_value2'], 'gateways': ['gateways_value1', 'gateways_value2'], 'rules': [{'matches': [{'method': {'type_': 1, 'grpc_service': 'grpc_service_value', 'grpc_method': 'grpc_method_value', 'case_sensitive': True}, 'headers': [{'type_': 1, 'key': 'key_value', 'value': 'value_value'}]}], 'action': {'destinations': [{'service_name': 'service_name_value', 'weight': 648}], 'fault_injection_policy': {'delay': {'fixed_delay': {'seconds': 751, 'nanos': 543}, 'percentage': 1054}, 'abort': {'http_status': 1219, 'percentage': 1054}}, 'timeout': {}, 'retry_policy': {'retry_conditions': ['retry_conditions_value1', 'retry_conditions_value2'], 'num_retries': 1197}}}]}
    test_field = gcn_grpc_route.UpdateGrpcRouteRequest.meta.fields['grpc_route']

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
    for (field, value) in request_init['grpc_route'].items():
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
                for i in range(0, len(request_init['grpc_route'][field])):
                    del request_init['grpc_route'][field][i][subfield]
            else:
                del request_init['grpc_route'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_grpc_route(request)
    assert response.operation.name == 'operations/spam'

def test_update_grpc_route_rest_required_fields(request_type=gcn_grpc_route.UpdateGrpcRouteRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_grpc_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_grpc_route._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_grpc_route(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_grpc_route_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_grpc_route._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('grpcRoute',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_grpc_route_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_update_grpc_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_update_grpc_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_grpc_route.UpdateGrpcRouteRequest.pb(gcn_grpc_route.UpdateGrpcRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_grpc_route.UpdateGrpcRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_grpc_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_grpc_route_rest_bad_request(transport: str='rest', request_type=gcn_grpc_route.UpdateGrpcRouteRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'grpc_route': {'name': 'projects/sample1/locations/sample2/grpcRoutes/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_grpc_route(request)

def test_update_grpc_route_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'grpc_route': {'name': 'projects/sample1/locations/sample2/grpcRoutes/sample3'}}
        mock_args = dict(grpc_route=gcn_grpc_route.GrpcRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_grpc_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{grpc_route.name=projects/*/locations/*/grpcRoutes/*}' % client.transport._host, args[1])

def test_update_grpc_route_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_grpc_route(gcn_grpc_route.UpdateGrpcRouteRequest(), grpc_route=gcn_grpc_route.GrpcRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_grpc_route_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [grpc_route.DeleteGrpcRouteRequest, dict])
def test_delete_grpc_route_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/grpcRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_grpc_route(request)
    assert response.operation.name == 'operations/spam'

def test_delete_grpc_route_rest_required_fields(request_type=grpc_route.DeleteGrpcRouteRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_grpc_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_grpc_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_grpc_route(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_grpc_route_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_grpc_route._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_grpc_route_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_delete_grpc_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_delete_grpc_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grpc_route.DeleteGrpcRouteRequest.pb(grpc_route.DeleteGrpcRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = grpc_route.DeleteGrpcRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_grpc_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_grpc_route_rest_bad_request(transport: str='rest', request_type=grpc_route.DeleteGrpcRouteRequest):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/grpcRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_grpc_route(request)

def test_delete_grpc_route_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/grpcRoutes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_grpc_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/grpcRoutes/*}' % client.transport._host, args[1])

def test_delete_grpc_route_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_grpc_route(grpc_route.DeleteGrpcRouteRequest(), name='name_value')

def test_delete_grpc_route_rest_error():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [http_route.ListHttpRoutesRequest, dict])
def test_list_http_routes_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = http_route.ListHttpRoutesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = http_route.ListHttpRoutesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_http_routes(request)
    assert isinstance(response, pagers.ListHttpRoutesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_http_routes_rest_required_fields(request_type=http_route.ListHttpRoutesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_http_routes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_http_routes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = http_route.ListHttpRoutesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = http_route.ListHttpRoutesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_http_routes(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_http_routes_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_http_routes._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_http_routes_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_list_http_routes') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_list_http_routes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = http_route.ListHttpRoutesRequest.pb(http_route.ListHttpRoutesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = http_route.ListHttpRoutesResponse.to_json(http_route.ListHttpRoutesResponse())
        request = http_route.ListHttpRoutesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = http_route.ListHttpRoutesResponse()
        client.list_http_routes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_http_routes_rest_bad_request(transport: str='rest', request_type=http_route.ListHttpRoutesRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_http_routes(request)

def test_list_http_routes_rest_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = http_route.ListHttpRoutesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = http_route.ListHttpRoutesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_http_routes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/httpRoutes' % client.transport._host, args[1])

def test_list_http_routes_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_http_routes(http_route.ListHttpRoutesRequest(), parent='parent_value')

def test_list_http_routes_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute(), http_route.HttpRoute(), http_route.HttpRoute()], next_page_token='abc'), http_route.ListHttpRoutesResponse(http_routes=[], next_page_token='def'), http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute()], next_page_token='ghi'), http_route.ListHttpRoutesResponse(http_routes=[http_route.HttpRoute(), http_route.HttpRoute()]))
        response = response + response
        response = tuple((http_route.ListHttpRoutesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_http_routes(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, http_route.HttpRoute) for i in results))
        pages = list(client.list_http_routes(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [http_route.GetHttpRouteRequest, dict])
def test_get_http_route_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/httpRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = http_route.HttpRoute(name='name_value', self_link='self_link_value', description='description_value', hostnames=['hostnames_value'], meshes=['meshes_value'], gateways=['gateways_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = http_route.HttpRoute.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_http_route(request)
    assert isinstance(response, http_route.HttpRoute)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.hostnames == ['hostnames_value']
    assert response.meshes == ['meshes_value']
    assert response.gateways == ['gateways_value']

def test_get_http_route_rest_required_fields(request_type=http_route.GetHttpRouteRequest):
    if False:
        return 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_http_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_http_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = http_route.HttpRoute()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = http_route.HttpRoute.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_http_route(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_http_route_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_http_route._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_http_route_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_get_http_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_get_http_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = http_route.GetHttpRouteRequest.pb(http_route.GetHttpRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = http_route.HttpRoute.to_json(http_route.HttpRoute())
        request = http_route.GetHttpRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = http_route.HttpRoute()
        client.get_http_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_http_route_rest_bad_request(transport: str='rest', request_type=http_route.GetHttpRouteRequest):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/httpRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_http_route(request)

def test_get_http_route_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = http_route.HttpRoute()
        sample_request = {'name': 'projects/sample1/locations/sample2/httpRoutes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = http_route.HttpRoute.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_http_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/httpRoutes/*}' % client.transport._host, args[1])

def test_get_http_route_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_http_route(http_route.GetHttpRouteRequest(), name='name_value')

def test_get_http_route_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_http_route.CreateHttpRouteRequest, dict])
def test_create_http_route_rest(request_type):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['http_route'] = {'name': 'name_value', 'self_link': 'self_link_value', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'hostnames': ['hostnames_value1', 'hostnames_value2'], 'meshes': ['meshes_value1', 'meshes_value2'], 'gateways': ['gateways_value1', 'gateways_value2'], 'labels': {}, 'rules': [{'matches': [{'full_path_match': 'full_path_match_value', 'prefix_match': 'prefix_match_value', 'regex_match': 'regex_match_value', 'ignore_case': True, 'headers': [{'exact_match': 'exact_match_value', 'regex_match': 'regex_match_value', 'prefix_match': 'prefix_match_value', 'present_match': True, 'suffix_match': 'suffix_match_value', 'range_match': {'start': 558, 'end': 311}, 'header': 'header_value', 'invert_match': True}], 'query_parameters': [{'exact_match': 'exact_match_value', 'regex_match': 'regex_match_value', 'present_match': True, 'query_parameter': 'query_parameter_value'}]}], 'action': {'destinations': [{'service_name': 'service_name_value', 'weight': 648}], 'redirect': {'host_redirect': 'host_redirect_value', 'path_redirect': 'path_redirect_value', 'prefix_rewrite': 'prefix_rewrite_value', 'response_code': 1, 'https_redirect': True, 'strip_query': True, 'port_redirect': 1398}, 'fault_injection_policy': {'delay': {'fixed_delay': {'seconds': 751, 'nanos': 543}, 'percentage': 1054}, 'abort': {'http_status': 1219, 'percentage': 1054}}, 'request_header_modifier': {'set': {}, 'add': {}, 'remove': ['remove_value1', 'remove_value2']}, 'response_header_modifier': {}, 'url_rewrite': {'path_prefix_rewrite': 'path_prefix_rewrite_value', 'host_rewrite': 'host_rewrite_value'}, 'timeout': {}, 'retry_policy': {'retry_conditions': ['retry_conditions_value1', 'retry_conditions_value2'], 'num_retries': 1197, 'per_try_timeout': {}}, 'request_mirror_policy': {'destination': {}}, 'cors_policy': {'allow_origins': ['allow_origins_value1', 'allow_origins_value2'], 'allow_origin_regexes': ['allow_origin_regexes_value1', 'allow_origin_regexes_value2'], 'allow_methods': ['allow_methods_value1', 'allow_methods_value2'], 'allow_headers': ['allow_headers_value1', 'allow_headers_value2'], 'expose_headers': ['expose_headers_value1', 'expose_headers_value2'], 'max_age': 'max_age_value', 'allow_credentials': True, 'disabled': True}}}]}
    test_field = gcn_http_route.CreateHttpRouteRequest.meta.fields['http_route']

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
    for (field, value) in request_init['http_route'].items():
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
                for i in range(0, len(request_init['http_route'][field])):
                    del request_init['http_route'][field][i][subfield]
            else:
                del request_init['http_route'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_http_route(request)
    assert response.operation.name == 'operations/spam'

def test_create_http_route_rest_required_fields(request_type=gcn_http_route.CreateHttpRouteRequest):
    if False:
        return 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['http_route_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'httpRouteId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_http_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'httpRouteId' in jsonified_request
    assert jsonified_request['httpRouteId'] == request_init['http_route_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['httpRouteId'] = 'http_route_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_http_route._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('http_route_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'httpRouteId' in jsonified_request
    assert jsonified_request['httpRouteId'] == 'http_route_id_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_http_route(request)
            expected_params = [('httpRouteId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_http_route_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_http_route._get_unset_required_fields({})
    assert set(unset_fields) == set(('httpRouteId',)) & set(('parent', 'httpRouteId', 'httpRoute'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_http_route_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_create_http_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_create_http_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_http_route.CreateHttpRouteRequest.pb(gcn_http_route.CreateHttpRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_http_route.CreateHttpRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_http_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_http_route_rest_bad_request(transport: str='rest', request_type=gcn_http_route.CreateHttpRouteRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_http_route(request)

def test_create_http_route_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', http_route=gcn_http_route.HttpRoute(name='name_value'), http_route_id='http_route_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_http_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/httpRoutes' % client.transport._host, args[1])

def test_create_http_route_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_http_route(gcn_http_route.CreateHttpRouteRequest(), parent='parent_value', http_route=gcn_http_route.HttpRoute(name='name_value'), http_route_id='http_route_id_value')

def test_create_http_route_rest_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_http_route.UpdateHttpRouteRequest, dict])
def test_update_http_route_rest(request_type):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'http_route': {'name': 'projects/sample1/locations/sample2/httpRoutes/sample3'}}
    request_init['http_route'] = {'name': 'projects/sample1/locations/sample2/httpRoutes/sample3', 'self_link': 'self_link_value', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'hostnames': ['hostnames_value1', 'hostnames_value2'], 'meshes': ['meshes_value1', 'meshes_value2'], 'gateways': ['gateways_value1', 'gateways_value2'], 'labels': {}, 'rules': [{'matches': [{'full_path_match': 'full_path_match_value', 'prefix_match': 'prefix_match_value', 'regex_match': 'regex_match_value', 'ignore_case': True, 'headers': [{'exact_match': 'exact_match_value', 'regex_match': 'regex_match_value', 'prefix_match': 'prefix_match_value', 'present_match': True, 'suffix_match': 'suffix_match_value', 'range_match': {'start': 558, 'end': 311}, 'header': 'header_value', 'invert_match': True}], 'query_parameters': [{'exact_match': 'exact_match_value', 'regex_match': 'regex_match_value', 'present_match': True, 'query_parameter': 'query_parameter_value'}]}], 'action': {'destinations': [{'service_name': 'service_name_value', 'weight': 648}], 'redirect': {'host_redirect': 'host_redirect_value', 'path_redirect': 'path_redirect_value', 'prefix_rewrite': 'prefix_rewrite_value', 'response_code': 1, 'https_redirect': True, 'strip_query': True, 'port_redirect': 1398}, 'fault_injection_policy': {'delay': {'fixed_delay': {'seconds': 751, 'nanos': 543}, 'percentage': 1054}, 'abort': {'http_status': 1219, 'percentage': 1054}}, 'request_header_modifier': {'set': {}, 'add': {}, 'remove': ['remove_value1', 'remove_value2']}, 'response_header_modifier': {}, 'url_rewrite': {'path_prefix_rewrite': 'path_prefix_rewrite_value', 'host_rewrite': 'host_rewrite_value'}, 'timeout': {}, 'retry_policy': {'retry_conditions': ['retry_conditions_value1', 'retry_conditions_value2'], 'num_retries': 1197, 'per_try_timeout': {}}, 'request_mirror_policy': {'destination': {}}, 'cors_policy': {'allow_origins': ['allow_origins_value1', 'allow_origins_value2'], 'allow_origin_regexes': ['allow_origin_regexes_value1', 'allow_origin_regexes_value2'], 'allow_methods': ['allow_methods_value1', 'allow_methods_value2'], 'allow_headers': ['allow_headers_value1', 'allow_headers_value2'], 'expose_headers': ['expose_headers_value1', 'expose_headers_value2'], 'max_age': 'max_age_value', 'allow_credentials': True, 'disabled': True}}}]}
    test_field = gcn_http_route.UpdateHttpRouteRequest.meta.fields['http_route']

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
    for (field, value) in request_init['http_route'].items():
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
                for i in range(0, len(request_init['http_route'][field])):
                    del request_init['http_route'][field][i][subfield]
            else:
                del request_init['http_route'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_http_route(request)
    assert response.operation.name == 'operations/spam'

def test_update_http_route_rest_required_fields(request_type=gcn_http_route.UpdateHttpRouteRequest):
    if False:
        return 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_http_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_http_route._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_http_route(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_http_route_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_http_route._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('httpRoute',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_http_route_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_update_http_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_update_http_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_http_route.UpdateHttpRouteRequest.pb(gcn_http_route.UpdateHttpRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_http_route.UpdateHttpRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_http_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_http_route_rest_bad_request(transport: str='rest', request_type=gcn_http_route.UpdateHttpRouteRequest):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'http_route': {'name': 'projects/sample1/locations/sample2/httpRoutes/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_http_route(request)

def test_update_http_route_rest_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'http_route': {'name': 'projects/sample1/locations/sample2/httpRoutes/sample3'}}
        mock_args = dict(http_route=gcn_http_route.HttpRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_http_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{http_route.name=projects/*/locations/*/httpRoutes/*}' % client.transport._host, args[1])

def test_update_http_route_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_http_route(gcn_http_route.UpdateHttpRouteRequest(), http_route=gcn_http_route.HttpRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_http_route_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [http_route.DeleteHttpRouteRequest, dict])
def test_delete_http_route_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/httpRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_http_route(request)
    assert response.operation.name == 'operations/spam'

def test_delete_http_route_rest_required_fields(request_type=http_route.DeleteHttpRouteRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_http_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_http_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_http_route(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_http_route_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_http_route._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_http_route_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_delete_http_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_delete_http_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = http_route.DeleteHttpRouteRequest.pb(http_route.DeleteHttpRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = http_route.DeleteHttpRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_http_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_http_route_rest_bad_request(transport: str='rest', request_type=http_route.DeleteHttpRouteRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/httpRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_http_route(request)

def test_delete_http_route_rest_flattened():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/httpRoutes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_http_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/httpRoutes/*}' % client.transport._host, args[1])

def test_delete_http_route_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_http_route(http_route.DeleteHttpRouteRequest(), name='name_value')

def test_delete_http_route_rest_error():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tcp_route.ListTcpRoutesRequest, dict])
def test_list_tcp_routes_rest(request_type):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tcp_route.ListTcpRoutesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tcp_route.ListTcpRoutesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_tcp_routes(request)
    assert isinstance(response, pagers.ListTcpRoutesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tcp_routes_rest_required_fields(request_type=tcp_route.ListTcpRoutesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tcp_routes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tcp_routes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tcp_route.ListTcpRoutesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tcp_route.ListTcpRoutesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_tcp_routes(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_tcp_routes_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_tcp_routes._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_tcp_routes_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_list_tcp_routes') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_list_tcp_routes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tcp_route.ListTcpRoutesRequest.pb(tcp_route.ListTcpRoutesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tcp_route.ListTcpRoutesResponse.to_json(tcp_route.ListTcpRoutesResponse())
        request = tcp_route.ListTcpRoutesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tcp_route.ListTcpRoutesResponse()
        client.list_tcp_routes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_tcp_routes_rest_bad_request(transport: str='rest', request_type=tcp_route.ListTcpRoutesRequest):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_tcp_routes(request)

def test_list_tcp_routes_rest_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tcp_route.ListTcpRoutesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tcp_route.ListTcpRoutesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_tcp_routes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/tcpRoutes' % client.transport._host, args[1])

def test_list_tcp_routes_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_tcp_routes(tcp_route.ListTcpRoutesRequest(), parent='parent_value')

def test_list_tcp_routes_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute(), tcp_route.TcpRoute(), tcp_route.TcpRoute()], next_page_token='abc'), tcp_route.ListTcpRoutesResponse(tcp_routes=[], next_page_token='def'), tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute()], next_page_token='ghi'), tcp_route.ListTcpRoutesResponse(tcp_routes=[tcp_route.TcpRoute(), tcp_route.TcpRoute()]))
        response = response + response
        response = tuple((tcp_route.ListTcpRoutesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_tcp_routes(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tcp_route.TcpRoute) for i in results))
        pages = list(client.list_tcp_routes(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tcp_route.GetTcpRouteRequest, dict])
def test_get_tcp_route_rest(request_type):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/tcpRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tcp_route.TcpRoute(name='name_value', self_link='self_link_value', description='description_value', meshes=['meshes_value'], gateways=['gateways_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = tcp_route.TcpRoute.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_tcp_route(request)
    assert isinstance(response, tcp_route.TcpRoute)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.meshes == ['meshes_value']
    assert response.gateways == ['gateways_value']

def test_get_tcp_route_rest_required_fields(request_type=tcp_route.GetTcpRouteRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_tcp_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_tcp_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tcp_route.TcpRoute()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tcp_route.TcpRoute.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_tcp_route(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_tcp_route_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_tcp_route._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_tcp_route_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_get_tcp_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_get_tcp_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tcp_route.GetTcpRouteRequest.pb(tcp_route.GetTcpRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tcp_route.TcpRoute.to_json(tcp_route.TcpRoute())
        request = tcp_route.GetTcpRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tcp_route.TcpRoute()
        client.get_tcp_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_tcp_route_rest_bad_request(transport: str='rest', request_type=tcp_route.GetTcpRouteRequest):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/tcpRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_tcp_route(request)

def test_get_tcp_route_rest_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tcp_route.TcpRoute()
        sample_request = {'name': 'projects/sample1/locations/sample2/tcpRoutes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tcp_route.TcpRoute.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_tcp_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/tcpRoutes/*}' % client.transport._host, args[1])

def test_get_tcp_route_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_tcp_route(tcp_route.GetTcpRouteRequest(), name='name_value')

def test_get_tcp_route_rest_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_tcp_route.CreateTcpRouteRequest, dict])
def test_create_tcp_route_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['tcp_route'] = {'name': 'name_value', 'self_link': 'self_link_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'description': 'description_value', 'rules': [{'matches': [{'address': 'address_value', 'port': 'port_value'}], 'action': {'destinations': [{'service_name': 'service_name_value', 'weight': 648}], 'original_destination': True}}], 'meshes': ['meshes_value1', 'meshes_value2'], 'gateways': ['gateways_value1', 'gateways_value2'], 'labels': {}}
    test_field = gcn_tcp_route.CreateTcpRouteRequest.meta.fields['tcp_route']

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
    for (field, value) in request_init['tcp_route'].items():
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
                for i in range(0, len(request_init['tcp_route'][field])):
                    del request_init['tcp_route'][field][i][subfield]
            else:
                del request_init['tcp_route'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_tcp_route(request)
    assert response.operation.name == 'operations/spam'

def test_create_tcp_route_rest_required_fields(request_type=gcn_tcp_route.CreateTcpRouteRequest):
    if False:
        return 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['tcp_route_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'tcpRouteId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_tcp_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'tcpRouteId' in jsonified_request
    assert jsonified_request['tcpRouteId'] == request_init['tcp_route_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['tcpRouteId'] = 'tcp_route_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_tcp_route._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('tcp_route_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'tcpRouteId' in jsonified_request
    assert jsonified_request['tcpRouteId'] == 'tcp_route_id_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_tcp_route(request)
            expected_params = [('tcpRouteId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_tcp_route_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_tcp_route._get_unset_required_fields({})
    assert set(unset_fields) == set(('tcpRouteId',)) & set(('parent', 'tcpRouteId', 'tcpRoute'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_tcp_route_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_create_tcp_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_create_tcp_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_tcp_route.CreateTcpRouteRequest.pb(gcn_tcp_route.CreateTcpRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_tcp_route.CreateTcpRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_tcp_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_tcp_route_rest_bad_request(transport: str='rest', request_type=gcn_tcp_route.CreateTcpRouteRequest):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_tcp_route(request)

def test_create_tcp_route_rest_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', tcp_route=gcn_tcp_route.TcpRoute(name='name_value'), tcp_route_id='tcp_route_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_tcp_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/tcpRoutes' % client.transport._host, args[1])

def test_create_tcp_route_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_tcp_route(gcn_tcp_route.CreateTcpRouteRequest(), parent='parent_value', tcp_route=gcn_tcp_route.TcpRoute(name='name_value'), tcp_route_id='tcp_route_id_value')

def test_create_tcp_route_rest_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_tcp_route.UpdateTcpRouteRequest, dict])
def test_update_tcp_route_rest(request_type):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'tcp_route': {'name': 'projects/sample1/locations/sample2/tcpRoutes/sample3'}}
    request_init['tcp_route'] = {'name': 'projects/sample1/locations/sample2/tcpRoutes/sample3', 'self_link': 'self_link_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'description': 'description_value', 'rules': [{'matches': [{'address': 'address_value', 'port': 'port_value'}], 'action': {'destinations': [{'service_name': 'service_name_value', 'weight': 648}], 'original_destination': True}}], 'meshes': ['meshes_value1', 'meshes_value2'], 'gateways': ['gateways_value1', 'gateways_value2'], 'labels': {}}
    test_field = gcn_tcp_route.UpdateTcpRouteRequest.meta.fields['tcp_route']

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
    for (field, value) in request_init['tcp_route'].items():
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
                for i in range(0, len(request_init['tcp_route'][field])):
                    del request_init['tcp_route'][field][i][subfield]
            else:
                del request_init['tcp_route'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_tcp_route(request)
    assert response.operation.name == 'operations/spam'

def test_update_tcp_route_rest_required_fields(request_type=gcn_tcp_route.UpdateTcpRouteRequest):
    if False:
        return 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_tcp_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_tcp_route._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_tcp_route(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_tcp_route_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_tcp_route._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('tcpRoute',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_tcp_route_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_update_tcp_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_update_tcp_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_tcp_route.UpdateTcpRouteRequest.pb(gcn_tcp_route.UpdateTcpRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_tcp_route.UpdateTcpRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_tcp_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_tcp_route_rest_bad_request(transport: str='rest', request_type=gcn_tcp_route.UpdateTcpRouteRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'tcp_route': {'name': 'projects/sample1/locations/sample2/tcpRoutes/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_tcp_route(request)

def test_update_tcp_route_rest_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'tcp_route': {'name': 'projects/sample1/locations/sample2/tcpRoutes/sample3'}}
        mock_args = dict(tcp_route=gcn_tcp_route.TcpRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_tcp_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{tcp_route.name=projects/*/locations/*/tcpRoutes/*}' % client.transport._host, args[1])

def test_update_tcp_route_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_tcp_route(gcn_tcp_route.UpdateTcpRouteRequest(), tcp_route=gcn_tcp_route.TcpRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_tcp_route_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tcp_route.DeleteTcpRouteRequest, dict])
def test_delete_tcp_route_rest(request_type):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/tcpRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_tcp_route(request)
    assert response.operation.name == 'operations/spam'

def test_delete_tcp_route_rest_required_fields(request_type=tcp_route.DeleteTcpRouteRequest):
    if False:
        return 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_tcp_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_tcp_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_tcp_route(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_tcp_route_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_tcp_route._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_tcp_route_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_delete_tcp_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_delete_tcp_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tcp_route.DeleteTcpRouteRequest.pb(tcp_route.DeleteTcpRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = tcp_route.DeleteTcpRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_tcp_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_tcp_route_rest_bad_request(transport: str='rest', request_type=tcp_route.DeleteTcpRouteRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/tcpRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_tcp_route(request)

def test_delete_tcp_route_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/tcpRoutes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_tcp_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/tcpRoutes/*}' % client.transport._host, args[1])

def test_delete_tcp_route_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_tcp_route(tcp_route.DeleteTcpRouteRequest(), name='name_value')

def test_delete_tcp_route_rest_error():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tls_route.ListTlsRoutesRequest, dict])
def test_list_tls_routes_rest(request_type):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tls_route.ListTlsRoutesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tls_route.ListTlsRoutesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_tls_routes(request)
    assert isinstance(response, pagers.ListTlsRoutesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tls_routes_rest_required_fields(request_type=tls_route.ListTlsRoutesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tls_routes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tls_routes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tls_route.ListTlsRoutesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tls_route.ListTlsRoutesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_tls_routes(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_tls_routes_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_tls_routes._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_tls_routes_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_list_tls_routes') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_list_tls_routes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tls_route.ListTlsRoutesRequest.pb(tls_route.ListTlsRoutesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tls_route.ListTlsRoutesResponse.to_json(tls_route.ListTlsRoutesResponse())
        request = tls_route.ListTlsRoutesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tls_route.ListTlsRoutesResponse()
        client.list_tls_routes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_tls_routes_rest_bad_request(transport: str='rest', request_type=tls_route.ListTlsRoutesRequest):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_tls_routes(request)

def test_list_tls_routes_rest_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tls_route.ListTlsRoutesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tls_route.ListTlsRoutesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_tls_routes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/tlsRoutes' % client.transport._host, args[1])

def test_list_tls_routes_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_tls_routes(tls_route.ListTlsRoutesRequest(), parent='parent_value')

def test_list_tls_routes_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute(), tls_route.TlsRoute(), tls_route.TlsRoute()], next_page_token='abc'), tls_route.ListTlsRoutesResponse(tls_routes=[], next_page_token='def'), tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute()], next_page_token='ghi'), tls_route.ListTlsRoutesResponse(tls_routes=[tls_route.TlsRoute(), tls_route.TlsRoute()]))
        response = response + response
        response = tuple((tls_route.ListTlsRoutesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_tls_routes(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tls_route.TlsRoute) for i in results))
        pages = list(client.list_tls_routes(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tls_route.GetTlsRouteRequest, dict])
def test_get_tls_route_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/tlsRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tls_route.TlsRoute(name='name_value', self_link='self_link_value', description='description_value', meshes=['meshes_value'], gateways=['gateways_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = tls_route.TlsRoute.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_tls_route(request)
    assert isinstance(response, tls_route.TlsRoute)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.meshes == ['meshes_value']
    assert response.gateways == ['gateways_value']

def test_get_tls_route_rest_required_fields(request_type=tls_route.GetTlsRouteRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_tls_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_tls_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tls_route.TlsRoute()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tls_route.TlsRoute.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_tls_route(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_tls_route_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_tls_route._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_tls_route_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_get_tls_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_get_tls_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tls_route.GetTlsRouteRequest.pb(tls_route.GetTlsRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tls_route.TlsRoute.to_json(tls_route.TlsRoute())
        request = tls_route.GetTlsRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tls_route.TlsRoute()
        client.get_tls_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_tls_route_rest_bad_request(transport: str='rest', request_type=tls_route.GetTlsRouteRequest):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/tlsRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_tls_route(request)

def test_get_tls_route_rest_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tls_route.TlsRoute()
        sample_request = {'name': 'projects/sample1/locations/sample2/tlsRoutes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tls_route.TlsRoute.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_tls_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/tlsRoutes/*}' % client.transport._host, args[1])

def test_get_tls_route_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_tls_route(tls_route.GetTlsRouteRequest(), name='name_value')

def test_get_tls_route_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_tls_route.CreateTlsRouteRequest, dict])
def test_create_tls_route_rest(request_type):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['tls_route'] = {'name': 'name_value', 'self_link': 'self_link_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'description': 'description_value', 'rules': [{'matches': [{'sni_host': ['sni_host_value1', 'sni_host_value2'], 'alpn': ['alpn_value1', 'alpn_value2']}], 'action': {'destinations': [{'service_name': 'service_name_value', 'weight': 648}]}}], 'meshes': ['meshes_value1', 'meshes_value2'], 'gateways': ['gateways_value1', 'gateways_value2']}
    test_field = gcn_tls_route.CreateTlsRouteRequest.meta.fields['tls_route']

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
    for (field, value) in request_init['tls_route'].items():
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
                for i in range(0, len(request_init['tls_route'][field])):
                    del request_init['tls_route'][field][i][subfield]
            else:
                del request_init['tls_route'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_tls_route(request)
    assert response.operation.name == 'operations/spam'

def test_create_tls_route_rest_required_fields(request_type=gcn_tls_route.CreateTlsRouteRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['tls_route_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'tlsRouteId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_tls_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'tlsRouteId' in jsonified_request
    assert jsonified_request['tlsRouteId'] == request_init['tls_route_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['tlsRouteId'] = 'tls_route_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_tls_route._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('tls_route_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'tlsRouteId' in jsonified_request
    assert jsonified_request['tlsRouteId'] == 'tls_route_id_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_tls_route(request)
            expected_params = [('tlsRouteId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_tls_route_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_tls_route._get_unset_required_fields({})
    assert set(unset_fields) == set(('tlsRouteId',)) & set(('parent', 'tlsRouteId', 'tlsRoute'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_tls_route_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_create_tls_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_create_tls_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_tls_route.CreateTlsRouteRequest.pb(gcn_tls_route.CreateTlsRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_tls_route.CreateTlsRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_tls_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_tls_route_rest_bad_request(transport: str='rest', request_type=gcn_tls_route.CreateTlsRouteRequest):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_tls_route(request)

def test_create_tls_route_rest_flattened():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', tls_route=gcn_tls_route.TlsRoute(name='name_value'), tls_route_id='tls_route_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_tls_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/tlsRoutes' % client.transport._host, args[1])

def test_create_tls_route_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_tls_route(gcn_tls_route.CreateTlsRouteRequest(), parent='parent_value', tls_route=gcn_tls_route.TlsRoute(name='name_value'), tls_route_id='tls_route_id_value')

def test_create_tls_route_rest_error():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_tls_route.UpdateTlsRouteRequest, dict])
def test_update_tls_route_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'tls_route': {'name': 'projects/sample1/locations/sample2/tlsRoutes/sample3'}}
    request_init['tls_route'] = {'name': 'projects/sample1/locations/sample2/tlsRoutes/sample3', 'self_link': 'self_link_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'description': 'description_value', 'rules': [{'matches': [{'sni_host': ['sni_host_value1', 'sni_host_value2'], 'alpn': ['alpn_value1', 'alpn_value2']}], 'action': {'destinations': [{'service_name': 'service_name_value', 'weight': 648}]}}], 'meshes': ['meshes_value1', 'meshes_value2'], 'gateways': ['gateways_value1', 'gateways_value2']}
    test_field = gcn_tls_route.UpdateTlsRouteRequest.meta.fields['tls_route']

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
    for (field, value) in request_init['tls_route'].items():
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
                for i in range(0, len(request_init['tls_route'][field])):
                    del request_init['tls_route'][field][i][subfield]
            else:
                del request_init['tls_route'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_tls_route(request)
    assert response.operation.name == 'operations/spam'

def test_update_tls_route_rest_required_fields(request_type=gcn_tls_route.UpdateTlsRouteRequest):
    if False:
        return 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_tls_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_tls_route._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_tls_route(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_tls_route_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_tls_route._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('tlsRoute',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_tls_route_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_update_tls_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_update_tls_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_tls_route.UpdateTlsRouteRequest.pb(gcn_tls_route.UpdateTlsRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_tls_route.UpdateTlsRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_tls_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_tls_route_rest_bad_request(transport: str='rest', request_type=gcn_tls_route.UpdateTlsRouteRequest):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'tls_route': {'name': 'projects/sample1/locations/sample2/tlsRoutes/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_tls_route(request)

def test_update_tls_route_rest_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'tls_route': {'name': 'projects/sample1/locations/sample2/tlsRoutes/sample3'}}
        mock_args = dict(tls_route=gcn_tls_route.TlsRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_tls_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{tls_route.name=projects/*/locations/*/tlsRoutes/*}' % client.transport._host, args[1])

def test_update_tls_route_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_tls_route(gcn_tls_route.UpdateTlsRouteRequest(), tls_route=gcn_tls_route.TlsRoute(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_tls_route_rest_error():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tls_route.DeleteTlsRouteRequest, dict])
def test_delete_tls_route_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/tlsRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_tls_route(request)
    assert response.operation.name == 'operations/spam'

def test_delete_tls_route_rest_required_fields(request_type=tls_route.DeleteTlsRouteRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_tls_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_tls_route._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_tls_route(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_tls_route_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_tls_route._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_tls_route_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_delete_tls_route') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_delete_tls_route') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tls_route.DeleteTlsRouteRequest.pb(tls_route.DeleteTlsRouteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = tls_route.DeleteTlsRouteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_tls_route(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_tls_route_rest_bad_request(transport: str='rest', request_type=tls_route.DeleteTlsRouteRequest):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/tlsRoutes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_tls_route(request)

def test_delete_tls_route_rest_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/tlsRoutes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_tls_route(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/tlsRoutes/*}' % client.transport._host, args[1])

def test_delete_tls_route_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_tls_route(tls_route.DeleteTlsRouteRequest(), name='name_value')

def test_delete_tls_route_rest_error():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service_binding.ListServiceBindingsRequest, dict])
def test_list_service_bindings_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service_binding.ListServiceBindingsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = service_binding.ListServiceBindingsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_service_bindings(request)
    assert isinstance(response, pagers.ListServiceBindingsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_service_bindings_rest_required_fields(request_type=service_binding.ListServiceBindingsRequest):
    if False:
        return 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_service_bindings._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_service_bindings._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service_binding.ListServiceBindingsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service_binding.ListServiceBindingsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_service_bindings(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_service_bindings_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_service_bindings._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_service_bindings_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_list_service_bindings') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_list_service_bindings') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service_binding.ListServiceBindingsRequest.pb(service_binding.ListServiceBindingsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service_binding.ListServiceBindingsResponse.to_json(service_binding.ListServiceBindingsResponse())
        request = service_binding.ListServiceBindingsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service_binding.ListServiceBindingsResponse()
        client.list_service_bindings(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_service_bindings_rest_bad_request(transport: str='rest', request_type=service_binding.ListServiceBindingsRequest):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_service_bindings(request)

def test_list_service_bindings_rest_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service_binding.ListServiceBindingsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service_binding.ListServiceBindingsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_service_bindings(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/serviceBindings' % client.transport._host, args[1])

def test_list_service_bindings_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_service_bindings(service_binding.ListServiceBindingsRequest(), parent='parent_value')

def test_list_service_bindings_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding(), service_binding.ServiceBinding(), service_binding.ServiceBinding()], next_page_token='abc'), service_binding.ListServiceBindingsResponse(service_bindings=[], next_page_token='def'), service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding()], next_page_token='ghi'), service_binding.ListServiceBindingsResponse(service_bindings=[service_binding.ServiceBinding(), service_binding.ServiceBinding()]))
        response = response + response
        response = tuple((service_binding.ListServiceBindingsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_service_bindings(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service_binding.ServiceBinding) for i in results))
        pages = list(client.list_service_bindings(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service_binding.GetServiceBindingRequest, dict])
def test_get_service_binding_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/serviceBindings/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service_binding.ServiceBinding(name='name_value', description='description_value', service='service_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = service_binding.ServiceBinding.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_service_binding(request)
    assert isinstance(response, service_binding.ServiceBinding)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.service == 'service_value'

def test_get_service_binding_rest_required_fields(request_type=service_binding.GetServiceBindingRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_service_binding._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_service_binding._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service_binding.ServiceBinding()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service_binding.ServiceBinding.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_service_binding(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_service_binding_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_service_binding._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_service_binding_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_get_service_binding') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_get_service_binding') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service_binding.GetServiceBindingRequest.pb(service_binding.GetServiceBindingRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service_binding.ServiceBinding.to_json(service_binding.ServiceBinding())
        request = service_binding.GetServiceBindingRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service_binding.ServiceBinding()
        client.get_service_binding(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_service_binding_rest_bad_request(transport: str='rest', request_type=service_binding.GetServiceBindingRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/serviceBindings/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_service_binding(request)

def test_get_service_binding_rest_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service_binding.ServiceBinding()
        sample_request = {'name': 'projects/sample1/locations/sample2/serviceBindings/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service_binding.ServiceBinding.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_service_binding(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/serviceBindings/*}' % client.transport._host, args[1])

def test_get_service_binding_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_service_binding(service_binding.GetServiceBindingRequest(), name='name_value')

def test_get_service_binding_rest_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_service_binding.CreateServiceBindingRequest, dict])
def test_create_service_binding_rest(request_type):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['service_binding'] = {'name': 'name_value', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'service': 'service_value', 'labels': {}}
    test_field = gcn_service_binding.CreateServiceBindingRequest.meta.fields['service_binding']

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
    for (field, value) in request_init['service_binding'].items():
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
                for i in range(0, len(request_init['service_binding'][field])):
                    del request_init['service_binding'][field][i][subfield]
            else:
                del request_init['service_binding'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_service_binding(request)
    assert response.operation.name == 'operations/spam'

def test_create_service_binding_rest_required_fields(request_type=gcn_service_binding.CreateServiceBindingRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['service_binding_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'serviceBindingId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_service_binding._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'serviceBindingId' in jsonified_request
    assert jsonified_request['serviceBindingId'] == request_init['service_binding_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['serviceBindingId'] = 'service_binding_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_service_binding._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('service_binding_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'serviceBindingId' in jsonified_request
    assert jsonified_request['serviceBindingId'] == 'service_binding_id_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_service_binding(request)
            expected_params = [('serviceBindingId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_service_binding_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_service_binding._get_unset_required_fields({})
    assert set(unset_fields) == set(('serviceBindingId',)) & set(('parent', 'serviceBindingId', 'serviceBinding'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_service_binding_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_create_service_binding') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_create_service_binding') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_service_binding.CreateServiceBindingRequest.pb(gcn_service_binding.CreateServiceBindingRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_service_binding.CreateServiceBindingRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_service_binding(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_service_binding_rest_bad_request(transport: str='rest', request_type=gcn_service_binding.CreateServiceBindingRequest):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_service_binding(request)

def test_create_service_binding_rest_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', service_binding=gcn_service_binding.ServiceBinding(name='name_value'), service_binding_id='service_binding_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_service_binding(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/serviceBindings' % client.transport._host, args[1])

def test_create_service_binding_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_service_binding(gcn_service_binding.CreateServiceBindingRequest(), parent='parent_value', service_binding=gcn_service_binding.ServiceBinding(name='name_value'), service_binding_id='service_binding_id_value')

def test_create_service_binding_rest_error():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service_binding.DeleteServiceBindingRequest, dict])
def test_delete_service_binding_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/serviceBindings/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_service_binding(request)
    assert response.operation.name == 'operations/spam'

def test_delete_service_binding_rest_required_fields(request_type=service_binding.DeleteServiceBindingRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_service_binding._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_service_binding._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_service_binding(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_service_binding_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_service_binding._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_service_binding_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_delete_service_binding') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_delete_service_binding') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service_binding.DeleteServiceBindingRequest.pb(service_binding.DeleteServiceBindingRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service_binding.DeleteServiceBindingRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_service_binding(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_service_binding_rest_bad_request(transport: str='rest', request_type=service_binding.DeleteServiceBindingRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/serviceBindings/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_service_binding(request)

def test_delete_service_binding_rest_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/serviceBindings/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_service_binding(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/serviceBindings/*}' % client.transport._host, args[1])

def test_delete_service_binding_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_service_binding(service_binding.DeleteServiceBindingRequest(), name='name_value')

def test_delete_service_binding_rest_error():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [mesh.ListMeshesRequest, dict])
def test_list_meshes_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = mesh.ListMeshesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = mesh.ListMeshesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_meshes(request)
    assert isinstance(response, pagers.ListMeshesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_meshes_rest_required_fields(request_type=mesh.ListMeshesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_meshes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_meshes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = mesh.ListMeshesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = mesh.ListMeshesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_meshes(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_meshes_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_meshes._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_meshes_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_list_meshes') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_list_meshes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = mesh.ListMeshesRequest.pb(mesh.ListMeshesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = mesh.ListMeshesResponse.to_json(mesh.ListMeshesResponse())
        request = mesh.ListMeshesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = mesh.ListMeshesResponse()
        client.list_meshes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_meshes_rest_bad_request(transport: str='rest', request_type=mesh.ListMeshesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_meshes(request)

def test_list_meshes_rest_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = mesh.ListMeshesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = mesh.ListMeshesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_meshes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/meshes' % client.transport._host, args[1])

def test_list_meshes_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_meshes(mesh.ListMeshesRequest(), parent='parent_value')

def test_list_meshes_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (mesh.ListMeshesResponse(meshes=[mesh.Mesh(), mesh.Mesh(), mesh.Mesh()], next_page_token='abc'), mesh.ListMeshesResponse(meshes=[], next_page_token='def'), mesh.ListMeshesResponse(meshes=[mesh.Mesh()], next_page_token='ghi'), mesh.ListMeshesResponse(meshes=[mesh.Mesh(), mesh.Mesh()]))
        response = response + response
        response = tuple((mesh.ListMeshesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_meshes(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, mesh.Mesh) for i in results))
        pages = list(client.list_meshes(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [mesh.GetMeshRequest, dict])
def test_get_mesh_rest(request_type):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/meshes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = mesh.Mesh(name='name_value', self_link='self_link_value', description='description_value', interception_port=1848)
        response_value = Response()
        response_value.status_code = 200
        return_value = mesh.Mesh.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_mesh(request)
    assert isinstance(response, mesh.Mesh)
    assert response.name == 'name_value'
    assert response.self_link == 'self_link_value'
    assert response.description == 'description_value'
    assert response.interception_port == 1848

def test_get_mesh_rest_required_fields(request_type=mesh.GetMeshRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_mesh._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_mesh._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = mesh.Mesh()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = mesh.Mesh.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_mesh(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_mesh_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_mesh._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_mesh_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_get_mesh') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_get_mesh') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = mesh.GetMeshRequest.pb(mesh.GetMeshRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = mesh.Mesh.to_json(mesh.Mesh())
        request = mesh.GetMeshRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = mesh.Mesh()
        client.get_mesh(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_mesh_rest_bad_request(transport: str='rest', request_type=mesh.GetMeshRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/meshes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_mesh(request)

def test_get_mesh_rest_flattened():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = mesh.Mesh()
        sample_request = {'name': 'projects/sample1/locations/sample2/meshes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = mesh.Mesh.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_mesh(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/meshes/*}' % client.transport._host, args[1])

def test_get_mesh_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_mesh(mesh.GetMeshRequest(), name='name_value')

def test_get_mesh_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_mesh.CreateMeshRequest, dict])
def test_create_mesh_rest(request_type):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['mesh'] = {'name': 'name_value', 'self_link': 'self_link_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'interception_port': 1848}
    test_field = gcn_mesh.CreateMeshRequest.meta.fields['mesh']

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
    for (field, value) in request_init['mesh'].items():
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
                for i in range(0, len(request_init['mesh'][field])):
                    del request_init['mesh'][field][i][subfield]
            else:
                del request_init['mesh'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_mesh(request)
    assert response.operation.name == 'operations/spam'

def test_create_mesh_rest_required_fields(request_type=gcn_mesh.CreateMeshRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['mesh_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'meshId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_mesh._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'meshId' in jsonified_request
    assert jsonified_request['meshId'] == request_init['mesh_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['meshId'] = 'mesh_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_mesh._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('mesh_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'meshId' in jsonified_request
    assert jsonified_request['meshId'] == 'mesh_id_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_mesh(request)
            expected_params = [('meshId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_mesh_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_mesh._get_unset_required_fields({})
    assert set(unset_fields) == set(('meshId',)) & set(('parent', 'meshId', 'mesh'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_mesh_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_create_mesh') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_create_mesh') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_mesh.CreateMeshRequest.pb(gcn_mesh.CreateMeshRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_mesh.CreateMeshRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_mesh(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_mesh_rest_bad_request(transport: str='rest', request_type=gcn_mesh.CreateMeshRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_mesh(request)

def test_create_mesh_rest_flattened():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', mesh=gcn_mesh.Mesh(name='name_value'), mesh_id='mesh_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_mesh(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/meshes' % client.transport._host, args[1])

def test_create_mesh_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_mesh(gcn_mesh.CreateMeshRequest(), parent='parent_value', mesh=gcn_mesh.Mesh(name='name_value'), mesh_id='mesh_id_value')

def test_create_mesh_rest_error():
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcn_mesh.UpdateMeshRequest, dict])
def test_update_mesh_rest(request_type):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'mesh': {'name': 'projects/sample1/locations/sample2/meshes/sample3'}}
    request_init['mesh'] = {'name': 'projects/sample1/locations/sample2/meshes/sample3', 'self_link': 'self_link_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'interception_port': 1848}
    test_field = gcn_mesh.UpdateMeshRequest.meta.fields['mesh']

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
    for (field, value) in request_init['mesh'].items():
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
                for i in range(0, len(request_init['mesh'][field])):
                    del request_init['mesh'][field][i][subfield]
            else:
                del request_init['mesh'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_mesh(request)
    assert response.operation.name == 'operations/spam'

def test_update_mesh_rest_required_fields(request_type=gcn_mesh.UpdateMeshRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_mesh._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_mesh._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_mesh(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_mesh_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_mesh._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('mesh',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_mesh_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_update_mesh') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_update_mesh') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcn_mesh.UpdateMeshRequest.pb(gcn_mesh.UpdateMeshRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcn_mesh.UpdateMeshRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_mesh(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_mesh_rest_bad_request(transport: str='rest', request_type=gcn_mesh.UpdateMeshRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'mesh': {'name': 'projects/sample1/locations/sample2/meshes/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_mesh(request)

def test_update_mesh_rest_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'mesh': {'name': 'projects/sample1/locations/sample2/meshes/sample3'}}
        mock_args = dict(mesh=gcn_mesh.Mesh(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_mesh(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{mesh.name=projects/*/locations/*/meshes/*}' % client.transport._host, args[1])

def test_update_mesh_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_mesh(gcn_mesh.UpdateMeshRequest(), mesh=gcn_mesh.Mesh(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_mesh_rest_error():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [mesh.DeleteMeshRequest, dict])
def test_delete_mesh_rest(request_type):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/meshes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_mesh(request)
    assert response.operation.name == 'operations/spam'

def test_delete_mesh_rest_required_fields(request_type=mesh.DeleteMeshRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NetworkServicesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_mesh._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_mesh._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_mesh(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_mesh_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_mesh._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_mesh_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NetworkServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NetworkServicesRestInterceptor())
    client = NetworkServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.NetworkServicesRestInterceptor, 'post_delete_mesh') as post, mock.patch.object(transports.NetworkServicesRestInterceptor, 'pre_delete_mesh') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = mesh.DeleteMeshRequest.pb(mesh.DeleteMeshRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = mesh.DeleteMeshRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_mesh(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_mesh_rest_bad_request(transport: str='rest', request_type=mesh.DeleteMeshRequest):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/meshes/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_mesh(request)

def test_delete_mesh_rest_flattened():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/meshes/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_mesh(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/meshes/*}' % client.transport._host, args[1])

def test_delete_mesh_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_mesh(mesh.DeleteMeshRequest(), name='name_value')

def test_delete_mesh_rest_error():
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkServicesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.NetworkServicesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NetworkServicesClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.NetworkServicesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = NetworkServicesClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = NetworkServicesClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.NetworkServicesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NetworkServicesClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.NetworkServicesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = NetworkServicesClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.NetworkServicesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.NetworkServicesGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.NetworkServicesGrpcTransport, transports.NetworkServicesGrpcAsyncIOTransport, transports.NetworkServicesRestTransport])
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
    transport = NetworkServicesClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.NetworkServicesGrpcTransport)

def test_network_services_base_transport_error():
    if False:
        return 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.NetworkServicesTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_network_services_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.network_services_v1.services.network_services.transports.NetworkServicesTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.NetworkServicesTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_endpoint_policies', 'get_endpoint_policy', 'create_endpoint_policy', 'update_endpoint_policy', 'delete_endpoint_policy', 'list_gateways', 'get_gateway', 'create_gateway', 'update_gateway', 'delete_gateway', 'list_grpc_routes', 'get_grpc_route', 'create_grpc_route', 'update_grpc_route', 'delete_grpc_route', 'list_http_routes', 'get_http_route', 'create_http_route', 'update_http_route', 'delete_http_route', 'list_tcp_routes', 'get_tcp_route', 'create_tcp_route', 'update_tcp_route', 'delete_tcp_route', 'list_tls_routes', 'get_tls_route', 'create_tls_route', 'update_tls_route', 'delete_tls_route', 'list_service_bindings', 'get_service_binding', 'create_service_binding', 'delete_service_binding', 'list_meshes', 'get_mesh', 'create_mesh', 'update_mesh', 'delete_mesh', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_network_services_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.network_services_v1.services.network_services.transports.NetworkServicesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.NetworkServicesTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_network_services_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.network_services_v1.services.network_services.transports.NetworkServicesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.NetworkServicesTransport()
        adc.assert_called_once()

def test_network_services_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        NetworkServicesClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.NetworkServicesGrpcTransport, transports.NetworkServicesGrpcAsyncIOTransport])
def test_network_services_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.NetworkServicesGrpcTransport, transports.NetworkServicesGrpcAsyncIOTransport, transports.NetworkServicesRestTransport])
def test_network_services_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.NetworkServicesGrpcTransport, grpc_helpers), (transports.NetworkServicesGrpcAsyncIOTransport, grpc_helpers_async)])
def test_network_services_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('networkservices.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='networkservices.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.NetworkServicesGrpcTransport, transports.NetworkServicesGrpcAsyncIOTransport])
def test_network_services_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        while True:
            i = 10
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

def test_network_services_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.NetworkServicesRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_network_services_rest_lro_client():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_network_services_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='networkservices.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('networkservices.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://networkservices.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_network_services_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='networkservices.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('networkservices.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://networkservices.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_network_services_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = NetworkServicesClient(credentials=creds1, transport=transport_name)
    client2 = NetworkServicesClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_endpoint_policies._session
    session2 = client2.transport.list_endpoint_policies._session
    assert session1 != session2
    session1 = client1.transport.get_endpoint_policy._session
    session2 = client2.transport.get_endpoint_policy._session
    assert session1 != session2
    session1 = client1.transport.create_endpoint_policy._session
    session2 = client2.transport.create_endpoint_policy._session
    assert session1 != session2
    session1 = client1.transport.update_endpoint_policy._session
    session2 = client2.transport.update_endpoint_policy._session
    assert session1 != session2
    session1 = client1.transport.delete_endpoint_policy._session
    session2 = client2.transport.delete_endpoint_policy._session
    assert session1 != session2
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
    session1 = client1.transport.list_grpc_routes._session
    session2 = client2.transport.list_grpc_routes._session
    assert session1 != session2
    session1 = client1.transport.get_grpc_route._session
    session2 = client2.transport.get_grpc_route._session
    assert session1 != session2
    session1 = client1.transport.create_grpc_route._session
    session2 = client2.transport.create_grpc_route._session
    assert session1 != session2
    session1 = client1.transport.update_grpc_route._session
    session2 = client2.transport.update_grpc_route._session
    assert session1 != session2
    session1 = client1.transport.delete_grpc_route._session
    session2 = client2.transport.delete_grpc_route._session
    assert session1 != session2
    session1 = client1.transport.list_http_routes._session
    session2 = client2.transport.list_http_routes._session
    assert session1 != session2
    session1 = client1.transport.get_http_route._session
    session2 = client2.transport.get_http_route._session
    assert session1 != session2
    session1 = client1.transport.create_http_route._session
    session2 = client2.transport.create_http_route._session
    assert session1 != session2
    session1 = client1.transport.update_http_route._session
    session2 = client2.transport.update_http_route._session
    assert session1 != session2
    session1 = client1.transport.delete_http_route._session
    session2 = client2.transport.delete_http_route._session
    assert session1 != session2
    session1 = client1.transport.list_tcp_routes._session
    session2 = client2.transport.list_tcp_routes._session
    assert session1 != session2
    session1 = client1.transport.get_tcp_route._session
    session2 = client2.transport.get_tcp_route._session
    assert session1 != session2
    session1 = client1.transport.create_tcp_route._session
    session2 = client2.transport.create_tcp_route._session
    assert session1 != session2
    session1 = client1.transport.update_tcp_route._session
    session2 = client2.transport.update_tcp_route._session
    assert session1 != session2
    session1 = client1.transport.delete_tcp_route._session
    session2 = client2.transport.delete_tcp_route._session
    assert session1 != session2
    session1 = client1.transport.list_tls_routes._session
    session2 = client2.transport.list_tls_routes._session
    assert session1 != session2
    session1 = client1.transport.get_tls_route._session
    session2 = client2.transport.get_tls_route._session
    assert session1 != session2
    session1 = client1.transport.create_tls_route._session
    session2 = client2.transport.create_tls_route._session
    assert session1 != session2
    session1 = client1.transport.update_tls_route._session
    session2 = client2.transport.update_tls_route._session
    assert session1 != session2
    session1 = client1.transport.delete_tls_route._session
    session2 = client2.transport.delete_tls_route._session
    assert session1 != session2
    session1 = client1.transport.list_service_bindings._session
    session2 = client2.transport.list_service_bindings._session
    assert session1 != session2
    session1 = client1.transport.get_service_binding._session
    session2 = client2.transport.get_service_binding._session
    assert session1 != session2
    session1 = client1.transport.create_service_binding._session
    session2 = client2.transport.create_service_binding._session
    assert session1 != session2
    session1 = client1.transport.delete_service_binding._session
    session2 = client2.transport.delete_service_binding._session
    assert session1 != session2
    session1 = client1.transport.list_meshes._session
    session2 = client2.transport.list_meshes._session
    assert session1 != session2
    session1 = client1.transport.get_mesh._session
    session2 = client2.transport.get_mesh._session
    assert session1 != session2
    session1 = client1.transport.create_mesh._session
    session2 = client2.transport.create_mesh._session
    assert session1 != session2
    session1 = client1.transport.update_mesh._session
    session2 = client2.transport.update_mesh._session
    assert session1 != session2
    session1 = client1.transport.delete_mesh._session
    session2 = client2.transport.delete_mesh._session
    assert session1 != session2

def test_network_services_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.NetworkServicesGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_network_services_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.NetworkServicesGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.NetworkServicesGrpcTransport, transports.NetworkServicesGrpcAsyncIOTransport])
def test_network_services_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.NetworkServicesGrpcTransport, transports.NetworkServicesGrpcAsyncIOTransport])
def test_network_services_transport_channel_mtls_with_adc(transport_class):
    if False:
        return 10
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

def test_network_services_grpc_lro_client():
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_network_services_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_authorization_policy_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    authorization_policy = 'whelk'
    expected = 'projects/{project}/locations/{location}/authorizationPolicies/{authorization_policy}'.format(project=project, location=location, authorization_policy=authorization_policy)
    actual = NetworkServicesClient.authorization_policy_path(project, location, authorization_policy)
    assert expected == actual

def test_parse_authorization_policy_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus', 'location': 'oyster', 'authorization_policy': 'nudibranch'}
    path = NetworkServicesClient.authorization_policy_path(**expected)
    actual = NetworkServicesClient.parse_authorization_policy_path(path)
    assert expected == actual

def test_backend_service_path():
    if False:
        return 10
    project = 'cuttlefish'
    location = 'mussel'
    backend_service = 'winkle'
    expected = 'projects/{project}/locations/{location}/backendServices/{backend_service}'.format(project=project, location=location, backend_service=backend_service)
    actual = NetworkServicesClient.backend_service_path(project, location, backend_service)
    assert expected == actual

def test_parse_backend_service_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nautilus', 'location': 'scallop', 'backend_service': 'abalone'}
    path = NetworkServicesClient.backend_service_path(**expected)
    actual = NetworkServicesClient.parse_backend_service_path(path)
    assert expected == actual

def test_client_tls_policy_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    client_tls_policy = 'whelk'
    expected = 'projects/{project}/locations/{location}/clientTlsPolicies/{client_tls_policy}'.format(project=project, location=location, client_tls_policy=client_tls_policy)
    actual = NetworkServicesClient.client_tls_policy_path(project, location, client_tls_policy)
    assert expected == actual

def test_parse_client_tls_policy_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus', 'location': 'oyster', 'client_tls_policy': 'nudibranch'}
    path = NetworkServicesClient.client_tls_policy_path(**expected)
    actual = NetworkServicesClient.parse_client_tls_policy_path(path)
    assert expected == actual

def test_endpoint_policy_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    endpoint_policy = 'winkle'
    expected = 'projects/{project}/locations/{location}/endpointPolicies/{endpoint_policy}'.format(project=project, location=location, endpoint_policy=endpoint_policy)
    actual = NetworkServicesClient.endpoint_policy_path(project, location, endpoint_policy)
    assert expected == actual

def test_parse_endpoint_policy_path():
    if False:
        return 10
    expected = {'project': 'nautilus', 'location': 'scallop', 'endpoint_policy': 'abalone'}
    path = NetworkServicesClient.endpoint_policy_path(**expected)
    actual = NetworkServicesClient.parse_endpoint_policy_path(path)
    assert expected == actual

def test_gateway_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    gateway = 'whelk'
    expected = 'projects/{project}/locations/{location}/gateways/{gateway}'.format(project=project, location=location, gateway=gateway)
    actual = NetworkServicesClient.gateway_path(project, location, gateway)
    assert expected == actual

def test_parse_gateway_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus', 'location': 'oyster', 'gateway': 'nudibranch'}
    path = NetworkServicesClient.gateway_path(**expected)
    actual = NetworkServicesClient.parse_gateway_path(path)
    assert expected == actual

def test_grpc_route_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    grpc_route = 'winkle'
    expected = 'projects/{project}/locations/{location}/grpcRoutes/{grpc_route}'.format(project=project, location=location, grpc_route=grpc_route)
    actual = NetworkServicesClient.grpc_route_path(project, location, grpc_route)
    assert expected == actual

def test_parse_grpc_route_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nautilus', 'location': 'scallop', 'grpc_route': 'abalone'}
    path = NetworkServicesClient.grpc_route_path(**expected)
    actual = NetworkServicesClient.parse_grpc_route_path(path)
    assert expected == actual

def test_http_route_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    http_route = 'whelk'
    expected = 'projects/{project}/locations/{location}/httpRoutes/{http_route}'.format(project=project, location=location, http_route=http_route)
    actual = NetworkServicesClient.http_route_path(project, location, http_route)
    assert expected == actual

def test_parse_http_route_path():
    if False:
        return 10
    expected = {'project': 'octopus', 'location': 'oyster', 'http_route': 'nudibranch'}
    path = NetworkServicesClient.http_route_path(**expected)
    actual = NetworkServicesClient.parse_http_route_path(path)
    assert expected == actual

def test_mesh_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    mesh = 'winkle'
    expected = 'projects/{project}/locations/{location}/meshes/{mesh}'.format(project=project, location=location, mesh=mesh)
    actual = NetworkServicesClient.mesh_path(project, location, mesh)
    assert expected == actual

def test_parse_mesh_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nautilus', 'location': 'scallop', 'mesh': 'abalone'}
    path = NetworkServicesClient.mesh_path(**expected)
    actual = NetworkServicesClient.parse_mesh_path(path)
    assert expected == actual

def test_server_tls_policy_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    server_tls_policy = 'whelk'
    expected = 'projects/{project}/locations/{location}/serverTlsPolicies/{server_tls_policy}'.format(project=project, location=location, server_tls_policy=server_tls_policy)
    actual = NetworkServicesClient.server_tls_policy_path(project, location, server_tls_policy)
    assert expected == actual

def test_parse_server_tls_policy_path():
    if False:
        return 10
    expected = {'project': 'octopus', 'location': 'oyster', 'server_tls_policy': 'nudibranch'}
    path = NetworkServicesClient.server_tls_policy_path(**expected)
    actual = NetworkServicesClient.parse_server_tls_policy_path(path)
    assert expected == actual

def test_service_binding_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    service_binding = 'winkle'
    expected = 'projects/{project}/locations/{location}/serviceBindings/{service_binding}'.format(project=project, location=location, service_binding=service_binding)
    actual = NetworkServicesClient.service_binding_path(project, location, service_binding)
    assert expected == actual

def test_parse_service_binding_path():
    if False:
        return 10
    expected = {'project': 'nautilus', 'location': 'scallop', 'service_binding': 'abalone'}
    path = NetworkServicesClient.service_binding_path(**expected)
    actual = NetworkServicesClient.parse_service_binding_path(path)
    assert expected == actual

def test_tcp_route_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    tcp_route = 'whelk'
    expected = 'projects/{project}/locations/{location}/tcpRoutes/{tcp_route}'.format(project=project, location=location, tcp_route=tcp_route)
    actual = NetworkServicesClient.tcp_route_path(project, location, tcp_route)
    assert expected == actual

def test_parse_tcp_route_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'location': 'oyster', 'tcp_route': 'nudibranch'}
    path = NetworkServicesClient.tcp_route_path(**expected)
    actual = NetworkServicesClient.parse_tcp_route_path(path)
    assert expected == actual

def test_tls_route_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    tls_route = 'winkle'
    expected = 'projects/{project}/locations/{location}/tlsRoutes/{tls_route}'.format(project=project, location=location, tls_route=tls_route)
    actual = NetworkServicesClient.tls_route_path(project, location, tls_route)
    assert expected == actual

def test_parse_tls_route_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nautilus', 'location': 'scallop', 'tls_route': 'abalone'}
    path = NetworkServicesClient.tls_route_path(**expected)
    actual = NetworkServicesClient.parse_tls_route_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = NetworkServicesClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'clam'}
    path = NetworkServicesClient.common_billing_account_path(**expected)
    actual = NetworkServicesClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = NetworkServicesClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'octopus'}
    path = NetworkServicesClient.common_folder_path(**expected)
    actual = NetworkServicesClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = NetworkServicesClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'nudibranch'}
    path = NetworkServicesClient.common_organization_path(**expected)
    actual = NetworkServicesClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = NetworkServicesClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'mussel'}
    path = NetworkServicesClient.common_project_path(**expected)
    actual = NetworkServicesClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = NetworkServicesClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = NetworkServicesClient.common_location_path(**expected)
    actual = NetworkServicesClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.NetworkServicesTransport, '_prep_wrapped_messages') as prep:
        client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.NetworkServicesTransport, '_prep_wrapped_messages') as prep:
        transport_class = NetworkServicesClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/edgeCacheKeysets/sample3'}, request)
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/edgeCacheKeysets/sample3'}
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
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/edgeCacheKeysets/sample3'}, request)
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/edgeCacheKeysets/sample3'}
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/edgeCacheKeysets/sample3'}, request)
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/edgeCacheKeysets/sample3'}
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        return 10
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = NetworkServicesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = NetworkServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(NetworkServicesClient, transports.NetworkServicesGrpcTransport), (NetworkServicesAsyncClient, transports.NetworkServicesGrpcAsyncIOTransport)])
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