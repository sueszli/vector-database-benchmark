import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
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
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from google.cloud.network_security_v1beta1.services.network_security import NetworkSecurityAsyncClient, NetworkSecurityClient, pagers, transports
from google.cloud.network_security_v1beta1.types import authorization_policy as gcn_authorization_policy
from google.cloud.network_security_v1beta1.types import client_tls_policy as gcn_client_tls_policy
from google.cloud.network_security_v1beta1.types import server_tls_policy as gcn_server_tls_policy
from google.cloud.network_security_v1beta1.types import authorization_policy
from google.cloud.network_security_v1beta1.types import client_tls_policy
from google.cloud.network_security_v1beta1.types import common
from google.cloud.network_security_v1beta1.types import server_tls_policy
from google.cloud.network_security_v1beta1.types import tls

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
    assert NetworkSecurityClient._get_default_mtls_endpoint(None) is None
    assert NetworkSecurityClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert NetworkSecurityClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert NetworkSecurityClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert NetworkSecurityClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert NetworkSecurityClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(NetworkSecurityClient, 'grpc'), (NetworkSecurityAsyncClient, 'grpc_asyncio')])
def test_network_security_client_from_service_account_info(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == 'networksecurity.googleapis.com:443'

@pytest.mark.parametrize('transport_class,transport_name', [(transports.NetworkSecurityGrpcTransport, 'grpc'), (transports.NetworkSecurityGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_network_security_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(NetworkSecurityClient, 'grpc'), (NetworkSecurityAsyncClient, 'grpc_asyncio')])
def test_network_security_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == 'networksecurity.googleapis.com:443'

def test_network_security_client_get_transport_class():
    if False:
        return 10
    transport = NetworkSecurityClient.get_transport_class()
    available_transports = [transports.NetworkSecurityGrpcTransport]
    assert transport in available_transports
    transport = NetworkSecurityClient.get_transport_class('grpc')
    assert transport == transports.NetworkSecurityGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(NetworkSecurityClient, transports.NetworkSecurityGrpcTransport, 'grpc'), (NetworkSecurityAsyncClient, transports.NetworkSecurityGrpcAsyncIOTransport, 'grpc_asyncio')])
@mock.patch.object(NetworkSecurityClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetworkSecurityClient))
@mock.patch.object(NetworkSecurityAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetworkSecurityAsyncClient))
def test_network_security_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(NetworkSecurityClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(NetworkSecurityClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(NetworkSecurityClient, transports.NetworkSecurityGrpcTransport, 'grpc', 'true'), (NetworkSecurityAsyncClient, transports.NetworkSecurityGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (NetworkSecurityClient, transports.NetworkSecurityGrpcTransport, 'grpc', 'false'), (NetworkSecurityAsyncClient, transports.NetworkSecurityGrpcAsyncIOTransport, 'grpc_asyncio', 'false')])
@mock.patch.object(NetworkSecurityClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetworkSecurityClient))
@mock.patch.object(NetworkSecurityAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetworkSecurityAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_network_security_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [NetworkSecurityClient, NetworkSecurityAsyncClient])
@mock.patch.object(NetworkSecurityClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetworkSecurityClient))
@mock.patch.object(NetworkSecurityAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NetworkSecurityAsyncClient))
def test_network_security_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        i = 10
        return i + 15
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(NetworkSecurityClient, transports.NetworkSecurityGrpcTransport, 'grpc'), (NetworkSecurityAsyncClient, transports.NetworkSecurityGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_network_security_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(NetworkSecurityClient, transports.NetworkSecurityGrpcTransport, 'grpc', grpc_helpers), (NetworkSecurityAsyncClient, transports.NetworkSecurityGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_network_security_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_network_security_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.network_security_v1beta1.services.network_security.transports.NetworkSecurityGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = NetworkSecurityClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(NetworkSecurityClient, transports.NetworkSecurityGrpcTransport, 'grpc', grpc_helpers), (NetworkSecurityAsyncClient, transports.NetworkSecurityGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_network_security_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
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
        create_channel.assert_called_with('networksecurity.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='networksecurity.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [authorization_policy.ListAuthorizationPoliciesRequest, dict])
def test_list_authorization_policies(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_authorization_policies), '__call__') as call:
        call.return_value = authorization_policy.ListAuthorizationPoliciesResponse(next_page_token='next_page_token_value')
        response = client.list_authorization_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == authorization_policy.ListAuthorizationPoliciesRequest()
    assert isinstance(response, pagers.ListAuthorizationPoliciesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_authorization_policies_empty_call():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_authorization_policies), '__call__') as call:
        client.list_authorization_policies()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == authorization_policy.ListAuthorizationPoliciesRequest()

@pytest.mark.asyncio
async def test_list_authorization_policies_async(transport: str='grpc_asyncio', request_type=authorization_policy.ListAuthorizationPoliciesRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_authorization_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(authorization_policy.ListAuthorizationPoliciesResponse(next_page_token='next_page_token_value'))
        response = await client.list_authorization_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == authorization_policy.ListAuthorizationPoliciesRequest()
    assert isinstance(response, pagers.ListAuthorizationPoliciesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_authorization_policies_async_from_dict():
    await test_list_authorization_policies_async(request_type=dict)

def test_list_authorization_policies_field_headers():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = authorization_policy.ListAuthorizationPoliciesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_authorization_policies), '__call__') as call:
        call.return_value = authorization_policy.ListAuthorizationPoliciesResponse()
        client.list_authorization_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_authorization_policies_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = authorization_policy.ListAuthorizationPoliciesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_authorization_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(authorization_policy.ListAuthorizationPoliciesResponse())
        await client.list_authorization_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_authorization_policies_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_authorization_policies), '__call__') as call:
        call.return_value = authorization_policy.ListAuthorizationPoliciesResponse()
        client.list_authorization_policies(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_authorization_policies_flattened_error():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_authorization_policies(authorization_policy.ListAuthorizationPoliciesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_authorization_policies_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_authorization_policies), '__call__') as call:
        call.return_value = authorization_policy.ListAuthorizationPoliciesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(authorization_policy.ListAuthorizationPoliciesResponse())
        response = await client.list_authorization_policies(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_authorization_policies_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_authorization_policies(authorization_policy.ListAuthorizationPoliciesRequest(), parent='parent_value')

def test_list_authorization_policies_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_authorization_policies), '__call__') as call:
        call.side_effect = (authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[authorization_policy.AuthorizationPolicy(), authorization_policy.AuthorizationPolicy(), authorization_policy.AuthorizationPolicy()], next_page_token='abc'), authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[], next_page_token='def'), authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[authorization_policy.AuthorizationPolicy()], next_page_token='ghi'), authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[authorization_policy.AuthorizationPolicy(), authorization_policy.AuthorizationPolicy()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_authorization_policies(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, authorization_policy.AuthorizationPolicy) for i in results))

def test_list_authorization_policies_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_authorization_policies), '__call__') as call:
        call.side_effect = (authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[authorization_policy.AuthorizationPolicy(), authorization_policy.AuthorizationPolicy(), authorization_policy.AuthorizationPolicy()], next_page_token='abc'), authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[], next_page_token='def'), authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[authorization_policy.AuthorizationPolicy()], next_page_token='ghi'), authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[authorization_policy.AuthorizationPolicy(), authorization_policy.AuthorizationPolicy()]), RuntimeError)
        pages = list(client.list_authorization_policies(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_authorization_policies_async_pager():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_authorization_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[authorization_policy.AuthorizationPolicy(), authorization_policy.AuthorizationPolicy(), authorization_policy.AuthorizationPolicy()], next_page_token='abc'), authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[], next_page_token='def'), authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[authorization_policy.AuthorizationPolicy()], next_page_token='ghi'), authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[authorization_policy.AuthorizationPolicy(), authorization_policy.AuthorizationPolicy()]), RuntimeError)
        async_pager = await client.list_authorization_policies(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, authorization_policy.AuthorizationPolicy) for i in responses))

@pytest.mark.asyncio
async def test_list_authorization_policies_async_pages():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_authorization_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[authorization_policy.AuthorizationPolicy(), authorization_policy.AuthorizationPolicy(), authorization_policy.AuthorizationPolicy()], next_page_token='abc'), authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[], next_page_token='def'), authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[authorization_policy.AuthorizationPolicy()], next_page_token='ghi'), authorization_policy.ListAuthorizationPoliciesResponse(authorization_policies=[authorization_policy.AuthorizationPolicy(), authorization_policy.AuthorizationPolicy()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_authorization_policies(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [authorization_policy.GetAuthorizationPolicyRequest, dict])
def test_get_authorization_policy(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_authorization_policy), '__call__') as call:
        call.return_value = authorization_policy.AuthorizationPolicy(name='name_value', description='description_value', action=authorization_policy.AuthorizationPolicy.Action.ALLOW)
        response = client.get_authorization_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == authorization_policy.GetAuthorizationPolicyRequest()
    assert isinstance(response, authorization_policy.AuthorizationPolicy)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.action == authorization_policy.AuthorizationPolicy.Action.ALLOW

def test_get_authorization_policy_empty_call():
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_authorization_policy), '__call__') as call:
        client.get_authorization_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == authorization_policy.GetAuthorizationPolicyRequest()

@pytest.mark.asyncio
async def test_get_authorization_policy_async(transport: str='grpc_asyncio', request_type=authorization_policy.GetAuthorizationPolicyRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_authorization_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(authorization_policy.AuthorizationPolicy(name='name_value', description='description_value', action=authorization_policy.AuthorizationPolicy.Action.ALLOW))
        response = await client.get_authorization_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == authorization_policy.GetAuthorizationPolicyRequest()
    assert isinstance(response, authorization_policy.AuthorizationPolicy)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.action == authorization_policy.AuthorizationPolicy.Action.ALLOW

@pytest.mark.asyncio
async def test_get_authorization_policy_async_from_dict():
    await test_get_authorization_policy_async(request_type=dict)

def test_get_authorization_policy_field_headers():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = authorization_policy.GetAuthorizationPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_authorization_policy), '__call__') as call:
        call.return_value = authorization_policy.AuthorizationPolicy()
        client.get_authorization_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_authorization_policy_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = authorization_policy.GetAuthorizationPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_authorization_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(authorization_policy.AuthorizationPolicy())
        await client.get_authorization_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_authorization_policy_flattened():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_authorization_policy), '__call__') as call:
        call.return_value = authorization_policy.AuthorizationPolicy()
        client.get_authorization_policy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_authorization_policy_flattened_error():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_authorization_policy(authorization_policy.GetAuthorizationPolicyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_authorization_policy_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_authorization_policy), '__call__') as call:
        call.return_value = authorization_policy.AuthorizationPolicy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(authorization_policy.AuthorizationPolicy())
        response = await client.get_authorization_policy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_authorization_policy_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_authorization_policy(authorization_policy.GetAuthorizationPolicyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_authorization_policy.CreateAuthorizationPolicyRequest, dict])
def test_create_authorization_policy(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_authorization_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_authorization_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_authorization_policy.CreateAuthorizationPolicyRequest()
    assert isinstance(response, future.Future)

def test_create_authorization_policy_empty_call():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_authorization_policy), '__call__') as call:
        client.create_authorization_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_authorization_policy.CreateAuthorizationPolicyRequest()

@pytest.mark.asyncio
async def test_create_authorization_policy_async(transport: str='grpc_asyncio', request_type=gcn_authorization_policy.CreateAuthorizationPolicyRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_authorization_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_authorization_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_authorization_policy.CreateAuthorizationPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_authorization_policy_async_from_dict():
    await test_create_authorization_policy_async(request_type=dict)

def test_create_authorization_policy_field_headers():
    if False:
        while True:
            i = 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_authorization_policy.CreateAuthorizationPolicyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_authorization_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_authorization_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_authorization_policy_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_authorization_policy.CreateAuthorizationPolicyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_authorization_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_authorization_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_authorization_policy_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_authorization_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_authorization_policy(parent='parent_value', authorization_policy=gcn_authorization_policy.AuthorizationPolicy(name='name_value'), authorization_policy_id='authorization_policy_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].authorization_policy
        mock_val = gcn_authorization_policy.AuthorizationPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].authorization_policy_id
        mock_val = 'authorization_policy_id_value'
        assert arg == mock_val

def test_create_authorization_policy_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_authorization_policy(gcn_authorization_policy.CreateAuthorizationPolicyRequest(), parent='parent_value', authorization_policy=gcn_authorization_policy.AuthorizationPolicy(name='name_value'), authorization_policy_id='authorization_policy_id_value')

@pytest.mark.asyncio
async def test_create_authorization_policy_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_authorization_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_authorization_policy(parent='parent_value', authorization_policy=gcn_authorization_policy.AuthorizationPolicy(name='name_value'), authorization_policy_id='authorization_policy_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].authorization_policy
        mock_val = gcn_authorization_policy.AuthorizationPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].authorization_policy_id
        mock_val = 'authorization_policy_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_authorization_policy_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_authorization_policy(gcn_authorization_policy.CreateAuthorizationPolicyRequest(), parent='parent_value', authorization_policy=gcn_authorization_policy.AuthorizationPolicy(name='name_value'), authorization_policy_id='authorization_policy_id_value')

@pytest.mark.parametrize('request_type', [gcn_authorization_policy.UpdateAuthorizationPolicyRequest, dict])
def test_update_authorization_policy(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_authorization_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_authorization_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_authorization_policy.UpdateAuthorizationPolicyRequest()
    assert isinstance(response, future.Future)

def test_update_authorization_policy_empty_call():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_authorization_policy), '__call__') as call:
        client.update_authorization_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_authorization_policy.UpdateAuthorizationPolicyRequest()

@pytest.mark.asyncio
async def test_update_authorization_policy_async(transport: str='grpc_asyncio', request_type=gcn_authorization_policy.UpdateAuthorizationPolicyRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_authorization_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_authorization_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_authorization_policy.UpdateAuthorizationPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_authorization_policy_async_from_dict():
    await test_update_authorization_policy_async(request_type=dict)

def test_update_authorization_policy_field_headers():
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_authorization_policy.UpdateAuthorizationPolicyRequest()
    request.authorization_policy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_authorization_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_authorization_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'authorization_policy.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_authorization_policy_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_authorization_policy.UpdateAuthorizationPolicyRequest()
    request.authorization_policy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_authorization_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_authorization_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'authorization_policy.name=name_value') in kw['metadata']

def test_update_authorization_policy_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_authorization_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_authorization_policy(authorization_policy=gcn_authorization_policy.AuthorizationPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].authorization_policy
        mock_val = gcn_authorization_policy.AuthorizationPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_authorization_policy_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_authorization_policy(gcn_authorization_policy.UpdateAuthorizationPolicyRequest(), authorization_policy=gcn_authorization_policy.AuthorizationPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_authorization_policy_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_authorization_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_authorization_policy(authorization_policy=gcn_authorization_policy.AuthorizationPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].authorization_policy
        mock_val = gcn_authorization_policy.AuthorizationPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_authorization_policy_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_authorization_policy(gcn_authorization_policy.UpdateAuthorizationPolicyRequest(), authorization_policy=gcn_authorization_policy.AuthorizationPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [authorization_policy.DeleteAuthorizationPolicyRequest, dict])
def test_delete_authorization_policy(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_authorization_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_authorization_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == authorization_policy.DeleteAuthorizationPolicyRequest()
    assert isinstance(response, future.Future)

def test_delete_authorization_policy_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_authorization_policy), '__call__') as call:
        client.delete_authorization_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == authorization_policy.DeleteAuthorizationPolicyRequest()

@pytest.mark.asyncio
async def test_delete_authorization_policy_async(transport: str='grpc_asyncio', request_type=authorization_policy.DeleteAuthorizationPolicyRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_authorization_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_authorization_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == authorization_policy.DeleteAuthorizationPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_authorization_policy_async_from_dict():
    await test_delete_authorization_policy_async(request_type=dict)

def test_delete_authorization_policy_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = authorization_policy.DeleteAuthorizationPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_authorization_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_authorization_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_authorization_policy_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = authorization_policy.DeleteAuthorizationPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_authorization_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_authorization_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_authorization_policy_flattened():
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_authorization_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_authorization_policy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_authorization_policy_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_authorization_policy(authorization_policy.DeleteAuthorizationPolicyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_authorization_policy_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_authorization_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_authorization_policy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_authorization_policy_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_authorization_policy(authorization_policy.DeleteAuthorizationPolicyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [server_tls_policy.ListServerTlsPoliciesRequest, dict])
def test_list_server_tls_policies(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_server_tls_policies), '__call__') as call:
        call.return_value = server_tls_policy.ListServerTlsPoliciesResponse(next_page_token='next_page_token_value')
        response = client.list_server_tls_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == server_tls_policy.ListServerTlsPoliciesRequest()
    assert isinstance(response, pagers.ListServerTlsPoliciesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_server_tls_policies_empty_call():
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_server_tls_policies), '__call__') as call:
        client.list_server_tls_policies()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == server_tls_policy.ListServerTlsPoliciesRequest()

@pytest.mark.asyncio
async def test_list_server_tls_policies_async(transport: str='grpc_asyncio', request_type=server_tls_policy.ListServerTlsPoliciesRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_server_tls_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(server_tls_policy.ListServerTlsPoliciesResponse(next_page_token='next_page_token_value'))
        response = await client.list_server_tls_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == server_tls_policy.ListServerTlsPoliciesRequest()
    assert isinstance(response, pagers.ListServerTlsPoliciesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_server_tls_policies_async_from_dict():
    await test_list_server_tls_policies_async(request_type=dict)

def test_list_server_tls_policies_field_headers():
    if False:
        while True:
            i = 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = server_tls_policy.ListServerTlsPoliciesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_server_tls_policies), '__call__') as call:
        call.return_value = server_tls_policy.ListServerTlsPoliciesResponse()
        client.list_server_tls_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_server_tls_policies_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = server_tls_policy.ListServerTlsPoliciesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_server_tls_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(server_tls_policy.ListServerTlsPoliciesResponse())
        await client.list_server_tls_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_server_tls_policies_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_server_tls_policies), '__call__') as call:
        call.return_value = server_tls_policy.ListServerTlsPoliciesResponse()
        client.list_server_tls_policies(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_server_tls_policies_flattened_error():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_server_tls_policies(server_tls_policy.ListServerTlsPoliciesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_server_tls_policies_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_server_tls_policies), '__call__') as call:
        call.return_value = server_tls_policy.ListServerTlsPoliciesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(server_tls_policy.ListServerTlsPoliciesResponse())
        response = await client.list_server_tls_policies(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_server_tls_policies_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_server_tls_policies(server_tls_policy.ListServerTlsPoliciesRequest(), parent='parent_value')

def test_list_server_tls_policies_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_server_tls_policies), '__call__') as call:
        call.side_effect = (server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[server_tls_policy.ServerTlsPolicy(), server_tls_policy.ServerTlsPolicy(), server_tls_policy.ServerTlsPolicy()], next_page_token='abc'), server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[], next_page_token='def'), server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[server_tls_policy.ServerTlsPolicy()], next_page_token='ghi'), server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[server_tls_policy.ServerTlsPolicy(), server_tls_policy.ServerTlsPolicy()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_server_tls_policies(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, server_tls_policy.ServerTlsPolicy) for i in results))

def test_list_server_tls_policies_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_server_tls_policies), '__call__') as call:
        call.side_effect = (server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[server_tls_policy.ServerTlsPolicy(), server_tls_policy.ServerTlsPolicy(), server_tls_policy.ServerTlsPolicy()], next_page_token='abc'), server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[], next_page_token='def'), server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[server_tls_policy.ServerTlsPolicy()], next_page_token='ghi'), server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[server_tls_policy.ServerTlsPolicy(), server_tls_policy.ServerTlsPolicy()]), RuntimeError)
        pages = list(client.list_server_tls_policies(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_server_tls_policies_async_pager():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_server_tls_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[server_tls_policy.ServerTlsPolicy(), server_tls_policy.ServerTlsPolicy(), server_tls_policy.ServerTlsPolicy()], next_page_token='abc'), server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[], next_page_token='def'), server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[server_tls_policy.ServerTlsPolicy()], next_page_token='ghi'), server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[server_tls_policy.ServerTlsPolicy(), server_tls_policy.ServerTlsPolicy()]), RuntimeError)
        async_pager = await client.list_server_tls_policies(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, server_tls_policy.ServerTlsPolicy) for i in responses))

@pytest.mark.asyncio
async def test_list_server_tls_policies_async_pages():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_server_tls_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[server_tls_policy.ServerTlsPolicy(), server_tls_policy.ServerTlsPolicy(), server_tls_policy.ServerTlsPolicy()], next_page_token='abc'), server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[], next_page_token='def'), server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[server_tls_policy.ServerTlsPolicy()], next_page_token='ghi'), server_tls_policy.ListServerTlsPoliciesResponse(server_tls_policies=[server_tls_policy.ServerTlsPolicy(), server_tls_policy.ServerTlsPolicy()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_server_tls_policies(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [server_tls_policy.GetServerTlsPolicyRequest, dict])
def test_get_server_tls_policy(request_type, transport: str='grpc'):
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_server_tls_policy), '__call__') as call:
        call.return_value = server_tls_policy.ServerTlsPolicy(name='name_value', description='description_value', allow_open=True)
        response = client.get_server_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == server_tls_policy.GetServerTlsPolicyRequest()
    assert isinstance(response, server_tls_policy.ServerTlsPolicy)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.allow_open is True

def test_get_server_tls_policy_empty_call():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_server_tls_policy), '__call__') as call:
        client.get_server_tls_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == server_tls_policy.GetServerTlsPolicyRequest()

@pytest.mark.asyncio
async def test_get_server_tls_policy_async(transport: str='grpc_asyncio', request_type=server_tls_policy.GetServerTlsPolicyRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_server_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(server_tls_policy.ServerTlsPolicy(name='name_value', description='description_value', allow_open=True))
        response = await client.get_server_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == server_tls_policy.GetServerTlsPolicyRequest()
    assert isinstance(response, server_tls_policy.ServerTlsPolicy)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.allow_open is True

@pytest.mark.asyncio
async def test_get_server_tls_policy_async_from_dict():
    await test_get_server_tls_policy_async(request_type=dict)

def test_get_server_tls_policy_field_headers():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = server_tls_policy.GetServerTlsPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_server_tls_policy), '__call__') as call:
        call.return_value = server_tls_policy.ServerTlsPolicy()
        client.get_server_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_server_tls_policy_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = server_tls_policy.GetServerTlsPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_server_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(server_tls_policy.ServerTlsPolicy())
        await client.get_server_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_server_tls_policy_flattened():
    if False:
        while True:
            i = 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_server_tls_policy), '__call__') as call:
        call.return_value = server_tls_policy.ServerTlsPolicy()
        client.get_server_tls_policy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_server_tls_policy_flattened_error():
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_server_tls_policy(server_tls_policy.GetServerTlsPolicyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_server_tls_policy_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_server_tls_policy), '__call__') as call:
        call.return_value = server_tls_policy.ServerTlsPolicy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(server_tls_policy.ServerTlsPolicy())
        response = await client.get_server_tls_policy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_server_tls_policy_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_server_tls_policy(server_tls_policy.GetServerTlsPolicyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_server_tls_policy.CreateServerTlsPolicyRequest, dict])
def test_create_server_tls_policy(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_server_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_server_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_server_tls_policy.CreateServerTlsPolicyRequest()
    assert isinstance(response, future.Future)

def test_create_server_tls_policy_empty_call():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_server_tls_policy), '__call__') as call:
        client.create_server_tls_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_server_tls_policy.CreateServerTlsPolicyRequest()

@pytest.mark.asyncio
async def test_create_server_tls_policy_async(transport: str='grpc_asyncio', request_type=gcn_server_tls_policy.CreateServerTlsPolicyRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_server_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_server_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_server_tls_policy.CreateServerTlsPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_server_tls_policy_async_from_dict():
    await test_create_server_tls_policy_async(request_type=dict)

def test_create_server_tls_policy_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_server_tls_policy.CreateServerTlsPolicyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_server_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_server_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_server_tls_policy_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_server_tls_policy.CreateServerTlsPolicyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_server_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_server_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_server_tls_policy_flattened():
    if False:
        while True:
            i = 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_server_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_server_tls_policy(parent='parent_value', server_tls_policy=gcn_server_tls_policy.ServerTlsPolicy(name='name_value'), server_tls_policy_id='server_tls_policy_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].server_tls_policy
        mock_val = gcn_server_tls_policy.ServerTlsPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].server_tls_policy_id
        mock_val = 'server_tls_policy_id_value'
        assert arg == mock_val

def test_create_server_tls_policy_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_server_tls_policy(gcn_server_tls_policy.CreateServerTlsPolicyRequest(), parent='parent_value', server_tls_policy=gcn_server_tls_policy.ServerTlsPolicy(name='name_value'), server_tls_policy_id='server_tls_policy_id_value')

@pytest.mark.asyncio
async def test_create_server_tls_policy_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_server_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_server_tls_policy(parent='parent_value', server_tls_policy=gcn_server_tls_policy.ServerTlsPolicy(name='name_value'), server_tls_policy_id='server_tls_policy_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].server_tls_policy
        mock_val = gcn_server_tls_policy.ServerTlsPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].server_tls_policy_id
        mock_val = 'server_tls_policy_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_server_tls_policy_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_server_tls_policy(gcn_server_tls_policy.CreateServerTlsPolicyRequest(), parent='parent_value', server_tls_policy=gcn_server_tls_policy.ServerTlsPolicy(name='name_value'), server_tls_policy_id='server_tls_policy_id_value')

@pytest.mark.parametrize('request_type', [gcn_server_tls_policy.UpdateServerTlsPolicyRequest, dict])
def test_update_server_tls_policy(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_server_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_server_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_server_tls_policy.UpdateServerTlsPolicyRequest()
    assert isinstance(response, future.Future)

def test_update_server_tls_policy_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_server_tls_policy), '__call__') as call:
        client.update_server_tls_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_server_tls_policy.UpdateServerTlsPolicyRequest()

@pytest.mark.asyncio
async def test_update_server_tls_policy_async(transport: str='grpc_asyncio', request_type=gcn_server_tls_policy.UpdateServerTlsPolicyRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_server_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_server_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_server_tls_policy.UpdateServerTlsPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_server_tls_policy_async_from_dict():
    await test_update_server_tls_policy_async(request_type=dict)

def test_update_server_tls_policy_field_headers():
    if False:
        while True:
            i = 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_server_tls_policy.UpdateServerTlsPolicyRequest()
    request.server_tls_policy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_server_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_server_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'server_tls_policy.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_server_tls_policy_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_server_tls_policy.UpdateServerTlsPolicyRequest()
    request.server_tls_policy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_server_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_server_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'server_tls_policy.name=name_value') in kw['metadata']

def test_update_server_tls_policy_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_server_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_server_tls_policy(server_tls_policy=gcn_server_tls_policy.ServerTlsPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].server_tls_policy
        mock_val = gcn_server_tls_policy.ServerTlsPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_server_tls_policy_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_server_tls_policy(gcn_server_tls_policy.UpdateServerTlsPolicyRequest(), server_tls_policy=gcn_server_tls_policy.ServerTlsPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_server_tls_policy_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_server_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_server_tls_policy(server_tls_policy=gcn_server_tls_policy.ServerTlsPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].server_tls_policy
        mock_val = gcn_server_tls_policy.ServerTlsPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_server_tls_policy_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_server_tls_policy(gcn_server_tls_policy.UpdateServerTlsPolicyRequest(), server_tls_policy=gcn_server_tls_policy.ServerTlsPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [server_tls_policy.DeleteServerTlsPolicyRequest, dict])
def test_delete_server_tls_policy(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_server_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_server_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == server_tls_policy.DeleteServerTlsPolicyRequest()
    assert isinstance(response, future.Future)

def test_delete_server_tls_policy_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_server_tls_policy), '__call__') as call:
        client.delete_server_tls_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == server_tls_policy.DeleteServerTlsPolicyRequest()

@pytest.mark.asyncio
async def test_delete_server_tls_policy_async(transport: str='grpc_asyncio', request_type=server_tls_policy.DeleteServerTlsPolicyRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_server_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_server_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == server_tls_policy.DeleteServerTlsPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_server_tls_policy_async_from_dict():
    await test_delete_server_tls_policy_async(request_type=dict)

def test_delete_server_tls_policy_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = server_tls_policy.DeleteServerTlsPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_server_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_server_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_server_tls_policy_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = server_tls_policy.DeleteServerTlsPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_server_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_server_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_server_tls_policy_flattened():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_server_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_server_tls_policy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_server_tls_policy_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_server_tls_policy(server_tls_policy.DeleteServerTlsPolicyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_server_tls_policy_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_server_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_server_tls_policy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_server_tls_policy_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_server_tls_policy(server_tls_policy.DeleteServerTlsPolicyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [client_tls_policy.ListClientTlsPoliciesRequest, dict])
def test_list_client_tls_policies(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_client_tls_policies), '__call__') as call:
        call.return_value = client_tls_policy.ListClientTlsPoliciesResponse(next_page_token='next_page_token_value')
        response = client.list_client_tls_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_tls_policy.ListClientTlsPoliciesRequest()
    assert isinstance(response, pagers.ListClientTlsPoliciesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_client_tls_policies_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_client_tls_policies), '__call__') as call:
        client.list_client_tls_policies()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_tls_policy.ListClientTlsPoliciesRequest()

@pytest.mark.asyncio
async def test_list_client_tls_policies_async(transport: str='grpc_asyncio', request_type=client_tls_policy.ListClientTlsPoliciesRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_client_tls_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(client_tls_policy.ListClientTlsPoliciesResponse(next_page_token='next_page_token_value'))
        response = await client.list_client_tls_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_tls_policy.ListClientTlsPoliciesRequest()
    assert isinstance(response, pagers.ListClientTlsPoliciesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_client_tls_policies_async_from_dict():
    await test_list_client_tls_policies_async(request_type=dict)

def test_list_client_tls_policies_field_headers():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_tls_policy.ListClientTlsPoliciesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_client_tls_policies), '__call__') as call:
        call.return_value = client_tls_policy.ListClientTlsPoliciesResponse()
        client.list_client_tls_policies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_client_tls_policies_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_tls_policy.ListClientTlsPoliciesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_client_tls_policies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(client_tls_policy.ListClientTlsPoliciesResponse())
        await client.list_client_tls_policies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_client_tls_policies_flattened():
    if False:
        while True:
            i = 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_client_tls_policies), '__call__') as call:
        call.return_value = client_tls_policy.ListClientTlsPoliciesResponse()
        client.list_client_tls_policies(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_client_tls_policies_flattened_error():
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_client_tls_policies(client_tls_policy.ListClientTlsPoliciesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_client_tls_policies_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_client_tls_policies), '__call__') as call:
        call.return_value = client_tls_policy.ListClientTlsPoliciesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(client_tls_policy.ListClientTlsPoliciesResponse())
        response = await client.list_client_tls_policies(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_client_tls_policies_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_client_tls_policies(client_tls_policy.ListClientTlsPoliciesRequest(), parent='parent_value')

def test_list_client_tls_policies_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_client_tls_policies), '__call__') as call:
        call.side_effect = (client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[client_tls_policy.ClientTlsPolicy(), client_tls_policy.ClientTlsPolicy(), client_tls_policy.ClientTlsPolicy()], next_page_token='abc'), client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[], next_page_token='def'), client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[client_tls_policy.ClientTlsPolicy()], next_page_token='ghi'), client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[client_tls_policy.ClientTlsPolicy(), client_tls_policy.ClientTlsPolicy()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_client_tls_policies(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, client_tls_policy.ClientTlsPolicy) for i in results))

def test_list_client_tls_policies_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_client_tls_policies), '__call__') as call:
        call.side_effect = (client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[client_tls_policy.ClientTlsPolicy(), client_tls_policy.ClientTlsPolicy(), client_tls_policy.ClientTlsPolicy()], next_page_token='abc'), client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[], next_page_token='def'), client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[client_tls_policy.ClientTlsPolicy()], next_page_token='ghi'), client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[client_tls_policy.ClientTlsPolicy(), client_tls_policy.ClientTlsPolicy()]), RuntimeError)
        pages = list(client.list_client_tls_policies(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_client_tls_policies_async_pager():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_client_tls_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[client_tls_policy.ClientTlsPolicy(), client_tls_policy.ClientTlsPolicy(), client_tls_policy.ClientTlsPolicy()], next_page_token='abc'), client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[], next_page_token='def'), client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[client_tls_policy.ClientTlsPolicy()], next_page_token='ghi'), client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[client_tls_policy.ClientTlsPolicy(), client_tls_policy.ClientTlsPolicy()]), RuntimeError)
        async_pager = await client.list_client_tls_policies(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, client_tls_policy.ClientTlsPolicy) for i in responses))

@pytest.mark.asyncio
async def test_list_client_tls_policies_async_pages():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_client_tls_policies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[client_tls_policy.ClientTlsPolicy(), client_tls_policy.ClientTlsPolicy(), client_tls_policy.ClientTlsPolicy()], next_page_token='abc'), client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[], next_page_token='def'), client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[client_tls_policy.ClientTlsPolicy()], next_page_token='ghi'), client_tls_policy.ListClientTlsPoliciesResponse(client_tls_policies=[client_tls_policy.ClientTlsPolicy(), client_tls_policy.ClientTlsPolicy()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_client_tls_policies(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [client_tls_policy.GetClientTlsPolicyRequest, dict])
def test_get_client_tls_policy(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_client_tls_policy), '__call__') as call:
        call.return_value = client_tls_policy.ClientTlsPolicy(name='name_value', description='description_value', sni='sni_value')
        response = client.get_client_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_tls_policy.GetClientTlsPolicyRequest()
    assert isinstance(response, client_tls_policy.ClientTlsPolicy)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.sni == 'sni_value'

def test_get_client_tls_policy_empty_call():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_client_tls_policy), '__call__') as call:
        client.get_client_tls_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_tls_policy.GetClientTlsPolicyRequest()

@pytest.mark.asyncio
async def test_get_client_tls_policy_async(transport: str='grpc_asyncio', request_type=client_tls_policy.GetClientTlsPolicyRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_client_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(client_tls_policy.ClientTlsPolicy(name='name_value', description='description_value', sni='sni_value'))
        response = await client.get_client_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_tls_policy.GetClientTlsPolicyRequest()
    assert isinstance(response, client_tls_policy.ClientTlsPolicy)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.sni == 'sni_value'

@pytest.mark.asyncio
async def test_get_client_tls_policy_async_from_dict():
    await test_get_client_tls_policy_async(request_type=dict)

def test_get_client_tls_policy_field_headers():
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_tls_policy.GetClientTlsPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_client_tls_policy), '__call__') as call:
        call.return_value = client_tls_policy.ClientTlsPolicy()
        client.get_client_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_client_tls_policy_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_tls_policy.GetClientTlsPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_client_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(client_tls_policy.ClientTlsPolicy())
        await client.get_client_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_client_tls_policy_flattened():
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_client_tls_policy), '__call__') as call:
        call.return_value = client_tls_policy.ClientTlsPolicy()
        client.get_client_tls_policy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_client_tls_policy_flattened_error():
    if False:
        while True:
            i = 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_client_tls_policy(client_tls_policy.GetClientTlsPolicyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_client_tls_policy_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_client_tls_policy), '__call__') as call:
        call.return_value = client_tls_policy.ClientTlsPolicy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(client_tls_policy.ClientTlsPolicy())
        response = await client.get_client_tls_policy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_client_tls_policy_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_client_tls_policy(client_tls_policy.GetClientTlsPolicyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_client_tls_policy.CreateClientTlsPolicyRequest, dict])
def test_create_client_tls_policy(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_client_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_client_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_client_tls_policy.CreateClientTlsPolicyRequest()
    assert isinstance(response, future.Future)

def test_create_client_tls_policy_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_client_tls_policy), '__call__') as call:
        client.create_client_tls_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_client_tls_policy.CreateClientTlsPolicyRequest()

@pytest.mark.asyncio
async def test_create_client_tls_policy_async(transport: str='grpc_asyncio', request_type=gcn_client_tls_policy.CreateClientTlsPolicyRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_client_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_client_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_client_tls_policy.CreateClientTlsPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_client_tls_policy_async_from_dict():
    await test_create_client_tls_policy_async(request_type=dict)

def test_create_client_tls_policy_field_headers():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_client_tls_policy.CreateClientTlsPolicyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_client_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_client_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_client_tls_policy_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_client_tls_policy.CreateClientTlsPolicyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_client_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_client_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_client_tls_policy_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_client_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_client_tls_policy(parent='parent_value', client_tls_policy=gcn_client_tls_policy.ClientTlsPolicy(name='name_value'), client_tls_policy_id='client_tls_policy_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].client_tls_policy
        mock_val = gcn_client_tls_policy.ClientTlsPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].client_tls_policy_id
        mock_val = 'client_tls_policy_id_value'
        assert arg == mock_val

def test_create_client_tls_policy_flattened_error():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_client_tls_policy(gcn_client_tls_policy.CreateClientTlsPolicyRequest(), parent='parent_value', client_tls_policy=gcn_client_tls_policy.ClientTlsPolicy(name='name_value'), client_tls_policy_id='client_tls_policy_id_value')

@pytest.mark.asyncio
async def test_create_client_tls_policy_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_client_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_client_tls_policy(parent='parent_value', client_tls_policy=gcn_client_tls_policy.ClientTlsPolicy(name='name_value'), client_tls_policy_id='client_tls_policy_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].client_tls_policy
        mock_val = gcn_client_tls_policy.ClientTlsPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].client_tls_policy_id
        mock_val = 'client_tls_policy_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_client_tls_policy_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_client_tls_policy(gcn_client_tls_policy.CreateClientTlsPolicyRequest(), parent='parent_value', client_tls_policy=gcn_client_tls_policy.ClientTlsPolicy(name='name_value'), client_tls_policy_id='client_tls_policy_id_value')

@pytest.mark.parametrize('request_type', [gcn_client_tls_policy.UpdateClientTlsPolicyRequest, dict])
def test_update_client_tls_policy(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_client_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_client_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_client_tls_policy.UpdateClientTlsPolicyRequest()
    assert isinstance(response, future.Future)

def test_update_client_tls_policy_empty_call():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_client_tls_policy), '__call__') as call:
        client.update_client_tls_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_client_tls_policy.UpdateClientTlsPolicyRequest()

@pytest.mark.asyncio
async def test_update_client_tls_policy_async(transport: str='grpc_asyncio', request_type=gcn_client_tls_policy.UpdateClientTlsPolicyRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_client_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_client_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_client_tls_policy.UpdateClientTlsPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_client_tls_policy_async_from_dict():
    await test_update_client_tls_policy_async(request_type=dict)

def test_update_client_tls_policy_field_headers():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_client_tls_policy.UpdateClientTlsPolicyRequest()
    request.client_tls_policy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_client_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_client_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'client_tls_policy.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_client_tls_policy_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_client_tls_policy.UpdateClientTlsPolicyRequest()
    request.client_tls_policy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_client_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_client_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'client_tls_policy.name=name_value') in kw['metadata']

def test_update_client_tls_policy_flattened():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_client_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_client_tls_policy(client_tls_policy=gcn_client_tls_policy.ClientTlsPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].client_tls_policy
        mock_val = gcn_client_tls_policy.ClientTlsPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_client_tls_policy_flattened_error():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_client_tls_policy(gcn_client_tls_policy.UpdateClientTlsPolicyRequest(), client_tls_policy=gcn_client_tls_policy.ClientTlsPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_client_tls_policy_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_client_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_client_tls_policy(client_tls_policy=gcn_client_tls_policy.ClientTlsPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].client_tls_policy
        mock_val = gcn_client_tls_policy.ClientTlsPolicy(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_client_tls_policy_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_client_tls_policy(gcn_client_tls_policy.UpdateClientTlsPolicyRequest(), client_tls_policy=gcn_client_tls_policy.ClientTlsPolicy(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [client_tls_policy.DeleteClientTlsPolicyRequest, dict])
def test_delete_client_tls_policy(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_client_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_client_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_tls_policy.DeleteClientTlsPolicyRequest()
    assert isinstance(response, future.Future)

def test_delete_client_tls_policy_empty_call():
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_client_tls_policy), '__call__') as call:
        client.delete_client_tls_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_tls_policy.DeleteClientTlsPolicyRequest()

@pytest.mark.asyncio
async def test_delete_client_tls_policy_async(transport: str='grpc_asyncio', request_type=client_tls_policy.DeleteClientTlsPolicyRequest):
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_client_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_client_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == client_tls_policy.DeleteClientTlsPolicyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_client_tls_policy_async_from_dict():
    await test_delete_client_tls_policy_async(request_type=dict)

def test_delete_client_tls_policy_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_tls_policy.DeleteClientTlsPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_client_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_client_tls_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_client_tls_policy_field_headers_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = client_tls_policy.DeleteClientTlsPolicyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_client_tls_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_client_tls_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_client_tls_policy_flattened():
    if False:
        while True:
            i = 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_client_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_client_tls_policy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_client_tls_policy_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_client_tls_policy(client_tls_policy.DeleteClientTlsPolicyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_client_tls_policy_flattened_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_client_tls_policy), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_client_tls_policy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_client_tls_policy_flattened_error_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_client_tls_policy(client_tls_policy.DeleteClientTlsPolicyRequest(), name='name_value')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkSecurityGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.NetworkSecurityGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NetworkSecurityClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.NetworkSecurityGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = NetworkSecurityClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = NetworkSecurityClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.NetworkSecurityGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NetworkSecurityClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NetworkSecurityGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = NetworkSecurityClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.NetworkSecurityGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.NetworkSecurityGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.NetworkSecurityGrpcTransport, transports.NetworkSecurityGrpcAsyncIOTransport])
def test_transport_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc'])
def test_transport_kind(transport_name):
    if False:
        for i in range(10):
            print('nop')
    transport = NetworkSecurityClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.NetworkSecurityGrpcTransport)

def test_network_security_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.NetworkSecurityTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_network_security_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.network_security_v1beta1.services.network_security.transports.NetworkSecurityTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.NetworkSecurityTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_authorization_policies', 'get_authorization_policy', 'create_authorization_policy', 'update_authorization_policy', 'delete_authorization_policy', 'list_server_tls_policies', 'get_server_tls_policy', 'create_server_tls_policy', 'update_server_tls_policy', 'delete_server_tls_policy', 'list_client_tls_policies', 'get_client_tls_policy', 'create_client_tls_policy', 'update_client_tls_policy', 'delete_client_tls_policy', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_network_security_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.network_security_v1beta1.services.network_security.transports.NetworkSecurityTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.NetworkSecurityTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_network_security_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.network_security_v1beta1.services.network_security.transports.NetworkSecurityTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.NetworkSecurityTransport()
        adc.assert_called_once()

def test_network_security_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        NetworkSecurityClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.NetworkSecurityGrpcTransport, transports.NetworkSecurityGrpcAsyncIOTransport])
def test_network_security_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.NetworkSecurityGrpcTransport, transports.NetworkSecurityGrpcAsyncIOTransport])
def test_network_security_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.NetworkSecurityGrpcTransport, grpc_helpers), (transports.NetworkSecurityGrpcAsyncIOTransport, grpc_helpers_async)])
def test_network_security_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('networksecurity.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='networksecurity.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.NetworkSecurityGrpcTransport, transports.NetworkSecurityGrpcAsyncIOTransport])
def test_network_security_grpc_transport_client_cert_source_for_mtls(transport_class):
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

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_network_security_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='networksecurity.googleapis.com'), transport=transport_name)
    assert client.transport._host == 'networksecurity.googleapis.com:443'

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_network_security_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='networksecurity.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == 'networksecurity.googleapis.com:8000'

def test_network_security_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.NetworkSecurityGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_network_security_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.NetworkSecurityGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.NetworkSecurityGrpcTransport, transports.NetworkSecurityGrpcAsyncIOTransport])
def test_network_security_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.NetworkSecurityGrpcTransport, transports.NetworkSecurityGrpcAsyncIOTransport])
def test_network_security_transport_channel_mtls_with_adc(transport_class):
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

def test_network_security_grpc_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_network_security_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_authorization_policy_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    authorization_policy = 'whelk'
    expected = 'projects/{project}/locations/{location}/authorizationPolicies/{authorization_policy}'.format(project=project, location=location, authorization_policy=authorization_policy)
    actual = NetworkSecurityClient.authorization_policy_path(project, location, authorization_policy)
    assert expected == actual

def test_parse_authorization_policy_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus', 'location': 'oyster', 'authorization_policy': 'nudibranch'}
    path = NetworkSecurityClient.authorization_policy_path(**expected)
    actual = NetworkSecurityClient.parse_authorization_policy_path(path)
    assert expected == actual

def test_client_tls_policy_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    location = 'mussel'
    client_tls_policy = 'winkle'
    expected = 'projects/{project}/locations/{location}/clientTlsPolicies/{client_tls_policy}'.format(project=project, location=location, client_tls_policy=client_tls_policy)
    actual = NetworkSecurityClient.client_tls_policy_path(project, location, client_tls_policy)
    assert expected == actual

def test_parse_client_tls_policy_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nautilus', 'location': 'scallop', 'client_tls_policy': 'abalone'}
    path = NetworkSecurityClient.client_tls_policy_path(**expected)
    actual = NetworkSecurityClient.parse_client_tls_policy_path(path)
    assert expected == actual

def test_server_tls_policy_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    server_tls_policy = 'whelk'
    expected = 'projects/{project}/locations/{location}/serverTlsPolicies/{server_tls_policy}'.format(project=project, location=location, server_tls_policy=server_tls_policy)
    actual = NetworkSecurityClient.server_tls_policy_path(project, location, server_tls_policy)
    assert expected == actual

def test_parse_server_tls_policy_path():
    if False:
        return 10
    expected = {'project': 'octopus', 'location': 'oyster', 'server_tls_policy': 'nudibranch'}
    path = NetworkSecurityClient.server_tls_policy_path(**expected)
    actual = NetworkSecurityClient.parse_server_tls_policy_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = NetworkSecurityClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'mussel'}
    path = NetworkSecurityClient.common_billing_account_path(**expected)
    actual = NetworkSecurityClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = NetworkSecurityClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'nautilus'}
    path = NetworkSecurityClient.common_folder_path(**expected)
    actual = NetworkSecurityClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = NetworkSecurityClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'abalone'}
    path = NetworkSecurityClient.common_organization_path(**expected)
    actual = NetworkSecurityClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = NetworkSecurityClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'clam'}
    path = NetworkSecurityClient.common_project_path(**expected)
    actual = NetworkSecurityClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = NetworkSecurityClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = NetworkSecurityClient.common_location_path(**expected)
    actual = NetworkSecurityClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.NetworkSecurityTransport, '_prep_wrapped_messages') as prep:
        client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.NetworkSecurityTransport, '_prep_wrapped_messages') as prep:
        transport_class = NetworkSecurityClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_delete_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        return 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = NetworkSecurityAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        return 10
    transports = {'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['grpc']
    for transport in transports:
        client = NetworkSecurityClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(NetworkSecurityClient, transports.NetworkSecurityGrpcTransport), (NetworkSecurityAsyncClient, transports.NetworkSecurityGrpcAsyncIOTransport)])
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