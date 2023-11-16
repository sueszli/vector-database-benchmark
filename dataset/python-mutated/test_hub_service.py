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
from google.cloud.networkconnectivity_v1.services.hub_service import HubServiceAsyncClient, HubServiceClient, pagers, transports
from google.cloud.networkconnectivity_v1.types import common
from google.cloud.networkconnectivity_v1.types import hub
from google.cloud.networkconnectivity_v1.types import hub as gcn_hub

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
        while True:
            i = 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert HubServiceClient._get_default_mtls_endpoint(None) is None
    assert HubServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert HubServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert HubServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert HubServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert HubServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(HubServiceClient, 'grpc'), (HubServiceAsyncClient, 'grpc_asyncio')])
def test_hub_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == 'networkconnectivity.googleapis.com:443'

@pytest.mark.parametrize('transport_class,transport_name', [(transports.HubServiceGrpcTransport, 'grpc'), (transports.HubServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_hub_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(HubServiceClient, 'grpc'), (HubServiceAsyncClient, 'grpc_asyncio')])
def test_hub_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == 'networkconnectivity.googleapis.com:443'

def test_hub_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = HubServiceClient.get_transport_class()
    available_transports = [transports.HubServiceGrpcTransport]
    assert transport in available_transports
    transport = HubServiceClient.get_transport_class('grpc')
    assert transport == transports.HubServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(HubServiceClient, transports.HubServiceGrpcTransport, 'grpc'), (HubServiceAsyncClient, transports.HubServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
@mock.patch.object(HubServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(HubServiceClient))
@mock.patch.object(HubServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(HubServiceAsyncClient))
def test_hub_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(HubServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(HubServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(HubServiceClient, transports.HubServiceGrpcTransport, 'grpc', 'true'), (HubServiceAsyncClient, transports.HubServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (HubServiceClient, transports.HubServiceGrpcTransport, 'grpc', 'false'), (HubServiceAsyncClient, transports.HubServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false')])
@mock.patch.object(HubServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(HubServiceClient))
@mock.patch.object(HubServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(HubServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_hub_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [HubServiceClient, HubServiceAsyncClient])
@mock.patch.object(HubServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(HubServiceClient))
@mock.patch.object(HubServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(HubServiceAsyncClient))
def test_hub_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(HubServiceClient, transports.HubServiceGrpcTransport, 'grpc'), (HubServiceAsyncClient, transports.HubServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_hub_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(HubServiceClient, transports.HubServiceGrpcTransport, 'grpc', grpc_helpers), (HubServiceAsyncClient, transports.HubServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_hub_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_hub_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.networkconnectivity_v1.services.hub_service.transports.HubServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = HubServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(HubServiceClient, transports.HubServiceGrpcTransport, 'grpc', grpc_helpers), (HubServiceAsyncClient, transports.HubServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_hub_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('networkconnectivity.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='networkconnectivity.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [hub.ListHubsRequest, dict])
def test_list_hubs(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_hubs), '__call__') as call:
        call.return_value = hub.ListHubsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_hubs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListHubsRequest()
    assert isinstance(response, pagers.ListHubsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_hubs_empty_call():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_hubs), '__call__') as call:
        client.list_hubs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListHubsRequest()

@pytest.mark.asyncio
async def test_list_hubs_async(transport: str='grpc_asyncio', request_type=hub.ListHubsRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_hubs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListHubsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_hubs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListHubsRequest()
    assert isinstance(response, pagers.ListHubsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_hubs_async_from_dict():
    await test_list_hubs_async(request_type=dict)

def test_list_hubs_field_headers():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.ListHubsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_hubs), '__call__') as call:
        call.return_value = hub.ListHubsResponse()
        client.list_hubs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_hubs_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.ListHubsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_hubs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListHubsResponse())
        await client.list_hubs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_hubs_flattened():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_hubs), '__call__') as call:
        call.return_value = hub.ListHubsResponse()
        client.list_hubs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_hubs_flattened_error():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_hubs(hub.ListHubsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_hubs_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_hubs), '__call__') as call:
        call.return_value = hub.ListHubsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListHubsResponse())
        response = await client.list_hubs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_hubs_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_hubs(hub.ListHubsRequest(), parent='parent_value')

def test_list_hubs_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_hubs), '__call__') as call:
        call.side_effect = (hub.ListHubsResponse(hubs=[hub.Hub(), hub.Hub(), hub.Hub()], next_page_token='abc'), hub.ListHubsResponse(hubs=[], next_page_token='def'), hub.ListHubsResponse(hubs=[hub.Hub()], next_page_token='ghi'), hub.ListHubsResponse(hubs=[hub.Hub(), hub.Hub()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_hubs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, hub.Hub) for i in results))

def test_list_hubs_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_hubs), '__call__') as call:
        call.side_effect = (hub.ListHubsResponse(hubs=[hub.Hub(), hub.Hub(), hub.Hub()], next_page_token='abc'), hub.ListHubsResponse(hubs=[], next_page_token='def'), hub.ListHubsResponse(hubs=[hub.Hub()], next_page_token='ghi'), hub.ListHubsResponse(hubs=[hub.Hub(), hub.Hub()]), RuntimeError)
        pages = list(client.list_hubs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_hubs_async_pager():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_hubs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (hub.ListHubsResponse(hubs=[hub.Hub(), hub.Hub(), hub.Hub()], next_page_token='abc'), hub.ListHubsResponse(hubs=[], next_page_token='def'), hub.ListHubsResponse(hubs=[hub.Hub()], next_page_token='ghi'), hub.ListHubsResponse(hubs=[hub.Hub(), hub.Hub()]), RuntimeError)
        async_pager = await client.list_hubs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, hub.Hub) for i in responses))

@pytest.mark.asyncio
async def test_list_hubs_async_pages():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_hubs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (hub.ListHubsResponse(hubs=[hub.Hub(), hub.Hub(), hub.Hub()], next_page_token='abc'), hub.ListHubsResponse(hubs=[], next_page_token='def'), hub.ListHubsResponse(hubs=[hub.Hub()], next_page_token='ghi'), hub.ListHubsResponse(hubs=[hub.Hub(), hub.Hub()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_hubs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [hub.GetHubRequest, dict])
def test_get_hub(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_hub), '__call__') as call:
        call.return_value = hub.Hub(name='name_value', description='description_value', unique_id='unique_id_value', state=hub.State.CREATING, route_tables=['route_tables_value'])
        response = client.get_hub(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetHubRequest()
    assert isinstance(response, hub.Hub)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.unique_id == 'unique_id_value'
    assert response.state == hub.State.CREATING
    assert response.route_tables == ['route_tables_value']

def test_get_hub_empty_call():
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_hub), '__call__') as call:
        client.get_hub()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetHubRequest()

@pytest.mark.asyncio
async def test_get_hub_async(transport: str='grpc_asyncio', request_type=hub.GetHubRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_hub), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.Hub(name='name_value', description='description_value', unique_id='unique_id_value', state=hub.State.CREATING, route_tables=['route_tables_value']))
        response = await client.get_hub(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetHubRequest()
    assert isinstance(response, hub.Hub)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.unique_id == 'unique_id_value'
    assert response.state == hub.State.CREATING
    assert response.route_tables == ['route_tables_value']

@pytest.mark.asyncio
async def test_get_hub_async_from_dict():
    await test_get_hub_async(request_type=dict)

def test_get_hub_field_headers():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.GetHubRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_hub), '__call__') as call:
        call.return_value = hub.Hub()
        client.get_hub(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_hub_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.GetHubRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_hub), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.Hub())
        await client.get_hub(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_hub_flattened():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_hub), '__call__') as call:
        call.return_value = hub.Hub()
        client.get_hub(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_hub_flattened_error():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_hub(hub.GetHubRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_hub_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_hub), '__call__') as call:
        call.return_value = hub.Hub()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.Hub())
        response = await client.get_hub(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_hub_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_hub(hub.GetHubRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcn_hub.CreateHubRequest, dict])
def test_create_hub(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_hub), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_hub(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_hub.CreateHubRequest()
    assert isinstance(response, future.Future)

def test_create_hub_empty_call():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_hub), '__call__') as call:
        client.create_hub()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_hub.CreateHubRequest()

@pytest.mark.asyncio
async def test_create_hub_async(transport: str='grpc_asyncio', request_type=gcn_hub.CreateHubRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_hub), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_hub(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_hub.CreateHubRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_hub_async_from_dict():
    await test_create_hub_async(request_type=dict)

def test_create_hub_field_headers():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_hub.CreateHubRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_hub), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_hub(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_hub_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_hub.CreateHubRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_hub), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_hub(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_hub_flattened():
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_hub), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_hub(parent='parent_value', hub=gcn_hub.Hub(name='name_value'), hub_id='hub_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].hub
        mock_val = gcn_hub.Hub(name='name_value')
        assert arg == mock_val
        arg = args[0].hub_id
        mock_val = 'hub_id_value'
        assert arg == mock_val

def test_create_hub_flattened_error():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_hub(gcn_hub.CreateHubRequest(), parent='parent_value', hub=gcn_hub.Hub(name='name_value'), hub_id='hub_id_value')

@pytest.mark.asyncio
async def test_create_hub_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_hub), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_hub(parent='parent_value', hub=gcn_hub.Hub(name='name_value'), hub_id='hub_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].hub
        mock_val = gcn_hub.Hub(name='name_value')
        assert arg == mock_val
        arg = args[0].hub_id
        mock_val = 'hub_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_hub_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_hub(gcn_hub.CreateHubRequest(), parent='parent_value', hub=gcn_hub.Hub(name='name_value'), hub_id='hub_id_value')

@pytest.mark.parametrize('request_type', [gcn_hub.UpdateHubRequest, dict])
def test_update_hub(request_type, transport: str='grpc'):
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_hub), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_hub(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_hub.UpdateHubRequest()
    assert isinstance(response, future.Future)

def test_update_hub_empty_call():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_hub), '__call__') as call:
        client.update_hub()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_hub.UpdateHubRequest()

@pytest.mark.asyncio
async def test_update_hub_async(transport: str='grpc_asyncio', request_type=gcn_hub.UpdateHubRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_hub), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_hub(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcn_hub.UpdateHubRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_hub_async_from_dict():
    await test_update_hub_async(request_type=dict)

def test_update_hub_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_hub.UpdateHubRequest()
    request.hub.name = 'name_value'
    with mock.patch.object(type(client.transport.update_hub), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_hub(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'hub.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_hub_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcn_hub.UpdateHubRequest()
    request.hub.name = 'name_value'
    with mock.patch.object(type(client.transport.update_hub), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_hub(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'hub.name=name_value') in kw['metadata']

def test_update_hub_flattened():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_hub), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_hub(hub=gcn_hub.Hub(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].hub
        mock_val = gcn_hub.Hub(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_hub_flattened_error():
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_hub(gcn_hub.UpdateHubRequest(), hub=gcn_hub.Hub(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_hub_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_hub), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_hub(hub=gcn_hub.Hub(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].hub
        mock_val = gcn_hub.Hub(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_hub_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_hub(gcn_hub.UpdateHubRequest(), hub=gcn_hub.Hub(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [hub.DeleteHubRequest, dict])
def test_delete_hub(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_hub), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_hub(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.DeleteHubRequest()
    assert isinstance(response, future.Future)

def test_delete_hub_empty_call():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_hub), '__call__') as call:
        client.delete_hub()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.DeleteHubRequest()

@pytest.mark.asyncio
async def test_delete_hub_async(transport: str='grpc_asyncio', request_type=hub.DeleteHubRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_hub), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_hub(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.DeleteHubRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_hub_async_from_dict():
    await test_delete_hub_async(request_type=dict)

def test_delete_hub_field_headers():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.DeleteHubRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_hub), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_hub(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_hub_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.DeleteHubRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_hub), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_hub(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_hub_flattened():
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_hub), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_hub(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_hub_flattened_error():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_hub(hub.DeleteHubRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_hub_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_hub), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_hub(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_hub_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_hub(hub.DeleteHubRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [hub.ListHubSpokesRequest, dict])
def test_list_hub_spokes(request_type, transport: str='grpc'):
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_hub_spokes), '__call__') as call:
        call.return_value = hub.ListHubSpokesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_hub_spokes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListHubSpokesRequest()
    assert isinstance(response, pagers.ListHubSpokesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_hub_spokes_empty_call():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_hub_spokes), '__call__') as call:
        client.list_hub_spokes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListHubSpokesRequest()

@pytest.mark.asyncio
async def test_list_hub_spokes_async(transport: str='grpc_asyncio', request_type=hub.ListHubSpokesRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_hub_spokes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListHubSpokesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_hub_spokes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListHubSpokesRequest()
    assert isinstance(response, pagers.ListHubSpokesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_hub_spokes_async_from_dict():
    await test_list_hub_spokes_async(request_type=dict)

def test_list_hub_spokes_field_headers():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.ListHubSpokesRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.list_hub_spokes), '__call__') as call:
        call.return_value = hub.ListHubSpokesResponse()
        client.list_hub_spokes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_hub_spokes_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.ListHubSpokesRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.list_hub_spokes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListHubSpokesResponse())
        await client.list_hub_spokes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_list_hub_spokes_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_hub_spokes), '__call__') as call:
        call.return_value = hub.ListHubSpokesResponse()
        client.list_hub_spokes(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_list_hub_spokes_flattened_error():
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_hub_spokes(hub.ListHubSpokesRequest(), name='name_value')

@pytest.mark.asyncio
async def test_list_hub_spokes_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_hub_spokes), '__call__') as call:
        call.return_value = hub.ListHubSpokesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListHubSpokesResponse())
        response = await client.list_hub_spokes(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_hub_spokes_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_hub_spokes(hub.ListHubSpokesRequest(), name='name_value')

def test_list_hub_spokes_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_hub_spokes), '__call__') as call:
        call.side_effect = (hub.ListHubSpokesResponse(spokes=[hub.Spoke(), hub.Spoke(), hub.Spoke()], next_page_token='abc'), hub.ListHubSpokesResponse(spokes=[], next_page_token='def'), hub.ListHubSpokesResponse(spokes=[hub.Spoke()], next_page_token='ghi'), hub.ListHubSpokesResponse(spokes=[hub.Spoke(), hub.Spoke()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', ''),)),)
        pager = client.list_hub_spokes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, hub.Spoke) for i in results))

def test_list_hub_spokes_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_hub_spokes), '__call__') as call:
        call.side_effect = (hub.ListHubSpokesResponse(spokes=[hub.Spoke(), hub.Spoke(), hub.Spoke()], next_page_token='abc'), hub.ListHubSpokesResponse(spokes=[], next_page_token='def'), hub.ListHubSpokesResponse(spokes=[hub.Spoke()], next_page_token='ghi'), hub.ListHubSpokesResponse(spokes=[hub.Spoke(), hub.Spoke()]), RuntimeError)
        pages = list(client.list_hub_spokes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_hub_spokes_async_pager():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_hub_spokes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (hub.ListHubSpokesResponse(spokes=[hub.Spoke(), hub.Spoke(), hub.Spoke()], next_page_token='abc'), hub.ListHubSpokesResponse(spokes=[], next_page_token='def'), hub.ListHubSpokesResponse(spokes=[hub.Spoke()], next_page_token='ghi'), hub.ListHubSpokesResponse(spokes=[hub.Spoke(), hub.Spoke()]), RuntimeError)
        async_pager = await client.list_hub_spokes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, hub.Spoke) for i in responses))

@pytest.mark.asyncio
async def test_list_hub_spokes_async_pages():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_hub_spokes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (hub.ListHubSpokesResponse(spokes=[hub.Spoke(), hub.Spoke(), hub.Spoke()], next_page_token='abc'), hub.ListHubSpokesResponse(spokes=[], next_page_token='def'), hub.ListHubSpokesResponse(spokes=[hub.Spoke()], next_page_token='ghi'), hub.ListHubSpokesResponse(spokes=[hub.Spoke(), hub.Spoke()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_hub_spokes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [hub.ListSpokesRequest, dict])
def test_list_spokes(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_spokes), '__call__') as call:
        call.return_value = hub.ListSpokesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_spokes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListSpokesRequest()
    assert isinstance(response, pagers.ListSpokesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_spokes_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_spokes), '__call__') as call:
        client.list_spokes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListSpokesRequest()

@pytest.mark.asyncio
async def test_list_spokes_async(transport: str='grpc_asyncio', request_type=hub.ListSpokesRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_spokes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListSpokesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_spokes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListSpokesRequest()
    assert isinstance(response, pagers.ListSpokesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_spokes_async_from_dict():
    await test_list_spokes_async(request_type=dict)

def test_list_spokes_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.ListSpokesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_spokes), '__call__') as call:
        call.return_value = hub.ListSpokesResponse()
        client.list_spokes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_spokes_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.ListSpokesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_spokes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListSpokesResponse())
        await client.list_spokes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_spokes_flattened():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_spokes), '__call__') as call:
        call.return_value = hub.ListSpokesResponse()
        client.list_spokes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_spokes_flattened_error():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_spokes(hub.ListSpokesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_spokes_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_spokes), '__call__') as call:
        call.return_value = hub.ListSpokesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListSpokesResponse())
        response = await client.list_spokes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_spokes_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_spokes(hub.ListSpokesRequest(), parent='parent_value')

def test_list_spokes_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_spokes), '__call__') as call:
        call.side_effect = (hub.ListSpokesResponse(spokes=[hub.Spoke(), hub.Spoke(), hub.Spoke()], next_page_token='abc'), hub.ListSpokesResponse(spokes=[], next_page_token='def'), hub.ListSpokesResponse(spokes=[hub.Spoke()], next_page_token='ghi'), hub.ListSpokesResponse(spokes=[hub.Spoke(), hub.Spoke()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_spokes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, hub.Spoke) for i in results))

def test_list_spokes_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_spokes), '__call__') as call:
        call.side_effect = (hub.ListSpokesResponse(spokes=[hub.Spoke(), hub.Spoke(), hub.Spoke()], next_page_token='abc'), hub.ListSpokesResponse(spokes=[], next_page_token='def'), hub.ListSpokesResponse(spokes=[hub.Spoke()], next_page_token='ghi'), hub.ListSpokesResponse(spokes=[hub.Spoke(), hub.Spoke()]), RuntimeError)
        pages = list(client.list_spokes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_spokes_async_pager():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_spokes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (hub.ListSpokesResponse(spokes=[hub.Spoke(), hub.Spoke(), hub.Spoke()], next_page_token='abc'), hub.ListSpokesResponse(spokes=[], next_page_token='def'), hub.ListSpokesResponse(spokes=[hub.Spoke()], next_page_token='ghi'), hub.ListSpokesResponse(spokes=[hub.Spoke(), hub.Spoke()]), RuntimeError)
        async_pager = await client.list_spokes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, hub.Spoke) for i in responses))

@pytest.mark.asyncio
async def test_list_spokes_async_pages():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_spokes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (hub.ListSpokesResponse(spokes=[hub.Spoke(), hub.Spoke(), hub.Spoke()], next_page_token='abc'), hub.ListSpokesResponse(spokes=[], next_page_token='def'), hub.ListSpokesResponse(spokes=[hub.Spoke()], next_page_token='ghi'), hub.ListSpokesResponse(spokes=[hub.Spoke(), hub.Spoke()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_spokes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [hub.GetSpokeRequest, dict])
def test_get_spoke(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_spoke), '__call__') as call:
        call.return_value = hub.Spoke(name='name_value', description='description_value', hub='hub_value', group='group_value', unique_id='unique_id_value', state=hub.State.CREATING, spoke_type=hub.SpokeType.VPN_TUNNEL)
        response = client.get_spoke(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetSpokeRequest()
    assert isinstance(response, hub.Spoke)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.hub == 'hub_value'
    assert response.group == 'group_value'
    assert response.unique_id == 'unique_id_value'
    assert response.state == hub.State.CREATING
    assert response.spoke_type == hub.SpokeType.VPN_TUNNEL

def test_get_spoke_empty_call():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_spoke), '__call__') as call:
        client.get_spoke()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetSpokeRequest()

@pytest.mark.asyncio
async def test_get_spoke_async(transport: str='grpc_asyncio', request_type=hub.GetSpokeRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_spoke), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.Spoke(name='name_value', description='description_value', hub='hub_value', group='group_value', unique_id='unique_id_value', state=hub.State.CREATING, spoke_type=hub.SpokeType.VPN_TUNNEL))
        response = await client.get_spoke(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetSpokeRequest()
    assert isinstance(response, hub.Spoke)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.hub == 'hub_value'
    assert response.group == 'group_value'
    assert response.unique_id == 'unique_id_value'
    assert response.state == hub.State.CREATING
    assert response.spoke_type == hub.SpokeType.VPN_TUNNEL

@pytest.mark.asyncio
async def test_get_spoke_async_from_dict():
    await test_get_spoke_async(request_type=dict)

def test_get_spoke_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.GetSpokeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_spoke), '__call__') as call:
        call.return_value = hub.Spoke()
        client.get_spoke(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_spoke_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.GetSpokeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_spoke), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.Spoke())
        await client.get_spoke(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_spoke_flattened():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_spoke), '__call__') as call:
        call.return_value = hub.Spoke()
        client.get_spoke(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_spoke_flattened_error():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_spoke(hub.GetSpokeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_spoke_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_spoke), '__call__') as call:
        call.return_value = hub.Spoke()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.Spoke())
        response = await client.get_spoke(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_spoke_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_spoke(hub.GetSpokeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [hub.CreateSpokeRequest, dict])
def test_create_spoke(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_spoke(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.CreateSpokeRequest()
    assert isinstance(response, future.Future)

def test_create_spoke_empty_call():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_spoke), '__call__') as call:
        client.create_spoke()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.CreateSpokeRequest()

@pytest.mark.asyncio
async def test_create_spoke_async(transport: str='grpc_asyncio', request_type=hub.CreateSpokeRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_spoke), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_spoke(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.CreateSpokeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_spoke_async_from_dict():
    await test_create_spoke_async(request_type=dict)

def test_create_spoke_field_headers():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.CreateSpokeRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_spoke(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_spoke_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.CreateSpokeRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_spoke), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_spoke(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_spoke_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_spoke(parent='parent_value', spoke=hub.Spoke(name='name_value'), spoke_id='spoke_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].spoke
        mock_val = hub.Spoke(name='name_value')
        assert arg == mock_val
        arg = args[0].spoke_id
        mock_val = 'spoke_id_value'
        assert arg == mock_val

def test_create_spoke_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_spoke(hub.CreateSpokeRequest(), parent='parent_value', spoke=hub.Spoke(name='name_value'), spoke_id='spoke_id_value')

@pytest.mark.asyncio
async def test_create_spoke_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_spoke(parent='parent_value', spoke=hub.Spoke(name='name_value'), spoke_id='spoke_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].spoke
        mock_val = hub.Spoke(name='name_value')
        assert arg == mock_val
        arg = args[0].spoke_id
        mock_val = 'spoke_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_spoke_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_spoke(hub.CreateSpokeRequest(), parent='parent_value', spoke=hub.Spoke(name='name_value'), spoke_id='spoke_id_value')

@pytest.mark.parametrize('request_type', [hub.UpdateSpokeRequest, dict])
def test_update_spoke(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_spoke(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.UpdateSpokeRequest()
    assert isinstance(response, future.Future)

def test_update_spoke_empty_call():
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_spoke), '__call__') as call:
        client.update_spoke()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.UpdateSpokeRequest()

@pytest.mark.asyncio
async def test_update_spoke_async(transport: str='grpc_asyncio', request_type=hub.UpdateSpokeRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_spoke), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_spoke(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.UpdateSpokeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_spoke_async_from_dict():
    await test_update_spoke_async(request_type=dict)

def test_update_spoke_field_headers():
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.UpdateSpokeRequest()
    request.spoke.name = 'name_value'
    with mock.patch.object(type(client.transport.update_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_spoke(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'spoke.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_spoke_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.UpdateSpokeRequest()
    request.spoke.name = 'name_value'
    with mock.patch.object(type(client.transport.update_spoke), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_spoke(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'spoke.name=name_value') in kw['metadata']

def test_update_spoke_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_spoke(spoke=hub.Spoke(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].spoke
        mock_val = hub.Spoke(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_spoke_flattened_error():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_spoke(hub.UpdateSpokeRequest(), spoke=hub.Spoke(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_spoke_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_spoke(spoke=hub.Spoke(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].spoke
        mock_val = hub.Spoke(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_spoke_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_spoke(hub.UpdateSpokeRequest(), spoke=hub.Spoke(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [hub.RejectHubSpokeRequest, dict])
def test_reject_hub_spoke(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reject_hub_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.reject_hub_spoke(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.RejectHubSpokeRequest()
    assert isinstance(response, future.Future)

def test_reject_hub_spoke_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.reject_hub_spoke), '__call__') as call:
        client.reject_hub_spoke()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.RejectHubSpokeRequest()

@pytest.mark.asyncio
async def test_reject_hub_spoke_async(transport: str='grpc_asyncio', request_type=hub.RejectHubSpokeRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reject_hub_spoke), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reject_hub_spoke(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.RejectHubSpokeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_reject_hub_spoke_async_from_dict():
    await test_reject_hub_spoke_async(request_type=dict)

def test_reject_hub_spoke_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.RejectHubSpokeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.reject_hub_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reject_hub_spoke(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_reject_hub_spoke_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.RejectHubSpokeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.reject_hub_spoke), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.reject_hub_spoke(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_reject_hub_spoke_flattened():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reject_hub_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reject_hub_spoke(name='name_value', spoke_uri='spoke_uri_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].spoke_uri
        mock_val = 'spoke_uri_value'
        assert arg == mock_val

def test_reject_hub_spoke_flattened_error():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.reject_hub_spoke(hub.RejectHubSpokeRequest(), name='name_value', spoke_uri='spoke_uri_value')

@pytest.mark.asyncio
async def test_reject_hub_spoke_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reject_hub_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reject_hub_spoke(name='name_value', spoke_uri='spoke_uri_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].spoke_uri
        mock_val = 'spoke_uri_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_reject_hub_spoke_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.reject_hub_spoke(hub.RejectHubSpokeRequest(), name='name_value', spoke_uri='spoke_uri_value')

@pytest.mark.parametrize('request_type', [hub.AcceptHubSpokeRequest, dict])
def test_accept_hub_spoke(request_type, transport: str='grpc'):
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.accept_hub_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.accept_hub_spoke(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.AcceptHubSpokeRequest()
    assert isinstance(response, future.Future)

def test_accept_hub_spoke_empty_call():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.accept_hub_spoke), '__call__') as call:
        client.accept_hub_spoke()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.AcceptHubSpokeRequest()

@pytest.mark.asyncio
async def test_accept_hub_spoke_async(transport: str='grpc_asyncio', request_type=hub.AcceptHubSpokeRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.accept_hub_spoke), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.accept_hub_spoke(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.AcceptHubSpokeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_accept_hub_spoke_async_from_dict():
    await test_accept_hub_spoke_async(request_type=dict)

def test_accept_hub_spoke_field_headers():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.AcceptHubSpokeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.accept_hub_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.accept_hub_spoke(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_accept_hub_spoke_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.AcceptHubSpokeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.accept_hub_spoke), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.accept_hub_spoke(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_accept_hub_spoke_flattened():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.accept_hub_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.accept_hub_spoke(name='name_value', spoke_uri='spoke_uri_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].spoke_uri
        mock_val = 'spoke_uri_value'
        assert arg == mock_val

def test_accept_hub_spoke_flattened_error():
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.accept_hub_spoke(hub.AcceptHubSpokeRequest(), name='name_value', spoke_uri='spoke_uri_value')

@pytest.mark.asyncio
async def test_accept_hub_spoke_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.accept_hub_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.accept_hub_spoke(name='name_value', spoke_uri='spoke_uri_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].spoke_uri
        mock_val = 'spoke_uri_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_accept_hub_spoke_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.accept_hub_spoke(hub.AcceptHubSpokeRequest(), name='name_value', spoke_uri='spoke_uri_value')

@pytest.mark.parametrize('request_type', [hub.DeleteSpokeRequest, dict])
def test_delete_spoke(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_spoke(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.DeleteSpokeRequest()
    assert isinstance(response, future.Future)

def test_delete_spoke_empty_call():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_spoke), '__call__') as call:
        client.delete_spoke()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.DeleteSpokeRequest()

@pytest.mark.asyncio
async def test_delete_spoke_async(transport: str='grpc_asyncio', request_type=hub.DeleteSpokeRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_spoke), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_spoke(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.DeleteSpokeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_spoke_async_from_dict():
    await test_delete_spoke_async(request_type=dict)

def test_delete_spoke_field_headers():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.DeleteSpokeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_spoke(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_spoke_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.DeleteSpokeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_spoke), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_spoke(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_spoke_flattened():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_spoke(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_spoke_flattened_error():
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_spoke(hub.DeleteSpokeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_spoke_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_spoke), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_spoke(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_spoke_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_spoke(hub.DeleteSpokeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [hub.GetRouteTableRequest, dict])
def test_get_route_table(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_route_table), '__call__') as call:
        call.return_value = hub.RouteTable(name='name_value', description='description_value', uid='uid_value', state=hub.State.CREATING)
        response = client.get_route_table(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetRouteTableRequest()
    assert isinstance(response, hub.RouteTable)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.uid == 'uid_value'
    assert response.state == hub.State.CREATING

def test_get_route_table_empty_call():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_route_table), '__call__') as call:
        client.get_route_table()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetRouteTableRequest()

@pytest.mark.asyncio
async def test_get_route_table_async(transport: str='grpc_asyncio', request_type=hub.GetRouteTableRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_route_table), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.RouteTable(name='name_value', description='description_value', uid='uid_value', state=hub.State.CREATING))
        response = await client.get_route_table(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetRouteTableRequest()
    assert isinstance(response, hub.RouteTable)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.uid == 'uid_value'
    assert response.state == hub.State.CREATING

@pytest.mark.asyncio
async def test_get_route_table_async_from_dict():
    await test_get_route_table_async(request_type=dict)

def test_get_route_table_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.GetRouteTableRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_route_table), '__call__') as call:
        call.return_value = hub.RouteTable()
        client.get_route_table(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_route_table_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.GetRouteTableRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_route_table), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.RouteTable())
        await client.get_route_table(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_route_table_flattened():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_route_table), '__call__') as call:
        call.return_value = hub.RouteTable()
        client.get_route_table(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_route_table_flattened_error():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_route_table(hub.GetRouteTableRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_route_table_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_route_table), '__call__') as call:
        call.return_value = hub.RouteTable()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.RouteTable())
        response = await client.get_route_table(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_route_table_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_route_table(hub.GetRouteTableRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [hub.GetRouteRequest, dict])
def test_get_route(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_route), '__call__') as call:
        call.return_value = hub.Route(name='name_value', ip_cidr_range='ip_cidr_range_value', type_=hub.RouteType.VPC_PRIMARY_SUBNET, description='description_value', uid='uid_value', state=hub.State.CREATING, spoke='spoke_value', location='location_value')
        response = client.get_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetRouteRequest()
    assert isinstance(response, hub.Route)
    assert response.name == 'name_value'
    assert response.ip_cidr_range == 'ip_cidr_range_value'
    assert response.type_ == hub.RouteType.VPC_PRIMARY_SUBNET
    assert response.description == 'description_value'
    assert response.uid == 'uid_value'
    assert response.state == hub.State.CREATING
    assert response.spoke == 'spoke_value'
    assert response.location == 'location_value'

def test_get_route_empty_call():
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_route), '__call__') as call:
        client.get_route()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetRouteRequest()

@pytest.mark.asyncio
async def test_get_route_async(transport: str='grpc_asyncio', request_type=hub.GetRouteRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.Route(name='name_value', ip_cidr_range='ip_cidr_range_value', type_=hub.RouteType.VPC_PRIMARY_SUBNET, description='description_value', uid='uid_value', state=hub.State.CREATING, spoke='spoke_value', location='location_value'))
        response = await client.get_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetRouteRequest()
    assert isinstance(response, hub.Route)
    assert response.name == 'name_value'
    assert response.ip_cidr_range == 'ip_cidr_range_value'
    assert response.type_ == hub.RouteType.VPC_PRIMARY_SUBNET
    assert response.description == 'description_value'
    assert response.uid == 'uid_value'
    assert response.state == hub.State.CREATING
    assert response.spoke == 'spoke_value'
    assert response.location == 'location_value'

@pytest.mark.asyncio
async def test_get_route_async_from_dict():
    await test_get_route_async(request_type=dict)

def test_get_route_field_headers():
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.GetRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_route), '__call__') as call:
        call.return_value = hub.Route()
        client.get_route(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_route_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.GetRouteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_route), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.Route())
        await client.get_route(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_route_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_route), '__call__') as call:
        call.return_value = hub.Route()
        client.get_route(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_route_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_route(hub.GetRouteRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_route_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_route), '__call__') as call:
        call.return_value = hub.Route()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.Route())
        response = await client.get_route(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_route_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_route(hub.GetRouteRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [hub.ListRoutesRequest, dict])
def test_list_routes(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_routes), '__call__') as call:
        call.return_value = hub.ListRoutesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_routes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListRoutesRequest()
    assert isinstance(response, pagers.ListRoutesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_routes_empty_call():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_routes), '__call__') as call:
        client.list_routes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListRoutesRequest()

@pytest.mark.asyncio
async def test_list_routes_async(transport: str='grpc_asyncio', request_type=hub.ListRoutesRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_routes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListRoutesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_routes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListRoutesRequest()
    assert isinstance(response, pagers.ListRoutesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_routes_async_from_dict():
    await test_list_routes_async(request_type=dict)

def test_list_routes_field_headers():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.ListRoutesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_routes), '__call__') as call:
        call.return_value = hub.ListRoutesResponse()
        client.list_routes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_routes_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.ListRoutesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_routes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListRoutesResponse())
        await client.list_routes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_routes_flattened():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_routes), '__call__') as call:
        call.return_value = hub.ListRoutesResponse()
        client.list_routes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_routes_flattened_error():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_routes(hub.ListRoutesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_routes_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_routes), '__call__') as call:
        call.return_value = hub.ListRoutesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListRoutesResponse())
        response = await client.list_routes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_routes_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_routes(hub.ListRoutesRequest(), parent='parent_value')

def test_list_routes_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_routes), '__call__') as call:
        call.side_effect = (hub.ListRoutesResponse(routes=[hub.Route(), hub.Route(), hub.Route()], next_page_token='abc'), hub.ListRoutesResponse(routes=[], next_page_token='def'), hub.ListRoutesResponse(routes=[hub.Route()], next_page_token='ghi'), hub.ListRoutesResponse(routes=[hub.Route(), hub.Route()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_routes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, hub.Route) for i in results))

def test_list_routes_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_routes), '__call__') as call:
        call.side_effect = (hub.ListRoutesResponse(routes=[hub.Route(), hub.Route(), hub.Route()], next_page_token='abc'), hub.ListRoutesResponse(routes=[], next_page_token='def'), hub.ListRoutesResponse(routes=[hub.Route()], next_page_token='ghi'), hub.ListRoutesResponse(routes=[hub.Route(), hub.Route()]), RuntimeError)
        pages = list(client.list_routes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_routes_async_pager():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_routes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (hub.ListRoutesResponse(routes=[hub.Route(), hub.Route(), hub.Route()], next_page_token='abc'), hub.ListRoutesResponse(routes=[], next_page_token='def'), hub.ListRoutesResponse(routes=[hub.Route()], next_page_token='ghi'), hub.ListRoutesResponse(routes=[hub.Route(), hub.Route()]), RuntimeError)
        async_pager = await client.list_routes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, hub.Route) for i in responses))

@pytest.mark.asyncio
async def test_list_routes_async_pages():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_routes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (hub.ListRoutesResponse(routes=[hub.Route(), hub.Route(), hub.Route()], next_page_token='abc'), hub.ListRoutesResponse(routes=[], next_page_token='def'), hub.ListRoutesResponse(routes=[hub.Route()], next_page_token='ghi'), hub.ListRoutesResponse(routes=[hub.Route(), hub.Route()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_routes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [hub.ListRouteTablesRequest, dict])
def test_list_route_tables(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_route_tables), '__call__') as call:
        call.return_value = hub.ListRouteTablesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_route_tables(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListRouteTablesRequest()
    assert isinstance(response, pagers.ListRouteTablesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_route_tables_empty_call():
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_route_tables), '__call__') as call:
        client.list_route_tables()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListRouteTablesRequest()

@pytest.mark.asyncio
async def test_list_route_tables_async(transport: str='grpc_asyncio', request_type=hub.ListRouteTablesRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_route_tables), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListRouteTablesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_route_tables(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListRouteTablesRequest()
    assert isinstance(response, pagers.ListRouteTablesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_route_tables_async_from_dict():
    await test_list_route_tables_async(request_type=dict)

def test_list_route_tables_field_headers():
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.ListRouteTablesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_route_tables), '__call__') as call:
        call.return_value = hub.ListRouteTablesResponse()
        client.list_route_tables(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_route_tables_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.ListRouteTablesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_route_tables), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListRouteTablesResponse())
        await client.list_route_tables(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_route_tables_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_route_tables), '__call__') as call:
        call.return_value = hub.ListRouteTablesResponse()
        client.list_route_tables(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_route_tables_flattened_error():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_route_tables(hub.ListRouteTablesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_route_tables_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_route_tables), '__call__') as call:
        call.return_value = hub.ListRouteTablesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListRouteTablesResponse())
        response = await client.list_route_tables(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_route_tables_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_route_tables(hub.ListRouteTablesRequest(), parent='parent_value')

def test_list_route_tables_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_route_tables), '__call__') as call:
        call.side_effect = (hub.ListRouteTablesResponse(route_tables=[hub.RouteTable(), hub.RouteTable(), hub.RouteTable()], next_page_token='abc'), hub.ListRouteTablesResponse(route_tables=[], next_page_token='def'), hub.ListRouteTablesResponse(route_tables=[hub.RouteTable()], next_page_token='ghi'), hub.ListRouteTablesResponse(route_tables=[hub.RouteTable(), hub.RouteTable()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_route_tables(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, hub.RouteTable) for i in results))

def test_list_route_tables_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_route_tables), '__call__') as call:
        call.side_effect = (hub.ListRouteTablesResponse(route_tables=[hub.RouteTable(), hub.RouteTable(), hub.RouteTable()], next_page_token='abc'), hub.ListRouteTablesResponse(route_tables=[], next_page_token='def'), hub.ListRouteTablesResponse(route_tables=[hub.RouteTable()], next_page_token='ghi'), hub.ListRouteTablesResponse(route_tables=[hub.RouteTable(), hub.RouteTable()]), RuntimeError)
        pages = list(client.list_route_tables(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_route_tables_async_pager():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_route_tables), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (hub.ListRouteTablesResponse(route_tables=[hub.RouteTable(), hub.RouteTable(), hub.RouteTable()], next_page_token='abc'), hub.ListRouteTablesResponse(route_tables=[], next_page_token='def'), hub.ListRouteTablesResponse(route_tables=[hub.RouteTable()], next_page_token='ghi'), hub.ListRouteTablesResponse(route_tables=[hub.RouteTable(), hub.RouteTable()]), RuntimeError)
        async_pager = await client.list_route_tables(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, hub.RouteTable) for i in responses))

@pytest.mark.asyncio
async def test_list_route_tables_async_pages():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_route_tables), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (hub.ListRouteTablesResponse(route_tables=[hub.RouteTable(), hub.RouteTable(), hub.RouteTable()], next_page_token='abc'), hub.ListRouteTablesResponse(route_tables=[], next_page_token='def'), hub.ListRouteTablesResponse(route_tables=[hub.RouteTable()], next_page_token='ghi'), hub.ListRouteTablesResponse(route_tables=[hub.RouteTable(), hub.RouteTable()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_route_tables(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [hub.GetGroupRequest, dict])
def test_get_group(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_group), '__call__') as call:
        call.return_value = hub.Group(name='name_value', description='description_value', uid='uid_value', state=hub.State.CREATING)
        response = client.get_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetGroupRequest()
    assert isinstance(response, hub.Group)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.uid == 'uid_value'
    assert response.state == hub.State.CREATING

def test_get_group_empty_call():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_group), '__call__') as call:
        client.get_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetGroupRequest()

@pytest.mark.asyncio
async def test_get_group_async(transport: str='grpc_asyncio', request_type=hub.GetGroupRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.Group(name='name_value', description='description_value', uid='uid_value', state=hub.State.CREATING))
        response = await client.get_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.GetGroupRequest()
    assert isinstance(response, hub.Group)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.uid == 'uid_value'
    assert response.state == hub.State.CREATING

@pytest.mark.asyncio
async def test_get_group_async_from_dict():
    await test_get_group_async(request_type=dict)

def test_get_group_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.GetGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_group), '__call__') as call:
        call.return_value = hub.Group()
        client.get_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_group_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.GetGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.Group())
        await client.get_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_group_flattened():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_group), '__call__') as call:
        call.return_value = hub.Group()
        client.get_group(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_group_flattened_error():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_group(hub.GetGroupRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_group_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_group), '__call__') as call:
        call.return_value = hub.Group()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.Group())
        response = await client.get_group(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_group_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_group(hub.GetGroupRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [hub.ListGroupsRequest, dict])
def test_list_groups(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.return_value = hub.ListGroupsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_groups(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListGroupsRequest()
    assert isinstance(response, pagers.ListGroupsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_groups_empty_call():
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        client.list_groups()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListGroupsRequest()

@pytest.mark.asyncio
async def test_list_groups_async(transport: str='grpc_asyncio', request_type=hub.ListGroupsRequest):
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListGroupsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_groups(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == hub.ListGroupsRequest()
    assert isinstance(response, pagers.ListGroupsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_groups_async_from_dict():
    await test_list_groups_async(request_type=dict)

def test_list_groups_field_headers():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.ListGroupsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.return_value = hub.ListGroupsResponse()
        client.list_groups(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_groups_field_headers_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = hub.ListGroupsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListGroupsResponse())
        await client.list_groups(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_groups_flattened():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.return_value = hub.ListGroupsResponse()
        client.list_groups(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_groups_flattened_error():
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_groups(hub.ListGroupsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_groups_flattened_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.return_value = hub.ListGroupsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(hub.ListGroupsResponse())
        response = await client.list_groups(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_groups_flattened_error_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_groups(hub.ListGroupsRequest(), parent='parent_value')

def test_list_groups_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.side_effect = (hub.ListGroupsResponse(groups=[hub.Group(), hub.Group(), hub.Group()], next_page_token='abc'), hub.ListGroupsResponse(groups=[], next_page_token='def'), hub.ListGroupsResponse(groups=[hub.Group()], next_page_token='ghi'), hub.ListGroupsResponse(groups=[hub.Group(), hub.Group()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_groups(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, hub.Group) for i in results))

def test_list_groups_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_groups), '__call__') as call:
        call.side_effect = (hub.ListGroupsResponse(groups=[hub.Group(), hub.Group(), hub.Group()], next_page_token='abc'), hub.ListGroupsResponse(groups=[], next_page_token='def'), hub.ListGroupsResponse(groups=[hub.Group()], next_page_token='ghi'), hub.ListGroupsResponse(groups=[hub.Group(), hub.Group()]), RuntimeError)
        pages = list(client.list_groups(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_groups_async_pager():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_groups), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (hub.ListGroupsResponse(groups=[hub.Group(), hub.Group(), hub.Group()], next_page_token='abc'), hub.ListGroupsResponse(groups=[], next_page_token='def'), hub.ListGroupsResponse(groups=[hub.Group()], next_page_token='ghi'), hub.ListGroupsResponse(groups=[hub.Group(), hub.Group()]), RuntimeError)
        async_pager = await client.list_groups(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, hub.Group) for i in responses))

@pytest.mark.asyncio
async def test_list_groups_async_pages():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_groups), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (hub.ListGroupsResponse(groups=[hub.Group(), hub.Group(), hub.Group()], next_page_token='abc'), hub.ListGroupsResponse(groups=[], next_page_token='def'), hub.ListGroupsResponse(groups=[hub.Group()], next_page_token='ghi'), hub.ListGroupsResponse(groups=[hub.Group(), hub.Group()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_groups(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.HubServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.HubServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = HubServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.HubServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = HubServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = HubServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.HubServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = HubServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.HubServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = HubServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.HubServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.HubServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.HubServiceGrpcTransport, transports.HubServiceGrpcAsyncIOTransport])
def test_transport_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc'])
def test_transport_kind(transport_name):
    if False:
        print('Hello World!')
    transport = HubServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.HubServiceGrpcTransport)

def test_hub_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.HubServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_hub_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.networkconnectivity_v1.services.hub_service.transports.HubServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.HubServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_hubs', 'get_hub', 'create_hub', 'update_hub', 'delete_hub', 'list_hub_spokes', 'list_spokes', 'get_spoke', 'create_spoke', 'update_spoke', 'reject_hub_spoke', 'accept_hub_spoke', 'delete_spoke', 'get_route_table', 'get_route', 'list_routes', 'list_route_tables', 'get_group', 'list_groups', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_hub_service_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.networkconnectivity_v1.services.hub_service.transports.HubServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.HubServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_hub_service_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.networkconnectivity_v1.services.hub_service.transports.HubServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.HubServiceTransport()
        adc.assert_called_once()

def test_hub_service_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        HubServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.HubServiceGrpcTransport, transports.HubServiceGrpcAsyncIOTransport])
def test_hub_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.HubServiceGrpcTransport, transports.HubServiceGrpcAsyncIOTransport])
def test_hub_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.HubServiceGrpcTransport, grpc_helpers), (transports.HubServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_hub_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('networkconnectivity.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='networkconnectivity.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.HubServiceGrpcTransport, transports.HubServiceGrpcAsyncIOTransport])
def test_hub_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_hub_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='networkconnectivity.googleapis.com'), transport=transport_name)
    assert client.transport._host == 'networkconnectivity.googleapis.com:443'

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_hub_service_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='networkconnectivity.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == 'networkconnectivity.googleapis.com:8000'

def test_hub_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.HubServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_hub_service_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.HubServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.HubServiceGrpcTransport, transports.HubServiceGrpcAsyncIOTransport])
def test_hub_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.HubServiceGrpcTransport, transports.HubServiceGrpcAsyncIOTransport])
def test_hub_service_transport_channel_mtls_with_adc(transport_class):
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

def test_hub_service_grpc_lro_client():
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_hub_service_grpc_lro_async_client():
    if False:
        print('Hello World!')
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_group_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    hub = 'clam'
    group = 'whelk'
    expected = 'projects/{project}/locations/global/hubs/{hub}/groups/{group}'.format(project=project, hub=hub, group=group)
    actual = HubServiceClient.group_path(project, hub, group)
    assert expected == actual

def test_parse_group_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus', 'hub': 'oyster', 'group': 'nudibranch'}
    path = HubServiceClient.group_path(**expected)
    actual = HubServiceClient.parse_group_path(path)
    assert expected == actual

def test_hub_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    hub = 'mussel'
    expected = 'projects/{project}/locations/global/hubs/{hub}'.format(project=project, hub=hub)
    actual = HubServiceClient.hub_path(project, hub)
    assert expected == actual

def test_parse_hub_path():
    if False:
        return 10
    expected = {'project': 'winkle', 'hub': 'nautilus'}
    path = HubServiceClient.hub_path(**expected)
    actual = HubServiceClient.parse_hub_path(path)
    assert expected == actual

def test_hub_route_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    hub = 'abalone'
    route_table = 'squid'
    route = 'clam'
    expected = 'projects/{project}/locations/global/hubs/{hub}/routeTables/{route_table}/routes/{route}'.format(project=project, hub=hub, route_table=route_table, route=route)
    actual = HubServiceClient.hub_route_path(project, hub, route_table, route)
    assert expected == actual

def test_parse_hub_route_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'whelk', 'hub': 'octopus', 'route_table': 'oyster', 'route': 'nudibranch'}
    path = HubServiceClient.hub_route_path(**expected)
    actual = HubServiceClient.parse_hub_route_path(path)
    assert expected == actual

def test_instance_path():
    if False:
        return 10
    project = 'cuttlefish'
    zone = 'mussel'
    instance = 'winkle'
    expected = 'projects/{project}/zones/{zone}/instances/{instance}'.format(project=project, zone=zone, instance=instance)
    actual = HubServiceClient.instance_path(project, zone, instance)
    assert expected == actual

def test_parse_instance_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nautilus', 'zone': 'scallop', 'instance': 'abalone'}
    path = HubServiceClient.instance_path(**expected)
    actual = HubServiceClient.parse_instance_path(path)
    assert expected == actual

def test_interconnect_attachment_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    region = 'clam'
    resource_id = 'whelk'
    expected = 'projects/{project}/regions/{region}/interconnectAttachments/{resource_id}'.format(project=project, region=region, resource_id=resource_id)
    actual = HubServiceClient.interconnect_attachment_path(project, region, resource_id)
    assert expected == actual

def test_parse_interconnect_attachment_path():
    if False:
        return 10
    expected = {'project': 'octopus', 'region': 'oyster', 'resource_id': 'nudibranch'}
    path = HubServiceClient.interconnect_attachment_path(**expected)
    actual = HubServiceClient.parse_interconnect_attachment_path(path)
    assert expected == actual

def test_network_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    resource_id = 'mussel'
    expected = 'projects/{project}/global/networks/{resource_id}'.format(project=project, resource_id=resource_id)
    actual = HubServiceClient.network_path(project, resource_id)
    assert expected == actual

def test_parse_network_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'winkle', 'resource_id': 'nautilus'}
    path = HubServiceClient.network_path(**expected)
    actual = HubServiceClient.parse_network_path(path)
    assert expected == actual

def test_route_table_path():
    if False:
        print('Hello World!')
    project = 'scallop'
    hub = 'abalone'
    route_table = 'squid'
    expected = 'projects/{project}/locations/global/hubs/{hub}/routeTables/{route_table}'.format(project=project, hub=hub, route_table=route_table)
    actual = HubServiceClient.route_table_path(project, hub, route_table)
    assert expected == actual

def test_parse_route_table_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'clam', 'hub': 'whelk', 'route_table': 'octopus'}
    path = HubServiceClient.route_table_path(**expected)
    actual = HubServiceClient.parse_route_table_path(path)
    assert expected == actual

def test_spoke_path():
    if False:
        return 10
    project = 'oyster'
    location = 'nudibranch'
    spoke = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/spokes/{spoke}'.format(project=project, location=location, spoke=spoke)
    actual = HubServiceClient.spoke_path(project, location, spoke)
    assert expected == actual

def test_parse_spoke_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'mussel', 'location': 'winkle', 'spoke': 'nautilus'}
    path = HubServiceClient.spoke_path(**expected)
    actual = HubServiceClient.parse_spoke_path(path)
    assert expected == actual

def test_vpn_tunnel_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    region = 'abalone'
    resource_id = 'squid'
    expected = 'projects/{project}/regions/{region}/vpnTunnels/{resource_id}'.format(project=project, region=region, resource_id=resource_id)
    actual = HubServiceClient.vpn_tunnel_path(project, region, resource_id)
    assert expected == actual

def test_parse_vpn_tunnel_path():
    if False:
        print('Hello World!')
    expected = {'project': 'clam', 'region': 'whelk', 'resource_id': 'octopus'}
    path = HubServiceClient.vpn_tunnel_path(**expected)
    actual = HubServiceClient.parse_vpn_tunnel_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = HubServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'nudibranch'}
    path = HubServiceClient.common_billing_account_path(**expected)
    actual = HubServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = HubServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'mussel'}
    path = HubServiceClient.common_folder_path(**expected)
    actual = HubServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = HubServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'nautilus'}
    path = HubServiceClient.common_organization_path(**expected)
    actual = HubServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = HubServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'abalone'}
    path = HubServiceClient.common_project_path(**expected)
    actual = HubServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = HubServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = HubServiceClient.common_location_path(**expected)
    actual = HubServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.HubServiceTransport, '_prep_wrapped_messages') as prep:
        client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.HubServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = HubServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_delete_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = HubServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        while True:
            i = 10
    transports = ['grpc']
    for transport in transports:
        client = HubServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(HubServiceClient, transports.HubServiceGrpcTransport), (HubServiceAsyncClient, transports.HubServiceGrpcAsyncIOTransport)])
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