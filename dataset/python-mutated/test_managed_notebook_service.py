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
from google.cloud.notebooks_v1.services.managed_notebook_service import ManagedNotebookServiceAsyncClient, ManagedNotebookServiceClient, pagers, transports
from google.cloud.notebooks_v1.types import diagnostic_config as gcn_diagnostic_config
from google.cloud.notebooks_v1.types import environment, event, managed_service
from google.cloud.notebooks_v1.types import runtime
from google.cloud.notebooks_v1.types import runtime as gcn_runtime
from google.cloud.notebooks_v1.types import service

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        for i in range(10):
            print('nop')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert ManagedNotebookServiceClient._get_default_mtls_endpoint(None) is None
    assert ManagedNotebookServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ManagedNotebookServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ManagedNotebookServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ManagedNotebookServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ManagedNotebookServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ManagedNotebookServiceClient, 'grpc'), (ManagedNotebookServiceAsyncClient, 'grpc_asyncio')])
def test_managed_notebook_service_client_from_service_account_info(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == 'notebooks.googleapis.com:443'

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ManagedNotebookServiceGrpcTransport, 'grpc'), (transports.ManagedNotebookServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_managed_notebook_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ManagedNotebookServiceClient, 'grpc'), (ManagedNotebookServiceAsyncClient, 'grpc_asyncio')])
def test_managed_notebook_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == 'notebooks.googleapis.com:443'

def test_managed_notebook_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = ManagedNotebookServiceClient.get_transport_class()
    available_transports = [transports.ManagedNotebookServiceGrpcTransport]
    assert transport in available_transports
    transport = ManagedNotebookServiceClient.get_transport_class('grpc')
    assert transport == transports.ManagedNotebookServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ManagedNotebookServiceClient, transports.ManagedNotebookServiceGrpcTransport, 'grpc'), (ManagedNotebookServiceAsyncClient, transports.ManagedNotebookServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
@mock.patch.object(ManagedNotebookServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ManagedNotebookServiceClient))
@mock.patch.object(ManagedNotebookServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ManagedNotebookServiceAsyncClient))
def test_managed_notebook_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(ManagedNotebookServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ManagedNotebookServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ManagedNotebookServiceClient, transports.ManagedNotebookServiceGrpcTransport, 'grpc', 'true'), (ManagedNotebookServiceAsyncClient, transports.ManagedNotebookServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ManagedNotebookServiceClient, transports.ManagedNotebookServiceGrpcTransport, 'grpc', 'false'), (ManagedNotebookServiceAsyncClient, transports.ManagedNotebookServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false')])
@mock.patch.object(ManagedNotebookServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ManagedNotebookServiceClient))
@mock.patch.object(ManagedNotebookServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ManagedNotebookServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_managed_notebook_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [ManagedNotebookServiceClient, ManagedNotebookServiceAsyncClient])
@mock.patch.object(ManagedNotebookServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ManagedNotebookServiceClient))
@mock.patch.object(ManagedNotebookServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ManagedNotebookServiceAsyncClient))
def test_managed_notebook_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ManagedNotebookServiceClient, transports.ManagedNotebookServiceGrpcTransport, 'grpc'), (ManagedNotebookServiceAsyncClient, transports.ManagedNotebookServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_managed_notebook_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ManagedNotebookServiceClient, transports.ManagedNotebookServiceGrpcTransport, 'grpc', grpc_helpers), (ManagedNotebookServiceAsyncClient, transports.ManagedNotebookServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_managed_notebook_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_managed_notebook_service_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.notebooks_v1.services.managed_notebook_service.transports.ManagedNotebookServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ManagedNotebookServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ManagedNotebookServiceClient, transports.ManagedNotebookServiceGrpcTransport, 'grpc', grpc_helpers), (ManagedNotebookServiceAsyncClient, transports.ManagedNotebookServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_managed_notebook_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
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
        create_channel.assert_called_with('notebooks.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='notebooks.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [managed_service.ListRuntimesRequest, dict])
def test_list_runtimes(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        call.return_value = managed_service.ListRuntimesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_runtimes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.ListRuntimesRequest()
    assert isinstance(response, pagers.ListRuntimesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_runtimes_empty_call():
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        client.list_runtimes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.ListRuntimesRequest()

@pytest.mark.asyncio
async def test_list_runtimes_async(transport: str='grpc_asyncio', request_type=managed_service.ListRuntimesRequest):
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(managed_service.ListRuntimesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_runtimes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.ListRuntimesRequest()
    assert isinstance(response, pagers.ListRuntimesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_runtimes_async_from_dict():
    await test_list_runtimes_async(request_type=dict)

def test_list_runtimes_field_headers():
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.ListRuntimesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        call.return_value = managed_service.ListRuntimesResponse()
        client.list_runtimes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_runtimes_field_headers_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.ListRuntimesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(managed_service.ListRuntimesResponse())
        await client.list_runtimes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_runtimes_flattened():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        call.return_value = managed_service.ListRuntimesResponse()
        client.list_runtimes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_runtimes_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_runtimes(managed_service.ListRuntimesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_runtimes_flattened_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        call.return_value = managed_service.ListRuntimesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(managed_service.ListRuntimesResponse())
        response = await client.list_runtimes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_runtimes_flattened_error_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_runtimes(managed_service.ListRuntimesRequest(), parent='parent_value')

def test_list_runtimes_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        call.side_effect = (managed_service.ListRuntimesResponse(runtimes=[runtime.Runtime(), runtime.Runtime(), runtime.Runtime()], next_page_token='abc'), managed_service.ListRuntimesResponse(runtimes=[], next_page_token='def'), managed_service.ListRuntimesResponse(runtimes=[runtime.Runtime()], next_page_token='ghi'), managed_service.ListRuntimesResponse(runtimes=[runtime.Runtime(), runtime.Runtime()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_runtimes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, runtime.Runtime) for i in results))

def test_list_runtimes_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        call.side_effect = (managed_service.ListRuntimesResponse(runtimes=[runtime.Runtime(), runtime.Runtime(), runtime.Runtime()], next_page_token='abc'), managed_service.ListRuntimesResponse(runtimes=[], next_page_token='def'), managed_service.ListRuntimesResponse(runtimes=[runtime.Runtime()], next_page_token='ghi'), managed_service.ListRuntimesResponse(runtimes=[runtime.Runtime(), runtime.Runtime()]), RuntimeError)
        pages = list(client.list_runtimes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_runtimes_async_pager():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_runtimes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (managed_service.ListRuntimesResponse(runtimes=[runtime.Runtime(), runtime.Runtime(), runtime.Runtime()], next_page_token='abc'), managed_service.ListRuntimesResponse(runtimes=[], next_page_token='def'), managed_service.ListRuntimesResponse(runtimes=[runtime.Runtime()], next_page_token='ghi'), managed_service.ListRuntimesResponse(runtimes=[runtime.Runtime(), runtime.Runtime()]), RuntimeError)
        async_pager = await client.list_runtimes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, runtime.Runtime) for i in responses))

@pytest.mark.asyncio
async def test_list_runtimes_async_pages():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_runtimes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (managed_service.ListRuntimesResponse(runtimes=[runtime.Runtime(), runtime.Runtime(), runtime.Runtime()], next_page_token='abc'), managed_service.ListRuntimesResponse(runtimes=[], next_page_token='def'), managed_service.ListRuntimesResponse(runtimes=[runtime.Runtime()], next_page_token='ghi'), managed_service.ListRuntimesResponse(runtimes=[runtime.Runtime(), runtime.Runtime()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_runtimes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [managed_service.GetRuntimeRequest, dict])
def test_get_runtime(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_runtime), '__call__') as call:
        call.return_value = runtime.Runtime(name='name_value', state=runtime.Runtime.State.STARTING, health_state=runtime.Runtime.HealthState.HEALTHY)
        response = client.get_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.GetRuntimeRequest()
    assert isinstance(response, runtime.Runtime)
    assert response.name == 'name_value'
    assert response.state == runtime.Runtime.State.STARTING
    assert response.health_state == runtime.Runtime.HealthState.HEALTHY

def test_get_runtime_empty_call():
    if False:
        while True:
            i = 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_runtime), '__call__') as call:
        client.get_runtime()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.GetRuntimeRequest()

@pytest.mark.asyncio
async def test_get_runtime_async(transport: str='grpc_asyncio', request_type=managed_service.GetRuntimeRequest):
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(runtime.Runtime(name='name_value', state=runtime.Runtime.State.STARTING, health_state=runtime.Runtime.HealthState.HEALTHY))
        response = await client.get_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.GetRuntimeRequest()
    assert isinstance(response, runtime.Runtime)
    assert response.name == 'name_value'
    assert response.state == runtime.Runtime.State.STARTING
    assert response.health_state == runtime.Runtime.HealthState.HEALTHY

@pytest.mark.asyncio
async def test_get_runtime_async_from_dict():
    await test_get_runtime_async(request_type=dict)

def test_get_runtime_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.GetRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_runtime), '__call__') as call:
        call.return_value = runtime.Runtime()
        client.get_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_runtime_field_headers_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.GetRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(runtime.Runtime())
        await client.get_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_runtime_flattened():
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_runtime), '__call__') as call:
        call.return_value = runtime.Runtime()
        client.get_runtime(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_runtime_flattened_error():
    if False:
        while True:
            i = 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_runtime(managed_service.GetRuntimeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_runtime_flattened_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_runtime), '__call__') as call:
        call.return_value = runtime.Runtime()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(runtime.Runtime())
        response = await client.get_runtime(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_runtime_flattened_error_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_runtime(managed_service.GetRuntimeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [managed_service.CreateRuntimeRequest, dict])
def test_create_runtime(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.CreateRuntimeRequest()
    assert isinstance(response, future.Future)

def test_create_runtime_empty_call():
    if False:
        while True:
            i = 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_runtime), '__call__') as call:
        client.create_runtime()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.CreateRuntimeRequest()

@pytest.mark.asyncio
async def test_create_runtime_async(transport: str='grpc_asyncio', request_type=managed_service.CreateRuntimeRequest):
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.CreateRuntimeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_runtime_async_from_dict():
    await test_create_runtime_async(request_type=dict)

def test_create_runtime_field_headers():
    if False:
        while True:
            i = 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.CreateRuntimeRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_runtime_field_headers_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.CreateRuntimeRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_runtime_flattened():
    if False:
        while True:
            i = 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_runtime(parent='parent_value', runtime_id='runtime_id_value', runtime=gcn_runtime.Runtime(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].runtime_id
        mock_val = 'runtime_id_value'
        assert arg == mock_val
        arg = args[0].runtime
        mock_val = gcn_runtime.Runtime(name='name_value')
        assert arg == mock_val

def test_create_runtime_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_runtime(managed_service.CreateRuntimeRequest(), parent='parent_value', runtime_id='runtime_id_value', runtime=gcn_runtime.Runtime(name='name_value'))

@pytest.mark.asyncio
async def test_create_runtime_flattened_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_runtime(parent='parent_value', runtime_id='runtime_id_value', runtime=gcn_runtime.Runtime(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].runtime_id
        mock_val = 'runtime_id_value'
        assert arg == mock_val
        arg = args[0].runtime
        mock_val = gcn_runtime.Runtime(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_runtime_flattened_error_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_runtime(managed_service.CreateRuntimeRequest(), parent='parent_value', runtime_id='runtime_id_value', runtime=gcn_runtime.Runtime(name='name_value'))

@pytest.mark.parametrize('request_type', [managed_service.UpdateRuntimeRequest, dict])
def test_update_runtime(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.UpdateRuntimeRequest()
    assert isinstance(response, future.Future)

def test_update_runtime_empty_call():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_runtime), '__call__') as call:
        client.update_runtime()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.UpdateRuntimeRequest()

@pytest.mark.asyncio
async def test_update_runtime_async(transport: str='grpc_asyncio', request_type=managed_service.UpdateRuntimeRequest):
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.UpdateRuntimeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_runtime_async_from_dict():
    await test_update_runtime_async(request_type=dict)

def test_update_runtime_field_headers():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.UpdateRuntimeRequest()
    request.runtime.name = 'name_value'
    with mock.patch.object(type(client.transport.update_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'runtime.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_runtime_field_headers_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.UpdateRuntimeRequest()
    request.runtime.name = 'name_value'
    with mock.patch.object(type(client.transport.update_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'runtime.name=name_value') in kw['metadata']

def test_update_runtime_flattened():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_runtime(runtime=gcn_runtime.Runtime(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].runtime
        mock_val = gcn_runtime.Runtime(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_runtime_flattened_error():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_runtime(managed_service.UpdateRuntimeRequest(), runtime=gcn_runtime.Runtime(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_runtime_flattened_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_runtime(runtime=gcn_runtime.Runtime(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].runtime
        mock_val = gcn_runtime.Runtime(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_runtime_flattened_error_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_runtime(managed_service.UpdateRuntimeRequest(), runtime=gcn_runtime.Runtime(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [managed_service.DeleteRuntimeRequest, dict])
def test_delete_runtime(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.DeleteRuntimeRequest()
    assert isinstance(response, future.Future)

def test_delete_runtime_empty_call():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_runtime), '__call__') as call:
        client.delete_runtime()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.DeleteRuntimeRequest()

@pytest.mark.asyncio
async def test_delete_runtime_async(transport: str='grpc_asyncio', request_type=managed_service.DeleteRuntimeRequest):
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.DeleteRuntimeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_runtime_async_from_dict():
    await test_delete_runtime_async(request_type=dict)

def test_delete_runtime_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.DeleteRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_runtime_field_headers_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.DeleteRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_runtime_flattened():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_runtime(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_runtime_flattened_error():
    if False:
        return 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_runtime(managed_service.DeleteRuntimeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_runtime_flattened_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_runtime(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_runtime_flattened_error_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_runtime(managed_service.DeleteRuntimeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [managed_service.StartRuntimeRequest, dict])
def test_start_runtime(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.start_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.StartRuntimeRequest()
    assert isinstance(response, future.Future)

def test_start_runtime_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.start_runtime), '__call__') as call:
        client.start_runtime()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.StartRuntimeRequest()

@pytest.mark.asyncio
async def test_start_runtime_async(transport: str='grpc_asyncio', request_type=managed_service.StartRuntimeRequest):
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.start_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.StartRuntimeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_start_runtime_async_from_dict():
    await test_start_runtime_async(request_type=dict)

def test_start_runtime_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.StartRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.start_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.start_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_start_runtime_field_headers_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.StartRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.start_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.start_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_start_runtime_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.start_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.start_runtime(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_start_runtime_flattened_error():
    if False:
        while True:
            i = 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.start_runtime(managed_service.StartRuntimeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_start_runtime_flattened_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.start_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.start_runtime(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_start_runtime_flattened_error_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.start_runtime(managed_service.StartRuntimeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [managed_service.StopRuntimeRequest, dict])
def test_stop_runtime(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.stop_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.stop_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.StopRuntimeRequest()
    assert isinstance(response, future.Future)

def test_stop_runtime_empty_call():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.stop_runtime), '__call__') as call:
        client.stop_runtime()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.StopRuntimeRequest()

@pytest.mark.asyncio
async def test_stop_runtime_async(transport: str='grpc_asyncio', request_type=managed_service.StopRuntimeRequest):
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.stop_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.stop_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.StopRuntimeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_stop_runtime_async_from_dict():
    await test_stop_runtime_async(request_type=dict)

def test_stop_runtime_field_headers():
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.StopRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.stop_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.stop_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_stop_runtime_field_headers_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.StopRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.stop_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.stop_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_stop_runtime_flattened():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.stop_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.stop_runtime(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_stop_runtime_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.stop_runtime(managed_service.StopRuntimeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_stop_runtime_flattened_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.stop_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.stop_runtime(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_stop_runtime_flattened_error_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.stop_runtime(managed_service.StopRuntimeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [managed_service.SwitchRuntimeRequest, dict])
def test_switch_runtime(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.switch_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.switch_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.SwitchRuntimeRequest()
    assert isinstance(response, future.Future)

def test_switch_runtime_empty_call():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.switch_runtime), '__call__') as call:
        client.switch_runtime()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.SwitchRuntimeRequest()

@pytest.mark.asyncio
async def test_switch_runtime_async(transport: str='grpc_asyncio', request_type=managed_service.SwitchRuntimeRequest):
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.switch_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.switch_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.SwitchRuntimeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_switch_runtime_async_from_dict():
    await test_switch_runtime_async(request_type=dict)

def test_switch_runtime_field_headers():
    if False:
        return 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.SwitchRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.switch_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.switch_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_switch_runtime_field_headers_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.SwitchRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.switch_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.switch_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_switch_runtime_flattened():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.switch_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.switch_runtime(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_switch_runtime_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.switch_runtime(managed_service.SwitchRuntimeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_switch_runtime_flattened_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.switch_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.switch_runtime(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_switch_runtime_flattened_error_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.switch_runtime(managed_service.SwitchRuntimeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [managed_service.ResetRuntimeRequest, dict])
def test_reset_runtime(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reset_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.reset_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.ResetRuntimeRequest()
    assert isinstance(response, future.Future)

def test_reset_runtime_empty_call():
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.reset_runtime), '__call__') as call:
        client.reset_runtime()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.ResetRuntimeRequest()

@pytest.mark.asyncio
async def test_reset_runtime_async(transport: str='grpc_asyncio', request_type=managed_service.ResetRuntimeRequest):
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reset_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reset_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.ResetRuntimeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_reset_runtime_async_from_dict():
    await test_reset_runtime_async(request_type=dict)

def test_reset_runtime_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.ResetRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.reset_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reset_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_reset_runtime_field_headers_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.ResetRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.reset_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.reset_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_reset_runtime_flattened():
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reset_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reset_runtime(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_reset_runtime_flattened_error():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.reset_runtime(managed_service.ResetRuntimeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_reset_runtime_flattened_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reset_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reset_runtime(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_reset_runtime_flattened_error_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.reset_runtime(managed_service.ResetRuntimeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [managed_service.UpgradeRuntimeRequest, dict])
def test_upgrade_runtime(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.upgrade_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.upgrade_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.UpgradeRuntimeRequest()
    assert isinstance(response, future.Future)

def test_upgrade_runtime_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.upgrade_runtime), '__call__') as call:
        client.upgrade_runtime()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.UpgradeRuntimeRequest()

@pytest.mark.asyncio
async def test_upgrade_runtime_async(transport: str='grpc_asyncio', request_type=managed_service.UpgradeRuntimeRequest):
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.upgrade_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.upgrade_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.UpgradeRuntimeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_upgrade_runtime_async_from_dict():
    await test_upgrade_runtime_async(request_type=dict)

def test_upgrade_runtime_field_headers():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.UpgradeRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.upgrade_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.upgrade_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_upgrade_runtime_field_headers_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.UpgradeRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.upgrade_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.upgrade_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_upgrade_runtime_flattened():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.upgrade_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.upgrade_runtime(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_upgrade_runtime_flattened_error():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.upgrade_runtime(managed_service.UpgradeRuntimeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_upgrade_runtime_flattened_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.upgrade_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.upgrade_runtime(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_upgrade_runtime_flattened_error_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.upgrade_runtime(managed_service.UpgradeRuntimeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [managed_service.ReportRuntimeEventRequest, dict])
def test_report_runtime_event(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.report_runtime_event), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.report_runtime_event(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.ReportRuntimeEventRequest()
    assert isinstance(response, future.Future)

def test_report_runtime_event_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.report_runtime_event), '__call__') as call:
        client.report_runtime_event()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.ReportRuntimeEventRequest()

@pytest.mark.asyncio
async def test_report_runtime_event_async(transport: str='grpc_asyncio', request_type=managed_service.ReportRuntimeEventRequest):
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.report_runtime_event), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.report_runtime_event(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.ReportRuntimeEventRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_report_runtime_event_async_from_dict():
    await test_report_runtime_event_async(request_type=dict)

def test_report_runtime_event_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.ReportRuntimeEventRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.report_runtime_event), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.report_runtime_event(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_report_runtime_event_field_headers_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.ReportRuntimeEventRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.report_runtime_event), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.report_runtime_event(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_report_runtime_event_flattened():
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.report_runtime_event), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.report_runtime_event(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_report_runtime_event_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.report_runtime_event(managed_service.ReportRuntimeEventRequest(), name='name_value')

@pytest.mark.asyncio
async def test_report_runtime_event_flattened_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.report_runtime_event), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.report_runtime_event(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_report_runtime_event_flattened_error_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.report_runtime_event(managed_service.ReportRuntimeEventRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [managed_service.RefreshRuntimeTokenInternalRequest, dict])
def test_refresh_runtime_token_internal(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.refresh_runtime_token_internal), '__call__') as call:
        call.return_value = managed_service.RefreshRuntimeTokenInternalResponse(access_token='access_token_value')
        response = client.refresh_runtime_token_internal(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.RefreshRuntimeTokenInternalRequest()
    assert isinstance(response, managed_service.RefreshRuntimeTokenInternalResponse)
    assert response.access_token == 'access_token_value'

def test_refresh_runtime_token_internal_empty_call():
    if False:
        return 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.refresh_runtime_token_internal), '__call__') as call:
        client.refresh_runtime_token_internal()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.RefreshRuntimeTokenInternalRequest()

@pytest.mark.asyncio
async def test_refresh_runtime_token_internal_async(transport: str='grpc_asyncio', request_type=managed_service.RefreshRuntimeTokenInternalRequest):
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.refresh_runtime_token_internal), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(managed_service.RefreshRuntimeTokenInternalResponse(access_token='access_token_value'))
        response = await client.refresh_runtime_token_internal(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.RefreshRuntimeTokenInternalRequest()
    assert isinstance(response, managed_service.RefreshRuntimeTokenInternalResponse)
    assert response.access_token == 'access_token_value'

@pytest.mark.asyncio
async def test_refresh_runtime_token_internal_async_from_dict():
    await test_refresh_runtime_token_internal_async(request_type=dict)

def test_refresh_runtime_token_internal_field_headers():
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.RefreshRuntimeTokenInternalRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.refresh_runtime_token_internal), '__call__') as call:
        call.return_value = managed_service.RefreshRuntimeTokenInternalResponse()
        client.refresh_runtime_token_internal(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_refresh_runtime_token_internal_field_headers_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.RefreshRuntimeTokenInternalRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.refresh_runtime_token_internal), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(managed_service.RefreshRuntimeTokenInternalResponse())
        await client.refresh_runtime_token_internal(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_refresh_runtime_token_internal_flattened():
    if False:
        return 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.refresh_runtime_token_internal), '__call__') as call:
        call.return_value = managed_service.RefreshRuntimeTokenInternalResponse()
        client.refresh_runtime_token_internal(name='name_value', vm_id='vm_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].vm_id
        mock_val = 'vm_id_value'
        assert arg == mock_val

def test_refresh_runtime_token_internal_flattened_error():
    if False:
        return 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.refresh_runtime_token_internal(managed_service.RefreshRuntimeTokenInternalRequest(), name='name_value', vm_id='vm_id_value')

@pytest.mark.asyncio
async def test_refresh_runtime_token_internal_flattened_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.refresh_runtime_token_internal), '__call__') as call:
        call.return_value = managed_service.RefreshRuntimeTokenInternalResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(managed_service.RefreshRuntimeTokenInternalResponse())
        response = await client.refresh_runtime_token_internal(name='name_value', vm_id='vm_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].vm_id
        mock_val = 'vm_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_refresh_runtime_token_internal_flattened_error_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.refresh_runtime_token_internal(managed_service.RefreshRuntimeTokenInternalRequest(), name='name_value', vm_id='vm_id_value')

@pytest.mark.parametrize('request_type', [managed_service.DiagnoseRuntimeRequest, dict])
def test_diagnose_runtime(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.diagnose_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.diagnose_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.DiagnoseRuntimeRequest()
    assert isinstance(response, future.Future)

def test_diagnose_runtime_empty_call():
    if False:
        while True:
            i = 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.diagnose_runtime), '__call__') as call:
        client.diagnose_runtime()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.DiagnoseRuntimeRequest()

@pytest.mark.asyncio
async def test_diagnose_runtime_async(transport: str='grpc_asyncio', request_type=managed_service.DiagnoseRuntimeRequest):
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.diagnose_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.diagnose_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_service.DiagnoseRuntimeRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_diagnose_runtime_async_from_dict():
    await test_diagnose_runtime_async(request_type=dict)

def test_diagnose_runtime_field_headers():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.DiagnoseRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.diagnose_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.diagnose_runtime(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_diagnose_runtime_field_headers_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_service.DiagnoseRuntimeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.diagnose_runtime), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.diagnose_runtime(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_diagnose_runtime_flattened():
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.diagnose_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.diagnose_runtime(name='name_value', diagnostic_config=gcn_diagnostic_config.DiagnosticConfig(gcs_bucket='gcs_bucket_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].diagnostic_config
        mock_val = gcn_diagnostic_config.DiagnosticConfig(gcs_bucket='gcs_bucket_value')
        assert arg == mock_val

def test_diagnose_runtime_flattened_error():
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.diagnose_runtime(managed_service.DiagnoseRuntimeRequest(), name='name_value', diagnostic_config=gcn_diagnostic_config.DiagnosticConfig(gcs_bucket='gcs_bucket_value'))

@pytest.mark.asyncio
async def test_diagnose_runtime_flattened_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.diagnose_runtime), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.diagnose_runtime(name='name_value', diagnostic_config=gcn_diagnostic_config.DiagnosticConfig(gcs_bucket='gcs_bucket_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].diagnostic_config
        mock_val = gcn_diagnostic_config.DiagnosticConfig(gcs_bucket='gcs_bucket_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_diagnose_runtime_flattened_error_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.diagnose_runtime(managed_service.DiagnoseRuntimeRequest(), name='name_value', diagnostic_config=gcn_diagnostic_config.DiagnosticConfig(gcs_bucket='gcs_bucket_value'))

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ManagedNotebookServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ManagedNotebookServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ManagedNotebookServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ManagedNotebookServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ManagedNotebookServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ManagedNotebookServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ManagedNotebookServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ManagedNotebookServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ManagedNotebookServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ManagedNotebookServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.ManagedNotebookServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ManagedNotebookServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ManagedNotebookServiceGrpcTransport, transports.ManagedNotebookServiceGrpcAsyncIOTransport])
def test_transport_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc'])
def test_transport_kind(transport_name):
    if False:
        print('Hello World!')
    transport = ManagedNotebookServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ManagedNotebookServiceGrpcTransport)

def test_managed_notebook_service_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ManagedNotebookServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_managed_notebook_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.notebooks_v1.services.managed_notebook_service.transports.ManagedNotebookServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ManagedNotebookServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_runtimes', 'get_runtime', 'create_runtime', 'update_runtime', 'delete_runtime', 'start_runtime', 'stop_runtime', 'switch_runtime', 'reset_runtime', 'upgrade_runtime', 'report_runtime_event', 'refresh_runtime_token_internal', 'diagnose_runtime', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_managed_notebook_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.notebooks_v1.services.managed_notebook_service.transports.ManagedNotebookServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ManagedNotebookServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_managed_notebook_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.notebooks_v1.services.managed_notebook_service.transports.ManagedNotebookServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ManagedNotebookServiceTransport()
        adc.assert_called_once()

def test_managed_notebook_service_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ManagedNotebookServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ManagedNotebookServiceGrpcTransport, transports.ManagedNotebookServiceGrpcAsyncIOTransport])
def test_managed_notebook_service_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ManagedNotebookServiceGrpcTransport, transports.ManagedNotebookServiceGrpcAsyncIOTransport])
def test_managed_notebook_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ManagedNotebookServiceGrpcTransport, grpc_helpers), (transports.ManagedNotebookServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_managed_notebook_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('notebooks.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='notebooks.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ManagedNotebookServiceGrpcTransport, transports.ManagedNotebookServiceGrpcAsyncIOTransport])
def test_managed_notebook_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_managed_notebook_service_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='notebooks.googleapis.com'), transport=transport_name)
    assert client.transport._host == 'notebooks.googleapis.com:443'

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_managed_notebook_service_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='notebooks.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == 'notebooks.googleapis.com:8000'

def test_managed_notebook_service_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ManagedNotebookServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_managed_notebook_service_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ManagedNotebookServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ManagedNotebookServiceGrpcTransport, transports.ManagedNotebookServiceGrpcAsyncIOTransport])
def test_managed_notebook_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.ManagedNotebookServiceGrpcTransport, transports.ManagedNotebookServiceGrpcAsyncIOTransport])
def test_managed_notebook_service_transport_channel_mtls_with_adc(transport_class):
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

def test_managed_notebook_service_grpc_lro_client():
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_managed_notebook_service_grpc_lro_async_client():
    if False:
        return 10
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_runtime_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    runtime = 'whelk'
    expected = 'projects/{project}/locations/{location}/runtimes/{runtime}'.format(project=project, location=location, runtime=runtime)
    actual = ManagedNotebookServiceClient.runtime_path(project, location, runtime)
    assert expected == actual

def test_parse_runtime_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'runtime': 'nudibranch'}
    path = ManagedNotebookServiceClient.runtime_path(**expected)
    actual = ManagedNotebookServiceClient.parse_runtime_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ManagedNotebookServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'mussel'}
    path = ManagedNotebookServiceClient.common_billing_account_path(**expected)
    actual = ManagedNotebookServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ManagedNotebookServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'nautilus'}
    path = ManagedNotebookServiceClient.common_folder_path(**expected)
    actual = ManagedNotebookServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ManagedNotebookServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'abalone'}
    path = ManagedNotebookServiceClient.common_organization_path(**expected)
    actual = ManagedNotebookServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = ManagedNotebookServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'clam'}
    path = ManagedNotebookServiceClient.common_project_path(**expected)
    actual = ManagedNotebookServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ManagedNotebookServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = ManagedNotebookServiceClient.common_location_path(**expected)
    actual = ManagedNotebookServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ManagedNotebookServiceTransport, '_prep_wrapped_messages') as prep:
        client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ManagedNotebookServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = ManagedNotebookServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_delete_operation(transport: str='grpc'):
    if False:
        return 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = ManagedNotebookServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['grpc']
    for transport in transports:
        client = ManagedNotebookServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ManagedNotebookServiceClient, transports.ManagedNotebookServiceGrpcTransport), (ManagedNotebookServiceAsyncClient, transports.ManagedNotebookServiceGrpcAsyncIOTransport)])
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