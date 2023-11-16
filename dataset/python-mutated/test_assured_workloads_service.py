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
from google.protobuf import duration_pb2
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
from google.cloud.assuredworkloads_v1.services.assured_workloads_service import AssuredWorkloadsServiceAsyncClient, AssuredWorkloadsServiceClient, pagers, transports
from google.cloud.assuredworkloads_v1.types import assuredworkloads

def client_cert_source_callback():
    if False:
        for i in range(10):
            print('nop')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        for i in range(10):
            print('nop')
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
    assert AssuredWorkloadsServiceClient._get_default_mtls_endpoint(None) is None
    assert AssuredWorkloadsServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert AssuredWorkloadsServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert AssuredWorkloadsServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert AssuredWorkloadsServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert AssuredWorkloadsServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(AssuredWorkloadsServiceClient, 'grpc'), (AssuredWorkloadsServiceAsyncClient, 'grpc_asyncio'), (AssuredWorkloadsServiceClient, 'rest')])
def test_assured_workloads_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('assuredworkloads.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://assuredworkloads.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.AssuredWorkloadsServiceGrpcTransport, 'grpc'), (transports.AssuredWorkloadsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.AssuredWorkloadsServiceRestTransport, 'rest')])
def test_assured_workloads_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(AssuredWorkloadsServiceClient, 'grpc'), (AssuredWorkloadsServiceAsyncClient, 'grpc_asyncio'), (AssuredWorkloadsServiceClient, 'rest')])
def test_assured_workloads_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('assuredworkloads.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://assuredworkloads.googleapis.com')

def test_assured_workloads_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = AssuredWorkloadsServiceClient.get_transport_class()
    available_transports = [transports.AssuredWorkloadsServiceGrpcTransport, transports.AssuredWorkloadsServiceRestTransport]
    assert transport in available_transports
    transport = AssuredWorkloadsServiceClient.get_transport_class('grpc')
    assert transport == transports.AssuredWorkloadsServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AssuredWorkloadsServiceClient, transports.AssuredWorkloadsServiceGrpcTransport, 'grpc'), (AssuredWorkloadsServiceAsyncClient, transports.AssuredWorkloadsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (AssuredWorkloadsServiceClient, transports.AssuredWorkloadsServiceRestTransport, 'rest')])
@mock.patch.object(AssuredWorkloadsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AssuredWorkloadsServiceClient))
@mock.patch.object(AssuredWorkloadsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AssuredWorkloadsServiceAsyncClient))
def test_assured_workloads_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(AssuredWorkloadsServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(AssuredWorkloadsServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(AssuredWorkloadsServiceClient, transports.AssuredWorkloadsServiceGrpcTransport, 'grpc', 'true'), (AssuredWorkloadsServiceAsyncClient, transports.AssuredWorkloadsServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (AssuredWorkloadsServiceClient, transports.AssuredWorkloadsServiceGrpcTransport, 'grpc', 'false'), (AssuredWorkloadsServiceAsyncClient, transports.AssuredWorkloadsServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (AssuredWorkloadsServiceClient, transports.AssuredWorkloadsServiceRestTransport, 'rest', 'true'), (AssuredWorkloadsServiceClient, transports.AssuredWorkloadsServiceRestTransport, 'rest', 'false')])
@mock.patch.object(AssuredWorkloadsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AssuredWorkloadsServiceClient))
@mock.patch.object(AssuredWorkloadsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AssuredWorkloadsServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_assured_workloads_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
    if False:
        i = 10
        return i + 15
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

@pytest.mark.parametrize('client_class', [AssuredWorkloadsServiceClient, AssuredWorkloadsServiceAsyncClient])
@mock.patch.object(AssuredWorkloadsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AssuredWorkloadsServiceClient))
@mock.patch.object(AssuredWorkloadsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AssuredWorkloadsServiceAsyncClient))
def test_assured_workloads_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AssuredWorkloadsServiceClient, transports.AssuredWorkloadsServiceGrpcTransport, 'grpc'), (AssuredWorkloadsServiceAsyncClient, transports.AssuredWorkloadsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (AssuredWorkloadsServiceClient, transports.AssuredWorkloadsServiceRestTransport, 'rest')])
def test_assured_workloads_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AssuredWorkloadsServiceClient, transports.AssuredWorkloadsServiceGrpcTransport, 'grpc', grpc_helpers), (AssuredWorkloadsServiceAsyncClient, transports.AssuredWorkloadsServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (AssuredWorkloadsServiceClient, transports.AssuredWorkloadsServiceRestTransport, 'rest', None)])
def test_assured_workloads_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_assured_workloads_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.assuredworkloads_v1.services.assured_workloads_service.transports.AssuredWorkloadsServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = AssuredWorkloadsServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AssuredWorkloadsServiceClient, transports.AssuredWorkloadsServiceGrpcTransport, 'grpc', grpc_helpers), (AssuredWorkloadsServiceAsyncClient, transports.AssuredWorkloadsServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_assured_workloads_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('assuredworkloads.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='assuredworkloads.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [assuredworkloads.CreateWorkloadRequest, dict])
def test_create_workload(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_workload), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_workload(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.CreateWorkloadRequest()
    assert isinstance(response, future.Future)

def test_create_workload_empty_call():
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_workload), '__call__') as call:
        client.create_workload()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.CreateWorkloadRequest()

@pytest.mark.asyncio
async def test_create_workload_async(transport: str='grpc_asyncio', request_type=assuredworkloads.CreateWorkloadRequest):
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_workload), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_workload(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.CreateWorkloadRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_workload_async_from_dict():
    await test_create_workload_async(request_type=dict)

def test_create_workload_field_headers():
    if False:
        i = 10
        return i + 15
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = assuredworkloads.CreateWorkloadRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_workload), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_workload(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_workload_field_headers_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = assuredworkloads.CreateWorkloadRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_workload), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_workload(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_workload_flattened():
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_workload), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_workload(parent='parent_value', workload=assuredworkloads.Workload(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].workload
        mock_val = assuredworkloads.Workload(name='name_value')
        assert arg == mock_val

def test_create_workload_flattened_error():
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_workload(assuredworkloads.CreateWorkloadRequest(), parent='parent_value', workload=assuredworkloads.Workload(name='name_value'))

@pytest.mark.asyncio
async def test_create_workload_flattened_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_workload), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_workload(parent='parent_value', workload=assuredworkloads.Workload(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].workload
        mock_val = assuredworkloads.Workload(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_workload_flattened_error_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_workload(assuredworkloads.CreateWorkloadRequest(), parent='parent_value', workload=assuredworkloads.Workload(name='name_value'))

@pytest.mark.parametrize('request_type', [assuredworkloads.UpdateWorkloadRequest, dict])
def test_update_workload(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_workload), '__call__') as call:
        call.return_value = assuredworkloads.Workload(name='name_value', display_name='display_name_value', compliance_regime=assuredworkloads.Workload.ComplianceRegime.IL4, billing_account='billing_account_value', etag='etag_value', provisioned_resources_parent='provisioned_resources_parent_value', kaj_enrollment_state=assuredworkloads.Workload.KajEnrollmentState.KAJ_ENROLLMENT_STATE_PENDING, enable_sovereign_controls=True, compliant_but_disallowed_services=['compliant_but_disallowed_services_value'], partner=assuredworkloads.Workload.Partner.LOCAL_CONTROLS_BY_S3NS)
        response = client.update_workload(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.UpdateWorkloadRequest()
    assert isinstance(response, assuredworkloads.Workload)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.compliance_regime == assuredworkloads.Workload.ComplianceRegime.IL4
    assert response.billing_account == 'billing_account_value'
    assert response.etag == 'etag_value'
    assert response.provisioned_resources_parent == 'provisioned_resources_parent_value'
    assert response.kaj_enrollment_state == assuredworkloads.Workload.KajEnrollmentState.KAJ_ENROLLMENT_STATE_PENDING
    assert response.enable_sovereign_controls is True
    assert response.compliant_but_disallowed_services == ['compliant_but_disallowed_services_value']
    assert response.partner == assuredworkloads.Workload.Partner.LOCAL_CONTROLS_BY_S3NS

def test_update_workload_empty_call():
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_workload), '__call__') as call:
        client.update_workload()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.UpdateWorkloadRequest()

@pytest.mark.asyncio
async def test_update_workload_async(transport: str='grpc_asyncio', request_type=assuredworkloads.UpdateWorkloadRequest):
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_workload), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.Workload(name='name_value', display_name='display_name_value', compliance_regime=assuredworkloads.Workload.ComplianceRegime.IL4, billing_account='billing_account_value', etag='etag_value', provisioned_resources_parent='provisioned_resources_parent_value', kaj_enrollment_state=assuredworkloads.Workload.KajEnrollmentState.KAJ_ENROLLMENT_STATE_PENDING, enable_sovereign_controls=True, compliant_but_disallowed_services=['compliant_but_disallowed_services_value'], partner=assuredworkloads.Workload.Partner.LOCAL_CONTROLS_BY_S3NS))
        response = await client.update_workload(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.UpdateWorkloadRequest()
    assert isinstance(response, assuredworkloads.Workload)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.compliance_regime == assuredworkloads.Workload.ComplianceRegime.IL4
    assert response.billing_account == 'billing_account_value'
    assert response.etag == 'etag_value'
    assert response.provisioned_resources_parent == 'provisioned_resources_parent_value'
    assert response.kaj_enrollment_state == assuredworkloads.Workload.KajEnrollmentState.KAJ_ENROLLMENT_STATE_PENDING
    assert response.enable_sovereign_controls is True
    assert response.compliant_but_disallowed_services == ['compliant_but_disallowed_services_value']
    assert response.partner == assuredworkloads.Workload.Partner.LOCAL_CONTROLS_BY_S3NS

@pytest.mark.asyncio
async def test_update_workload_async_from_dict():
    await test_update_workload_async(request_type=dict)

def test_update_workload_field_headers():
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = assuredworkloads.UpdateWorkloadRequest()
    request.workload.name = 'name_value'
    with mock.patch.object(type(client.transport.update_workload), '__call__') as call:
        call.return_value = assuredworkloads.Workload()
        client.update_workload(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'workload.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_workload_field_headers_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = assuredworkloads.UpdateWorkloadRequest()
    request.workload.name = 'name_value'
    with mock.patch.object(type(client.transport.update_workload), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.Workload())
        await client.update_workload(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'workload.name=name_value') in kw['metadata']

def test_update_workload_flattened():
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_workload), '__call__') as call:
        call.return_value = assuredworkloads.Workload()
        client.update_workload(workload=assuredworkloads.Workload(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].workload
        mock_val = assuredworkloads.Workload(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_workload_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_workload(assuredworkloads.UpdateWorkloadRequest(), workload=assuredworkloads.Workload(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_workload_flattened_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_workload), '__call__') as call:
        call.return_value = assuredworkloads.Workload()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.Workload())
        response = await client.update_workload(workload=assuredworkloads.Workload(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].workload
        mock_val = assuredworkloads.Workload(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_workload_flattened_error_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_workload(assuredworkloads.UpdateWorkloadRequest(), workload=assuredworkloads.Workload(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [assuredworkloads.RestrictAllowedResourcesRequest, dict])
def test_restrict_allowed_resources(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.restrict_allowed_resources), '__call__') as call:
        call.return_value = assuredworkloads.RestrictAllowedResourcesResponse()
        response = client.restrict_allowed_resources(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.RestrictAllowedResourcesRequest()
    assert isinstance(response, assuredworkloads.RestrictAllowedResourcesResponse)

def test_restrict_allowed_resources_empty_call():
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.restrict_allowed_resources), '__call__') as call:
        client.restrict_allowed_resources()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.RestrictAllowedResourcesRequest()

@pytest.mark.asyncio
async def test_restrict_allowed_resources_async(transport: str='grpc_asyncio', request_type=assuredworkloads.RestrictAllowedResourcesRequest):
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.restrict_allowed_resources), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.RestrictAllowedResourcesResponse())
        response = await client.restrict_allowed_resources(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.RestrictAllowedResourcesRequest()
    assert isinstance(response, assuredworkloads.RestrictAllowedResourcesResponse)

@pytest.mark.asyncio
async def test_restrict_allowed_resources_async_from_dict():
    await test_restrict_allowed_resources_async(request_type=dict)

def test_restrict_allowed_resources_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = assuredworkloads.RestrictAllowedResourcesRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.restrict_allowed_resources), '__call__') as call:
        call.return_value = assuredworkloads.RestrictAllowedResourcesResponse()
        client.restrict_allowed_resources(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_restrict_allowed_resources_field_headers_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = assuredworkloads.RestrictAllowedResourcesRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.restrict_allowed_resources), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.RestrictAllowedResourcesResponse())
        await client.restrict_allowed_resources(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [assuredworkloads.DeleteWorkloadRequest, dict])
def test_delete_workload(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_workload), '__call__') as call:
        call.return_value = None
        response = client.delete_workload(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.DeleteWorkloadRequest()
    assert response is None

def test_delete_workload_empty_call():
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_workload), '__call__') as call:
        client.delete_workload()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.DeleteWorkloadRequest()

@pytest.mark.asyncio
async def test_delete_workload_async(transport: str='grpc_asyncio', request_type=assuredworkloads.DeleteWorkloadRequest):
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_workload), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_workload(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.DeleteWorkloadRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_workload_async_from_dict():
    await test_delete_workload_async(request_type=dict)

def test_delete_workload_field_headers():
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = assuredworkloads.DeleteWorkloadRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_workload), '__call__') as call:
        call.return_value = None
        client.delete_workload(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_workload_field_headers_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = assuredworkloads.DeleteWorkloadRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_workload), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_workload(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_workload_flattened():
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_workload), '__call__') as call:
        call.return_value = None
        client.delete_workload(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_workload_flattened_error():
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_workload(assuredworkloads.DeleteWorkloadRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_workload_flattened_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_workload), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_workload(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_workload_flattened_error_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_workload(assuredworkloads.DeleteWorkloadRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [assuredworkloads.GetWorkloadRequest, dict])
def test_get_workload(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_workload), '__call__') as call:
        call.return_value = assuredworkloads.Workload(name='name_value', display_name='display_name_value', compliance_regime=assuredworkloads.Workload.ComplianceRegime.IL4, billing_account='billing_account_value', etag='etag_value', provisioned_resources_parent='provisioned_resources_parent_value', kaj_enrollment_state=assuredworkloads.Workload.KajEnrollmentState.KAJ_ENROLLMENT_STATE_PENDING, enable_sovereign_controls=True, compliant_but_disallowed_services=['compliant_but_disallowed_services_value'], partner=assuredworkloads.Workload.Partner.LOCAL_CONTROLS_BY_S3NS)
        response = client.get_workload(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.GetWorkloadRequest()
    assert isinstance(response, assuredworkloads.Workload)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.compliance_regime == assuredworkloads.Workload.ComplianceRegime.IL4
    assert response.billing_account == 'billing_account_value'
    assert response.etag == 'etag_value'
    assert response.provisioned_resources_parent == 'provisioned_resources_parent_value'
    assert response.kaj_enrollment_state == assuredworkloads.Workload.KajEnrollmentState.KAJ_ENROLLMENT_STATE_PENDING
    assert response.enable_sovereign_controls is True
    assert response.compliant_but_disallowed_services == ['compliant_but_disallowed_services_value']
    assert response.partner == assuredworkloads.Workload.Partner.LOCAL_CONTROLS_BY_S3NS

def test_get_workload_empty_call():
    if False:
        i = 10
        return i + 15
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_workload), '__call__') as call:
        client.get_workload()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.GetWorkloadRequest()

@pytest.mark.asyncio
async def test_get_workload_async(transport: str='grpc_asyncio', request_type=assuredworkloads.GetWorkloadRequest):
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_workload), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.Workload(name='name_value', display_name='display_name_value', compliance_regime=assuredworkloads.Workload.ComplianceRegime.IL4, billing_account='billing_account_value', etag='etag_value', provisioned_resources_parent='provisioned_resources_parent_value', kaj_enrollment_state=assuredworkloads.Workload.KajEnrollmentState.KAJ_ENROLLMENT_STATE_PENDING, enable_sovereign_controls=True, compliant_but_disallowed_services=['compliant_but_disallowed_services_value'], partner=assuredworkloads.Workload.Partner.LOCAL_CONTROLS_BY_S3NS))
        response = await client.get_workload(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.GetWorkloadRequest()
    assert isinstance(response, assuredworkloads.Workload)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.compliance_regime == assuredworkloads.Workload.ComplianceRegime.IL4
    assert response.billing_account == 'billing_account_value'
    assert response.etag == 'etag_value'
    assert response.provisioned_resources_parent == 'provisioned_resources_parent_value'
    assert response.kaj_enrollment_state == assuredworkloads.Workload.KajEnrollmentState.KAJ_ENROLLMENT_STATE_PENDING
    assert response.enable_sovereign_controls is True
    assert response.compliant_but_disallowed_services == ['compliant_but_disallowed_services_value']
    assert response.partner == assuredworkloads.Workload.Partner.LOCAL_CONTROLS_BY_S3NS

@pytest.mark.asyncio
async def test_get_workload_async_from_dict():
    await test_get_workload_async(request_type=dict)

def test_get_workload_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = assuredworkloads.GetWorkloadRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_workload), '__call__') as call:
        call.return_value = assuredworkloads.Workload()
        client.get_workload(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_workload_field_headers_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = assuredworkloads.GetWorkloadRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_workload), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.Workload())
        await client.get_workload(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_workload_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_workload), '__call__') as call:
        call.return_value = assuredworkloads.Workload()
        client.get_workload(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_workload_flattened_error():
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_workload(assuredworkloads.GetWorkloadRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_workload_flattened_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_workload), '__call__') as call:
        call.return_value = assuredworkloads.Workload()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.Workload())
        response = await client.get_workload(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_workload_flattened_error_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_workload(assuredworkloads.GetWorkloadRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [assuredworkloads.ListWorkloadsRequest, dict])
def test_list_workloads(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_workloads), '__call__') as call:
        call.return_value = assuredworkloads.ListWorkloadsResponse(next_page_token='next_page_token_value')
        response = client.list_workloads(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.ListWorkloadsRequest()
    assert isinstance(response, pagers.ListWorkloadsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_workloads_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_workloads), '__call__') as call:
        client.list_workloads()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.ListWorkloadsRequest()

@pytest.mark.asyncio
async def test_list_workloads_async(transport: str='grpc_asyncio', request_type=assuredworkloads.ListWorkloadsRequest):
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_workloads), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.ListWorkloadsResponse(next_page_token='next_page_token_value'))
        response = await client.list_workloads(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.ListWorkloadsRequest()
    assert isinstance(response, pagers.ListWorkloadsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_workloads_async_from_dict():
    await test_list_workloads_async(request_type=dict)

def test_list_workloads_field_headers():
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = assuredworkloads.ListWorkloadsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_workloads), '__call__') as call:
        call.return_value = assuredworkloads.ListWorkloadsResponse()
        client.list_workloads(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_workloads_field_headers_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = assuredworkloads.ListWorkloadsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_workloads), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.ListWorkloadsResponse())
        await client.list_workloads(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_workloads_flattened():
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_workloads), '__call__') as call:
        call.return_value = assuredworkloads.ListWorkloadsResponse()
        client.list_workloads(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_workloads_flattened_error():
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_workloads(assuredworkloads.ListWorkloadsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_workloads_flattened_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_workloads), '__call__') as call:
        call.return_value = assuredworkloads.ListWorkloadsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.ListWorkloadsResponse())
        response = await client.list_workloads(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_workloads_flattened_error_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_workloads(assuredworkloads.ListWorkloadsRequest(), parent='parent_value')

def test_list_workloads_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_workloads), '__call__') as call:
        call.side_effect = (assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload(), assuredworkloads.Workload(), assuredworkloads.Workload()], next_page_token='abc'), assuredworkloads.ListWorkloadsResponse(workloads=[], next_page_token='def'), assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload()], next_page_token='ghi'), assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload(), assuredworkloads.Workload()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_workloads(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, assuredworkloads.Workload) for i in results))

def test_list_workloads_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_workloads), '__call__') as call:
        call.side_effect = (assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload(), assuredworkloads.Workload(), assuredworkloads.Workload()], next_page_token='abc'), assuredworkloads.ListWorkloadsResponse(workloads=[], next_page_token='def'), assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload()], next_page_token='ghi'), assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload(), assuredworkloads.Workload()]), RuntimeError)
        pages = list(client.list_workloads(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_workloads_async_pager():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_workloads), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload(), assuredworkloads.Workload(), assuredworkloads.Workload()], next_page_token='abc'), assuredworkloads.ListWorkloadsResponse(workloads=[], next_page_token='def'), assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload()], next_page_token='ghi'), assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload(), assuredworkloads.Workload()]), RuntimeError)
        async_pager = await client.list_workloads(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, assuredworkloads.Workload) for i in responses))

@pytest.mark.asyncio
async def test_list_workloads_async_pages():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_workloads), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload(), assuredworkloads.Workload(), assuredworkloads.Workload()], next_page_token='abc'), assuredworkloads.ListWorkloadsResponse(workloads=[], next_page_token='def'), assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload()], next_page_token='ghi'), assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload(), assuredworkloads.Workload()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_workloads(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [assuredworkloads.ListViolationsRequest, dict])
def test_list_violations(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_violations), '__call__') as call:
        call.return_value = assuredworkloads.ListViolationsResponse(next_page_token='next_page_token_value')
        response = client.list_violations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.ListViolationsRequest()
    assert isinstance(response, pagers.ListViolationsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_violations_empty_call():
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_violations), '__call__') as call:
        client.list_violations()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.ListViolationsRequest()

@pytest.mark.asyncio
async def test_list_violations_async(transport: str='grpc_asyncio', request_type=assuredworkloads.ListViolationsRequest):
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_violations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.ListViolationsResponse(next_page_token='next_page_token_value'))
        response = await client.list_violations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.ListViolationsRequest()
    assert isinstance(response, pagers.ListViolationsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_violations_async_from_dict():
    await test_list_violations_async(request_type=dict)

def test_list_violations_flattened():
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_violations), '__call__') as call:
        call.return_value = assuredworkloads.ListViolationsResponse()
        client.list_violations(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_violations_flattened_error():
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_violations(assuredworkloads.ListViolationsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_violations_flattened_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_violations), '__call__') as call:
        call.return_value = assuredworkloads.ListViolationsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.ListViolationsResponse())
        response = await client.list_violations(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_violations_flattened_error_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_violations(assuredworkloads.ListViolationsRequest(), parent='parent_value')

def test_list_violations_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_violations), '__call__') as call:
        call.side_effect = (assuredworkloads.ListViolationsResponse(violations=[assuredworkloads.Violation(), assuredworkloads.Violation(), assuredworkloads.Violation()], next_page_token='abc'), assuredworkloads.ListViolationsResponse(violations=[], next_page_token='def'), assuredworkloads.ListViolationsResponse(violations=[assuredworkloads.Violation()], next_page_token='ghi'), assuredworkloads.ListViolationsResponse(violations=[assuredworkloads.Violation(), assuredworkloads.Violation()]), RuntimeError)
        metadata = ()
        pager = client.list_violations(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, assuredworkloads.Violation) for i in results))

def test_list_violations_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_violations), '__call__') as call:
        call.side_effect = (assuredworkloads.ListViolationsResponse(violations=[assuredworkloads.Violation(), assuredworkloads.Violation(), assuredworkloads.Violation()], next_page_token='abc'), assuredworkloads.ListViolationsResponse(violations=[], next_page_token='def'), assuredworkloads.ListViolationsResponse(violations=[assuredworkloads.Violation()], next_page_token='ghi'), assuredworkloads.ListViolationsResponse(violations=[assuredworkloads.Violation(), assuredworkloads.Violation()]), RuntimeError)
        pages = list(client.list_violations(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_violations_async_pager():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_violations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (assuredworkloads.ListViolationsResponse(violations=[assuredworkloads.Violation(), assuredworkloads.Violation(), assuredworkloads.Violation()], next_page_token='abc'), assuredworkloads.ListViolationsResponse(violations=[], next_page_token='def'), assuredworkloads.ListViolationsResponse(violations=[assuredworkloads.Violation()], next_page_token='ghi'), assuredworkloads.ListViolationsResponse(violations=[assuredworkloads.Violation(), assuredworkloads.Violation()]), RuntimeError)
        async_pager = await client.list_violations(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, assuredworkloads.Violation) for i in responses))

@pytest.mark.asyncio
async def test_list_violations_async_pages():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_violations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (assuredworkloads.ListViolationsResponse(violations=[assuredworkloads.Violation(), assuredworkloads.Violation(), assuredworkloads.Violation()], next_page_token='abc'), assuredworkloads.ListViolationsResponse(violations=[], next_page_token='def'), assuredworkloads.ListViolationsResponse(violations=[assuredworkloads.Violation()], next_page_token='ghi'), assuredworkloads.ListViolationsResponse(violations=[assuredworkloads.Violation(), assuredworkloads.Violation()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_violations(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [assuredworkloads.GetViolationRequest, dict])
def test_get_violation(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_violation), '__call__') as call:
        call.return_value = assuredworkloads.Violation(name='name_value', description='description_value', category='category_value', state=assuredworkloads.Violation.State.RESOLVED, org_policy_constraint='org_policy_constraint_value', audit_log_link='audit_log_link_value', non_compliant_org_policy='non_compliant_org_policy_value', acknowledged=True, exception_audit_log_link='exception_audit_log_link_value')
        response = client.get_violation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.GetViolationRequest()
    assert isinstance(response, assuredworkloads.Violation)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.category == 'category_value'
    assert response.state == assuredworkloads.Violation.State.RESOLVED
    assert response.org_policy_constraint == 'org_policy_constraint_value'
    assert response.audit_log_link == 'audit_log_link_value'
    assert response.non_compliant_org_policy == 'non_compliant_org_policy_value'
    assert response.acknowledged is True
    assert response.exception_audit_log_link == 'exception_audit_log_link_value'

def test_get_violation_empty_call():
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_violation), '__call__') as call:
        client.get_violation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.GetViolationRequest()

@pytest.mark.asyncio
async def test_get_violation_async(transport: str='grpc_asyncio', request_type=assuredworkloads.GetViolationRequest):
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_violation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.Violation(name='name_value', description='description_value', category='category_value', state=assuredworkloads.Violation.State.RESOLVED, org_policy_constraint='org_policy_constraint_value', audit_log_link='audit_log_link_value', non_compliant_org_policy='non_compliant_org_policy_value', acknowledged=True, exception_audit_log_link='exception_audit_log_link_value'))
        response = await client.get_violation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.GetViolationRequest()
    assert isinstance(response, assuredworkloads.Violation)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.category == 'category_value'
    assert response.state == assuredworkloads.Violation.State.RESOLVED
    assert response.org_policy_constraint == 'org_policy_constraint_value'
    assert response.audit_log_link == 'audit_log_link_value'
    assert response.non_compliant_org_policy == 'non_compliant_org_policy_value'
    assert response.acknowledged is True
    assert response.exception_audit_log_link == 'exception_audit_log_link_value'

@pytest.mark.asyncio
async def test_get_violation_async_from_dict():
    await test_get_violation_async(request_type=dict)

def test_get_violation_flattened():
    if False:
        i = 10
        return i + 15
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_violation), '__call__') as call:
        call.return_value = assuredworkloads.Violation()
        client.get_violation(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_violation_flattened_error():
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_violation(assuredworkloads.GetViolationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_violation_flattened_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_violation), '__call__') as call:
        call.return_value = assuredworkloads.Violation()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.Violation())
        response = await client.get_violation(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_violation_flattened_error_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_violation(assuredworkloads.GetViolationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [assuredworkloads.AcknowledgeViolationRequest, dict])
def test_acknowledge_violation(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.acknowledge_violation), '__call__') as call:
        call.return_value = assuredworkloads.AcknowledgeViolationResponse()
        response = client.acknowledge_violation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.AcknowledgeViolationRequest()
    assert isinstance(response, assuredworkloads.AcknowledgeViolationResponse)

def test_acknowledge_violation_empty_call():
    if False:
        i = 10
        return i + 15
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.acknowledge_violation), '__call__') as call:
        client.acknowledge_violation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.AcknowledgeViolationRequest()

@pytest.mark.asyncio
async def test_acknowledge_violation_async(transport: str='grpc_asyncio', request_type=assuredworkloads.AcknowledgeViolationRequest):
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.acknowledge_violation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(assuredworkloads.AcknowledgeViolationResponse())
        response = await client.acknowledge_violation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == assuredworkloads.AcknowledgeViolationRequest()
    assert isinstance(response, assuredworkloads.AcknowledgeViolationResponse)

@pytest.mark.asyncio
async def test_acknowledge_violation_async_from_dict():
    await test_acknowledge_violation_async(request_type=dict)

@pytest.mark.parametrize('request_type', [assuredworkloads.CreateWorkloadRequest, dict])
def test_create_workload_rest(request_type):
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'organizations/sample1/locations/sample2'}
    request_init['workload'] = {'name': 'name_value', 'display_name': 'display_name_value', 'resources': [{'resource_id': 1172, 'resource_type': 1}], 'compliance_regime': 1, 'create_time': {'seconds': 751, 'nanos': 543}, 'billing_account': 'billing_account_value', 'etag': 'etag_value', 'labels': {}, 'provisioned_resources_parent': 'provisioned_resources_parent_value', 'kms_settings': {'next_rotation_time': {}, 'rotation_period': {'seconds': 751, 'nanos': 543}}, 'resource_settings': [{'resource_id': 'resource_id_value', 'resource_type': 1, 'display_name': 'display_name_value'}], 'kaj_enrollment_state': 1, 'enable_sovereign_controls': True, 'saa_enrollment_response': {'setup_status': 1, 'setup_errors': [1]}, 'compliant_but_disallowed_services': ['compliant_but_disallowed_services_value1', 'compliant_but_disallowed_services_value2'], 'partner': 1}
    test_field = assuredworkloads.CreateWorkloadRequest.meta.fields['workload']

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
    for (field, value) in request_init['workload'].items():
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
                for i in range(0, len(request_init['workload'][field])):
                    del request_init['workload'][field][i][subfield]
            else:
                del request_init['workload'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_workload(request)
    assert response.operation.name == 'operations/spam'

def test_create_workload_rest_required_fields(request_type=assuredworkloads.CreateWorkloadRequest):
    if False:
        return 10
    transport_class = transports.AssuredWorkloadsServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_workload._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_workload._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('external_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_workload(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_workload_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AssuredWorkloadsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_workload._get_unset_required_fields({})
    assert set(unset_fields) == set(('externalId',)) & set(('parent', 'workload'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_workload_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AssuredWorkloadsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssuredWorkloadsServiceRestInterceptor())
    client = AssuredWorkloadsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AssuredWorkloadsServiceRestInterceptor, 'post_create_workload') as post, mock.patch.object(transports.AssuredWorkloadsServiceRestInterceptor, 'pre_create_workload') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = assuredworkloads.CreateWorkloadRequest.pb(assuredworkloads.CreateWorkloadRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = assuredworkloads.CreateWorkloadRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_workload(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_workload_rest_bad_request(transport: str='rest', request_type=assuredworkloads.CreateWorkloadRequest):
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'organizations/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_workload(request)

def test_create_workload_rest_flattened():
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'organizations/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', workload=assuredworkloads.Workload(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_workload(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=organizations/*/locations/*}/workloads' % client.transport._host, args[1])

def test_create_workload_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_workload(assuredworkloads.CreateWorkloadRequest(), parent='parent_value', workload=assuredworkloads.Workload(name='name_value'))

def test_create_workload_rest_error():
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [assuredworkloads.UpdateWorkloadRequest, dict])
def test_update_workload_rest(request_type):
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'workload': {'name': 'organizations/sample1/locations/sample2/workloads/sample3'}}
    request_init['workload'] = {'name': 'organizations/sample1/locations/sample2/workloads/sample3', 'display_name': 'display_name_value', 'resources': [{'resource_id': 1172, 'resource_type': 1}], 'compliance_regime': 1, 'create_time': {'seconds': 751, 'nanos': 543}, 'billing_account': 'billing_account_value', 'etag': 'etag_value', 'labels': {}, 'provisioned_resources_parent': 'provisioned_resources_parent_value', 'kms_settings': {'next_rotation_time': {}, 'rotation_period': {'seconds': 751, 'nanos': 543}}, 'resource_settings': [{'resource_id': 'resource_id_value', 'resource_type': 1, 'display_name': 'display_name_value'}], 'kaj_enrollment_state': 1, 'enable_sovereign_controls': True, 'saa_enrollment_response': {'setup_status': 1, 'setup_errors': [1]}, 'compliant_but_disallowed_services': ['compliant_but_disallowed_services_value1', 'compliant_but_disallowed_services_value2'], 'partner': 1}
    test_field = assuredworkloads.UpdateWorkloadRequest.meta.fields['workload']

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
    for (field, value) in request_init['workload'].items():
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
                for i in range(0, len(request_init['workload'][field])):
                    del request_init['workload'][field][i][subfield]
            else:
                del request_init['workload'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = assuredworkloads.Workload(name='name_value', display_name='display_name_value', compliance_regime=assuredworkloads.Workload.ComplianceRegime.IL4, billing_account='billing_account_value', etag='etag_value', provisioned_resources_parent='provisioned_resources_parent_value', kaj_enrollment_state=assuredworkloads.Workload.KajEnrollmentState.KAJ_ENROLLMENT_STATE_PENDING, enable_sovereign_controls=True, compliant_but_disallowed_services=['compliant_but_disallowed_services_value'], partner=assuredworkloads.Workload.Partner.LOCAL_CONTROLS_BY_S3NS)
        response_value = Response()
        response_value.status_code = 200
        return_value = assuredworkloads.Workload.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_workload(request)
    assert isinstance(response, assuredworkloads.Workload)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.compliance_regime == assuredworkloads.Workload.ComplianceRegime.IL4
    assert response.billing_account == 'billing_account_value'
    assert response.etag == 'etag_value'
    assert response.provisioned_resources_parent == 'provisioned_resources_parent_value'
    assert response.kaj_enrollment_state == assuredworkloads.Workload.KajEnrollmentState.KAJ_ENROLLMENT_STATE_PENDING
    assert response.enable_sovereign_controls is True
    assert response.compliant_but_disallowed_services == ['compliant_but_disallowed_services_value']
    assert response.partner == assuredworkloads.Workload.Partner.LOCAL_CONTROLS_BY_S3NS

def test_update_workload_rest_required_fields(request_type=assuredworkloads.UpdateWorkloadRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AssuredWorkloadsServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_workload._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_workload._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = assuredworkloads.Workload()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = assuredworkloads.Workload.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_workload(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_workload_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AssuredWorkloadsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_workload._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('workload', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_workload_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AssuredWorkloadsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssuredWorkloadsServiceRestInterceptor())
    client = AssuredWorkloadsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssuredWorkloadsServiceRestInterceptor, 'post_update_workload') as post, mock.patch.object(transports.AssuredWorkloadsServiceRestInterceptor, 'pre_update_workload') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = assuredworkloads.UpdateWorkloadRequest.pb(assuredworkloads.UpdateWorkloadRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = assuredworkloads.Workload.to_json(assuredworkloads.Workload())
        request = assuredworkloads.UpdateWorkloadRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = assuredworkloads.Workload()
        client.update_workload(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_workload_rest_bad_request(transport: str='rest', request_type=assuredworkloads.UpdateWorkloadRequest):
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'workload': {'name': 'organizations/sample1/locations/sample2/workloads/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_workload(request)

def test_update_workload_rest_flattened():
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = assuredworkloads.Workload()
        sample_request = {'workload': {'name': 'organizations/sample1/locations/sample2/workloads/sample3'}}
        mock_args = dict(workload=assuredworkloads.Workload(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = assuredworkloads.Workload.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_workload(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{workload.name=organizations/*/locations/*/workloads/*}' % client.transport._host, args[1])

def test_update_workload_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_workload(assuredworkloads.UpdateWorkloadRequest(), workload=assuredworkloads.Workload(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_workload_rest_error():
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [assuredworkloads.RestrictAllowedResourcesRequest, dict])
def test_restrict_allowed_resources_rest(request_type):
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/locations/sample2/workloads/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = assuredworkloads.RestrictAllowedResourcesResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = assuredworkloads.RestrictAllowedResourcesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.restrict_allowed_resources(request)
    assert isinstance(response, assuredworkloads.RestrictAllowedResourcesResponse)

def test_restrict_allowed_resources_rest_required_fields(request_type=assuredworkloads.RestrictAllowedResourcesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AssuredWorkloadsServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).restrict_allowed_resources._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).restrict_allowed_resources._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = assuredworkloads.RestrictAllowedResourcesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = assuredworkloads.RestrictAllowedResourcesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.restrict_allowed_resources(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_restrict_allowed_resources_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AssuredWorkloadsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.restrict_allowed_resources._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'restrictionType'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_restrict_allowed_resources_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AssuredWorkloadsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssuredWorkloadsServiceRestInterceptor())
    client = AssuredWorkloadsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssuredWorkloadsServiceRestInterceptor, 'post_restrict_allowed_resources') as post, mock.patch.object(transports.AssuredWorkloadsServiceRestInterceptor, 'pre_restrict_allowed_resources') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = assuredworkloads.RestrictAllowedResourcesRequest.pb(assuredworkloads.RestrictAllowedResourcesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = assuredworkloads.RestrictAllowedResourcesResponse.to_json(assuredworkloads.RestrictAllowedResourcesResponse())
        request = assuredworkloads.RestrictAllowedResourcesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = assuredworkloads.RestrictAllowedResourcesResponse()
        client.restrict_allowed_resources(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_restrict_allowed_resources_rest_bad_request(transport: str='rest', request_type=assuredworkloads.RestrictAllowedResourcesRequest):
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'organizations/sample1/locations/sample2/workloads/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.restrict_allowed_resources(request)

def test_restrict_allowed_resources_rest_error():
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [assuredworkloads.DeleteWorkloadRequest, dict])
def test_delete_workload_rest(request_type):
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/locations/sample2/workloads/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_workload(request)
    assert response is None

def test_delete_workload_rest_required_fields(request_type=assuredworkloads.DeleteWorkloadRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AssuredWorkloadsServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_workload._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_workload._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('etag',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_workload(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_workload_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AssuredWorkloadsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_workload._get_unset_required_fields({})
    assert set(unset_fields) == set(('etag',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_workload_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AssuredWorkloadsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssuredWorkloadsServiceRestInterceptor())
    client = AssuredWorkloadsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssuredWorkloadsServiceRestInterceptor, 'pre_delete_workload') as pre:
        pre.assert_not_called()
        pb_message = assuredworkloads.DeleteWorkloadRequest.pb(assuredworkloads.DeleteWorkloadRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = assuredworkloads.DeleteWorkloadRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_workload(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_workload_rest_bad_request(transport: str='rest', request_type=assuredworkloads.DeleteWorkloadRequest):
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'organizations/sample1/locations/sample2/workloads/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_workload(request)

def test_delete_workload_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'organizations/sample1/locations/sample2/workloads/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_workload(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=organizations/*/locations/*/workloads/*}' % client.transport._host, args[1])

def test_delete_workload_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_workload(assuredworkloads.DeleteWorkloadRequest(), name='name_value')

def test_delete_workload_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [assuredworkloads.GetWorkloadRequest, dict])
def test_get_workload_rest(request_type):
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/locations/sample2/workloads/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = assuredworkloads.Workload(name='name_value', display_name='display_name_value', compliance_regime=assuredworkloads.Workload.ComplianceRegime.IL4, billing_account='billing_account_value', etag='etag_value', provisioned_resources_parent='provisioned_resources_parent_value', kaj_enrollment_state=assuredworkloads.Workload.KajEnrollmentState.KAJ_ENROLLMENT_STATE_PENDING, enable_sovereign_controls=True, compliant_but_disallowed_services=['compliant_but_disallowed_services_value'], partner=assuredworkloads.Workload.Partner.LOCAL_CONTROLS_BY_S3NS)
        response_value = Response()
        response_value.status_code = 200
        return_value = assuredworkloads.Workload.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_workload(request)
    assert isinstance(response, assuredworkloads.Workload)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.compliance_regime == assuredworkloads.Workload.ComplianceRegime.IL4
    assert response.billing_account == 'billing_account_value'
    assert response.etag == 'etag_value'
    assert response.provisioned_resources_parent == 'provisioned_resources_parent_value'
    assert response.kaj_enrollment_state == assuredworkloads.Workload.KajEnrollmentState.KAJ_ENROLLMENT_STATE_PENDING
    assert response.enable_sovereign_controls is True
    assert response.compliant_but_disallowed_services == ['compliant_but_disallowed_services_value']
    assert response.partner == assuredworkloads.Workload.Partner.LOCAL_CONTROLS_BY_S3NS

def test_get_workload_rest_required_fields(request_type=assuredworkloads.GetWorkloadRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AssuredWorkloadsServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_workload._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_workload._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = assuredworkloads.Workload()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = assuredworkloads.Workload.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_workload(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_workload_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.AssuredWorkloadsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_workload._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_workload_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AssuredWorkloadsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssuredWorkloadsServiceRestInterceptor())
    client = AssuredWorkloadsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssuredWorkloadsServiceRestInterceptor, 'post_get_workload') as post, mock.patch.object(transports.AssuredWorkloadsServiceRestInterceptor, 'pre_get_workload') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = assuredworkloads.GetWorkloadRequest.pb(assuredworkloads.GetWorkloadRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = assuredworkloads.Workload.to_json(assuredworkloads.Workload())
        request = assuredworkloads.GetWorkloadRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = assuredworkloads.Workload()
        client.get_workload(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_workload_rest_bad_request(transport: str='rest', request_type=assuredworkloads.GetWorkloadRequest):
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'organizations/sample1/locations/sample2/workloads/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_workload(request)

def test_get_workload_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = assuredworkloads.Workload()
        sample_request = {'name': 'organizations/sample1/locations/sample2/workloads/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = assuredworkloads.Workload.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_workload(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=organizations/*/locations/*/workloads/*}' % client.transport._host, args[1])

def test_get_workload_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_workload(assuredworkloads.GetWorkloadRequest(), name='name_value')

def test_get_workload_rest_error():
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [assuredworkloads.ListWorkloadsRequest, dict])
def test_list_workloads_rest(request_type):
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'organizations/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = assuredworkloads.ListWorkloadsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = assuredworkloads.ListWorkloadsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_workloads(request)
    assert isinstance(response, pagers.ListWorkloadsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_workloads_rest_required_fields(request_type=assuredworkloads.ListWorkloadsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AssuredWorkloadsServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_workloads._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_workloads._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = assuredworkloads.ListWorkloadsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = assuredworkloads.ListWorkloadsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_workloads(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_workloads_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AssuredWorkloadsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_workloads._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_workloads_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AssuredWorkloadsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AssuredWorkloadsServiceRestInterceptor())
    client = AssuredWorkloadsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AssuredWorkloadsServiceRestInterceptor, 'post_list_workloads') as post, mock.patch.object(transports.AssuredWorkloadsServiceRestInterceptor, 'pre_list_workloads') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = assuredworkloads.ListWorkloadsRequest.pb(assuredworkloads.ListWorkloadsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = assuredworkloads.ListWorkloadsResponse.to_json(assuredworkloads.ListWorkloadsResponse())
        request = assuredworkloads.ListWorkloadsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = assuredworkloads.ListWorkloadsResponse()
        client.list_workloads(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_workloads_rest_bad_request(transport: str='rest', request_type=assuredworkloads.ListWorkloadsRequest):
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'organizations/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_workloads(request)

def test_list_workloads_rest_flattened():
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = assuredworkloads.ListWorkloadsResponse()
        sample_request = {'parent': 'organizations/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = assuredworkloads.ListWorkloadsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_workloads(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=organizations/*/locations/*}/workloads' % client.transport._host, args[1])

def test_list_workloads_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_workloads(assuredworkloads.ListWorkloadsRequest(), parent='parent_value')

def test_list_workloads_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload(), assuredworkloads.Workload(), assuredworkloads.Workload()], next_page_token='abc'), assuredworkloads.ListWorkloadsResponse(workloads=[], next_page_token='def'), assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload()], next_page_token='ghi'), assuredworkloads.ListWorkloadsResponse(workloads=[assuredworkloads.Workload(), assuredworkloads.Workload()]))
        response = response + response
        response = tuple((assuredworkloads.ListWorkloadsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'organizations/sample1/locations/sample2'}
        pager = client.list_workloads(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, assuredworkloads.Workload) for i in results))
        pages = list(client.list_workloads(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_list_violations_rest_no_http_options():
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = assuredworkloads.ListViolationsRequest()
    with pytest.raises(RuntimeError):
        client.list_violations(request)

def test_get_violation_rest_no_http_options():
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = assuredworkloads.GetViolationRequest()
    with pytest.raises(RuntimeError):
        client.get_violation(request)

def test_acknowledge_violation_rest_no_http_options():
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = assuredworkloads.AcknowledgeViolationRequest()
    with pytest.raises(RuntimeError):
        client.acknowledge_violation(request)

def test_list_violations_rest_error():
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with pytest.raises(NotImplementedError) as not_implemented_error:
        client.list_violations({})
    assert 'Method ListViolations is not available over REST transport' in str(not_implemented_error.value)

def test_get_violation_rest_error():
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with pytest.raises(NotImplementedError) as not_implemented_error:
        client.get_violation({})
    assert 'Method GetViolation is not available over REST transport' in str(not_implemented_error.value)

def test_acknowledge_violation_rest_error():
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with pytest.raises(NotImplementedError) as not_implemented_error:
        client.acknowledge_violation({})
    assert 'Method AcknowledgeViolation is not available over REST transport' in str(not_implemented_error.value)

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.AssuredWorkloadsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.AssuredWorkloadsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AssuredWorkloadsServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.AssuredWorkloadsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AssuredWorkloadsServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AssuredWorkloadsServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.AssuredWorkloadsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AssuredWorkloadsServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.AssuredWorkloadsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = AssuredWorkloadsServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.AssuredWorkloadsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.AssuredWorkloadsServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.AssuredWorkloadsServiceGrpcTransport, transports.AssuredWorkloadsServiceGrpcAsyncIOTransport, transports.AssuredWorkloadsServiceRestTransport])
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
        for i in range(10):
            print('nop')
    transport = AssuredWorkloadsServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.AssuredWorkloadsServiceGrpcTransport)

def test_assured_workloads_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.AssuredWorkloadsServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_assured_workloads_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.assuredworkloads_v1.services.assured_workloads_service.transports.AssuredWorkloadsServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.AssuredWorkloadsServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_workload', 'update_workload', 'restrict_allowed_resources', 'delete_workload', 'get_workload', 'list_workloads', 'list_violations', 'get_violation', 'acknowledge_violation', 'get_operation', 'list_operations')
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

def test_assured_workloads_service_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.assuredworkloads_v1.services.assured_workloads_service.transports.AssuredWorkloadsServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AssuredWorkloadsServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_assured_workloads_service_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.assuredworkloads_v1.services.assured_workloads_service.transports.AssuredWorkloadsServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AssuredWorkloadsServiceTransport()
        adc.assert_called_once()

def test_assured_workloads_service_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        AssuredWorkloadsServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.AssuredWorkloadsServiceGrpcTransport, transports.AssuredWorkloadsServiceGrpcAsyncIOTransport])
def test_assured_workloads_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.AssuredWorkloadsServiceGrpcTransport, transports.AssuredWorkloadsServiceGrpcAsyncIOTransport, transports.AssuredWorkloadsServiceRestTransport])
def test_assured_workloads_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.AssuredWorkloadsServiceGrpcTransport, grpc_helpers), (transports.AssuredWorkloadsServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_assured_workloads_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('assuredworkloads.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='assuredworkloads.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.AssuredWorkloadsServiceGrpcTransport, transports.AssuredWorkloadsServiceGrpcAsyncIOTransport])
def test_assured_workloads_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_assured_workloads_service_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.AssuredWorkloadsServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_assured_workloads_service_rest_lro_client():
    if False:
        i = 10
        return i + 15
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_assured_workloads_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='assuredworkloads.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('assuredworkloads.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://assuredworkloads.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_assured_workloads_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='assuredworkloads.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('assuredworkloads.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://assuredworkloads.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_assured_workloads_service_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = AssuredWorkloadsServiceClient(credentials=creds1, transport=transport_name)
    client2 = AssuredWorkloadsServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_workload._session
    session2 = client2.transport.create_workload._session
    assert session1 != session2
    session1 = client1.transport.update_workload._session
    session2 = client2.transport.update_workload._session
    assert session1 != session2
    session1 = client1.transport.restrict_allowed_resources._session
    session2 = client2.transport.restrict_allowed_resources._session
    assert session1 != session2
    session1 = client1.transport.delete_workload._session
    session2 = client2.transport.delete_workload._session
    assert session1 != session2
    session1 = client1.transport.get_workload._session
    session2 = client2.transport.get_workload._session
    assert session1 != session2
    session1 = client1.transport.list_workloads._session
    session2 = client2.transport.list_workloads._session
    assert session1 != session2
    session1 = client1.transport.list_violations._session
    session2 = client2.transport.list_violations._session
    assert session1 != session2
    session1 = client1.transport.get_violation._session
    session2 = client2.transport.get_violation._session
    assert session1 != session2
    session1 = client1.transport.acknowledge_violation._session
    session2 = client2.transport.acknowledge_violation._session
    assert session1 != session2

def test_assured_workloads_service_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AssuredWorkloadsServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_assured_workloads_service_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AssuredWorkloadsServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.AssuredWorkloadsServiceGrpcTransport, transports.AssuredWorkloadsServiceGrpcAsyncIOTransport])
def test_assured_workloads_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.AssuredWorkloadsServiceGrpcTransport, transports.AssuredWorkloadsServiceGrpcAsyncIOTransport])
def test_assured_workloads_service_transport_channel_mtls_with_adc(transport_class):
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

def test_assured_workloads_service_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_assured_workloads_service_grpc_lro_async_client():
    if False:
        return 10
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_violation_path():
    if False:
        return 10
    organization = 'squid'
    location = 'clam'
    workload = 'whelk'
    violation = 'octopus'
    expected = 'organizations/{organization}/locations/{location}/workloads/{workload}/violations/{violation}'.format(organization=organization, location=location, workload=workload, violation=violation)
    actual = AssuredWorkloadsServiceClient.violation_path(organization, location, workload, violation)
    assert expected == actual

def test_parse_violation_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'oyster', 'location': 'nudibranch', 'workload': 'cuttlefish', 'violation': 'mussel'}
    path = AssuredWorkloadsServiceClient.violation_path(**expected)
    actual = AssuredWorkloadsServiceClient.parse_violation_path(path)
    assert expected == actual

def test_workload_path():
    if False:
        i = 10
        return i + 15
    organization = 'winkle'
    location = 'nautilus'
    workload = 'scallop'
    expected = 'organizations/{organization}/locations/{location}/workloads/{workload}'.format(organization=organization, location=location, workload=workload)
    actual = AssuredWorkloadsServiceClient.workload_path(organization, location, workload)
    assert expected == actual

def test_parse_workload_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'abalone', 'location': 'squid', 'workload': 'clam'}
    path = AssuredWorkloadsServiceClient.workload_path(**expected)
    actual = AssuredWorkloadsServiceClient.parse_workload_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = AssuredWorkloadsServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'octopus'}
    path = AssuredWorkloadsServiceClient.common_billing_account_path(**expected)
    actual = AssuredWorkloadsServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = AssuredWorkloadsServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'nudibranch'}
    path = AssuredWorkloadsServiceClient.common_folder_path(**expected)
    actual = AssuredWorkloadsServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = AssuredWorkloadsServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'mussel'}
    path = AssuredWorkloadsServiceClient.common_organization_path(**expected)
    actual = AssuredWorkloadsServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = AssuredWorkloadsServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nautilus'}
    path = AssuredWorkloadsServiceClient.common_project_path(**expected)
    actual = AssuredWorkloadsServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = AssuredWorkloadsServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'squid', 'location': 'clam'}
    path = AssuredWorkloadsServiceClient.common_location_path(**expected)
    actual = AssuredWorkloadsServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.AssuredWorkloadsServiceTransport, '_prep_wrapped_messages') as prep:
        client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.AssuredWorkloadsServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = AssuredWorkloadsServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'organizations/sample1/locations/sample2/operations/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.GetOperationRequest, dict])
def test_get_operation_rest(request_type):
    if False:
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/locations/sample2/operations/sample3'}
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
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'organizations/sample1/locations/sample2'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_operations(request)

@pytest.mark.parametrize('request_type', [operations_pb2.ListOperationsRequest, dict])
def test_list_operations_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/locations/sample2'}
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

def test_get_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = AssuredWorkloadsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = AssuredWorkloadsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(AssuredWorkloadsServiceClient, transports.AssuredWorkloadsServiceGrpcTransport), (AssuredWorkloadsServiceAsyncClient, transports.AssuredWorkloadsServiceGrpcAsyncIOTransport)])
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