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
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.devtools.cloudbuild_v2.services.repository_manager import RepositoryManagerAsyncClient, RepositoryManagerClient, pagers, transports
from google.cloud.devtools.cloudbuild_v2.types import cloudbuild, repositories

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
        for i in range(10):
            print('nop')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert RepositoryManagerClient._get_default_mtls_endpoint(None) is None
    assert RepositoryManagerClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert RepositoryManagerClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert RepositoryManagerClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert RepositoryManagerClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert RepositoryManagerClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(RepositoryManagerClient, 'grpc'), (RepositoryManagerAsyncClient, 'grpc_asyncio'), (RepositoryManagerClient, 'rest')])
def test_repository_manager_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('cloudbuild.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudbuild.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.RepositoryManagerGrpcTransport, 'grpc'), (transports.RepositoryManagerGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.RepositoryManagerRestTransport, 'rest')])
def test_repository_manager_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(RepositoryManagerClient, 'grpc'), (RepositoryManagerAsyncClient, 'grpc_asyncio'), (RepositoryManagerClient, 'rest')])
def test_repository_manager_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('cloudbuild.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudbuild.googleapis.com')

def test_repository_manager_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = RepositoryManagerClient.get_transport_class()
    available_transports = [transports.RepositoryManagerGrpcTransport, transports.RepositoryManagerRestTransport]
    assert transport in available_transports
    transport = RepositoryManagerClient.get_transport_class('grpc')
    assert transport == transports.RepositoryManagerGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(RepositoryManagerClient, transports.RepositoryManagerGrpcTransport, 'grpc'), (RepositoryManagerAsyncClient, transports.RepositoryManagerGrpcAsyncIOTransport, 'grpc_asyncio'), (RepositoryManagerClient, transports.RepositoryManagerRestTransport, 'rest')])
@mock.patch.object(RepositoryManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RepositoryManagerClient))
@mock.patch.object(RepositoryManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RepositoryManagerAsyncClient))
def test_repository_manager_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(RepositoryManagerClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(RepositoryManagerClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(RepositoryManagerClient, transports.RepositoryManagerGrpcTransport, 'grpc', 'true'), (RepositoryManagerAsyncClient, transports.RepositoryManagerGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (RepositoryManagerClient, transports.RepositoryManagerGrpcTransport, 'grpc', 'false'), (RepositoryManagerAsyncClient, transports.RepositoryManagerGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (RepositoryManagerClient, transports.RepositoryManagerRestTransport, 'rest', 'true'), (RepositoryManagerClient, transports.RepositoryManagerRestTransport, 'rest', 'false')])
@mock.patch.object(RepositoryManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RepositoryManagerClient))
@mock.patch.object(RepositoryManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RepositoryManagerAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_repository_manager_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [RepositoryManagerClient, RepositoryManagerAsyncClient])
@mock.patch.object(RepositoryManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RepositoryManagerClient))
@mock.patch.object(RepositoryManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RepositoryManagerAsyncClient))
def test_repository_manager_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(RepositoryManagerClient, transports.RepositoryManagerGrpcTransport, 'grpc'), (RepositoryManagerAsyncClient, transports.RepositoryManagerGrpcAsyncIOTransport, 'grpc_asyncio'), (RepositoryManagerClient, transports.RepositoryManagerRestTransport, 'rest')])
def test_repository_manager_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(RepositoryManagerClient, transports.RepositoryManagerGrpcTransport, 'grpc', grpc_helpers), (RepositoryManagerAsyncClient, transports.RepositoryManagerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (RepositoryManagerClient, transports.RepositoryManagerRestTransport, 'rest', None)])
def test_repository_manager_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_repository_manager_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.devtools.cloudbuild_v2.services.repository_manager.transports.RepositoryManagerGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = RepositoryManagerClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(RepositoryManagerClient, transports.RepositoryManagerGrpcTransport, 'grpc', grpc_helpers), (RepositoryManagerAsyncClient, transports.RepositoryManagerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_repository_manager_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('cloudbuild.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='cloudbuild.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [repositories.CreateConnectionRequest, dict])
def test_create_connection(request_type, transport: str='grpc'):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.CreateConnectionRequest()
    assert isinstance(response, future.Future)

def test_create_connection_empty_call():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_connection), '__call__') as call:
        client.create_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.CreateConnectionRequest()

@pytest.mark.asyncio
async def test_create_connection_async(transport: str='grpc_asyncio', request_type=repositories.CreateConnectionRequest):
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.CreateConnectionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_connection_async_from_dict():
    await test_create_connection_async(request_type=dict)

def test_create_connection_field_headers():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.CreateConnectionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_connection_field_headers_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.CreateConnectionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_connection_flattened():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_connection(parent='parent_value', connection=repositories.Connection(name='name_value'), connection_id='connection_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].connection
        mock_val = repositories.Connection(name='name_value')
        assert arg == mock_val
        arg = args[0].connection_id
        mock_val = 'connection_id_value'
        assert arg == mock_val

def test_create_connection_flattened_error():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_connection(repositories.CreateConnectionRequest(), parent='parent_value', connection=repositories.Connection(name='name_value'), connection_id='connection_id_value')

@pytest.mark.asyncio
async def test_create_connection_flattened_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_connection(parent='parent_value', connection=repositories.Connection(name='name_value'), connection_id='connection_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].connection
        mock_val = repositories.Connection(name='name_value')
        assert arg == mock_val
        arg = args[0].connection_id
        mock_val = 'connection_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_connection_flattened_error_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_connection(repositories.CreateConnectionRequest(), parent='parent_value', connection=repositories.Connection(name='name_value'), connection_id='connection_id_value')

@pytest.mark.parametrize('request_type', [repositories.GetConnectionRequest, dict])
def test_get_connection(request_type, transport: str='grpc'):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_connection), '__call__') as call:
        call.return_value = repositories.Connection(name='name_value', disabled=True, reconciling=True, etag='etag_value')
        response = client.get_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.GetConnectionRequest()
    assert isinstance(response, repositories.Connection)
    assert response.name == 'name_value'
    assert response.disabled is True
    assert response.reconciling is True
    assert response.etag == 'etag_value'

def test_get_connection_empty_call():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_connection), '__call__') as call:
        client.get_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.GetConnectionRequest()

@pytest.mark.asyncio
async def test_get_connection_async(transport: str='grpc_asyncio', request_type=repositories.GetConnectionRequest):
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.Connection(name='name_value', disabled=True, reconciling=True, etag='etag_value'))
        response = await client.get_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.GetConnectionRequest()
    assert isinstance(response, repositories.Connection)
    assert response.name == 'name_value'
    assert response.disabled is True
    assert response.reconciling is True
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_connection_async_from_dict():
    await test_get_connection_async(request_type=dict)

def test_get_connection_field_headers():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.GetConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_connection), '__call__') as call:
        call.return_value = repositories.Connection()
        client.get_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_connection_field_headers_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.GetConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.Connection())
        await client.get_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_connection_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_connection), '__call__') as call:
        call.return_value = repositories.Connection()
        client.get_connection(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_connection_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_connection(repositories.GetConnectionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_connection_flattened_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_connection), '__call__') as call:
        call.return_value = repositories.Connection()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.Connection())
        response = await client.get_connection(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_connection_flattened_error_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_connection(repositories.GetConnectionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [repositories.ListConnectionsRequest, dict])
def test_list_connections(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.return_value = repositories.ListConnectionsResponse(next_page_token='next_page_token_value')
        response = client.list_connections(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.ListConnectionsRequest()
    assert isinstance(response, pagers.ListConnectionsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_connections_empty_call():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        client.list_connections()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.ListConnectionsRequest()

@pytest.mark.asyncio
async def test_list_connections_async(transport: str='grpc_asyncio', request_type=repositories.ListConnectionsRequest):
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.ListConnectionsResponse(next_page_token='next_page_token_value'))
        response = await client.list_connections(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.ListConnectionsRequest()
    assert isinstance(response, pagers.ListConnectionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_connections_async_from_dict():
    await test_list_connections_async(request_type=dict)

def test_list_connections_field_headers():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.ListConnectionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.return_value = repositories.ListConnectionsResponse()
        client.list_connections(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_connections_field_headers_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.ListConnectionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.ListConnectionsResponse())
        await client.list_connections(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_connections_flattened():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.return_value = repositories.ListConnectionsResponse()
        client.list_connections(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_connections_flattened_error():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_connections(repositories.ListConnectionsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_connections_flattened_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.return_value = repositories.ListConnectionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.ListConnectionsResponse())
        response = await client.list_connections(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_connections_flattened_error_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_connections(repositories.ListConnectionsRequest(), parent='parent_value')

def test_list_connections_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.side_effect = (repositories.ListConnectionsResponse(connections=[repositories.Connection(), repositories.Connection(), repositories.Connection()], next_page_token='abc'), repositories.ListConnectionsResponse(connections=[], next_page_token='def'), repositories.ListConnectionsResponse(connections=[repositories.Connection()], next_page_token='ghi'), repositories.ListConnectionsResponse(connections=[repositories.Connection(), repositories.Connection()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_connections(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, repositories.Connection) for i in results))

def test_list_connections_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.side_effect = (repositories.ListConnectionsResponse(connections=[repositories.Connection(), repositories.Connection(), repositories.Connection()], next_page_token='abc'), repositories.ListConnectionsResponse(connections=[], next_page_token='def'), repositories.ListConnectionsResponse(connections=[repositories.Connection()], next_page_token='ghi'), repositories.ListConnectionsResponse(connections=[repositories.Connection(), repositories.Connection()]), RuntimeError)
        pages = list(client.list_connections(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_connections_async_pager():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_connections), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (repositories.ListConnectionsResponse(connections=[repositories.Connection(), repositories.Connection(), repositories.Connection()], next_page_token='abc'), repositories.ListConnectionsResponse(connections=[], next_page_token='def'), repositories.ListConnectionsResponse(connections=[repositories.Connection()], next_page_token='ghi'), repositories.ListConnectionsResponse(connections=[repositories.Connection(), repositories.Connection()]), RuntimeError)
        async_pager = await client.list_connections(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, repositories.Connection) for i in responses))

@pytest.mark.asyncio
async def test_list_connections_async_pages():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_connections), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (repositories.ListConnectionsResponse(connections=[repositories.Connection(), repositories.Connection(), repositories.Connection()], next_page_token='abc'), repositories.ListConnectionsResponse(connections=[], next_page_token='def'), repositories.ListConnectionsResponse(connections=[repositories.Connection()], next_page_token='ghi'), repositories.ListConnectionsResponse(connections=[repositories.Connection(), repositories.Connection()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_connections(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [repositories.UpdateConnectionRequest, dict])
def test_update_connection(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.UpdateConnectionRequest()
    assert isinstance(response, future.Future)

def test_update_connection_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_connection), '__call__') as call:
        client.update_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.UpdateConnectionRequest()

@pytest.mark.asyncio
async def test_update_connection_async(transport: str='grpc_asyncio', request_type=repositories.UpdateConnectionRequest):
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.UpdateConnectionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_connection_async_from_dict():
    await test_update_connection_async(request_type=dict)

def test_update_connection_field_headers():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.UpdateConnectionRequest()
    request.connection.name = 'name_value'
    with mock.patch.object(type(client.transport.update_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'connection.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_connection_field_headers_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.UpdateConnectionRequest()
    request.connection.name = 'name_value'
    with mock.patch.object(type(client.transport.update_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'connection.name=name_value') in kw['metadata']

def test_update_connection_flattened():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_connection(connection=repositories.Connection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].connection
        mock_val = repositories.Connection(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_connection_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_connection(repositories.UpdateConnectionRequest(), connection=repositories.Connection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_connection_flattened_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_connection(connection=repositories.Connection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].connection
        mock_val = repositories.Connection(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_connection_flattened_error_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_connection(repositories.UpdateConnectionRequest(), connection=repositories.Connection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [repositories.DeleteConnectionRequest, dict])
def test_delete_connection(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.DeleteConnectionRequest()
    assert isinstance(response, future.Future)

def test_delete_connection_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_connection), '__call__') as call:
        client.delete_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.DeleteConnectionRequest()

@pytest.mark.asyncio
async def test_delete_connection_async(transport: str='grpc_asyncio', request_type=repositories.DeleteConnectionRequest):
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.DeleteConnectionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_connection_async_from_dict():
    await test_delete_connection_async(request_type=dict)

def test_delete_connection_field_headers():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.DeleteConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_connection_field_headers_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.DeleteConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_connection_flattened():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_connection(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_connection_flattened_error():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_connection(repositories.DeleteConnectionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_connection_flattened_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_connection(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_connection_flattened_error_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_connection(repositories.DeleteConnectionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [repositories.CreateRepositoryRequest, dict])
def test_create_repository(request_type, transport: str='grpc'):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_repository(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.CreateRepositoryRequest()
    assert isinstance(response, future.Future)

def test_create_repository_empty_call():
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_repository), '__call__') as call:
        client.create_repository()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.CreateRepositoryRequest()

@pytest.mark.asyncio
async def test_create_repository_async(transport: str='grpc_asyncio', request_type=repositories.CreateRepositoryRequest):
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_repository), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_repository(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.CreateRepositoryRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_repository_async_from_dict():
    await test_create_repository_async(request_type=dict)

def test_create_repository_field_headers():
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.CreateRepositoryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_repository(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_repository_field_headers_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.CreateRepositoryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_repository), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_repository(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_repository_flattened():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_repository(parent='parent_value', repository=repositories.Repository(name='name_value'), repository_id='repository_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].repository
        mock_val = repositories.Repository(name='name_value')
        assert arg == mock_val
        arg = args[0].repository_id
        mock_val = 'repository_id_value'
        assert arg == mock_val

def test_create_repository_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_repository(repositories.CreateRepositoryRequest(), parent='parent_value', repository=repositories.Repository(name='name_value'), repository_id='repository_id_value')

@pytest.mark.asyncio
async def test_create_repository_flattened_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_repository(parent='parent_value', repository=repositories.Repository(name='name_value'), repository_id='repository_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].repository
        mock_val = repositories.Repository(name='name_value')
        assert arg == mock_val
        arg = args[0].repository_id
        mock_val = 'repository_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_repository_flattened_error_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_repository(repositories.CreateRepositoryRequest(), parent='parent_value', repository=repositories.Repository(name='name_value'), repository_id='repository_id_value')

@pytest.mark.parametrize('request_type', [repositories.BatchCreateRepositoriesRequest, dict])
def test_batch_create_repositories(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_repositories), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_create_repositories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.BatchCreateRepositoriesRequest()
    assert isinstance(response, future.Future)

def test_batch_create_repositories_empty_call():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_create_repositories), '__call__') as call:
        client.batch_create_repositories()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.BatchCreateRepositoriesRequest()

@pytest.mark.asyncio
async def test_batch_create_repositories_async(transport: str='grpc_asyncio', request_type=repositories.BatchCreateRepositoriesRequest):
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_repositories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_create_repositories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.BatchCreateRepositoriesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_create_repositories_async_from_dict():
    await test_batch_create_repositories_async(request_type=dict)

def test_batch_create_repositories_field_headers():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.BatchCreateRepositoriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_create_repositories), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_create_repositories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_create_repositories_field_headers_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.BatchCreateRepositoriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_create_repositories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_create_repositories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_batch_create_repositories_flattened():
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_create_repositories), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_create_repositories(parent='parent_value', requests=[repositories.CreateRepositoryRequest(parent='parent_value')])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].requests
        mock_val = [repositories.CreateRepositoryRequest(parent='parent_value')]
        assert arg == mock_val

def test_batch_create_repositories_flattened_error():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_create_repositories(repositories.BatchCreateRepositoriesRequest(), parent='parent_value', requests=[repositories.CreateRepositoryRequest(parent='parent_value')])

@pytest.mark.asyncio
async def test_batch_create_repositories_flattened_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_create_repositories), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_create_repositories(parent='parent_value', requests=[repositories.CreateRepositoryRequest(parent='parent_value')])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].requests
        mock_val = [repositories.CreateRepositoryRequest(parent='parent_value')]
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_create_repositories_flattened_error_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_create_repositories(repositories.BatchCreateRepositoriesRequest(), parent='parent_value', requests=[repositories.CreateRepositoryRequest(parent='parent_value')])

@pytest.mark.parametrize('request_type', [repositories.GetRepositoryRequest, dict])
def test_get_repository(request_type, transport: str='grpc'):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_repository), '__call__') as call:
        call.return_value = repositories.Repository(name='name_value', remote_uri='remote_uri_value', etag='etag_value', webhook_id='webhook_id_value')
        response = client.get_repository(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.GetRepositoryRequest()
    assert isinstance(response, repositories.Repository)
    assert response.name == 'name_value'
    assert response.remote_uri == 'remote_uri_value'
    assert response.etag == 'etag_value'
    assert response.webhook_id == 'webhook_id_value'

def test_get_repository_empty_call():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_repository), '__call__') as call:
        client.get_repository()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.GetRepositoryRequest()

@pytest.mark.asyncio
async def test_get_repository_async(transport: str='grpc_asyncio', request_type=repositories.GetRepositoryRequest):
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_repository), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.Repository(name='name_value', remote_uri='remote_uri_value', etag='etag_value', webhook_id='webhook_id_value'))
        response = await client.get_repository(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.GetRepositoryRequest()
    assert isinstance(response, repositories.Repository)
    assert response.name == 'name_value'
    assert response.remote_uri == 'remote_uri_value'
    assert response.etag == 'etag_value'
    assert response.webhook_id == 'webhook_id_value'

@pytest.mark.asyncio
async def test_get_repository_async_from_dict():
    await test_get_repository_async(request_type=dict)

def test_get_repository_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.GetRepositoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_repository), '__call__') as call:
        call.return_value = repositories.Repository()
        client.get_repository(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_repository_field_headers_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.GetRepositoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_repository), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.Repository())
        await client.get_repository(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_repository_flattened():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_repository), '__call__') as call:
        call.return_value = repositories.Repository()
        client.get_repository(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_repository_flattened_error():
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_repository(repositories.GetRepositoryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_repository_flattened_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_repository), '__call__') as call:
        call.return_value = repositories.Repository()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.Repository())
        response = await client.get_repository(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_repository_flattened_error_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_repository(repositories.GetRepositoryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [repositories.ListRepositoriesRequest, dict])
def test_list_repositories(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.return_value = repositories.ListRepositoriesResponse(next_page_token='next_page_token_value')
        response = client.list_repositories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.ListRepositoriesRequest()
    assert isinstance(response, pagers.ListRepositoriesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_repositories_empty_call():
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        client.list_repositories()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.ListRepositoriesRequest()

@pytest.mark.asyncio
async def test_list_repositories_async(transport: str='grpc_asyncio', request_type=repositories.ListRepositoriesRequest):
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.ListRepositoriesResponse(next_page_token='next_page_token_value'))
        response = await client.list_repositories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.ListRepositoriesRequest()
    assert isinstance(response, pagers.ListRepositoriesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_repositories_async_from_dict():
    await test_list_repositories_async(request_type=dict)

def test_list_repositories_field_headers():
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.ListRepositoriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.return_value = repositories.ListRepositoriesResponse()
        client.list_repositories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_repositories_field_headers_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.ListRepositoriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.ListRepositoriesResponse())
        await client.list_repositories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_repositories_flattened():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.return_value = repositories.ListRepositoriesResponse()
        client.list_repositories(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_repositories_flattened_error():
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_repositories(repositories.ListRepositoriesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_repositories_flattened_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.return_value = repositories.ListRepositoriesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.ListRepositoriesResponse())
        response = await client.list_repositories(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_repositories_flattened_error_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_repositories(repositories.ListRepositoriesRequest(), parent='parent_value')

def test_list_repositories_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.side_effect = (repositories.ListRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository(), repositories.Repository()], next_page_token='abc'), repositories.ListRepositoriesResponse(repositories=[], next_page_token='def'), repositories.ListRepositoriesResponse(repositories=[repositories.Repository()], next_page_token='ghi'), repositories.ListRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_repositories(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, repositories.Repository) for i in results))

def test_list_repositories_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_repositories), '__call__') as call:
        call.side_effect = (repositories.ListRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository(), repositories.Repository()], next_page_token='abc'), repositories.ListRepositoriesResponse(repositories=[], next_page_token='def'), repositories.ListRepositoriesResponse(repositories=[repositories.Repository()], next_page_token='ghi'), repositories.ListRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository()]), RuntimeError)
        pages = list(client.list_repositories(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_repositories_async_pager():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_repositories), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (repositories.ListRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository(), repositories.Repository()], next_page_token='abc'), repositories.ListRepositoriesResponse(repositories=[], next_page_token='def'), repositories.ListRepositoriesResponse(repositories=[repositories.Repository()], next_page_token='ghi'), repositories.ListRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository()]), RuntimeError)
        async_pager = await client.list_repositories(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, repositories.Repository) for i in responses))

@pytest.mark.asyncio
async def test_list_repositories_async_pages():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_repositories), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (repositories.ListRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository(), repositories.Repository()], next_page_token='abc'), repositories.ListRepositoriesResponse(repositories=[], next_page_token='def'), repositories.ListRepositoriesResponse(repositories=[repositories.Repository()], next_page_token='ghi'), repositories.ListRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_repositories(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [repositories.DeleteRepositoryRequest, dict])
def test_delete_repository(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_repository(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.DeleteRepositoryRequest()
    assert isinstance(response, future.Future)

def test_delete_repository_empty_call():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_repository), '__call__') as call:
        client.delete_repository()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.DeleteRepositoryRequest()

@pytest.mark.asyncio
async def test_delete_repository_async(transport: str='grpc_asyncio', request_type=repositories.DeleteRepositoryRequest):
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_repository), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_repository(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.DeleteRepositoryRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_repository_async_from_dict():
    await test_delete_repository_async(request_type=dict)

def test_delete_repository_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.DeleteRepositoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_repository(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_repository_field_headers_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.DeleteRepositoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_repository), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_repository(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_repository_flattened():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_repository(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_repository_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_repository(repositories.DeleteRepositoryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_repository_flattened_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_repository), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_repository(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_repository_flattened_error_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_repository(repositories.DeleteRepositoryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [repositories.FetchReadWriteTokenRequest, dict])
def test_fetch_read_write_token(request_type, transport: str='grpc'):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.fetch_read_write_token), '__call__') as call:
        call.return_value = repositories.FetchReadWriteTokenResponse(token='token_value')
        response = client.fetch_read_write_token(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.FetchReadWriteTokenRequest()
    assert isinstance(response, repositories.FetchReadWriteTokenResponse)
    assert response.token == 'token_value'

def test_fetch_read_write_token_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.fetch_read_write_token), '__call__') as call:
        client.fetch_read_write_token()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.FetchReadWriteTokenRequest()

@pytest.mark.asyncio
async def test_fetch_read_write_token_async(transport: str='grpc_asyncio', request_type=repositories.FetchReadWriteTokenRequest):
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.fetch_read_write_token), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.FetchReadWriteTokenResponse(token='token_value'))
        response = await client.fetch_read_write_token(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.FetchReadWriteTokenRequest()
    assert isinstance(response, repositories.FetchReadWriteTokenResponse)
    assert response.token == 'token_value'

@pytest.mark.asyncio
async def test_fetch_read_write_token_async_from_dict():
    await test_fetch_read_write_token_async(request_type=dict)

def test_fetch_read_write_token_field_headers():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.FetchReadWriteTokenRequest()
    request.repository = 'repository_value'
    with mock.patch.object(type(client.transport.fetch_read_write_token), '__call__') as call:
        call.return_value = repositories.FetchReadWriteTokenResponse()
        client.fetch_read_write_token(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'repository=repository_value') in kw['metadata']

@pytest.mark.asyncio
async def test_fetch_read_write_token_field_headers_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.FetchReadWriteTokenRequest()
    request.repository = 'repository_value'
    with mock.patch.object(type(client.transport.fetch_read_write_token), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.FetchReadWriteTokenResponse())
        await client.fetch_read_write_token(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'repository=repository_value') in kw['metadata']

def test_fetch_read_write_token_flattened():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.fetch_read_write_token), '__call__') as call:
        call.return_value = repositories.FetchReadWriteTokenResponse()
        client.fetch_read_write_token(repository='repository_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].repository
        mock_val = 'repository_value'
        assert arg == mock_val

def test_fetch_read_write_token_flattened_error():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.fetch_read_write_token(repositories.FetchReadWriteTokenRequest(), repository='repository_value')

@pytest.mark.asyncio
async def test_fetch_read_write_token_flattened_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.fetch_read_write_token), '__call__') as call:
        call.return_value = repositories.FetchReadWriteTokenResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.FetchReadWriteTokenResponse())
        response = await client.fetch_read_write_token(repository='repository_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].repository
        mock_val = 'repository_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_fetch_read_write_token_flattened_error_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.fetch_read_write_token(repositories.FetchReadWriteTokenRequest(), repository='repository_value')

@pytest.mark.parametrize('request_type', [repositories.FetchReadTokenRequest, dict])
def test_fetch_read_token(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.fetch_read_token), '__call__') as call:
        call.return_value = repositories.FetchReadTokenResponse(token='token_value')
        response = client.fetch_read_token(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.FetchReadTokenRequest()
    assert isinstance(response, repositories.FetchReadTokenResponse)
    assert response.token == 'token_value'

def test_fetch_read_token_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.fetch_read_token), '__call__') as call:
        client.fetch_read_token()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.FetchReadTokenRequest()

@pytest.mark.asyncio
async def test_fetch_read_token_async(transport: str='grpc_asyncio', request_type=repositories.FetchReadTokenRequest):
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.fetch_read_token), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.FetchReadTokenResponse(token='token_value'))
        response = await client.fetch_read_token(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.FetchReadTokenRequest()
    assert isinstance(response, repositories.FetchReadTokenResponse)
    assert response.token == 'token_value'

@pytest.mark.asyncio
async def test_fetch_read_token_async_from_dict():
    await test_fetch_read_token_async(request_type=dict)

def test_fetch_read_token_field_headers():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.FetchReadTokenRequest()
    request.repository = 'repository_value'
    with mock.patch.object(type(client.transport.fetch_read_token), '__call__') as call:
        call.return_value = repositories.FetchReadTokenResponse()
        client.fetch_read_token(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'repository=repository_value') in kw['metadata']

@pytest.mark.asyncio
async def test_fetch_read_token_field_headers_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.FetchReadTokenRequest()
    request.repository = 'repository_value'
    with mock.patch.object(type(client.transport.fetch_read_token), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.FetchReadTokenResponse())
        await client.fetch_read_token(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'repository=repository_value') in kw['metadata']

def test_fetch_read_token_flattened():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.fetch_read_token), '__call__') as call:
        call.return_value = repositories.FetchReadTokenResponse()
        client.fetch_read_token(repository='repository_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].repository
        mock_val = 'repository_value'
        assert arg == mock_val

def test_fetch_read_token_flattened_error():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.fetch_read_token(repositories.FetchReadTokenRequest(), repository='repository_value')

@pytest.mark.asyncio
async def test_fetch_read_token_flattened_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.fetch_read_token), '__call__') as call:
        call.return_value = repositories.FetchReadTokenResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.FetchReadTokenResponse())
        response = await client.fetch_read_token(repository='repository_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].repository
        mock_val = 'repository_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_fetch_read_token_flattened_error_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.fetch_read_token(repositories.FetchReadTokenRequest(), repository='repository_value')

@pytest.mark.parametrize('request_type', [repositories.FetchLinkableRepositoriesRequest, dict])
def test_fetch_linkable_repositories(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.fetch_linkable_repositories), '__call__') as call:
        call.return_value = repositories.FetchLinkableRepositoriesResponse(next_page_token='next_page_token_value')
        response = client.fetch_linkable_repositories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.FetchLinkableRepositoriesRequest()
    assert isinstance(response, pagers.FetchLinkableRepositoriesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_fetch_linkable_repositories_empty_call():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.fetch_linkable_repositories), '__call__') as call:
        client.fetch_linkable_repositories()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.FetchLinkableRepositoriesRequest()

@pytest.mark.asyncio
async def test_fetch_linkable_repositories_async(transport: str='grpc_asyncio', request_type=repositories.FetchLinkableRepositoriesRequest):
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.fetch_linkable_repositories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.FetchLinkableRepositoriesResponse(next_page_token='next_page_token_value'))
        response = await client.fetch_linkable_repositories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.FetchLinkableRepositoriesRequest()
    assert isinstance(response, pagers.FetchLinkableRepositoriesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_fetch_linkable_repositories_async_from_dict():
    await test_fetch_linkable_repositories_async(request_type=dict)

def test_fetch_linkable_repositories_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.FetchLinkableRepositoriesRequest()
    request.connection = 'connection_value'
    with mock.patch.object(type(client.transport.fetch_linkable_repositories), '__call__') as call:
        call.return_value = repositories.FetchLinkableRepositoriesResponse()
        client.fetch_linkable_repositories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'connection=connection_value') in kw['metadata']

@pytest.mark.asyncio
async def test_fetch_linkable_repositories_field_headers_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.FetchLinkableRepositoriesRequest()
    request.connection = 'connection_value'
    with mock.patch.object(type(client.transport.fetch_linkable_repositories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.FetchLinkableRepositoriesResponse())
        await client.fetch_linkable_repositories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'connection=connection_value') in kw['metadata']

def test_fetch_linkable_repositories_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.fetch_linkable_repositories), '__call__') as call:
        call.side_effect = (repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository(), repositories.Repository()], next_page_token='abc'), repositories.FetchLinkableRepositoriesResponse(repositories=[], next_page_token='def'), repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository()], next_page_token='ghi'), repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('connection', ''),)),)
        pager = client.fetch_linkable_repositories(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, repositories.Repository) for i in results))

def test_fetch_linkable_repositories_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.fetch_linkable_repositories), '__call__') as call:
        call.side_effect = (repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository(), repositories.Repository()], next_page_token='abc'), repositories.FetchLinkableRepositoriesResponse(repositories=[], next_page_token='def'), repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository()], next_page_token='ghi'), repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository()]), RuntimeError)
        pages = list(client.fetch_linkable_repositories(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_fetch_linkable_repositories_async_pager():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.fetch_linkable_repositories), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository(), repositories.Repository()], next_page_token='abc'), repositories.FetchLinkableRepositoriesResponse(repositories=[], next_page_token='def'), repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository()], next_page_token='ghi'), repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository()]), RuntimeError)
        async_pager = await client.fetch_linkable_repositories(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, repositories.Repository) for i in responses))

@pytest.mark.asyncio
async def test_fetch_linkable_repositories_async_pages():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.fetch_linkable_repositories), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository(), repositories.Repository()], next_page_token='abc'), repositories.FetchLinkableRepositoriesResponse(repositories=[], next_page_token='def'), repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository()], next_page_token='ghi'), repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository()]), RuntimeError)
        pages = []
        async for page_ in (await client.fetch_linkable_repositories(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [repositories.FetchGitRefsRequest, dict])
def test_fetch_git_refs(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.fetch_git_refs), '__call__') as call:
        call.return_value = repositories.FetchGitRefsResponse(ref_names=['ref_names_value'])
        response = client.fetch_git_refs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.FetchGitRefsRequest()
    assert isinstance(response, repositories.FetchGitRefsResponse)
    assert response.ref_names == ['ref_names_value']

def test_fetch_git_refs_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.fetch_git_refs), '__call__') as call:
        client.fetch_git_refs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.FetchGitRefsRequest()

@pytest.mark.asyncio
async def test_fetch_git_refs_async(transport: str='grpc_asyncio', request_type=repositories.FetchGitRefsRequest):
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.fetch_git_refs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.FetchGitRefsResponse(ref_names=['ref_names_value']))
        response = await client.fetch_git_refs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == repositories.FetchGitRefsRequest()
    assert isinstance(response, repositories.FetchGitRefsResponse)
    assert response.ref_names == ['ref_names_value']

@pytest.mark.asyncio
async def test_fetch_git_refs_async_from_dict():
    await test_fetch_git_refs_async(request_type=dict)

def test_fetch_git_refs_field_headers():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.FetchGitRefsRequest()
    request.repository = 'repository_value'
    with mock.patch.object(type(client.transport.fetch_git_refs), '__call__') as call:
        call.return_value = repositories.FetchGitRefsResponse()
        client.fetch_git_refs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'repository=repository_value') in kw['metadata']

@pytest.mark.asyncio
async def test_fetch_git_refs_field_headers_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = repositories.FetchGitRefsRequest()
    request.repository = 'repository_value'
    with mock.patch.object(type(client.transport.fetch_git_refs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.FetchGitRefsResponse())
        await client.fetch_git_refs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'repository=repository_value') in kw['metadata']

def test_fetch_git_refs_flattened():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.fetch_git_refs), '__call__') as call:
        call.return_value = repositories.FetchGitRefsResponse()
        client.fetch_git_refs(repository='repository_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].repository
        mock_val = 'repository_value'
        assert arg == mock_val

def test_fetch_git_refs_flattened_error():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.fetch_git_refs(repositories.FetchGitRefsRequest(), repository='repository_value')

@pytest.mark.asyncio
async def test_fetch_git_refs_flattened_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.fetch_git_refs), '__call__') as call:
        call.return_value = repositories.FetchGitRefsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(repositories.FetchGitRefsResponse())
        response = await client.fetch_git_refs(repository='repository_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].repository
        mock_val = 'repository_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_fetch_git_refs_flattened_error_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.fetch_git_refs(repositories.FetchGitRefsRequest(), repository='repository_value')

@pytest.mark.parametrize('request_type', [repositories.CreateConnectionRequest, dict])
def test_create_connection_rest(request_type):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['connection'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'github_config': {'authorizer_credential': {'oauth_token_secret_version': 'oauth_token_secret_version_value', 'username': 'username_value'}, 'app_installation_id': 2014}, 'github_enterprise_config': {'host_uri': 'host_uri_value', 'api_key': 'api_key_value', 'app_id': 621, 'app_slug': 'app_slug_value', 'private_key_secret_version': 'private_key_secret_version_value', 'webhook_secret_secret_version': 'webhook_secret_secret_version_value', 'app_installation_id': 2014, 'service_directory_config': {'service': 'service_value'}, 'ssl_ca': 'ssl_ca_value', 'server_version': 'server_version_value'}, 'gitlab_config': {'host_uri': 'host_uri_value', 'webhook_secret_secret_version': 'webhook_secret_secret_version_value', 'read_authorizer_credential': {'user_token_secret_version': 'user_token_secret_version_value', 'username': 'username_value'}, 'authorizer_credential': {}, 'service_directory_config': {}, 'ssl_ca': 'ssl_ca_value', 'server_version': 'server_version_value'}, 'installation_state': {'stage': 1, 'message': 'message_value', 'action_uri': 'action_uri_value'}, 'disabled': True, 'reconciling': True, 'annotations': {}, 'etag': 'etag_value'}
    test_field = repositories.CreateConnectionRequest.meta.fields['connection']

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
    for (field, value) in request_init['connection'].items():
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
                for i in range(0, len(request_init['connection'][field])):
                    del request_init['connection'][field][i][subfield]
            else:
                del request_init['connection'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_connection(request)
    assert response.operation.name == 'operations/spam'

def test_create_connection_rest_required_fields(request_type=repositories.CreateConnectionRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RepositoryManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['connection_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'connectionId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'connectionId' in jsonified_request
    assert jsonified_request['connectionId'] == request_init['connection_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['connectionId'] = 'connection_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_connection._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('connection_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'connectionId' in jsonified_request
    assert jsonified_request['connectionId'] == 'connection_id_value'
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_connection(request)
            expected_params = [('connectionId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_connection_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(('connectionId',)) & set(('parent', 'connection', 'connectionId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_connection_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RepositoryManagerRestInterceptor())
    client = RepositoryManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.RepositoryManagerRestInterceptor, 'post_create_connection') as post, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'pre_create_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = repositories.CreateConnectionRequest.pb(repositories.CreateConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = repositories.CreateConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_connection_rest_bad_request(transport: str='rest', request_type=repositories.CreateConnectionRequest):
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_connection(request)

def test_create_connection_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', connection=repositories.Connection(name='name_value'), connection_id='connection_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/connections' % client.transport._host, args[1])

def test_create_connection_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_connection(repositories.CreateConnectionRequest(), parent='parent_value', connection=repositories.Connection(name='name_value'), connection_id='connection_id_value')

def test_create_connection_rest_error():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [repositories.GetConnectionRequest, dict])
def test_get_connection_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.Connection(name='name_value', disabled=True, reconciling=True, etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.Connection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_connection(request)
    assert isinstance(response, repositories.Connection)
    assert response.name == 'name_value'
    assert response.disabled is True
    assert response.reconciling is True
    assert response.etag == 'etag_value'

def test_get_connection_rest_required_fields(request_type=repositories.GetConnectionRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RepositoryManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = repositories.Connection()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = repositories.Connection.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_connection_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_connection_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RepositoryManagerRestInterceptor())
    client = RepositoryManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'post_get_connection') as post, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'pre_get_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = repositories.GetConnectionRequest.pb(repositories.GetConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = repositories.Connection.to_json(repositories.Connection())
        request = repositories.GetConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = repositories.Connection()
        client.get_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_connection_rest_bad_request(transport: str='rest', request_type=repositories.GetConnectionRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_connection(request)

def test_get_connection_rest_flattened():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.Connection()
        sample_request = {'name': 'projects/sample1/locations/sample2/connections/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.Connection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/connections/*}' % client.transport._host, args[1])

def test_get_connection_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_connection(repositories.GetConnectionRequest(), name='name_value')

def test_get_connection_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [repositories.ListConnectionsRequest, dict])
def test_list_connections_rest(request_type):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.ListConnectionsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.ListConnectionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_connections(request)
    assert isinstance(response, pagers.ListConnectionsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_connections_rest_required_fields(request_type=repositories.ListConnectionsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RepositoryManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_connections._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_connections._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = repositories.ListConnectionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = repositories.ListConnectionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_connections(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_connections_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_connections._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_connections_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RepositoryManagerRestInterceptor())
    client = RepositoryManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'post_list_connections') as post, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'pre_list_connections') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = repositories.ListConnectionsRequest.pb(repositories.ListConnectionsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = repositories.ListConnectionsResponse.to_json(repositories.ListConnectionsResponse())
        request = repositories.ListConnectionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = repositories.ListConnectionsResponse()
        client.list_connections(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_connections_rest_bad_request(transport: str='rest', request_type=repositories.ListConnectionsRequest):
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_connections(request)

def test_list_connections_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.ListConnectionsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.ListConnectionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_connections(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/connections' % client.transport._host, args[1])

def test_list_connections_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_connections(repositories.ListConnectionsRequest(), parent='parent_value')

def test_list_connections_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (repositories.ListConnectionsResponse(connections=[repositories.Connection(), repositories.Connection(), repositories.Connection()], next_page_token='abc'), repositories.ListConnectionsResponse(connections=[], next_page_token='def'), repositories.ListConnectionsResponse(connections=[repositories.Connection()], next_page_token='ghi'), repositories.ListConnectionsResponse(connections=[repositories.Connection(), repositories.Connection()]))
        response = response + response
        response = tuple((repositories.ListConnectionsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_connections(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, repositories.Connection) for i in results))
        pages = list(client.list_connections(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [repositories.UpdateConnectionRequest, dict])
def test_update_connection_rest(request_type):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'connection': {'name': 'projects/sample1/locations/sample2/connections/sample3'}}
    request_init['connection'] = {'name': 'projects/sample1/locations/sample2/connections/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'github_config': {'authorizer_credential': {'oauth_token_secret_version': 'oauth_token_secret_version_value', 'username': 'username_value'}, 'app_installation_id': 2014}, 'github_enterprise_config': {'host_uri': 'host_uri_value', 'api_key': 'api_key_value', 'app_id': 621, 'app_slug': 'app_slug_value', 'private_key_secret_version': 'private_key_secret_version_value', 'webhook_secret_secret_version': 'webhook_secret_secret_version_value', 'app_installation_id': 2014, 'service_directory_config': {'service': 'service_value'}, 'ssl_ca': 'ssl_ca_value', 'server_version': 'server_version_value'}, 'gitlab_config': {'host_uri': 'host_uri_value', 'webhook_secret_secret_version': 'webhook_secret_secret_version_value', 'read_authorizer_credential': {'user_token_secret_version': 'user_token_secret_version_value', 'username': 'username_value'}, 'authorizer_credential': {}, 'service_directory_config': {}, 'ssl_ca': 'ssl_ca_value', 'server_version': 'server_version_value'}, 'installation_state': {'stage': 1, 'message': 'message_value', 'action_uri': 'action_uri_value'}, 'disabled': True, 'reconciling': True, 'annotations': {}, 'etag': 'etag_value'}
    test_field = repositories.UpdateConnectionRequest.meta.fields['connection']

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
    for (field, value) in request_init['connection'].items():
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
                for i in range(0, len(request_init['connection'][field])):
                    del request_init['connection'][field][i][subfield]
            else:
                del request_init['connection'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_connection(request)
    assert response.operation.name == 'operations/spam'

def test_update_connection_rest_required_fields(request_type=repositories.UpdateConnectionRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RepositoryManagerRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_connection._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'etag', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_connection_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'etag', 'updateMask')) & set(('connection',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_connection_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RepositoryManagerRestInterceptor())
    client = RepositoryManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.RepositoryManagerRestInterceptor, 'post_update_connection') as post, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'pre_update_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = repositories.UpdateConnectionRequest.pb(repositories.UpdateConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = repositories.UpdateConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_connection_rest_bad_request(transport: str='rest', request_type=repositories.UpdateConnectionRequest):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'connection': {'name': 'projects/sample1/locations/sample2/connections/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_connection(request)

def test_update_connection_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'connection': {'name': 'projects/sample1/locations/sample2/connections/sample3'}}
        mock_args = dict(connection=repositories.Connection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{connection.name=projects/*/locations/*/connections/*}' % client.transport._host, args[1])

def test_update_connection_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_connection(repositories.UpdateConnectionRequest(), connection=repositories.Connection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_connection_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [repositories.DeleteConnectionRequest, dict])
def test_delete_connection_rest(request_type):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_connection(request)
    assert response.operation.name == 'operations/spam'

def test_delete_connection_rest_required_fields(request_type=repositories.DeleteConnectionRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RepositoryManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_connection._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('etag', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_connection_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(('etag', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_connection_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RepositoryManagerRestInterceptor())
    client = RepositoryManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.RepositoryManagerRestInterceptor, 'post_delete_connection') as post, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'pre_delete_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = repositories.DeleteConnectionRequest.pb(repositories.DeleteConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = repositories.DeleteConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_connection_rest_bad_request(transport: str='rest', request_type=repositories.DeleteConnectionRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_connection(request)

def test_delete_connection_rest_flattened():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/connections/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/connections/*}' % client.transport._host, args[1])

def test_delete_connection_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_connection(repositories.DeleteConnectionRequest(), name='name_value')

def test_delete_connection_rest_error():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [repositories.CreateRepositoryRequest, dict])
def test_create_repository_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/connections/sample3'}
    request_init['repository'] = {'name': 'name_value', 'remote_uri': 'remote_uri_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'annotations': {}, 'etag': 'etag_value', 'webhook_id': 'webhook_id_value'}
    test_field = repositories.CreateRepositoryRequest.meta.fields['repository']

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
    for (field, value) in request_init['repository'].items():
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
                for i in range(0, len(request_init['repository'][field])):
                    del request_init['repository'][field][i][subfield]
            else:
                del request_init['repository'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_repository(request)
    assert response.operation.name == 'operations/spam'

def test_create_repository_rest_required_fields(request_type=repositories.CreateRepositoryRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RepositoryManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['repository_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'repositoryId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_repository._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'repositoryId' in jsonified_request
    assert jsonified_request['repositoryId'] == request_init['repository_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['repositoryId'] = 'repository_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_repository._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('repository_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'repositoryId' in jsonified_request
    assert jsonified_request['repositoryId'] == 'repository_id_value'
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_repository(request)
            expected_params = [('repositoryId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_repository_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_repository._get_unset_required_fields({})
    assert set(unset_fields) == set(('repositoryId',)) & set(('parent', 'repository', 'repositoryId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_repository_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RepositoryManagerRestInterceptor())
    client = RepositoryManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.RepositoryManagerRestInterceptor, 'post_create_repository') as post, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'pre_create_repository') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = repositories.CreateRepositoryRequest.pb(repositories.CreateRepositoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = repositories.CreateRepositoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_repository(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_repository_rest_bad_request(transport: str='rest', request_type=repositories.CreateRepositoryRequest):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_repository(request)

def test_create_repository_rest_flattened():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/connections/sample3'}
        mock_args = dict(parent='parent_value', repository=repositories.Repository(name='name_value'), repository_id='repository_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_repository(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*/connections/*}/repositories' % client.transport._host, args[1])

def test_create_repository_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_repository(repositories.CreateRepositoryRequest(), parent='parent_value', repository=repositories.Repository(name='name_value'), repository_id='repository_id_value')

def test_create_repository_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [repositories.BatchCreateRepositoriesRequest, dict])
def test_batch_create_repositories_rest(request_type):
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_create_repositories(request)
    assert response.operation.name == 'operations/spam'

def test_batch_create_repositories_rest_required_fields(request_type=repositories.BatchCreateRepositoriesRequest):
    if False:
        return 10
    transport_class = transports.RepositoryManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_repositories._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_repositories._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_create_repositories(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_create_repositories_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_create_repositories._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'requests'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_create_repositories_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RepositoryManagerRestInterceptor())
    client = RepositoryManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.RepositoryManagerRestInterceptor, 'post_batch_create_repositories') as post, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'pre_batch_create_repositories') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = repositories.BatchCreateRepositoriesRequest.pb(repositories.BatchCreateRepositoriesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = repositories.BatchCreateRepositoriesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_create_repositories(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_create_repositories_rest_bad_request(transport: str='rest', request_type=repositories.BatchCreateRepositoriesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_create_repositories(request)

def test_batch_create_repositories_rest_flattened():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/connections/sample3'}
        mock_args = dict(parent='parent_value', requests=[repositories.CreateRepositoryRequest(parent='parent_value')])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_create_repositories(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*/connections/*}/repositories:batchCreate' % client.transport._host, args[1])

def test_batch_create_repositories_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_create_repositories(repositories.BatchCreateRepositoriesRequest(), parent='parent_value', requests=[repositories.CreateRepositoryRequest(parent='parent_value')])

def test_batch_create_repositories_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [repositories.GetRepositoryRequest, dict])
def test_get_repository_rest(request_type):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.Repository(name='name_value', remote_uri='remote_uri_value', etag='etag_value', webhook_id='webhook_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.Repository.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_repository(request)
    assert isinstance(response, repositories.Repository)
    assert response.name == 'name_value'
    assert response.remote_uri == 'remote_uri_value'
    assert response.etag == 'etag_value'
    assert response.webhook_id == 'webhook_id_value'

def test_get_repository_rest_required_fields(request_type=repositories.GetRepositoryRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RepositoryManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_repository._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_repository._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = repositories.Repository()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = repositories.Repository.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_repository(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_repository_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_repository._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_repository_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RepositoryManagerRestInterceptor())
    client = RepositoryManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'post_get_repository') as post, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'pre_get_repository') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = repositories.GetRepositoryRequest.pb(repositories.GetRepositoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = repositories.Repository.to_json(repositories.Repository())
        request = repositories.GetRepositoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = repositories.Repository()
        client.get_repository(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_repository_rest_bad_request(transport: str='rest', request_type=repositories.GetRepositoryRequest):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_repository(request)

def test_get_repository_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.Repository()
        sample_request = {'name': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.Repository.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_repository(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/connections/*/repositories/*}' % client.transport._host, args[1])

def test_get_repository_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_repository(repositories.GetRepositoryRequest(), name='name_value')

def test_get_repository_rest_error():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [repositories.ListRepositoriesRequest, dict])
def test_list_repositories_rest(request_type):
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.ListRepositoriesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.ListRepositoriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_repositories(request)
    assert isinstance(response, pagers.ListRepositoriesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_repositories_rest_required_fields(request_type=repositories.ListRepositoriesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RepositoryManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_repositories._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_repositories._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = repositories.ListRepositoriesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = repositories.ListRepositoriesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_repositories(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_repositories_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_repositories._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_repositories_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RepositoryManagerRestInterceptor())
    client = RepositoryManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'post_list_repositories') as post, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'pre_list_repositories') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = repositories.ListRepositoriesRequest.pb(repositories.ListRepositoriesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = repositories.ListRepositoriesResponse.to_json(repositories.ListRepositoriesResponse())
        request = repositories.ListRepositoriesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = repositories.ListRepositoriesResponse()
        client.list_repositories(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_repositories_rest_bad_request(transport: str='rest', request_type=repositories.ListRepositoriesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_repositories(request)

def test_list_repositories_rest_flattened():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.ListRepositoriesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/connections/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.ListRepositoriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_repositories(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*/connections/*}/repositories' % client.transport._host, args[1])

def test_list_repositories_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_repositories(repositories.ListRepositoriesRequest(), parent='parent_value')

def test_list_repositories_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (repositories.ListRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository(), repositories.Repository()], next_page_token='abc'), repositories.ListRepositoriesResponse(repositories=[], next_page_token='def'), repositories.ListRepositoriesResponse(repositories=[repositories.Repository()], next_page_token='ghi'), repositories.ListRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository()]))
        response = response + response
        response = tuple((repositories.ListRepositoriesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/connections/sample3'}
        pager = client.list_repositories(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, repositories.Repository) for i in results))
        pages = list(client.list_repositories(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [repositories.DeleteRepositoryRequest, dict])
def test_delete_repository_rest(request_type):
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_repository(request)
    assert response.operation.name == 'operations/spam'

def test_delete_repository_rest_required_fields(request_type=repositories.DeleteRepositoryRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RepositoryManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_repository._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_repository._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('etag', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_repository(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_repository_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_repository._get_unset_required_fields({})
    assert set(unset_fields) == set(('etag', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_repository_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RepositoryManagerRestInterceptor())
    client = RepositoryManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.RepositoryManagerRestInterceptor, 'post_delete_repository') as post, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'pre_delete_repository') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = repositories.DeleteRepositoryRequest.pb(repositories.DeleteRepositoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = repositories.DeleteRepositoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_repository(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_repository_rest_bad_request(transport: str='rest', request_type=repositories.DeleteRepositoryRequest):
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_repository(request)

def test_delete_repository_rest_flattened():
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_repository(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/connections/*/repositories/*}' % client.transport._host, args[1])

def test_delete_repository_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_repository(repositories.DeleteRepositoryRequest(), name='name_value')

def test_delete_repository_rest_error():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [repositories.FetchReadWriteTokenRequest, dict])
def test_fetch_read_write_token_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'repository': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.FetchReadWriteTokenResponse(token='token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.FetchReadWriteTokenResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.fetch_read_write_token(request)
    assert isinstance(response, repositories.FetchReadWriteTokenResponse)
    assert response.token == 'token_value'

def test_fetch_read_write_token_rest_required_fields(request_type=repositories.FetchReadWriteTokenRequest):
    if False:
        return 10
    transport_class = transports.RepositoryManagerRestTransport
    request_init = {}
    request_init['repository'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).fetch_read_write_token._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['repository'] = 'repository_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).fetch_read_write_token._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'repository' in jsonified_request
    assert jsonified_request['repository'] == 'repository_value'
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = repositories.FetchReadWriteTokenResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = repositories.FetchReadWriteTokenResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.fetch_read_write_token(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_fetch_read_write_token_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.fetch_read_write_token._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('repository',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_fetch_read_write_token_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RepositoryManagerRestInterceptor())
    client = RepositoryManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'post_fetch_read_write_token') as post, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'pre_fetch_read_write_token') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = repositories.FetchReadWriteTokenRequest.pb(repositories.FetchReadWriteTokenRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = repositories.FetchReadWriteTokenResponse.to_json(repositories.FetchReadWriteTokenResponse())
        request = repositories.FetchReadWriteTokenRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = repositories.FetchReadWriteTokenResponse()
        client.fetch_read_write_token(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_fetch_read_write_token_rest_bad_request(transport: str='rest', request_type=repositories.FetchReadWriteTokenRequest):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'repository': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.fetch_read_write_token(request)

def test_fetch_read_write_token_rest_flattened():
    if False:
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.FetchReadWriteTokenResponse()
        sample_request = {'repository': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
        mock_args = dict(repository='repository_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.FetchReadWriteTokenResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.fetch_read_write_token(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{repository=projects/*/locations/*/connections/*/repositories/*}:accessReadWriteToken' % client.transport._host, args[1])

def test_fetch_read_write_token_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.fetch_read_write_token(repositories.FetchReadWriteTokenRequest(), repository='repository_value')

def test_fetch_read_write_token_rest_error():
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [repositories.FetchReadTokenRequest, dict])
def test_fetch_read_token_rest(request_type):
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'repository': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.FetchReadTokenResponse(token='token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.FetchReadTokenResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.fetch_read_token(request)
    assert isinstance(response, repositories.FetchReadTokenResponse)
    assert response.token == 'token_value'

def test_fetch_read_token_rest_required_fields(request_type=repositories.FetchReadTokenRequest):
    if False:
        return 10
    transport_class = transports.RepositoryManagerRestTransport
    request_init = {}
    request_init['repository'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).fetch_read_token._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['repository'] = 'repository_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).fetch_read_token._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'repository' in jsonified_request
    assert jsonified_request['repository'] == 'repository_value'
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = repositories.FetchReadTokenResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = repositories.FetchReadTokenResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.fetch_read_token(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_fetch_read_token_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.fetch_read_token._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('repository',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_fetch_read_token_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RepositoryManagerRestInterceptor())
    client = RepositoryManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'post_fetch_read_token') as post, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'pre_fetch_read_token') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = repositories.FetchReadTokenRequest.pb(repositories.FetchReadTokenRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = repositories.FetchReadTokenResponse.to_json(repositories.FetchReadTokenResponse())
        request = repositories.FetchReadTokenRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = repositories.FetchReadTokenResponse()
        client.fetch_read_token(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_fetch_read_token_rest_bad_request(transport: str='rest', request_type=repositories.FetchReadTokenRequest):
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'repository': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.fetch_read_token(request)

def test_fetch_read_token_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.FetchReadTokenResponse()
        sample_request = {'repository': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
        mock_args = dict(repository='repository_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.FetchReadTokenResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.fetch_read_token(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{repository=projects/*/locations/*/connections/*/repositories/*}:accessReadToken' % client.transport._host, args[1])

def test_fetch_read_token_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.fetch_read_token(repositories.FetchReadTokenRequest(), repository='repository_value')

def test_fetch_read_token_rest_error():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [repositories.FetchLinkableRepositoriesRequest, dict])
def test_fetch_linkable_repositories_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'connection': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.FetchLinkableRepositoriesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.FetchLinkableRepositoriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.fetch_linkable_repositories(request)
    assert isinstance(response, pagers.FetchLinkableRepositoriesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_fetch_linkable_repositories_rest_required_fields(request_type=repositories.FetchLinkableRepositoriesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RepositoryManagerRestTransport
    request_init = {}
    request_init['connection'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).fetch_linkable_repositories._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['connection'] = 'connection_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).fetch_linkable_repositories._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'connection' in jsonified_request
    assert jsonified_request['connection'] == 'connection_value'
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = repositories.FetchLinkableRepositoriesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = repositories.FetchLinkableRepositoriesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.fetch_linkable_repositories(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_fetch_linkable_repositories_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.fetch_linkable_repositories._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('connection',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_fetch_linkable_repositories_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RepositoryManagerRestInterceptor())
    client = RepositoryManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'post_fetch_linkable_repositories') as post, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'pre_fetch_linkable_repositories') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = repositories.FetchLinkableRepositoriesRequest.pb(repositories.FetchLinkableRepositoriesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = repositories.FetchLinkableRepositoriesResponse.to_json(repositories.FetchLinkableRepositoriesResponse())
        request = repositories.FetchLinkableRepositoriesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = repositories.FetchLinkableRepositoriesResponse()
        client.fetch_linkable_repositories(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_fetch_linkable_repositories_rest_bad_request(transport: str='rest', request_type=repositories.FetchLinkableRepositoriesRequest):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'connection': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.fetch_linkable_repositories(request)

def test_fetch_linkable_repositories_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository(), repositories.Repository()], next_page_token='abc'), repositories.FetchLinkableRepositoriesResponse(repositories=[], next_page_token='def'), repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository()], next_page_token='ghi'), repositories.FetchLinkableRepositoriesResponse(repositories=[repositories.Repository(), repositories.Repository()]))
        response = response + response
        response = tuple((repositories.FetchLinkableRepositoriesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'connection': 'projects/sample1/locations/sample2/connections/sample3'}
        pager = client.fetch_linkable_repositories(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, repositories.Repository) for i in results))
        pages = list(client.fetch_linkable_repositories(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [repositories.FetchGitRefsRequest, dict])
def test_fetch_git_refs_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'repository': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.FetchGitRefsResponse(ref_names=['ref_names_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.FetchGitRefsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.fetch_git_refs(request)
    assert isinstance(response, repositories.FetchGitRefsResponse)
    assert response.ref_names == ['ref_names_value']

def test_fetch_git_refs_rest_required_fields(request_type=repositories.FetchGitRefsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RepositoryManagerRestTransport
    request_init = {}
    request_init['repository'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).fetch_git_refs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['repository'] = 'repository_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).fetch_git_refs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('ref_type',))
    jsonified_request.update(unset_fields)
    assert 'repository' in jsonified_request
    assert jsonified_request['repository'] == 'repository_value'
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = repositories.FetchGitRefsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = repositories.FetchGitRefsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.fetch_git_refs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_fetch_git_refs_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.fetch_git_refs._get_unset_required_fields({})
    assert set(unset_fields) == set(('refType',)) & set(('repository',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_fetch_git_refs_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RepositoryManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RepositoryManagerRestInterceptor())
    client = RepositoryManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'post_fetch_git_refs') as post, mock.patch.object(transports.RepositoryManagerRestInterceptor, 'pre_fetch_git_refs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = repositories.FetchGitRefsRequest.pb(repositories.FetchGitRefsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = repositories.FetchGitRefsResponse.to_json(repositories.FetchGitRefsResponse())
        request = repositories.FetchGitRefsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = repositories.FetchGitRefsResponse()
        client.fetch_git_refs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_fetch_git_refs_rest_bad_request(transport: str='rest', request_type=repositories.FetchGitRefsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'repository': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.fetch_git_refs(request)

def test_fetch_git_refs_rest_flattened():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = repositories.FetchGitRefsResponse()
        sample_request = {'repository': 'projects/sample1/locations/sample2/connections/sample3/repositories/sample4'}
        mock_args = dict(repository='repository_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = repositories.FetchGitRefsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.fetch_git_refs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{repository=projects/*/locations/*/connections/*/repositories/*}:fetchGitRefs' % client.transport._host, args[1])

def test_fetch_git_refs_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.fetch_git_refs(repositories.FetchGitRefsRequest(), repository='repository_value')

def test_fetch_git_refs_rest_error():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RepositoryManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.RepositoryManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RepositoryManagerClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.RepositoryManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = RepositoryManagerClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = RepositoryManagerClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.RepositoryManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RepositoryManagerClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RepositoryManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = RepositoryManagerClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.RepositoryManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.RepositoryManagerGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.RepositoryManagerGrpcTransport, transports.RepositoryManagerGrpcAsyncIOTransport, transports.RepositoryManagerRestTransport])
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
        while True:
            i = 10
    transport = RepositoryManagerClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.RepositoryManagerGrpcTransport)

def test_repository_manager_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.RepositoryManagerTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_repository_manager_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.devtools.cloudbuild_v2.services.repository_manager.transports.RepositoryManagerTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.RepositoryManagerTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_connection', 'get_connection', 'list_connections', 'update_connection', 'delete_connection', 'create_repository', 'batch_create_repositories', 'get_repository', 'list_repositories', 'delete_repository', 'fetch_read_write_token', 'fetch_read_token', 'fetch_linkable_repositories', 'fetch_git_refs', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_operation', 'cancel_operation')
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

def test_repository_manager_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.devtools.cloudbuild_v2.services.repository_manager.transports.RepositoryManagerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.RepositoryManagerTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_repository_manager_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.devtools.cloudbuild_v2.services.repository_manager.transports.RepositoryManagerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.RepositoryManagerTransport()
        adc.assert_called_once()

def test_repository_manager_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        RepositoryManagerClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.RepositoryManagerGrpcTransport, transports.RepositoryManagerGrpcAsyncIOTransport])
def test_repository_manager_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.RepositoryManagerGrpcTransport, transports.RepositoryManagerGrpcAsyncIOTransport, transports.RepositoryManagerRestTransport])
def test_repository_manager_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.RepositoryManagerGrpcTransport, grpc_helpers), (transports.RepositoryManagerGrpcAsyncIOTransport, grpc_helpers_async)])
def test_repository_manager_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudbuild.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='cloudbuild.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.RepositoryManagerGrpcTransport, transports.RepositoryManagerGrpcAsyncIOTransport])
def test_repository_manager_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_repository_manager_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.RepositoryManagerRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_repository_manager_rest_lro_client():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_repository_manager_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudbuild.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudbuild.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudbuild.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_repository_manager_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudbuild.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudbuild.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudbuild.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_repository_manager_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = RepositoryManagerClient(credentials=creds1, transport=transport_name)
    client2 = RepositoryManagerClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_connection._session
    session2 = client2.transport.create_connection._session
    assert session1 != session2
    session1 = client1.transport.get_connection._session
    session2 = client2.transport.get_connection._session
    assert session1 != session2
    session1 = client1.transport.list_connections._session
    session2 = client2.transport.list_connections._session
    assert session1 != session2
    session1 = client1.transport.update_connection._session
    session2 = client2.transport.update_connection._session
    assert session1 != session2
    session1 = client1.transport.delete_connection._session
    session2 = client2.transport.delete_connection._session
    assert session1 != session2
    session1 = client1.transport.create_repository._session
    session2 = client2.transport.create_repository._session
    assert session1 != session2
    session1 = client1.transport.batch_create_repositories._session
    session2 = client2.transport.batch_create_repositories._session
    assert session1 != session2
    session1 = client1.transport.get_repository._session
    session2 = client2.transport.get_repository._session
    assert session1 != session2
    session1 = client1.transport.list_repositories._session
    session2 = client2.transport.list_repositories._session
    assert session1 != session2
    session1 = client1.transport.delete_repository._session
    session2 = client2.transport.delete_repository._session
    assert session1 != session2
    session1 = client1.transport.fetch_read_write_token._session
    session2 = client2.transport.fetch_read_write_token._session
    assert session1 != session2
    session1 = client1.transport.fetch_read_token._session
    session2 = client2.transport.fetch_read_token._session
    assert session1 != session2
    session1 = client1.transport.fetch_linkable_repositories._session
    session2 = client2.transport.fetch_linkable_repositories._session
    assert session1 != session2
    session1 = client1.transport.fetch_git_refs._session
    session2 = client2.transport.fetch_git_refs._session
    assert session1 != session2

def test_repository_manager_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.RepositoryManagerGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_repository_manager_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.RepositoryManagerGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.RepositoryManagerGrpcTransport, transports.RepositoryManagerGrpcAsyncIOTransport])
def test_repository_manager_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.RepositoryManagerGrpcTransport, transports.RepositoryManagerGrpcAsyncIOTransport])
def test_repository_manager_transport_channel_mtls_with_adc(transport_class):
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

def test_repository_manager_grpc_lro_client():
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_repository_manager_grpc_lro_async_client():
    if False:
        return 10
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_connection_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    connection = 'whelk'
    expected = 'projects/{project}/locations/{location}/connections/{connection}'.format(project=project, location=location, connection=connection)
    actual = RepositoryManagerClient.connection_path(project, location, connection)
    assert expected == actual

def test_parse_connection_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'location': 'oyster', 'connection': 'nudibranch'}
    path = RepositoryManagerClient.connection_path(**expected)
    actual = RepositoryManagerClient.parse_connection_path(path)
    assert expected == actual

def test_repository_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    connection = 'winkle'
    repository = 'nautilus'
    expected = 'projects/{project}/locations/{location}/connections/{connection}/repositories/{repository}'.format(project=project, location=location, connection=connection, repository=repository)
    actual = RepositoryManagerClient.repository_path(project, location, connection, repository)
    assert expected == actual

def test_parse_repository_path():
    if False:
        return 10
    expected = {'project': 'scallop', 'location': 'abalone', 'connection': 'squid', 'repository': 'clam'}
    path = RepositoryManagerClient.repository_path(**expected)
    actual = RepositoryManagerClient.parse_repository_path(path)
    assert expected == actual

def test_secret_version_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    secret = 'octopus'
    version = 'oyster'
    expected = 'projects/{project}/secrets/{secret}/versions/{version}'.format(project=project, secret=secret, version=version)
    actual = RepositoryManagerClient.secret_version_path(project, secret, version)
    assert expected == actual

def test_parse_secret_version_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nudibranch', 'secret': 'cuttlefish', 'version': 'mussel'}
    path = RepositoryManagerClient.secret_version_path(**expected)
    actual = RepositoryManagerClient.parse_secret_version_path(path)
    assert expected == actual

def test_service_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    namespace = 'scallop'
    service = 'abalone'
    expected = 'projects/{project}/locations/{location}/namespaces/{namespace}/services/{service}'.format(project=project, location=location, namespace=namespace, service=service)
    actual = RepositoryManagerClient.service_path(project, location, namespace, service)
    assert expected == actual

def test_parse_service_path():
    if False:
        return 10
    expected = {'project': 'squid', 'location': 'clam', 'namespace': 'whelk', 'service': 'octopus'}
    path = RepositoryManagerClient.service_path(**expected)
    actual = RepositoryManagerClient.parse_service_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = RepositoryManagerClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'nudibranch'}
    path = RepositoryManagerClient.common_billing_account_path(**expected)
    actual = RepositoryManagerClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = RepositoryManagerClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'mussel'}
    path = RepositoryManagerClient.common_folder_path(**expected)
    actual = RepositoryManagerClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = RepositoryManagerClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'nautilus'}
    path = RepositoryManagerClient.common_organization_path(**expected)
    actual = RepositoryManagerClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = RepositoryManagerClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'abalone'}
    path = RepositoryManagerClient.common_project_path(**expected)
    actual = RepositoryManagerClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = RepositoryManagerClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = RepositoryManagerClient.common_location_path(**expected)
    actual = RepositoryManagerClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.RepositoryManagerTransport, '_prep_wrapped_messages') as prep:
        client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.RepositoryManagerTransport, '_prep_wrapped_messages') as prep:
        transport_class = RepositoryManagerClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/connections/sample3'}, request)
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
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/connections/sample3'}
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
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/connections/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/connections/sample3'}
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
        i = 10
        return i + 15
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/connections/sample3'}, request)
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
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/connections/sample3'}
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
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_cancel_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = RepositoryManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = RepositoryManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(RepositoryManagerClient, transports.RepositoryManagerGrpcTransport), (RepositoryManagerAsyncClient, transports.RepositoryManagerGrpcAsyncIOTransport)])
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