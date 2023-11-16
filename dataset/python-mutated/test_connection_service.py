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
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
from google.oauth2 import service_account
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.type import expr_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.bigquery_connection_v1.services.connection_service import ConnectionServiceAsyncClient, ConnectionServiceClient, pagers, transports
from google.cloud.bigquery_connection_v1.types import connection as gcbc_connection
from google.cloud.bigquery_connection_v1.types import connection

def client_cert_source_callback():
    if False:
        for i in range(10):
            print('nop')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        print('Hello World!')
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
    assert ConnectionServiceClient._get_default_mtls_endpoint(None) is None
    assert ConnectionServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ConnectionServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ConnectionServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ConnectionServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ConnectionServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ConnectionServiceClient, 'grpc'), (ConnectionServiceAsyncClient, 'grpc_asyncio'), (ConnectionServiceClient, 'rest')])
def test_connection_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('bigqueryconnection.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://bigqueryconnection.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ConnectionServiceGrpcTransport, 'grpc'), (transports.ConnectionServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.ConnectionServiceRestTransport, 'rest')])
def test_connection_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ConnectionServiceClient, 'grpc'), (ConnectionServiceAsyncClient, 'grpc_asyncio'), (ConnectionServiceClient, 'rest')])
def test_connection_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('bigqueryconnection.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://bigqueryconnection.googleapis.com')

def test_connection_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = ConnectionServiceClient.get_transport_class()
    available_transports = [transports.ConnectionServiceGrpcTransport, transports.ConnectionServiceRestTransport]
    assert transport in available_transports
    transport = ConnectionServiceClient.get_transport_class('grpc')
    assert transport == transports.ConnectionServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ConnectionServiceClient, transports.ConnectionServiceGrpcTransport, 'grpc'), (ConnectionServiceAsyncClient, transports.ConnectionServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (ConnectionServiceClient, transports.ConnectionServiceRestTransport, 'rest')])
@mock.patch.object(ConnectionServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConnectionServiceClient))
@mock.patch.object(ConnectionServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConnectionServiceAsyncClient))
def test_connection_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(ConnectionServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ConnectionServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ConnectionServiceClient, transports.ConnectionServiceGrpcTransport, 'grpc', 'true'), (ConnectionServiceAsyncClient, transports.ConnectionServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ConnectionServiceClient, transports.ConnectionServiceGrpcTransport, 'grpc', 'false'), (ConnectionServiceAsyncClient, transports.ConnectionServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (ConnectionServiceClient, transports.ConnectionServiceRestTransport, 'rest', 'true'), (ConnectionServiceClient, transports.ConnectionServiceRestTransport, 'rest', 'false')])
@mock.patch.object(ConnectionServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConnectionServiceClient))
@mock.patch.object(ConnectionServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConnectionServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_connection_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [ConnectionServiceClient, ConnectionServiceAsyncClient])
@mock.patch.object(ConnectionServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConnectionServiceClient))
@mock.patch.object(ConnectionServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConnectionServiceAsyncClient))
def test_connection_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ConnectionServiceClient, transports.ConnectionServiceGrpcTransport, 'grpc'), (ConnectionServiceAsyncClient, transports.ConnectionServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (ConnectionServiceClient, transports.ConnectionServiceRestTransport, 'rest')])
def test_connection_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ConnectionServiceClient, transports.ConnectionServiceGrpcTransport, 'grpc', grpc_helpers), (ConnectionServiceAsyncClient, transports.ConnectionServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (ConnectionServiceClient, transports.ConnectionServiceRestTransport, 'rest', None)])
def test_connection_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_connection_service_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.bigquery_connection_v1.services.connection_service.transports.ConnectionServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ConnectionServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ConnectionServiceClient, transports.ConnectionServiceGrpcTransport, 'grpc', grpc_helpers), (ConnectionServiceAsyncClient, transports.ConnectionServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_connection_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('bigqueryconnection.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), scopes=None, default_host='bigqueryconnection.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [gcbc_connection.CreateConnectionRequest, dict])
def test_create_connection(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_connection), '__call__') as call:
        call.return_value = gcbc_connection.Connection(name='name_value', friendly_name='friendly_name_value', description='description_value', creation_time=1379, last_modified_time=1890, has_credential=True)
        response = client.create_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcbc_connection.CreateConnectionRequest()
    assert isinstance(response, gcbc_connection.Connection)
    assert response.name == 'name_value'
    assert response.friendly_name == 'friendly_name_value'
    assert response.description == 'description_value'
    assert response.creation_time == 1379
    assert response.last_modified_time == 1890
    assert response.has_credential is True

def test_create_connection_empty_call():
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_connection), '__call__') as call:
        client.create_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcbc_connection.CreateConnectionRequest()

@pytest.mark.asyncio
async def test_create_connection_async(transport: str='grpc_asyncio', request_type=gcbc_connection.CreateConnectionRequest):
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcbc_connection.Connection(name='name_value', friendly_name='friendly_name_value', description='description_value', creation_time=1379, last_modified_time=1890, has_credential=True))
        response = await client.create_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcbc_connection.CreateConnectionRequest()
    assert isinstance(response, gcbc_connection.Connection)
    assert response.name == 'name_value'
    assert response.friendly_name == 'friendly_name_value'
    assert response.description == 'description_value'
    assert response.creation_time == 1379
    assert response.last_modified_time == 1890
    assert response.has_credential is True

@pytest.mark.asyncio
async def test_create_connection_async_from_dict():
    await test_create_connection_async(request_type=dict)

def test_create_connection_field_headers():
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcbc_connection.CreateConnectionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_connection), '__call__') as call:
        call.return_value = gcbc_connection.Connection()
        client.create_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_connection_field_headers_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcbc_connection.CreateConnectionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcbc_connection.Connection())
        await client.create_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_connection_flattened():
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_connection), '__call__') as call:
        call.return_value = gcbc_connection.Connection()
        client.create_connection(parent='parent_value', connection=gcbc_connection.Connection(name='name_value'), connection_id='connection_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].connection
        mock_val = gcbc_connection.Connection(name='name_value')
        assert arg == mock_val
        arg = args[0].connection_id
        mock_val = 'connection_id_value'
        assert arg == mock_val

def test_create_connection_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_connection(gcbc_connection.CreateConnectionRequest(), parent='parent_value', connection=gcbc_connection.Connection(name='name_value'), connection_id='connection_id_value')

@pytest.mark.asyncio
async def test_create_connection_flattened_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_connection), '__call__') as call:
        call.return_value = gcbc_connection.Connection()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcbc_connection.Connection())
        response = await client.create_connection(parent='parent_value', connection=gcbc_connection.Connection(name='name_value'), connection_id='connection_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].connection
        mock_val = gcbc_connection.Connection(name='name_value')
        assert arg == mock_val
        arg = args[0].connection_id
        mock_val = 'connection_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_connection_flattened_error_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_connection(gcbc_connection.CreateConnectionRequest(), parent='parent_value', connection=gcbc_connection.Connection(name='name_value'), connection_id='connection_id_value')

@pytest.mark.parametrize('request_type', [connection.GetConnectionRequest, dict])
def test_get_connection(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_connection), '__call__') as call:
        call.return_value = connection.Connection(name='name_value', friendly_name='friendly_name_value', description='description_value', creation_time=1379, last_modified_time=1890, has_credential=True)
        response = client.get_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == connection.GetConnectionRequest()
    assert isinstance(response, connection.Connection)
    assert response.name == 'name_value'
    assert response.friendly_name == 'friendly_name_value'
    assert response.description == 'description_value'
    assert response.creation_time == 1379
    assert response.last_modified_time == 1890
    assert response.has_credential is True

def test_get_connection_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_connection), '__call__') as call:
        client.get_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == connection.GetConnectionRequest()

@pytest.mark.asyncio
async def test_get_connection_async(transport: str='grpc_asyncio', request_type=connection.GetConnectionRequest):
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(connection.Connection(name='name_value', friendly_name='friendly_name_value', description='description_value', creation_time=1379, last_modified_time=1890, has_credential=True))
        response = await client.get_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == connection.GetConnectionRequest()
    assert isinstance(response, connection.Connection)
    assert response.name == 'name_value'
    assert response.friendly_name == 'friendly_name_value'
    assert response.description == 'description_value'
    assert response.creation_time == 1379
    assert response.last_modified_time == 1890
    assert response.has_credential is True

@pytest.mark.asyncio
async def test_get_connection_async_from_dict():
    await test_get_connection_async(request_type=dict)

def test_get_connection_field_headers():
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = connection.GetConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_connection), '__call__') as call:
        call.return_value = connection.Connection()
        client.get_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_connection_field_headers_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = connection.GetConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(connection.Connection())
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
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_connection), '__call__') as call:
        call.return_value = connection.Connection()
        client.get_connection(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_connection_flattened_error():
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_connection(connection.GetConnectionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_connection_flattened_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_connection), '__call__') as call:
        call.return_value = connection.Connection()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(connection.Connection())
        response = await client.get_connection(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_connection_flattened_error_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_connection(connection.GetConnectionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [connection.ListConnectionsRequest, dict])
def test_list_connections(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.return_value = connection.ListConnectionsResponse(next_page_token='next_page_token_value')
        response = client.list_connections(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == connection.ListConnectionsRequest()
    assert isinstance(response, pagers.ListConnectionsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_connections_empty_call():
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        client.list_connections()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == connection.ListConnectionsRequest()

@pytest.mark.asyncio
async def test_list_connections_async(transport: str='grpc_asyncio', request_type=connection.ListConnectionsRequest):
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(connection.ListConnectionsResponse(next_page_token='next_page_token_value'))
        response = await client.list_connections(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == connection.ListConnectionsRequest()
    assert isinstance(response, pagers.ListConnectionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_connections_async_from_dict():
    await test_list_connections_async(request_type=dict)

def test_list_connections_field_headers():
    if False:
        while True:
            i = 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = connection.ListConnectionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.return_value = connection.ListConnectionsResponse()
        client.list_connections(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_connections_field_headers_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = connection.ListConnectionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(connection.ListConnectionsResponse())
        await client.list_connections(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_connections_flattened():
    if False:
        i = 10
        return i + 15
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.return_value = connection.ListConnectionsResponse()
        client.list_connections(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_connections_flattened_error():
    if False:
        while True:
            i = 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_connections(connection.ListConnectionsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_connections_flattened_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.return_value = connection.ListConnectionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(connection.ListConnectionsResponse())
        response = await client.list_connections(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_connections_flattened_error_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_connections(connection.ListConnectionsRequest(), parent='parent_value')

def test_list_connections_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.side_effect = (connection.ListConnectionsResponse(connections=[connection.Connection(), connection.Connection(), connection.Connection()], next_page_token='abc'), connection.ListConnectionsResponse(connections=[], next_page_token='def'), connection.ListConnectionsResponse(connections=[connection.Connection()], next_page_token='ghi'), connection.ListConnectionsResponse(connections=[connection.Connection(), connection.Connection()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_connections(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, connection.Connection) for i in results))

def test_list_connections_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_connections), '__call__') as call:
        call.side_effect = (connection.ListConnectionsResponse(connections=[connection.Connection(), connection.Connection(), connection.Connection()], next_page_token='abc'), connection.ListConnectionsResponse(connections=[], next_page_token='def'), connection.ListConnectionsResponse(connections=[connection.Connection()], next_page_token='ghi'), connection.ListConnectionsResponse(connections=[connection.Connection(), connection.Connection()]), RuntimeError)
        pages = list(client.list_connections(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_connections_async_pager():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_connections), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (connection.ListConnectionsResponse(connections=[connection.Connection(), connection.Connection(), connection.Connection()], next_page_token='abc'), connection.ListConnectionsResponse(connections=[], next_page_token='def'), connection.ListConnectionsResponse(connections=[connection.Connection()], next_page_token='ghi'), connection.ListConnectionsResponse(connections=[connection.Connection(), connection.Connection()]), RuntimeError)
        async_pager = await client.list_connections(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, connection.Connection) for i in responses))

@pytest.mark.asyncio
async def test_list_connections_async_pages():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_connections), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (connection.ListConnectionsResponse(connections=[connection.Connection(), connection.Connection(), connection.Connection()], next_page_token='abc'), connection.ListConnectionsResponse(connections=[], next_page_token='def'), connection.ListConnectionsResponse(connections=[connection.Connection()], next_page_token='ghi'), connection.ListConnectionsResponse(connections=[connection.Connection(), connection.Connection()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_connections(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [gcbc_connection.UpdateConnectionRequest, dict])
def test_update_connection(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_connection), '__call__') as call:
        call.return_value = gcbc_connection.Connection(name='name_value', friendly_name='friendly_name_value', description='description_value', creation_time=1379, last_modified_time=1890, has_credential=True)
        response = client.update_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcbc_connection.UpdateConnectionRequest()
    assert isinstance(response, gcbc_connection.Connection)
    assert response.name == 'name_value'
    assert response.friendly_name == 'friendly_name_value'
    assert response.description == 'description_value'
    assert response.creation_time == 1379
    assert response.last_modified_time == 1890
    assert response.has_credential is True

def test_update_connection_empty_call():
    if False:
        i = 10
        return i + 15
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_connection), '__call__') as call:
        client.update_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcbc_connection.UpdateConnectionRequest()

@pytest.mark.asyncio
async def test_update_connection_async(transport: str='grpc_asyncio', request_type=gcbc_connection.UpdateConnectionRequest):
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcbc_connection.Connection(name='name_value', friendly_name='friendly_name_value', description='description_value', creation_time=1379, last_modified_time=1890, has_credential=True))
        response = await client.update_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcbc_connection.UpdateConnectionRequest()
    assert isinstance(response, gcbc_connection.Connection)
    assert response.name == 'name_value'
    assert response.friendly_name == 'friendly_name_value'
    assert response.description == 'description_value'
    assert response.creation_time == 1379
    assert response.last_modified_time == 1890
    assert response.has_credential is True

@pytest.mark.asyncio
async def test_update_connection_async_from_dict():
    await test_update_connection_async(request_type=dict)

def test_update_connection_field_headers():
    if False:
        i = 10
        return i + 15
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcbc_connection.UpdateConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_connection), '__call__') as call:
        call.return_value = gcbc_connection.Connection()
        client.update_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_connection_field_headers_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcbc_connection.UpdateConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcbc_connection.Connection())
        await client.update_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_connection_flattened():
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_connection), '__call__') as call:
        call.return_value = gcbc_connection.Connection()
        client.update_connection(name='name_value', connection=gcbc_connection.Connection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].connection
        mock_val = gcbc_connection.Connection(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_connection_flattened_error():
    if False:
        while True:
            i = 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_connection(gcbc_connection.UpdateConnectionRequest(), name='name_value', connection=gcbc_connection.Connection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_connection_flattened_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_connection), '__call__') as call:
        call.return_value = gcbc_connection.Connection()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcbc_connection.Connection())
        response = await client.update_connection(name='name_value', connection=gcbc_connection.Connection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].connection
        mock_val = gcbc_connection.Connection(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_connection_flattened_error_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_connection(gcbc_connection.UpdateConnectionRequest(), name='name_value', connection=gcbc_connection.Connection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [connection.DeleteConnectionRequest, dict])
def test_delete_connection(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_connection), '__call__') as call:
        call.return_value = None
        response = client.delete_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == connection.DeleteConnectionRequest()
    assert response is None

def test_delete_connection_empty_call():
    if False:
        i = 10
        return i + 15
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_connection), '__call__') as call:
        client.delete_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == connection.DeleteConnectionRequest()

@pytest.mark.asyncio
async def test_delete_connection_async(transport: str='grpc_asyncio', request_type=connection.DeleteConnectionRequest):
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == connection.DeleteConnectionRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_connection_async_from_dict():
    await test_delete_connection_async(request_type=dict)

def test_delete_connection_field_headers():
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = connection.DeleteConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_connection), '__call__') as call:
        call.return_value = None
        client.delete_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_connection_field_headers_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = connection.DeleteConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_connection_flattened():
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_connection), '__call__') as call:
        call.return_value = None
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
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_connection(connection.DeleteConnectionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_connection_flattened_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_connection), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_connection(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_connection_flattened_error_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_connection(connection.DeleteConnectionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_empty_call():
    if False:
        while True:
            i = 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        client.get_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()

@pytest.mark.asyncio
async def test_get_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.GetIamPolicyRequest):
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_get_iam_policy_async_from_dict():
    await test_get_iam_policy_async(request_type=dict)

def test_get_iam_policy_field_headers():
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_iam_policy_field_headers_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_get_iam_policy_from_dict_foreign():
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_get_iam_policy_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(resource='resource_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

def test_get_iam_policy_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

@pytest.mark.asyncio
async def test_get_iam_policy_flattened_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(resource='resource_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_iam_policy_flattened_error_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_empty_call():
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        client.set_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()

@pytest.mark.asyncio
async def test_set_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.SetIamPolicyRequest):
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.set_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_set_iam_policy_async_from_dict():
    await test_set_iam_policy_async(request_type=dict)

def test_set_iam_policy_field_headers():
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_iam_policy_field_headers_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.set_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_set_iam_policy_from_dict_foreign():
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774), 'update_mask': field_mask_pb2.FieldMask(paths=['paths_value'])})
        call.assert_called()

def test_set_iam_policy_flattened():
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(resource='resource_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

def test_set_iam_policy_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

@pytest.mark.asyncio
async def test_set_iam_policy_flattened_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(resource='resource_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_set_iam_policy_flattened_error_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response = client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_empty_call():
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        client.test_iam_permissions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()

@pytest.mark.asyncio
async def test_test_iam_permissions_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value']))
        response = await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

@pytest.mark.asyncio
async def test_test_iam_permissions_async_from_dict():
    await test_test_iam_permissions_async(request_type=dict)

def test_test_iam_permissions_field_headers():
    if False:
        i = 10
        return i + 15
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_test_iam_permissions_field_headers_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_test_iam_permissions_from_dict_foreign():
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_test_iam_permissions_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(resource='resource_value', permissions=['permissions_value'])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val
        arg = args[0].permissions
        mock_val = ['permissions_value']
        assert arg == mock_val

def test_test_iam_permissions_flattened_error():
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

@pytest.mark.asyncio
async def test_test_iam_permissions_flattened_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(resource='resource_value', permissions=['permissions_value'])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val
        arg = args[0].permissions
        mock_val = ['permissions_value']
        assert arg == mock_val

@pytest.mark.asyncio
async def test_test_iam_permissions_flattened_error_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

@pytest.mark.parametrize('request_type', [gcbc_connection.CreateConnectionRequest, dict])
def test_create_connection_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['connection'] = {'name': 'name_value', 'friendly_name': 'friendly_name_value', 'description': 'description_value', 'cloud_sql': {'instance_id': 'instance_id_value', 'database': 'database_value', 'type_': 1, 'credential': {'username': 'username_value', 'password': 'password_value'}, 'service_account_id': 'service_account_id_value'}, 'aws': {'cross_account_role': {'iam_role_id': 'iam_role_id_value', 'iam_user_id': 'iam_user_id_value', 'external_id': 'external_id_value'}, 'access_role': {'iam_role_id': 'iam_role_id_value', 'identity': 'identity_value'}}, 'azure': {'application': 'application_value', 'client_id': 'client_id_value', 'object_id': 'object_id_value', 'customer_tenant_id': 'customer_tenant_id_value', 'redirect_uri': 'redirect_uri_value', 'federated_application_client_id': 'federated_application_client_id_value', 'identity': 'identity_value'}, 'cloud_spanner': {'database': 'database_value', 'use_parallelism': True, 'max_parallelism': 1595, 'use_serverless_analytics': True, 'use_data_boost': True, 'database_role': 'database_role_value'}, 'cloud_resource': {'service_account_id': 'service_account_id_value'}, 'spark': {'service_account_id': 'service_account_id_value', 'metastore_service_config': {'metastore_service': 'metastore_service_value'}, 'spark_history_server_config': {'dataproc_cluster': 'dataproc_cluster_value'}}, 'salesforce_data_cloud': {'instance_uri': 'instance_uri_value', 'identity': 'identity_value', 'tenant_id': 'tenant_id_value'}, 'creation_time': 1379, 'last_modified_time': 1890, 'has_credential': True}
    test_field = gcbc_connection.CreateConnectionRequest.meta.fields['connection']

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
        return_value = gcbc_connection.Connection(name='name_value', friendly_name='friendly_name_value', description='description_value', creation_time=1379, last_modified_time=1890, has_credential=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcbc_connection.Connection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_connection(request)
    assert isinstance(response, gcbc_connection.Connection)
    assert response.name == 'name_value'
    assert response.friendly_name == 'friendly_name_value'
    assert response.description == 'description_value'
    assert response.creation_time == 1379
    assert response.last_modified_time == 1890
    assert response.has_credential is True

def test_create_connection_rest_required_fields(request_type=gcbc_connection.CreateConnectionRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ConnectionServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_connection._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('connection_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcbc_connection.Connection()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcbc_connection.Connection.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_connection_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(('connectionId',)) & set(('parent', 'connection'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_connection_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConnectionServiceRestInterceptor())
    client = ConnectionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'post_create_connection') as post, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'pre_create_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcbc_connection.CreateConnectionRequest.pb(gcbc_connection.CreateConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcbc_connection.Connection.to_json(gcbc_connection.Connection())
        request = gcbc_connection.CreateConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcbc_connection.Connection()
        client.create_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_connection_rest_bad_request(transport: str='rest', request_type=gcbc_connection.CreateConnectionRequest):
    if False:
        while True:
            i = 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcbc_connection.Connection()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', connection=gcbc_connection.Connection(name='name_value'), connection_id='connection_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcbc_connection.Connection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/connections' % client.transport._host, args[1])

def test_create_connection_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_connection(gcbc_connection.CreateConnectionRequest(), parent='parent_value', connection=gcbc_connection.Connection(name='name_value'), connection_id='connection_id_value')

def test_create_connection_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [connection.GetConnectionRequest, dict])
def test_get_connection_rest(request_type):
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = connection.Connection(name='name_value', friendly_name='friendly_name_value', description='description_value', creation_time=1379, last_modified_time=1890, has_credential=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = connection.Connection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_connection(request)
    assert isinstance(response, connection.Connection)
    assert response.name == 'name_value'
    assert response.friendly_name == 'friendly_name_value'
    assert response.description == 'description_value'
    assert response.creation_time == 1379
    assert response.last_modified_time == 1890
    assert response.has_credential is True

def test_get_connection_rest_required_fields(request_type=connection.GetConnectionRequest):
    if False:
        return 10
    transport_class = transports.ConnectionServiceRestTransport
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
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = connection.Connection()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = connection.Connection.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_connection_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_connection_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConnectionServiceRestInterceptor())
    client = ConnectionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'post_get_connection') as post, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'pre_get_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = connection.GetConnectionRequest.pb(connection.GetConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = connection.Connection.to_json(connection.Connection())
        request = connection.GetConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = connection.Connection()
        client.get_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_connection_rest_bad_request(transport: str='rest', request_type=connection.GetConnectionRequest):
    if False:
        while True:
            i = 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = connection.Connection()
        sample_request = {'name': 'projects/sample1/locations/sample2/connections/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = connection.Connection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/connections/*}' % client.transport._host, args[1])

def test_get_connection_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_connection(connection.GetConnectionRequest(), name='name_value')

def test_get_connection_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [connection.ListConnectionsRequest, dict])
def test_list_connections_rest(request_type):
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = connection.ListConnectionsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = connection.ListConnectionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_connections(request)
    assert isinstance(response, pagers.ListConnectionsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_connections_rest_required_fields(request_type=connection.ListConnectionsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ConnectionServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['page_size'] = 0
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'pageSize' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_connections._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'pageSize' in jsonified_request
    assert jsonified_request['pageSize'] == request_init['page_size']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['pageSize'] = 951
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_connections._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'pageSize' in jsonified_request
    assert jsonified_request['pageSize'] == 951
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = connection.ListConnectionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = connection.ListConnectionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_connections(request)
            expected_params = [('pageSize', str(0)), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_connections_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_connections._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent', 'pageSize'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_connections_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConnectionServiceRestInterceptor())
    client = ConnectionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'post_list_connections') as post, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'pre_list_connections') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = connection.ListConnectionsRequest.pb(connection.ListConnectionsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = connection.ListConnectionsResponse.to_json(connection.ListConnectionsResponse())
        request = connection.ListConnectionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = connection.ListConnectionsResponse()
        client.list_connections(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_connections_rest_bad_request(transport: str='rest', request_type=connection.ListConnectionsRequest):
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = connection.ListConnectionsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = connection.ListConnectionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_connections(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/connections' % client.transport._host, args[1])

def test_list_connections_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_connections(connection.ListConnectionsRequest(), parent='parent_value')

def test_list_connections_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (connection.ListConnectionsResponse(connections=[connection.Connection(), connection.Connection(), connection.Connection()], next_page_token='abc'), connection.ListConnectionsResponse(connections=[], next_page_token='def'), connection.ListConnectionsResponse(connections=[connection.Connection()], next_page_token='ghi'), connection.ListConnectionsResponse(connections=[connection.Connection(), connection.Connection()]))
        response = response + response
        response = tuple((connection.ListConnectionsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_connections(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, connection.Connection) for i in results))
        pages = list(client.list_connections(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [gcbc_connection.UpdateConnectionRequest, dict])
def test_update_connection_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/connections/sample3'}
    request_init['connection'] = {'name': 'name_value', 'friendly_name': 'friendly_name_value', 'description': 'description_value', 'cloud_sql': {'instance_id': 'instance_id_value', 'database': 'database_value', 'type_': 1, 'credential': {'username': 'username_value', 'password': 'password_value'}, 'service_account_id': 'service_account_id_value'}, 'aws': {'cross_account_role': {'iam_role_id': 'iam_role_id_value', 'iam_user_id': 'iam_user_id_value', 'external_id': 'external_id_value'}, 'access_role': {'iam_role_id': 'iam_role_id_value', 'identity': 'identity_value'}}, 'azure': {'application': 'application_value', 'client_id': 'client_id_value', 'object_id': 'object_id_value', 'customer_tenant_id': 'customer_tenant_id_value', 'redirect_uri': 'redirect_uri_value', 'federated_application_client_id': 'federated_application_client_id_value', 'identity': 'identity_value'}, 'cloud_spanner': {'database': 'database_value', 'use_parallelism': True, 'max_parallelism': 1595, 'use_serverless_analytics': True, 'use_data_boost': True, 'database_role': 'database_role_value'}, 'cloud_resource': {'service_account_id': 'service_account_id_value'}, 'spark': {'service_account_id': 'service_account_id_value', 'metastore_service_config': {'metastore_service': 'metastore_service_value'}, 'spark_history_server_config': {'dataproc_cluster': 'dataproc_cluster_value'}}, 'salesforce_data_cloud': {'instance_uri': 'instance_uri_value', 'identity': 'identity_value', 'tenant_id': 'tenant_id_value'}, 'creation_time': 1379, 'last_modified_time': 1890, 'has_credential': True}
    test_field = gcbc_connection.UpdateConnectionRequest.meta.fields['connection']

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
        return_value = gcbc_connection.Connection(name='name_value', friendly_name='friendly_name_value', description='description_value', creation_time=1379, last_modified_time=1890, has_credential=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcbc_connection.Connection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_connection(request)
    assert isinstance(response, gcbc_connection.Connection)
    assert response.name == 'name_value'
    assert response.friendly_name == 'friendly_name_value'
    assert response.description == 'description_value'
    assert response.creation_time == 1379
    assert response.last_modified_time == 1890
    assert response.has_credential is True

def test_update_connection_rest_required_fields(request_type=gcbc_connection.UpdateConnectionRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ConnectionServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_connection._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcbc_connection.Connection()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcbc_connection.Connection.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_connection_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('name', 'connection', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_connection_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConnectionServiceRestInterceptor())
    client = ConnectionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'post_update_connection') as post, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'pre_update_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcbc_connection.UpdateConnectionRequest.pb(gcbc_connection.UpdateConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcbc_connection.Connection.to_json(gcbc_connection.Connection())
        request = gcbc_connection.UpdateConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcbc_connection.Connection()
        client.update_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_connection_rest_bad_request(transport: str='rest', request_type=gcbc_connection.UpdateConnectionRequest):
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_connection(request)

def test_update_connection_rest_flattened():
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcbc_connection.Connection()
        sample_request = {'name': 'projects/sample1/locations/sample2/connections/sample3'}
        mock_args = dict(name='name_value', connection=gcbc_connection.Connection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcbc_connection.Connection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/connections/*}' % client.transport._host, args[1])

def test_update_connection_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_connection(gcbc_connection.UpdateConnectionRequest(), name='name_value', connection=gcbc_connection.Connection(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_connection_rest_error():
    if False:
        i = 10
        return i + 15
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [connection.DeleteConnectionRequest, dict])
def test_delete_connection_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_connection(request)
    assert response is None

def test_delete_connection_rest_required_fields(request_type=connection.DeleteConnectionRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ConnectionServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_connection_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_connection_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConnectionServiceRestInterceptor())
    client = ConnectionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'pre_delete_connection') as pre:
        pre.assert_not_called()
        pb_message = connection.DeleteConnectionRequest.pb(connection.DeleteConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = connection.DeleteConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_connection_rest_bad_request(transport: str='rest', request_type=connection.DeleteConnectionRequest):
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/connections/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/connections/*}' % client.transport._host, args[1])

def test_delete_connection_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_connection(connection.DeleteConnectionRequest(), name='name_value')

def test_delete_connection_rest_error():
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_rest_required_fields(request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ConnectionServiceRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_iam_policy_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_iam_policy_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConnectionServiceRestInterceptor())
    client = ConnectionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'post_get_iam_policy') as post, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'pre_get_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.GetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.GetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.get_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

def test_get_iam_policy_rest_flattened():
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        sample_request = {'resource': 'projects/sample1/locations/sample2/connections/sample3'}
        mock_args = dict(resource='resource_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_iam_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{resource=projects/*/locations/*/connections/*}:getIamPolicy' % client.transport._host, args[1])

def test_get_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

def test_get_iam_policy_rest_error():
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_rest_required_fields(request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ConnectionServiceRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.set_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_iam_policy_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'policy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_iam_policy_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConnectionServiceRestInterceptor())
    client = ConnectionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'post_set_iam_policy') as post, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'pre_set_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.SetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.SetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.set_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

def test_set_iam_policy_rest_flattened():
    if False:
        while True:
            i = 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        sample_request = {'resource': 'projects/sample1/locations/sample2/connections/sample3'}
        mock_args = dict(resource='resource_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_iam_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{resource=projects/*/locations/*/connections/*}:setIamPolicy' % client.transport._host, args[1])

def test_set_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

def test_set_iam_policy_rest_error():
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        while True:
            i = 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.test_iam_permissions(request)
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_rest_required_fields(request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ConnectionServiceRestTransport
    request_init = {}
    request_init['resource'] = ''
    request_init['permissions'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    jsonified_request['permissions'] = 'permissions_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    assert 'permissions' in jsonified_request
    assert jsonified_request['permissions'] == 'permissions_value'
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = iam_policy_pb2.TestIamPermissionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.test_iam_permissions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_test_iam_permissions_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.test_iam_permissions._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'permissions'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_test_iam_permissions_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ConnectionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConnectionServiceRestInterceptor())
    client = ConnectionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'post_test_iam_permissions') as post, mock.patch.object(transports.ConnectionServiceRestInterceptor, 'pre_test_iam_permissions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.TestIamPermissionsRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(iam_policy_pb2.TestIamPermissionsResponse())
        request = iam_policy_pb2.TestIamPermissionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_test_iam_permissions_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/connections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

def test_test_iam_permissions_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse()
        sample_request = {'resource': 'projects/sample1/locations/sample2/connections/sample3'}
        mock_args = dict(resource='resource_value', permissions=['permissions_value'])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.test_iam_permissions(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{resource=projects/*/locations/*/connections/*}:testIamPermissions' % client.transport._host, args[1])

def test_test_iam_permissions_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

def test_test_iam_permissions_rest_error():
    if False:
        print('Hello World!')
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ConnectionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ConnectionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ConnectionServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ConnectionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ConnectionServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ConnectionServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ConnectionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ConnectionServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.ConnectionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ConnectionServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.ConnectionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ConnectionServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ConnectionServiceGrpcTransport, transports.ConnectionServiceGrpcAsyncIOTransport, transports.ConnectionServiceRestTransport])
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
        i = 10
        return i + 15
    transport = ConnectionServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ConnectionServiceGrpcTransport)

def test_connection_service_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ConnectionServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_connection_service_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.bigquery_connection_v1.services.connection_service.transports.ConnectionServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ConnectionServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_connection', 'get_connection', 'list_connections', 'update_connection', 'delete_connection', 'get_iam_policy', 'set_iam_policy', 'test_iam_permissions')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_connection_service_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.bigquery_connection_v1.services.connection_service.transports.ConnectionServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ConnectionServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

def test_connection_service_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.bigquery_connection_v1.services.connection_service.transports.ConnectionServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ConnectionServiceTransport()
        adc.assert_called_once()

def test_connection_service_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ConnectionServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ConnectionServiceGrpcTransport, transports.ConnectionServiceGrpcAsyncIOTransport])
def test_connection_service_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ConnectionServiceGrpcTransport, transports.ConnectionServiceGrpcAsyncIOTransport, transports.ConnectionServiceRestTransport])
def test_connection_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ConnectionServiceGrpcTransport, grpc_helpers), (transports.ConnectionServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_connection_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('bigqueryconnection.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/cloud-platform'), scopes=['1', '2'], default_host='bigqueryconnection.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ConnectionServiceGrpcTransport, transports.ConnectionServiceGrpcAsyncIOTransport])
def test_connection_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_connection_service_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.ConnectionServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_connection_service_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='bigqueryconnection.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('bigqueryconnection.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://bigqueryconnection.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_connection_service_host_with_port(transport_name):
    if False:
        return 10
    client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='bigqueryconnection.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('bigqueryconnection.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://bigqueryconnection.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_connection_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = ConnectionServiceClient(credentials=creds1, transport=transport_name)
    client2 = ConnectionServiceClient(credentials=creds2, transport=transport_name)
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
    session1 = client1.transport.get_iam_policy._session
    session2 = client2.transport.get_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.set_iam_policy._session
    session2 = client2.transport.set_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.test_iam_permissions._session
    session2 = client2.transport.test_iam_permissions._session
    assert session1 != session2

def test_connection_service_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ConnectionServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_connection_service_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ConnectionServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ConnectionServiceGrpcTransport, transports.ConnectionServiceGrpcAsyncIOTransport])
def test_connection_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.ConnectionServiceGrpcTransport, transports.ConnectionServiceGrpcAsyncIOTransport])
def test_connection_service_transport_channel_mtls_with_adc(transport_class):
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

def test_cluster_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    region = 'clam'
    cluster = 'whelk'
    expected = 'projects/{project}/regions/{region}/clusters/{cluster}'.format(project=project, region=region, cluster=cluster)
    actual = ConnectionServiceClient.cluster_path(project, region, cluster)
    assert expected == actual

def test_parse_cluster_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'region': 'oyster', 'cluster': 'nudibranch'}
    path = ConnectionServiceClient.cluster_path(**expected)
    actual = ConnectionServiceClient.parse_cluster_path(path)
    assert expected == actual

def test_connection_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    location = 'mussel'
    connection = 'winkle'
    expected = 'projects/{project}/locations/{location}/connections/{connection}'.format(project=project, location=location, connection=connection)
    actual = ConnectionServiceClient.connection_path(project, location, connection)
    assert expected == actual

def test_parse_connection_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nautilus', 'location': 'scallop', 'connection': 'abalone'}
    path = ConnectionServiceClient.connection_path(**expected)
    actual = ConnectionServiceClient.parse_connection_path(path)
    assert expected == actual

def test_service_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    service = 'whelk'
    expected = 'projects/{project}/locations/{location}/services/{service}'.format(project=project, location=location, service=service)
    actual = ConnectionServiceClient.service_path(project, location, service)
    assert expected == actual

def test_parse_service_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'service': 'nudibranch'}
    path = ConnectionServiceClient.service_path(**expected)
    actual = ConnectionServiceClient.parse_service_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ConnectionServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'mussel'}
    path = ConnectionServiceClient.common_billing_account_path(**expected)
    actual = ConnectionServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ConnectionServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'nautilus'}
    path = ConnectionServiceClient.common_folder_path(**expected)
    actual = ConnectionServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ConnectionServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'abalone'}
    path = ConnectionServiceClient.common_organization_path(**expected)
    actual = ConnectionServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = ConnectionServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'clam'}
    path = ConnectionServiceClient.common_project_path(**expected)
    actual = ConnectionServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ConnectionServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = ConnectionServiceClient.common_location_path(**expected)
    actual = ConnectionServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ConnectionServiceTransport, '_prep_wrapped_messages') as prep:
        client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ConnectionServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = ConnectionServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ConnectionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = ConnectionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ConnectionServiceClient, transports.ConnectionServiceGrpcTransport), (ConnectionServiceAsyncClient, transports.ConnectionServiceGrpcAsyncIOTransport)])
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