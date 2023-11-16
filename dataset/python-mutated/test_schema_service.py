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
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import empty_pb2
from google.protobuf import json_format
from google.protobuf import struct_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.discoveryengine_v1beta.services.schema_service import SchemaServiceAsyncClient, SchemaServiceClient, pagers, transports
from google.cloud.discoveryengine_v1beta.types import schema
from google.cloud.discoveryengine_v1beta.types import schema as gcd_schema
from google.cloud.discoveryengine_v1beta.types import schema_service

def client_cert_source_callback():
    if False:
        print('Hello World!')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        print('Hello World!')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert SchemaServiceClient._get_default_mtls_endpoint(None) is None
    assert SchemaServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert SchemaServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert SchemaServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert SchemaServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert SchemaServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(SchemaServiceClient, 'grpc'), (SchemaServiceAsyncClient, 'grpc_asyncio'), (SchemaServiceClient, 'rest')])
def test_schema_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('discoveryengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://discoveryengine.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.SchemaServiceGrpcTransport, 'grpc'), (transports.SchemaServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.SchemaServiceRestTransport, 'rest')])
def test_schema_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(SchemaServiceClient, 'grpc'), (SchemaServiceAsyncClient, 'grpc_asyncio'), (SchemaServiceClient, 'rest')])
def test_schema_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('discoveryengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://discoveryengine.googleapis.com')

def test_schema_service_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = SchemaServiceClient.get_transport_class()
    available_transports = [transports.SchemaServiceGrpcTransport, transports.SchemaServiceRestTransport]
    assert transport in available_transports
    transport = SchemaServiceClient.get_transport_class('grpc')
    assert transport == transports.SchemaServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(SchemaServiceClient, transports.SchemaServiceGrpcTransport, 'grpc'), (SchemaServiceAsyncClient, transports.SchemaServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (SchemaServiceClient, transports.SchemaServiceRestTransport, 'rest')])
@mock.patch.object(SchemaServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SchemaServiceClient))
@mock.patch.object(SchemaServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SchemaServiceAsyncClient))
def test_schema_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(SchemaServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(SchemaServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(SchemaServiceClient, transports.SchemaServiceGrpcTransport, 'grpc', 'true'), (SchemaServiceAsyncClient, transports.SchemaServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (SchemaServiceClient, transports.SchemaServiceGrpcTransport, 'grpc', 'false'), (SchemaServiceAsyncClient, transports.SchemaServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (SchemaServiceClient, transports.SchemaServiceRestTransport, 'rest', 'true'), (SchemaServiceClient, transports.SchemaServiceRestTransport, 'rest', 'false')])
@mock.patch.object(SchemaServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SchemaServiceClient))
@mock.patch.object(SchemaServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SchemaServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_schema_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [SchemaServiceClient, SchemaServiceAsyncClient])
@mock.patch.object(SchemaServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SchemaServiceClient))
@mock.patch.object(SchemaServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SchemaServiceAsyncClient))
def test_schema_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(SchemaServiceClient, transports.SchemaServiceGrpcTransport, 'grpc'), (SchemaServiceAsyncClient, transports.SchemaServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (SchemaServiceClient, transports.SchemaServiceRestTransport, 'rest')])
def test_schema_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(SchemaServiceClient, transports.SchemaServiceGrpcTransport, 'grpc', grpc_helpers), (SchemaServiceAsyncClient, transports.SchemaServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (SchemaServiceClient, transports.SchemaServiceRestTransport, 'rest', None)])
def test_schema_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_schema_service_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.discoveryengine_v1beta.services.schema_service.transports.SchemaServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = SchemaServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(SchemaServiceClient, transports.SchemaServiceGrpcTransport, 'grpc', grpc_helpers), (SchemaServiceAsyncClient, transports.SchemaServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_schema_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('discoveryengine.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='discoveryengine.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [schema_service.GetSchemaRequest, dict])
def test_get_schema(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_schema), '__call__') as call:
        call.return_value = schema.Schema(name='name_value', json_schema='json_schema_value')
        response = client.get_schema(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.GetSchemaRequest()
    assert isinstance(response, schema.Schema)
    assert response.name == 'name_value'

def test_get_schema_empty_call():
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_schema), '__call__') as call:
        client.get_schema()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.GetSchemaRequest()

@pytest.mark.asyncio
async def test_get_schema_async(transport: str='grpc_asyncio', request_type=schema_service.GetSchemaRequest):
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_schema), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(schema.Schema(name='name_value'))
        response = await client.get_schema(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.GetSchemaRequest()
    assert isinstance(response, schema.Schema)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_schema_async_from_dict():
    await test_get_schema_async(request_type=dict)

def test_get_schema_field_headers():
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = schema_service.GetSchemaRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_schema), '__call__') as call:
        call.return_value = schema.Schema()
        client.get_schema(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_schema_field_headers_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = schema_service.GetSchemaRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_schema), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(schema.Schema())
        await client.get_schema(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_schema_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_schema), '__call__') as call:
        call.return_value = schema.Schema()
        client.get_schema(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_schema_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_schema(schema_service.GetSchemaRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_schema_flattened_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_schema), '__call__') as call:
        call.return_value = schema.Schema()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(schema.Schema())
        response = await client.get_schema(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_schema_flattened_error_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_schema(schema_service.GetSchemaRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [schema_service.ListSchemasRequest, dict])
def test_list_schemas(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_schemas), '__call__') as call:
        call.return_value = schema_service.ListSchemasResponse(next_page_token='next_page_token_value')
        response = client.list_schemas(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.ListSchemasRequest()
    assert isinstance(response, pagers.ListSchemasPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_schemas_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_schemas), '__call__') as call:
        client.list_schemas()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.ListSchemasRequest()

@pytest.mark.asyncio
async def test_list_schemas_async(transport: str='grpc_asyncio', request_type=schema_service.ListSchemasRequest):
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_schemas), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(schema_service.ListSchemasResponse(next_page_token='next_page_token_value'))
        response = await client.list_schemas(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.ListSchemasRequest()
    assert isinstance(response, pagers.ListSchemasAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_schemas_async_from_dict():
    await test_list_schemas_async(request_type=dict)

def test_list_schemas_field_headers():
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = schema_service.ListSchemasRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_schemas), '__call__') as call:
        call.return_value = schema_service.ListSchemasResponse()
        client.list_schemas(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_schemas_field_headers_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = schema_service.ListSchemasRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_schemas), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(schema_service.ListSchemasResponse())
        await client.list_schemas(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_schemas_flattened():
    if False:
        print('Hello World!')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_schemas), '__call__') as call:
        call.return_value = schema_service.ListSchemasResponse()
        client.list_schemas(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_schemas_flattened_error():
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_schemas(schema_service.ListSchemasRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_schemas_flattened_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_schemas), '__call__') as call:
        call.return_value = schema_service.ListSchemasResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(schema_service.ListSchemasResponse())
        response = await client.list_schemas(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_schemas_flattened_error_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_schemas(schema_service.ListSchemasRequest(), parent='parent_value')

def test_list_schemas_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_schemas), '__call__') as call:
        call.side_effect = (schema_service.ListSchemasResponse(schemas=[schema.Schema(), schema.Schema(), schema.Schema()], next_page_token='abc'), schema_service.ListSchemasResponse(schemas=[], next_page_token='def'), schema_service.ListSchemasResponse(schemas=[schema.Schema()], next_page_token='ghi'), schema_service.ListSchemasResponse(schemas=[schema.Schema(), schema.Schema()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_schemas(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, schema.Schema) for i in results))

def test_list_schemas_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_schemas), '__call__') as call:
        call.side_effect = (schema_service.ListSchemasResponse(schemas=[schema.Schema(), schema.Schema(), schema.Schema()], next_page_token='abc'), schema_service.ListSchemasResponse(schemas=[], next_page_token='def'), schema_service.ListSchemasResponse(schemas=[schema.Schema()], next_page_token='ghi'), schema_service.ListSchemasResponse(schemas=[schema.Schema(), schema.Schema()]), RuntimeError)
        pages = list(client.list_schemas(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_schemas_async_pager():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_schemas), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (schema_service.ListSchemasResponse(schemas=[schema.Schema(), schema.Schema(), schema.Schema()], next_page_token='abc'), schema_service.ListSchemasResponse(schemas=[], next_page_token='def'), schema_service.ListSchemasResponse(schemas=[schema.Schema()], next_page_token='ghi'), schema_service.ListSchemasResponse(schemas=[schema.Schema(), schema.Schema()]), RuntimeError)
        async_pager = await client.list_schemas(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, schema.Schema) for i in responses))

@pytest.mark.asyncio
async def test_list_schemas_async_pages():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_schemas), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (schema_service.ListSchemasResponse(schemas=[schema.Schema(), schema.Schema(), schema.Schema()], next_page_token='abc'), schema_service.ListSchemasResponse(schemas=[], next_page_token='def'), schema_service.ListSchemasResponse(schemas=[schema.Schema()], next_page_token='ghi'), schema_service.ListSchemasResponse(schemas=[schema.Schema(), schema.Schema()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_schemas(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [schema_service.CreateSchemaRequest, dict])
def test_create_schema(request_type, transport: str='grpc'):
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_schema), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_schema(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.CreateSchemaRequest()
    assert isinstance(response, future.Future)

def test_create_schema_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_schema), '__call__') as call:
        client.create_schema()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.CreateSchemaRequest()

@pytest.mark.asyncio
async def test_create_schema_async(transport: str='grpc_asyncio', request_type=schema_service.CreateSchemaRequest):
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_schema), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_schema(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.CreateSchemaRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_schema_async_from_dict():
    await test_create_schema_async(request_type=dict)

def test_create_schema_field_headers():
    if False:
        i = 10
        return i + 15
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = schema_service.CreateSchemaRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_schema), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_schema(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_schema_field_headers_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = schema_service.CreateSchemaRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_schema), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_schema(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_schema_flattened():
    if False:
        print('Hello World!')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_schema), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_schema(parent='parent_value', schema=gcd_schema.Schema(struct_schema=struct_pb2.Struct(fields={'key_value': struct_pb2.Value(null_value=struct_pb2.NullValue.NULL_VALUE)})), schema_id='schema_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].schema
        mock_val = gcd_schema.Schema(struct_schema=struct_pb2.Struct(fields={'key_value': struct_pb2.Value(null_value=struct_pb2.NullValue.NULL_VALUE)}))
        assert arg == mock_val
        arg = args[0].schema_id
        mock_val = 'schema_id_value'
        assert arg == mock_val

def test_create_schema_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_schema(schema_service.CreateSchemaRequest(), parent='parent_value', schema=gcd_schema.Schema(struct_schema=struct_pb2.Struct(fields={'key_value': struct_pb2.Value(null_value=struct_pb2.NullValue.NULL_VALUE)})), schema_id='schema_id_value')

@pytest.mark.asyncio
async def test_create_schema_flattened_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_schema), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_schema(parent='parent_value', schema=gcd_schema.Schema(struct_schema=struct_pb2.Struct(fields={'key_value': struct_pb2.Value(null_value=struct_pb2.NullValue.NULL_VALUE)})), schema_id='schema_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].schema
        mock_val = gcd_schema.Schema(struct_schema=struct_pb2.Struct(fields={'key_value': struct_pb2.Value(null_value=struct_pb2.NullValue.NULL_VALUE)}))
        assert arg == mock_val
        arg = args[0].schema_id
        mock_val = 'schema_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_schema_flattened_error_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_schema(schema_service.CreateSchemaRequest(), parent='parent_value', schema=gcd_schema.Schema(struct_schema=struct_pb2.Struct(fields={'key_value': struct_pb2.Value(null_value=struct_pb2.NullValue.NULL_VALUE)})), schema_id='schema_id_value')

@pytest.mark.parametrize('request_type', [schema_service.UpdateSchemaRequest, dict])
def test_update_schema(request_type, transport: str='grpc'):
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_schema), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_schema(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.UpdateSchemaRequest()
    assert isinstance(response, future.Future)

def test_update_schema_empty_call():
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_schema), '__call__') as call:
        client.update_schema()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.UpdateSchemaRequest()

@pytest.mark.asyncio
async def test_update_schema_async(transport: str='grpc_asyncio', request_type=schema_service.UpdateSchemaRequest):
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_schema), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_schema(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.UpdateSchemaRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_schema_async_from_dict():
    await test_update_schema_async(request_type=dict)

def test_update_schema_field_headers():
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = schema_service.UpdateSchemaRequest()
    request.schema.name = 'name_value'
    with mock.patch.object(type(client.transport.update_schema), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_schema(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'schema.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_schema_field_headers_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = schema_service.UpdateSchemaRequest()
    request.schema.name = 'name_value'
    with mock.patch.object(type(client.transport.update_schema), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_schema(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'schema.name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [schema_service.DeleteSchemaRequest, dict])
def test_delete_schema(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_schema), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_schema(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.DeleteSchemaRequest()
    assert isinstance(response, future.Future)

def test_delete_schema_empty_call():
    if False:
        print('Hello World!')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_schema), '__call__') as call:
        client.delete_schema()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.DeleteSchemaRequest()

@pytest.mark.asyncio
async def test_delete_schema_async(transport: str='grpc_asyncio', request_type=schema_service.DeleteSchemaRequest):
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_schema), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_schema(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == schema_service.DeleteSchemaRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_schema_async_from_dict():
    await test_delete_schema_async(request_type=dict)

def test_delete_schema_field_headers():
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = schema_service.DeleteSchemaRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_schema), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_schema(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_schema_field_headers_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = schema_service.DeleteSchemaRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_schema), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_schema(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_schema_flattened():
    if False:
        while True:
            i = 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_schema), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_schema(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_schema_flattened_error():
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_schema(schema_service.DeleteSchemaRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_schema_flattened_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_schema), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_schema(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_schema_flattened_error_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_schema(schema_service.DeleteSchemaRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [schema_service.GetSchemaRequest, dict])
def test_get_schema_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/dataStores/sample3/schemas/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = schema.Schema(name='name_value', json_schema='json_schema_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = schema.Schema.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_schema(request)
    assert isinstance(response, schema.Schema)
    assert response.name == 'name_value'

def test_get_schema_rest_required_fields(request_type=schema_service.GetSchemaRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SchemaServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_schema._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_schema._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = schema.Schema()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = schema.Schema.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_schema(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_schema_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.SchemaServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_schema._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_schema_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SchemaServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SchemaServiceRestInterceptor())
    client = SchemaServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SchemaServiceRestInterceptor, 'post_get_schema') as post, mock.patch.object(transports.SchemaServiceRestInterceptor, 'pre_get_schema') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = schema_service.GetSchemaRequest.pb(schema_service.GetSchemaRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = schema.Schema.to_json(schema.Schema())
        request = schema_service.GetSchemaRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = schema.Schema()
        client.get_schema(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_schema_rest_bad_request(transport: str='rest', request_type=schema_service.GetSchemaRequest):
    if False:
        i = 10
        return i + 15
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/dataStores/sample3/schemas/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_schema(request)

def test_get_schema_rest_flattened():
    if False:
        print('Hello World!')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = schema.Schema()
        sample_request = {'name': 'projects/sample1/locations/sample2/dataStores/sample3/schemas/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = schema.Schema.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_schema(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/dataStores/*/schemas/*}' % client.transport._host, args[1])

def test_get_schema_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_schema(schema_service.GetSchemaRequest(), name='name_value')

def test_get_schema_rest_error():
    if False:
        while True:
            i = 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [schema_service.ListSchemasRequest, dict])
def test_list_schemas_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/dataStores/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = schema_service.ListSchemasResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = schema_service.ListSchemasResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_schemas(request)
    assert isinstance(response, pagers.ListSchemasPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_schemas_rest_required_fields(request_type=schema_service.ListSchemasRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SchemaServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_schemas._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_schemas._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = schema_service.ListSchemasResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = schema_service.ListSchemasResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_schemas(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_schemas_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SchemaServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_schemas._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_schemas_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SchemaServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SchemaServiceRestInterceptor())
    client = SchemaServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SchemaServiceRestInterceptor, 'post_list_schemas') as post, mock.patch.object(transports.SchemaServiceRestInterceptor, 'pre_list_schemas') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = schema_service.ListSchemasRequest.pb(schema_service.ListSchemasRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = schema_service.ListSchemasResponse.to_json(schema_service.ListSchemasResponse())
        request = schema_service.ListSchemasRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = schema_service.ListSchemasResponse()
        client.list_schemas(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_schemas_rest_bad_request(transport: str='rest', request_type=schema_service.ListSchemasRequest):
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/dataStores/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_schemas(request)

def test_list_schemas_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = schema_service.ListSchemasResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/dataStores/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = schema_service.ListSchemasResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_schemas(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/dataStores/*}/schemas' % client.transport._host, args[1])

def test_list_schemas_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_schemas(schema_service.ListSchemasRequest(), parent='parent_value')

def test_list_schemas_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (schema_service.ListSchemasResponse(schemas=[schema.Schema(), schema.Schema(), schema.Schema()], next_page_token='abc'), schema_service.ListSchemasResponse(schemas=[], next_page_token='def'), schema_service.ListSchemasResponse(schemas=[schema.Schema()], next_page_token='ghi'), schema_service.ListSchemasResponse(schemas=[schema.Schema(), schema.Schema()]))
        response = response + response
        response = tuple((schema_service.ListSchemasResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/dataStores/sample3'}
        pager = client.list_schemas(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, schema.Schema) for i in results))
        pages = list(client.list_schemas(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [schema_service.CreateSchemaRequest, dict])
def test_create_schema_rest(request_type):
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/dataStores/sample3'}
    request_init['schema'] = {'struct_schema': {'fields': {}}, 'json_schema': 'json_schema_value', 'name': 'name_value'}
    test_field = schema_service.CreateSchemaRequest.meta.fields['schema']

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
    for (field, value) in request_init['schema'].items():
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
                for i in range(0, len(request_init['schema'][field])):
                    del request_init['schema'][field][i][subfield]
            else:
                del request_init['schema'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_schema(request)
    assert response.operation.name == 'operations/spam'

def test_create_schema_rest_required_fields(request_type=schema_service.CreateSchemaRequest):
    if False:
        return 10
    transport_class = transports.SchemaServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['schema_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'schemaId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_schema._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'schemaId' in jsonified_request
    assert jsonified_request['schemaId'] == request_init['schema_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['schemaId'] = 'schema_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_schema._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('schema_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'schemaId' in jsonified_request
    assert jsonified_request['schemaId'] == 'schema_id_value'
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_schema(request)
            expected_params = [('schemaId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_schema_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.SchemaServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_schema._get_unset_required_fields({})
    assert set(unset_fields) == set(('schemaId',)) & set(('parent', 'schema', 'schemaId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_schema_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.SchemaServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SchemaServiceRestInterceptor())
    client = SchemaServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.SchemaServiceRestInterceptor, 'post_create_schema') as post, mock.patch.object(transports.SchemaServiceRestInterceptor, 'pre_create_schema') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = schema_service.CreateSchemaRequest.pb(schema_service.CreateSchemaRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = schema_service.CreateSchemaRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_schema(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_schema_rest_bad_request(transport: str='rest', request_type=schema_service.CreateSchemaRequest):
    if False:
        i = 10
        return i + 15
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/dataStores/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_schema(request)

def test_create_schema_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/dataStores/sample3'}
        mock_args = dict(parent='parent_value', schema=gcd_schema.Schema(struct_schema=struct_pb2.Struct(fields={'key_value': struct_pb2.Value(null_value=struct_pb2.NullValue.NULL_VALUE)})), schema_id='schema_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_schema(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/dataStores/*}/schemas' % client.transport._host, args[1])

def test_create_schema_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_schema(schema_service.CreateSchemaRequest(), parent='parent_value', schema=gcd_schema.Schema(struct_schema=struct_pb2.Struct(fields={'key_value': struct_pb2.Value(null_value=struct_pb2.NullValue.NULL_VALUE)})), schema_id='schema_id_value')

def test_create_schema_rest_error():
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [schema_service.UpdateSchemaRequest, dict])
def test_update_schema_rest(request_type):
    if False:
        while True:
            i = 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'schema': {'name': 'projects/sample1/locations/sample2/dataStores/sample3/schemas/sample4'}}
    request_init['schema'] = {'struct_schema': {'fields': {}}, 'json_schema': 'json_schema_value', 'name': 'projects/sample1/locations/sample2/dataStores/sample3/schemas/sample4'}
    test_field = schema_service.UpdateSchemaRequest.meta.fields['schema']

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
    for (field, value) in request_init['schema'].items():
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
                for i in range(0, len(request_init['schema'][field])):
                    del request_init['schema'][field][i][subfield]
            else:
                del request_init['schema'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_schema(request)
    assert response.operation.name == 'operations/spam'

def test_update_schema_rest_required_fields(request_type=schema_service.UpdateSchemaRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SchemaServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_schema._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_schema._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing',))
    jsonified_request.update(unset_fields)
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_schema(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_schema_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.SchemaServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_schema._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing',)) & set(('schema',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_schema_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SchemaServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SchemaServiceRestInterceptor())
    client = SchemaServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.SchemaServiceRestInterceptor, 'post_update_schema') as post, mock.patch.object(transports.SchemaServiceRestInterceptor, 'pre_update_schema') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = schema_service.UpdateSchemaRequest.pb(schema_service.UpdateSchemaRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = schema_service.UpdateSchemaRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_schema(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_schema_rest_bad_request(transport: str='rest', request_type=schema_service.UpdateSchemaRequest):
    if False:
        print('Hello World!')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'schema': {'name': 'projects/sample1/locations/sample2/dataStores/sample3/schemas/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_schema(request)

def test_update_schema_rest_error():
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [schema_service.DeleteSchemaRequest, dict])
def test_delete_schema_rest(request_type):
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/dataStores/sample3/schemas/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_schema(request)
    assert response.operation.name == 'operations/spam'

def test_delete_schema_rest_required_fields(request_type=schema_service.DeleteSchemaRequest):
    if False:
        print('Hello World!')
    transport_class = transports.SchemaServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_schema._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_schema._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_schema(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_schema_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.SchemaServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_schema._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_schema_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.SchemaServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SchemaServiceRestInterceptor())
    client = SchemaServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.SchemaServiceRestInterceptor, 'post_delete_schema') as post, mock.patch.object(transports.SchemaServiceRestInterceptor, 'pre_delete_schema') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = schema_service.DeleteSchemaRequest.pb(schema_service.DeleteSchemaRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = schema_service.DeleteSchemaRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_schema(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_schema_rest_bad_request(transport: str='rest', request_type=schema_service.DeleteSchemaRequest):
    if False:
        while True:
            i = 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/dataStores/sample3/schemas/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_schema(request)

def test_delete_schema_rest_flattened():
    if False:
        print('Hello World!')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/dataStores/sample3/schemas/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_schema(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/dataStores/*/schemas/*}' % client.transport._host, args[1])

def test_delete_schema_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_schema(schema_service.DeleteSchemaRequest(), name='name_value')

def test_delete_schema_rest_error():
    if False:
        i = 10
        return i + 15
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.SchemaServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.SchemaServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SchemaServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.SchemaServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = SchemaServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = SchemaServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.SchemaServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SchemaServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.SchemaServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = SchemaServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.SchemaServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.SchemaServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.SchemaServiceGrpcTransport, transports.SchemaServiceGrpcAsyncIOTransport, transports.SchemaServiceRestTransport])
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
    transport = SchemaServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.SchemaServiceGrpcTransport)

def test_schema_service_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.SchemaServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_schema_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.discoveryengine_v1beta.services.schema_service.transports.SchemaServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.SchemaServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_schema', 'list_schemas', 'create_schema', 'update_schema', 'delete_schema', 'get_operation', 'list_operations')
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

def test_schema_service_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.discoveryengine_v1beta.services.schema_service.transports.SchemaServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.SchemaServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_schema_service_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.discoveryengine_v1beta.services.schema_service.transports.SchemaServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.SchemaServiceTransport()
        adc.assert_called_once()

def test_schema_service_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        SchemaServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.SchemaServiceGrpcTransport, transports.SchemaServiceGrpcAsyncIOTransport])
def test_schema_service_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.SchemaServiceGrpcTransport, transports.SchemaServiceGrpcAsyncIOTransport, transports.SchemaServiceRestTransport])
def test_schema_service_transport_auth_gdch_credentials(transport_class):
    if False:
        i = 10
        return i + 15
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.SchemaServiceGrpcTransport, grpc_helpers), (transports.SchemaServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_schema_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('discoveryengine.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='discoveryengine.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.SchemaServiceGrpcTransport, transports.SchemaServiceGrpcAsyncIOTransport])
def test_schema_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_schema_service_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.SchemaServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_schema_service_rest_lro_client():
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_schema_service_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='discoveryengine.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('discoveryengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://discoveryengine.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_schema_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='discoveryengine.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('discoveryengine.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://discoveryengine.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_schema_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = SchemaServiceClient(credentials=creds1, transport=transport_name)
    client2 = SchemaServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.get_schema._session
    session2 = client2.transport.get_schema._session
    assert session1 != session2
    session1 = client1.transport.list_schemas._session
    session2 = client2.transport.list_schemas._session
    assert session1 != session2
    session1 = client1.transport.create_schema._session
    session2 = client2.transport.create_schema._session
    assert session1 != session2
    session1 = client1.transport.update_schema._session
    session2 = client2.transport.update_schema._session
    assert session1 != session2
    session1 = client1.transport.delete_schema._session
    session2 = client2.transport.delete_schema._session
    assert session1 != session2

def test_schema_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.SchemaServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_schema_service_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.SchemaServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.SchemaServiceGrpcTransport, transports.SchemaServiceGrpcAsyncIOTransport])
def test_schema_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.SchemaServiceGrpcTransport, transports.SchemaServiceGrpcAsyncIOTransport])
def test_schema_service_transport_channel_mtls_with_adc(transport_class):
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

def test_schema_service_grpc_lro_client():
    if False:
        print('Hello World!')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_schema_service_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_data_store_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    data_store = 'whelk'
    expected = 'projects/{project}/locations/{location}/dataStores/{data_store}'.format(project=project, location=location, data_store=data_store)
    actual = SchemaServiceClient.data_store_path(project, location, data_store)
    assert expected == actual

def test_parse_data_store_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus', 'location': 'oyster', 'data_store': 'nudibranch'}
    path = SchemaServiceClient.data_store_path(**expected)
    actual = SchemaServiceClient.parse_data_store_path(path)
    assert expected == actual

def test_schema_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    location = 'mussel'
    data_store = 'winkle'
    schema = 'nautilus'
    expected = 'projects/{project}/locations/{location}/dataStores/{data_store}/schemas/{schema}'.format(project=project, location=location, data_store=data_store, schema=schema)
    actual = SchemaServiceClient.schema_path(project, location, data_store, schema)
    assert expected == actual

def test_parse_schema_path():
    if False:
        return 10
    expected = {'project': 'scallop', 'location': 'abalone', 'data_store': 'squid', 'schema': 'clam'}
    path = SchemaServiceClient.schema_path(**expected)
    actual = SchemaServiceClient.parse_schema_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = SchemaServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'octopus'}
    path = SchemaServiceClient.common_billing_account_path(**expected)
    actual = SchemaServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = SchemaServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'nudibranch'}
    path = SchemaServiceClient.common_folder_path(**expected)
    actual = SchemaServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = SchemaServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'mussel'}
    path = SchemaServiceClient.common_organization_path(**expected)
    actual = SchemaServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = SchemaServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nautilus'}
    path = SchemaServiceClient.common_project_path(**expected)
    actual = SchemaServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = SchemaServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'squid', 'location': 'clam'}
    path = SchemaServiceClient.common_location_path(**expected)
    actual = SchemaServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.SchemaServiceTransport, '_prep_wrapped_messages') as prep:
        client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.SchemaServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = SchemaServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/collections/sample3/dataStores/sample4/branches/sample5/operations/sample6'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.GetOperationRequest, dict])
def test_get_operation_rest(request_type):
    if False:
        while True:
            i = 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/collections/sample3/dataStores/sample4/branches/sample5/operations/sample6'}
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
        i = 10
        return i + 15
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/collections/sample3/dataStores/sample4/branches/sample5'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_operations(request)

@pytest.mark.parametrize('request_type', [operations_pb2.ListOperationsRequest, dict])
def test_list_operations_rest(request_type):
    if False:
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/collections/sample3/dataStores/sample4/branches/sample5'}
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
        return 10
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = SchemaServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = SchemaServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(SchemaServiceClient, transports.SchemaServiceGrpcTransport), (SchemaServiceAsyncClient, transports.SchemaServiceGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)