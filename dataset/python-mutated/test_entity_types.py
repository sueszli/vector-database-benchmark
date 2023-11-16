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
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import struct_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.dialogflow_v2.services.entity_types import EntityTypesAsyncClient, EntityTypesClient, pagers, transports
from google.cloud.dialogflow_v2.types import entity_type
from google.cloud.dialogflow_v2.types import entity_type as gcd_entity_type

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
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
    assert EntityTypesClient._get_default_mtls_endpoint(None) is None
    assert EntityTypesClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert EntityTypesClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert EntityTypesClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert EntityTypesClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert EntityTypesClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(EntityTypesClient, 'grpc'), (EntityTypesAsyncClient, 'grpc_asyncio'), (EntityTypesClient, 'rest')])
def test_entity_types_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.EntityTypesGrpcTransport, 'grpc'), (transports.EntityTypesGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.EntityTypesRestTransport, 'rest')])
def test_entity_types_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(EntityTypesClient, 'grpc'), (EntityTypesAsyncClient, 'grpc_asyncio'), (EntityTypesClient, 'rest')])
def test_entity_types_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

def test_entity_types_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = EntityTypesClient.get_transport_class()
    available_transports = [transports.EntityTypesGrpcTransport, transports.EntityTypesRestTransport]
    assert transport in available_transports
    transport = EntityTypesClient.get_transport_class('grpc')
    assert transport == transports.EntityTypesGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(EntityTypesClient, transports.EntityTypesGrpcTransport, 'grpc'), (EntityTypesAsyncClient, transports.EntityTypesGrpcAsyncIOTransport, 'grpc_asyncio'), (EntityTypesClient, transports.EntityTypesRestTransport, 'rest')])
@mock.patch.object(EntityTypesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EntityTypesClient))
@mock.patch.object(EntityTypesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EntityTypesAsyncClient))
def test_entity_types_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(EntityTypesClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(EntityTypesClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(EntityTypesClient, transports.EntityTypesGrpcTransport, 'grpc', 'true'), (EntityTypesAsyncClient, transports.EntityTypesGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (EntityTypesClient, transports.EntityTypesGrpcTransport, 'grpc', 'false'), (EntityTypesAsyncClient, transports.EntityTypesGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (EntityTypesClient, transports.EntityTypesRestTransport, 'rest', 'true'), (EntityTypesClient, transports.EntityTypesRestTransport, 'rest', 'false')])
@mock.patch.object(EntityTypesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EntityTypesClient))
@mock.patch.object(EntityTypesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EntityTypesAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_entity_types_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [EntityTypesClient, EntityTypesAsyncClient])
@mock.patch.object(EntityTypesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EntityTypesClient))
@mock.patch.object(EntityTypesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EntityTypesAsyncClient))
def test_entity_types_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(EntityTypesClient, transports.EntityTypesGrpcTransport, 'grpc'), (EntityTypesAsyncClient, transports.EntityTypesGrpcAsyncIOTransport, 'grpc_asyncio'), (EntityTypesClient, transports.EntityTypesRestTransport, 'rest')])
def test_entity_types_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(EntityTypesClient, transports.EntityTypesGrpcTransport, 'grpc', grpc_helpers), (EntityTypesAsyncClient, transports.EntityTypesGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (EntityTypesClient, transports.EntityTypesRestTransport, 'rest', None)])
def test_entity_types_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_entity_types_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.dialogflow_v2.services.entity_types.transports.EntityTypesGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = EntityTypesClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(EntityTypesClient, transports.EntityTypesGrpcTransport, 'grpc', grpc_helpers), (EntityTypesAsyncClient, transports.EntityTypesGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_entity_types_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('dialogflow.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), scopes=None, default_host='dialogflow.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [entity_type.ListEntityTypesRequest, dict])
def test_list_entity_types(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_entity_types), '__call__') as call:
        call.return_value = entity_type.ListEntityTypesResponse(next_page_token='next_page_token_value')
        response = client.list_entity_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.ListEntityTypesRequest()
    assert isinstance(response, pagers.ListEntityTypesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_entity_types_empty_call():
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_entity_types), '__call__') as call:
        client.list_entity_types()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.ListEntityTypesRequest()

@pytest.mark.asyncio
async def test_list_entity_types_async(transport: str='grpc_asyncio', request_type=entity_type.ListEntityTypesRequest):
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_entity_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(entity_type.ListEntityTypesResponse(next_page_token='next_page_token_value'))
        response = await client.list_entity_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.ListEntityTypesRequest()
    assert isinstance(response, pagers.ListEntityTypesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_entity_types_async_from_dict():
    await test_list_entity_types_async(request_type=dict)

def test_list_entity_types_field_headers():
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.ListEntityTypesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_entity_types), '__call__') as call:
        call.return_value = entity_type.ListEntityTypesResponse()
        client.list_entity_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_entity_types_field_headers_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.ListEntityTypesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_entity_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(entity_type.ListEntityTypesResponse())
        await client.list_entity_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_entity_types_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_entity_types), '__call__') as call:
        call.return_value = entity_type.ListEntityTypesResponse()
        client.list_entity_types(parent='parent_value', language_code='language_code_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

def test_list_entity_types_flattened_error():
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_entity_types(entity_type.ListEntityTypesRequest(), parent='parent_value', language_code='language_code_value')

@pytest.mark.asyncio
async def test_list_entity_types_flattened_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_entity_types), '__call__') as call:
        call.return_value = entity_type.ListEntityTypesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(entity_type.ListEntityTypesResponse())
        response = await client.list_entity_types(parent='parent_value', language_code='language_code_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_entity_types_flattened_error_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_entity_types(entity_type.ListEntityTypesRequest(), parent='parent_value', language_code='language_code_value')

def test_list_entity_types_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_entity_types), '__call__') as call:
        call.side_effect = (entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType(), entity_type.EntityType(), entity_type.EntityType()], next_page_token='abc'), entity_type.ListEntityTypesResponse(entity_types=[], next_page_token='def'), entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType()], next_page_token='ghi'), entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType(), entity_type.EntityType()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_entity_types(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, entity_type.EntityType) for i in results))

def test_list_entity_types_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_entity_types), '__call__') as call:
        call.side_effect = (entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType(), entity_type.EntityType(), entity_type.EntityType()], next_page_token='abc'), entity_type.ListEntityTypesResponse(entity_types=[], next_page_token='def'), entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType()], next_page_token='ghi'), entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType(), entity_type.EntityType()]), RuntimeError)
        pages = list(client.list_entity_types(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_entity_types_async_pager():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_entity_types), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType(), entity_type.EntityType(), entity_type.EntityType()], next_page_token='abc'), entity_type.ListEntityTypesResponse(entity_types=[], next_page_token='def'), entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType()], next_page_token='ghi'), entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType(), entity_type.EntityType()]), RuntimeError)
        async_pager = await client.list_entity_types(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, entity_type.EntityType) for i in responses))

@pytest.mark.asyncio
async def test_list_entity_types_async_pages():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_entity_types), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType(), entity_type.EntityType(), entity_type.EntityType()], next_page_token='abc'), entity_type.ListEntityTypesResponse(entity_types=[], next_page_token='def'), entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType()], next_page_token='ghi'), entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType(), entity_type.EntityType()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_entity_types(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [entity_type.GetEntityTypeRequest, dict])
def test_get_entity_type(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_entity_type), '__call__') as call:
        call.return_value = entity_type.EntityType(name='name_value', display_name='display_name_value', kind=entity_type.EntityType.Kind.KIND_MAP, auto_expansion_mode=entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT, enable_fuzzy_extraction=True)
        response = client.get_entity_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.GetEntityTypeRequest()
    assert isinstance(response, entity_type.EntityType)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.kind == entity_type.EntityType.Kind.KIND_MAP
    assert response.auto_expansion_mode == entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT
    assert response.enable_fuzzy_extraction is True

def test_get_entity_type_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_entity_type), '__call__') as call:
        client.get_entity_type()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.GetEntityTypeRequest()

@pytest.mark.asyncio
async def test_get_entity_type_async(transport: str='grpc_asyncio', request_type=entity_type.GetEntityTypeRequest):
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_entity_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(entity_type.EntityType(name='name_value', display_name='display_name_value', kind=entity_type.EntityType.Kind.KIND_MAP, auto_expansion_mode=entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT, enable_fuzzy_extraction=True))
        response = await client.get_entity_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.GetEntityTypeRequest()
    assert isinstance(response, entity_type.EntityType)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.kind == entity_type.EntityType.Kind.KIND_MAP
    assert response.auto_expansion_mode == entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT
    assert response.enable_fuzzy_extraction is True

@pytest.mark.asyncio
async def test_get_entity_type_async_from_dict():
    await test_get_entity_type_async(request_type=dict)

def test_get_entity_type_field_headers():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.GetEntityTypeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_entity_type), '__call__') as call:
        call.return_value = entity_type.EntityType()
        client.get_entity_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_entity_type_field_headers_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.GetEntityTypeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_entity_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(entity_type.EntityType())
        await client.get_entity_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_entity_type_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_entity_type), '__call__') as call:
        call.return_value = entity_type.EntityType()
        client.get_entity_type(name='name_value', language_code='language_code_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

def test_get_entity_type_flattened_error():
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_entity_type(entity_type.GetEntityTypeRequest(), name='name_value', language_code='language_code_value')

@pytest.mark.asyncio
async def test_get_entity_type_flattened_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_entity_type), '__call__') as call:
        call.return_value = entity_type.EntityType()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(entity_type.EntityType())
        response = await client.get_entity_type(name='name_value', language_code='language_code_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_entity_type_flattened_error_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_entity_type(entity_type.GetEntityTypeRequest(), name='name_value', language_code='language_code_value')

@pytest.mark.parametrize('request_type', [gcd_entity_type.CreateEntityTypeRequest, dict])
def test_create_entity_type(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_entity_type), '__call__') as call:
        call.return_value = gcd_entity_type.EntityType(name='name_value', display_name='display_name_value', kind=gcd_entity_type.EntityType.Kind.KIND_MAP, auto_expansion_mode=gcd_entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT, enable_fuzzy_extraction=True)
        response = client.create_entity_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_entity_type.CreateEntityTypeRequest()
    assert isinstance(response, gcd_entity_type.EntityType)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.kind == gcd_entity_type.EntityType.Kind.KIND_MAP
    assert response.auto_expansion_mode == gcd_entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT
    assert response.enable_fuzzy_extraction is True

def test_create_entity_type_empty_call():
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_entity_type), '__call__') as call:
        client.create_entity_type()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_entity_type.CreateEntityTypeRequest()

@pytest.mark.asyncio
async def test_create_entity_type_async(transport: str='grpc_asyncio', request_type=gcd_entity_type.CreateEntityTypeRequest):
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_entity_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_entity_type.EntityType(name='name_value', display_name='display_name_value', kind=gcd_entity_type.EntityType.Kind.KIND_MAP, auto_expansion_mode=gcd_entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT, enable_fuzzy_extraction=True))
        response = await client.create_entity_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_entity_type.CreateEntityTypeRequest()
    assert isinstance(response, gcd_entity_type.EntityType)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.kind == gcd_entity_type.EntityType.Kind.KIND_MAP
    assert response.auto_expansion_mode == gcd_entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT
    assert response.enable_fuzzy_extraction is True

@pytest.mark.asyncio
async def test_create_entity_type_async_from_dict():
    await test_create_entity_type_async(request_type=dict)

def test_create_entity_type_field_headers():
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_entity_type.CreateEntityTypeRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_entity_type), '__call__') as call:
        call.return_value = gcd_entity_type.EntityType()
        client.create_entity_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_entity_type_field_headers_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_entity_type.CreateEntityTypeRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_entity_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_entity_type.EntityType())
        await client.create_entity_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_entity_type_flattened():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_entity_type), '__call__') as call:
        call.return_value = gcd_entity_type.EntityType()
        client.create_entity_type(parent='parent_value', entity_type=gcd_entity_type.EntityType(name='name_value'), language_code='language_code_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].entity_type
        mock_val = gcd_entity_type.EntityType(name='name_value')
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

def test_create_entity_type_flattened_error():
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_entity_type(gcd_entity_type.CreateEntityTypeRequest(), parent='parent_value', entity_type=gcd_entity_type.EntityType(name='name_value'), language_code='language_code_value')

@pytest.mark.asyncio
async def test_create_entity_type_flattened_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_entity_type), '__call__') as call:
        call.return_value = gcd_entity_type.EntityType()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_entity_type.EntityType())
        response = await client.create_entity_type(parent='parent_value', entity_type=gcd_entity_type.EntityType(name='name_value'), language_code='language_code_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].entity_type
        mock_val = gcd_entity_type.EntityType(name='name_value')
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_entity_type_flattened_error_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_entity_type(gcd_entity_type.CreateEntityTypeRequest(), parent='parent_value', entity_type=gcd_entity_type.EntityType(name='name_value'), language_code='language_code_value')

@pytest.mark.parametrize('request_type', [gcd_entity_type.UpdateEntityTypeRequest, dict])
def test_update_entity_type(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_entity_type), '__call__') as call:
        call.return_value = gcd_entity_type.EntityType(name='name_value', display_name='display_name_value', kind=gcd_entity_type.EntityType.Kind.KIND_MAP, auto_expansion_mode=gcd_entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT, enable_fuzzy_extraction=True)
        response = client.update_entity_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_entity_type.UpdateEntityTypeRequest()
    assert isinstance(response, gcd_entity_type.EntityType)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.kind == gcd_entity_type.EntityType.Kind.KIND_MAP
    assert response.auto_expansion_mode == gcd_entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT
    assert response.enable_fuzzy_extraction is True

def test_update_entity_type_empty_call():
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_entity_type), '__call__') as call:
        client.update_entity_type()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_entity_type.UpdateEntityTypeRequest()

@pytest.mark.asyncio
async def test_update_entity_type_async(transport: str='grpc_asyncio', request_type=gcd_entity_type.UpdateEntityTypeRequest):
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_entity_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_entity_type.EntityType(name='name_value', display_name='display_name_value', kind=gcd_entity_type.EntityType.Kind.KIND_MAP, auto_expansion_mode=gcd_entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT, enable_fuzzy_extraction=True))
        response = await client.update_entity_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_entity_type.UpdateEntityTypeRequest()
    assert isinstance(response, gcd_entity_type.EntityType)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.kind == gcd_entity_type.EntityType.Kind.KIND_MAP
    assert response.auto_expansion_mode == gcd_entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT
    assert response.enable_fuzzy_extraction is True

@pytest.mark.asyncio
async def test_update_entity_type_async_from_dict():
    await test_update_entity_type_async(request_type=dict)

def test_update_entity_type_field_headers():
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_entity_type.UpdateEntityTypeRequest()
    request.entity_type.name = 'name_value'
    with mock.patch.object(type(client.transport.update_entity_type), '__call__') as call:
        call.return_value = gcd_entity_type.EntityType()
        client.update_entity_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'entity_type.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_entity_type_field_headers_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_entity_type.UpdateEntityTypeRequest()
    request.entity_type.name = 'name_value'
    with mock.patch.object(type(client.transport.update_entity_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_entity_type.EntityType())
        await client.update_entity_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'entity_type.name=name_value') in kw['metadata']

def test_update_entity_type_flattened():
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_entity_type), '__call__') as call:
        call.return_value = gcd_entity_type.EntityType()
        client.update_entity_type(entity_type=gcd_entity_type.EntityType(name='name_value'), language_code='language_code_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].entity_type
        mock_val = gcd_entity_type.EntityType(name='name_value')
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

def test_update_entity_type_flattened_error():
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_entity_type(gcd_entity_type.UpdateEntityTypeRequest(), entity_type=gcd_entity_type.EntityType(name='name_value'), language_code='language_code_value')

@pytest.mark.asyncio
async def test_update_entity_type_flattened_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_entity_type), '__call__') as call:
        call.return_value = gcd_entity_type.EntityType()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_entity_type.EntityType())
        response = await client.update_entity_type(entity_type=gcd_entity_type.EntityType(name='name_value'), language_code='language_code_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].entity_type
        mock_val = gcd_entity_type.EntityType(name='name_value')
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_entity_type_flattened_error_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_entity_type(gcd_entity_type.UpdateEntityTypeRequest(), entity_type=gcd_entity_type.EntityType(name='name_value'), language_code='language_code_value')

@pytest.mark.parametrize('request_type', [entity_type.DeleteEntityTypeRequest, dict])
def test_delete_entity_type(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_entity_type), '__call__') as call:
        call.return_value = None
        response = client.delete_entity_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.DeleteEntityTypeRequest()
    assert response is None

def test_delete_entity_type_empty_call():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_entity_type), '__call__') as call:
        client.delete_entity_type()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.DeleteEntityTypeRequest()

@pytest.mark.asyncio
async def test_delete_entity_type_async(transport: str='grpc_asyncio', request_type=entity_type.DeleteEntityTypeRequest):
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_entity_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_entity_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.DeleteEntityTypeRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_entity_type_async_from_dict():
    await test_delete_entity_type_async(request_type=dict)

def test_delete_entity_type_field_headers():
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.DeleteEntityTypeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_entity_type), '__call__') as call:
        call.return_value = None
        client.delete_entity_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_entity_type_field_headers_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.DeleteEntityTypeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_entity_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_entity_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_entity_type_flattened():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_entity_type), '__call__') as call:
        call.return_value = None
        client.delete_entity_type(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_entity_type_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_entity_type(entity_type.DeleteEntityTypeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_entity_type_flattened_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_entity_type), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_entity_type(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_entity_type_flattened_error_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_entity_type(entity_type.DeleteEntityTypeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [entity_type.BatchUpdateEntityTypesRequest, dict])
def test_batch_update_entity_types(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_update_entity_types), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_update_entity_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchUpdateEntityTypesRequest()
    assert isinstance(response, future.Future)

def test_batch_update_entity_types_empty_call():
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_update_entity_types), '__call__') as call:
        client.batch_update_entity_types()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchUpdateEntityTypesRequest()

@pytest.mark.asyncio
async def test_batch_update_entity_types_async(transport: str='grpc_asyncio', request_type=entity_type.BatchUpdateEntityTypesRequest):
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_update_entity_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_update_entity_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchUpdateEntityTypesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_update_entity_types_async_from_dict():
    await test_batch_update_entity_types_async(request_type=dict)

def test_batch_update_entity_types_field_headers():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.BatchUpdateEntityTypesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_update_entity_types), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_update_entity_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_update_entity_types_field_headers_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.BatchUpdateEntityTypesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_update_entity_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_update_entity_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [entity_type.BatchDeleteEntityTypesRequest, dict])
def test_batch_delete_entity_types(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_delete_entity_types), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_delete_entity_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchDeleteEntityTypesRequest()
    assert isinstance(response, future.Future)

def test_batch_delete_entity_types_empty_call():
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_delete_entity_types), '__call__') as call:
        client.batch_delete_entity_types()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchDeleteEntityTypesRequest()

@pytest.mark.asyncio
async def test_batch_delete_entity_types_async(transport: str='grpc_asyncio', request_type=entity_type.BatchDeleteEntityTypesRequest):
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_delete_entity_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_delete_entity_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchDeleteEntityTypesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_delete_entity_types_async_from_dict():
    await test_batch_delete_entity_types_async(request_type=dict)

def test_batch_delete_entity_types_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.BatchDeleteEntityTypesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_delete_entity_types), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_delete_entity_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_delete_entity_types_field_headers_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.BatchDeleteEntityTypesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_delete_entity_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_delete_entity_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_batch_delete_entity_types_flattened():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_delete_entity_types), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_delete_entity_types(parent='parent_value', entity_type_names=['entity_type_names_value'])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].entity_type_names
        mock_val = ['entity_type_names_value']
        assert arg == mock_val

def test_batch_delete_entity_types_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_delete_entity_types(entity_type.BatchDeleteEntityTypesRequest(), parent='parent_value', entity_type_names=['entity_type_names_value'])

@pytest.mark.asyncio
async def test_batch_delete_entity_types_flattened_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_delete_entity_types), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_delete_entity_types(parent='parent_value', entity_type_names=['entity_type_names_value'])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].entity_type_names
        mock_val = ['entity_type_names_value']
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_delete_entity_types_flattened_error_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_delete_entity_types(entity_type.BatchDeleteEntityTypesRequest(), parent='parent_value', entity_type_names=['entity_type_names_value'])

@pytest.mark.parametrize('request_type', [entity_type.BatchCreateEntitiesRequest, dict])
def test_batch_create_entities(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_entities), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_create_entities(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchCreateEntitiesRequest()
    assert isinstance(response, future.Future)

def test_batch_create_entities_empty_call():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_create_entities), '__call__') as call:
        client.batch_create_entities()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchCreateEntitiesRequest()

@pytest.mark.asyncio
async def test_batch_create_entities_async(transport: str='grpc_asyncio', request_type=entity_type.BatchCreateEntitiesRequest):
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_entities), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_create_entities(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchCreateEntitiesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_create_entities_async_from_dict():
    await test_batch_create_entities_async(request_type=dict)

def test_batch_create_entities_field_headers():
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.BatchCreateEntitiesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_create_entities), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_create_entities(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_create_entities_field_headers_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.BatchCreateEntitiesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_create_entities), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_create_entities(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_batch_create_entities_flattened():
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_create_entities), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_create_entities(parent='parent_value', entities=[entity_type.EntityType.Entity(value='value_value')], language_code='language_code_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].entities
        mock_val = [entity_type.EntityType.Entity(value='value_value')]
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

def test_batch_create_entities_flattened_error():
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_create_entities(entity_type.BatchCreateEntitiesRequest(), parent='parent_value', entities=[entity_type.EntityType.Entity(value='value_value')], language_code='language_code_value')

@pytest.mark.asyncio
async def test_batch_create_entities_flattened_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_create_entities), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_create_entities(parent='parent_value', entities=[entity_type.EntityType.Entity(value='value_value')], language_code='language_code_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].entities
        mock_val = [entity_type.EntityType.Entity(value='value_value')]
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_create_entities_flattened_error_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_create_entities(entity_type.BatchCreateEntitiesRequest(), parent='parent_value', entities=[entity_type.EntityType.Entity(value='value_value')], language_code='language_code_value')

@pytest.mark.parametrize('request_type', [entity_type.BatchUpdateEntitiesRequest, dict])
def test_batch_update_entities(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_update_entities), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_update_entities(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchUpdateEntitiesRequest()
    assert isinstance(response, future.Future)

def test_batch_update_entities_empty_call():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_update_entities), '__call__') as call:
        client.batch_update_entities()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchUpdateEntitiesRequest()

@pytest.mark.asyncio
async def test_batch_update_entities_async(transport: str='grpc_asyncio', request_type=entity_type.BatchUpdateEntitiesRequest):
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_update_entities), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_update_entities(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchUpdateEntitiesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_update_entities_async_from_dict():
    await test_batch_update_entities_async(request_type=dict)

def test_batch_update_entities_field_headers():
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.BatchUpdateEntitiesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_update_entities), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_update_entities(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_update_entities_field_headers_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.BatchUpdateEntitiesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_update_entities), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_update_entities(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_batch_update_entities_flattened():
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_update_entities), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_update_entities(parent='parent_value', entities=[entity_type.EntityType.Entity(value='value_value')], language_code='language_code_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].entities
        mock_val = [entity_type.EntityType.Entity(value='value_value')]
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

def test_batch_update_entities_flattened_error():
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_update_entities(entity_type.BatchUpdateEntitiesRequest(), parent='parent_value', entities=[entity_type.EntityType.Entity(value='value_value')], language_code='language_code_value')

@pytest.mark.asyncio
async def test_batch_update_entities_flattened_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_update_entities), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_update_entities(parent='parent_value', entities=[entity_type.EntityType.Entity(value='value_value')], language_code='language_code_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].entities
        mock_val = [entity_type.EntityType.Entity(value='value_value')]
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_update_entities_flattened_error_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_update_entities(entity_type.BatchUpdateEntitiesRequest(), parent='parent_value', entities=[entity_type.EntityType.Entity(value='value_value')], language_code='language_code_value')

@pytest.mark.parametrize('request_type', [entity_type.BatchDeleteEntitiesRequest, dict])
def test_batch_delete_entities(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_delete_entities), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_delete_entities(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchDeleteEntitiesRequest()
    assert isinstance(response, future.Future)

def test_batch_delete_entities_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_delete_entities), '__call__') as call:
        client.batch_delete_entities()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchDeleteEntitiesRequest()

@pytest.mark.asyncio
async def test_batch_delete_entities_async(transport: str='grpc_asyncio', request_type=entity_type.BatchDeleteEntitiesRequest):
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_delete_entities), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_delete_entities(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == entity_type.BatchDeleteEntitiesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_delete_entities_async_from_dict():
    await test_batch_delete_entities_async(request_type=dict)

def test_batch_delete_entities_field_headers():
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.BatchDeleteEntitiesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_delete_entities), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_delete_entities(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_delete_entities_field_headers_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = entity_type.BatchDeleteEntitiesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_delete_entities), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_delete_entities(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_batch_delete_entities_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_delete_entities), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_delete_entities(parent='parent_value', entity_values=['entity_values_value'], language_code='language_code_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].entity_values
        mock_val = ['entity_values_value']
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

def test_batch_delete_entities_flattened_error():
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_delete_entities(entity_type.BatchDeleteEntitiesRequest(), parent='parent_value', entity_values=['entity_values_value'], language_code='language_code_value')

@pytest.mark.asyncio
async def test_batch_delete_entities_flattened_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_delete_entities), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_delete_entities(parent='parent_value', entity_values=['entity_values_value'], language_code='language_code_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].entity_values
        mock_val = ['entity_values_value']
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_delete_entities_flattened_error_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_delete_entities(entity_type.BatchDeleteEntitiesRequest(), parent='parent_value', entity_values=['entity_values_value'], language_code='language_code_value')

@pytest.mark.parametrize('request_type', [entity_type.ListEntityTypesRequest, dict])
def test_list_entity_types_rest(request_type):
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/agent'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = entity_type.ListEntityTypesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = entity_type.ListEntityTypesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_entity_types(request)
    assert isinstance(response, pagers.ListEntityTypesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_entity_types_rest_required_fields(request_type=entity_type.ListEntityTypesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EntityTypesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_entity_types._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_entity_types._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('language_code', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = entity_type.ListEntityTypesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = entity_type.ListEntityTypesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_entity_types(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_entity_types_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_entity_types._get_unset_required_fields({})
    assert set(unset_fields) == set(('languageCode', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_entity_types_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EntityTypesRestInterceptor())
    client = EntityTypesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EntityTypesRestInterceptor, 'post_list_entity_types') as post, mock.patch.object(transports.EntityTypesRestInterceptor, 'pre_list_entity_types') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = entity_type.ListEntityTypesRequest.pb(entity_type.ListEntityTypesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = entity_type.ListEntityTypesResponse.to_json(entity_type.ListEntityTypesResponse())
        request = entity_type.ListEntityTypesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = entity_type.ListEntityTypesResponse()
        client.list_entity_types(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_entity_types_rest_bad_request(transport: str='rest', request_type=entity_type.ListEntityTypesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/agent'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_entity_types(request)

def test_list_entity_types_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = entity_type.ListEntityTypesResponse()
        sample_request = {'parent': 'projects/sample1/agent'}
        mock_args = dict(parent='parent_value', language_code='language_code_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = entity_type.ListEntityTypesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_entity_types(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/agent}/entityTypes' % client.transport._host, args[1])

def test_list_entity_types_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_entity_types(entity_type.ListEntityTypesRequest(), parent='parent_value', language_code='language_code_value')

def test_list_entity_types_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType(), entity_type.EntityType(), entity_type.EntityType()], next_page_token='abc'), entity_type.ListEntityTypesResponse(entity_types=[], next_page_token='def'), entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType()], next_page_token='ghi'), entity_type.ListEntityTypesResponse(entity_types=[entity_type.EntityType(), entity_type.EntityType()]))
        response = response + response
        response = tuple((entity_type.ListEntityTypesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/agent'}
        pager = client.list_entity_types(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, entity_type.EntityType) for i in results))
        pages = list(client.list_entity_types(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [entity_type.GetEntityTypeRequest, dict])
def test_get_entity_type_rest(request_type):
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/agent/entityTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = entity_type.EntityType(name='name_value', display_name='display_name_value', kind=entity_type.EntityType.Kind.KIND_MAP, auto_expansion_mode=entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT, enable_fuzzy_extraction=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = entity_type.EntityType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_entity_type(request)
    assert isinstance(response, entity_type.EntityType)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.kind == entity_type.EntityType.Kind.KIND_MAP
    assert response.auto_expansion_mode == entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT
    assert response.enable_fuzzy_extraction is True

def test_get_entity_type_rest_required_fields(request_type=entity_type.GetEntityTypeRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EntityTypesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_entity_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_entity_type._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('language_code',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = entity_type.EntityType()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = entity_type.EntityType.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_entity_type(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_entity_type_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_entity_type._get_unset_required_fields({})
    assert set(unset_fields) == set(('languageCode',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_entity_type_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EntityTypesRestInterceptor())
    client = EntityTypesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EntityTypesRestInterceptor, 'post_get_entity_type') as post, mock.patch.object(transports.EntityTypesRestInterceptor, 'pre_get_entity_type') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = entity_type.GetEntityTypeRequest.pb(entity_type.GetEntityTypeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = entity_type.EntityType.to_json(entity_type.EntityType())
        request = entity_type.GetEntityTypeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = entity_type.EntityType()
        client.get_entity_type(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_entity_type_rest_bad_request(transport: str='rest', request_type=entity_type.GetEntityTypeRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/agent/entityTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_entity_type(request)

def test_get_entity_type_rest_flattened():
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = entity_type.EntityType()
        sample_request = {'name': 'projects/sample1/agent/entityTypes/sample2'}
        mock_args = dict(name='name_value', language_code='language_code_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = entity_type.EntityType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_entity_type(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/agent/entityTypes/*}' % client.transport._host, args[1])

def test_get_entity_type_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_entity_type(entity_type.GetEntityTypeRequest(), name='name_value', language_code='language_code_value')

def test_get_entity_type_rest_error():
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcd_entity_type.CreateEntityTypeRequest, dict])
def test_create_entity_type_rest(request_type):
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/agent'}
    request_init['entity_type'] = {'name': 'name_value', 'display_name': 'display_name_value', 'kind': 1, 'auto_expansion_mode': 1, 'entities': [{'value': 'value_value', 'synonyms': ['synonyms_value1', 'synonyms_value2']}], 'enable_fuzzy_extraction': True}
    test_field = gcd_entity_type.CreateEntityTypeRequest.meta.fields['entity_type']

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
    for (field, value) in request_init['entity_type'].items():
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
                for i in range(0, len(request_init['entity_type'][field])):
                    del request_init['entity_type'][field][i][subfield]
            else:
                del request_init['entity_type'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_entity_type.EntityType(name='name_value', display_name='display_name_value', kind=gcd_entity_type.EntityType.Kind.KIND_MAP, auto_expansion_mode=gcd_entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT, enable_fuzzy_extraction=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_entity_type.EntityType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_entity_type(request)
    assert isinstance(response, gcd_entity_type.EntityType)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.kind == gcd_entity_type.EntityType.Kind.KIND_MAP
    assert response.auto_expansion_mode == gcd_entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT
    assert response.enable_fuzzy_extraction is True

def test_create_entity_type_rest_required_fields(request_type=gcd_entity_type.CreateEntityTypeRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EntityTypesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_entity_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_entity_type._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('language_code',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcd_entity_type.EntityType()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcd_entity_type.EntityType.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_entity_type(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_entity_type_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_entity_type._get_unset_required_fields({})
    assert set(unset_fields) == set(('languageCode',)) & set(('parent', 'entityType'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_entity_type_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EntityTypesRestInterceptor())
    client = EntityTypesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EntityTypesRestInterceptor, 'post_create_entity_type') as post, mock.patch.object(transports.EntityTypesRestInterceptor, 'pre_create_entity_type') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcd_entity_type.CreateEntityTypeRequest.pb(gcd_entity_type.CreateEntityTypeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcd_entity_type.EntityType.to_json(gcd_entity_type.EntityType())
        request = gcd_entity_type.CreateEntityTypeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcd_entity_type.EntityType()
        client.create_entity_type(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_entity_type_rest_bad_request(transport: str='rest', request_type=gcd_entity_type.CreateEntityTypeRequest):
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/agent'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_entity_type(request)

def test_create_entity_type_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_entity_type.EntityType()
        sample_request = {'parent': 'projects/sample1/agent'}
        mock_args = dict(parent='parent_value', entity_type=gcd_entity_type.EntityType(name='name_value'), language_code='language_code_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_entity_type.EntityType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_entity_type(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/agent}/entityTypes' % client.transport._host, args[1])

def test_create_entity_type_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_entity_type(gcd_entity_type.CreateEntityTypeRequest(), parent='parent_value', entity_type=gcd_entity_type.EntityType(name='name_value'), language_code='language_code_value')

def test_create_entity_type_rest_error():
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcd_entity_type.UpdateEntityTypeRequest, dict])
def test_update_entity_type_rest(request_type):
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'entity_type': {'name': 'projects/sample1/agent/entityTypes/sample2'}}
    request_init['entity_type'] = {'name': 'projects/sample1/agent/entityTypes/sample2', 'display_name': 'display_name_value', 'kind': 1, 'auto_expansion_mode': 1, 'entities': [{'value': 'value_value', 'synonyms': ['synonyms_value1', 'synonyms_value2']}], 'enable_fuzzy_extraction': True}
    test_field = gcd_entity_type.UpdateEntityTypeRequest.meta.fields['entity_type']

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
    for (field, value) in request_init['entity_type'].items():
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
                for i in range(0, len(request_init['entity_type'][field])):
                    del request_init['entity_type'][field][i][subfield]
            else:
                del request_init['entity_type'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_entity_type.EntityType(name='name_value', display_name='display_name_value', kind=gcd_entity_type.EntityType.Kind.KIND_MAP, auto_expansion_mode=gcd_entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT, enable_fuzzy_extraction=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_entity_type.EntityType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_entity_type(request)
    assert isinstance(response, gcd_entity_type.EntityType)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.kind == gcd_entity_type.EntityType.Kind.KIND_MAP
    assert response.auto_expansion_mode == gcd_entity_type.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_DEFAULT
    assert response.enable_fuzzy_extraction is True

def test_update_entity_type_rest_required_fields(request_type=gcd_entity_type.UpdateEntityTypeRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EntityTypesRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_entity_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_entity_type._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('language_code', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcd_entity_type.EntityType()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcd_entity_type.EntityType.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_entity_type(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_entity_type_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_entity_type._get_unset_required_fields({})
    assert set(unset_fields) == set(('languageCode', 'updateMask')) & set(('entityType',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_entity_type_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EntityTypesRestInterceptor())
    client = EntityTypesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EntityTypesRestInterceptor, 'post_update_entity_type') as post, mock.patch.object(transports.EntityTypesRestInterceptor, 'pre_update_entity_type') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcd_entity_type.UpdateEntityTypeRequest.pb(gcd_entity_type.UpdateEntityTypeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcd_entity_type.EntityType.to_json(gcd_entity_type.EntityType())
        request = gcd_entity_type.UpdateEntityTypeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcd_entity_type.EntityType()
        client.update_entity_type(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_entity_type_rest_bad_request(transport: str='rest', request_type=gcd_entity_type.UpdateEntityTypeRequest):
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'entity_type': {'name': 'projects/sample1/agent/entityTypes/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_entity_type(request)

def test_update_entity_type_rest_flattened():
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_entity_type.EntityType()
        sample_request = {'entity_type': {'name': 'projects/sample1/agent/entityTypes/sample2'}}
        mock_args = dict(entity_type=gcd_entity_type.EntityType(name='name_value'), language_code='language_code_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_entity_type.EntityType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_entity_type(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{entity_type.name=projects/*/agent/entityTypes/*}' % client.transport._host, args[1])

def test_update_entity_type_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_entity_type(gcd_entity_type.UpdateEntityTypeRequest(), entity_type=gcd_entity_type.EntityType(name='name_value'), language_code='language_code_value')

def test_update_entity_type_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [entity_type.DeleteEntityTypeRequest, dict])
def test_delete_entity_type_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/agent/entityTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_entity_type(request)
    assert response is None

def test_delete_entity_type_rest_required_fields(request_type=entity_type.DeleteEntityTypeRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.EntityTypesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_entity_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_entity_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_entity_type(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_entity_type_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_entity_type._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_entity_type_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EntityTypesRestInterceptor())
    client = EntityTypesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EntityTypesRestInterceptor, 'pre_delete_entity_type') as pre:
        pre.assert_not_called()
        pb_message = entity_type.DeleteEntityTypeRequest.pb(entity_type.DeleteEntityTypeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = entity_type.DeleteEntityTypeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_entity_type(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_entity_type_rest_bad_request(transport: str='rest', request_type=entity_type.DeleteEntityTypeRequest):
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/agent/entityTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_entity_type(request)

def test_delete_entity_type_rest_flattened():
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/agent/entityTypes/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_entity_type(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/agent/entityTypes/*}' % client.transport._host, args[1])

def test_delete_entity_type_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_entity_type(entity_type.DeleteEntityTypeRequest(), name='name_value')

def test_delete_entity_type_rest_error():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [entity_type.BatchUpdateEntityTypesRequest, dict])
def test_batch_update_entity_types_rest(request_type):
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/agent'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_update_entity_types(request)
    assert response.operation.name == 'operations/spam'

def test_batch_update_entity_types_rest_required_fields(request_type=entity_type.BatchUpdateEntityTypesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EntityTypesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_update_entity_types._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_update_entity_types._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_update_entity_types(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_update_entity_types_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_update_entity_types._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_update_entity_types_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EntityTypesRestInterceptor())
    client = EntityTypesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EntityTypesRestInterceptor, 'post_batch_update_entity_types') as post, mock.patch.object(transports.EntityTypesRestInterceptor, 'pre_batch_update_entity_types') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = entity_type.BatchUpdateEntityTypesRequest.pb(entity_type.BatchUpdateEntityTypesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = entity_type.BatchUpdateEntityTypesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_update_entity_types(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_update_entity_types_rest_bad_request(transport: str='rest', request_type=entity_type.BatchUpdateEntityTypesRequest):
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/agent'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_update_entity_types(request)

def test_batch_update_entity_types_rest_error():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [entity_type.BatchDeleteEntityTypesRequest, dict])
def test_batch_delete_entity_types_rest(request_type):
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/agent'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_delete_entity_types(request)
    assert response.operation.name == 'operations/spam'

def test_batch_delete_entity_types_rest_required_fields(request_type=entity_type.BatchDeleteEntityTypesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.EntityTypesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['entity_type_names'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_delete_entity_types._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['entityTypeNames'] = 'entity_type_names_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_delete_entity_types._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'entityTypeNames' in jsonified_request
    assert jsonified_request['entityTypeNames'] == 'entity_type_names_value'
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_delete_entity_types(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_delete_entity_types_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_delete_entity_types._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'entityTypeNames'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_delete_entity_types_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EntityTypesRestInterceptor())
    client = EntityTypesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EntityTypesRestInterceptor, 'post_batch_delete_entity_types') as post, mock.patch.object(transports.EntityTypesRestInterceptor, 'pre_batch_delete_entity_types') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = entity_type.BatchDeleteEntityTypesRequest.pb(entity_type.BatchDeleteEntityTypesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = entity_type.BatchDeleteEntityTypesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_delete_entity_types(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_delete_entity_types_rest_bad_request(transport: str='rest', request_type=entity_type.BatchDeleteEntityTypesRequest):
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/agent'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_delete_entity_types(request)

def test_batch_delete_entity_types_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/agent'}
        mock_args = dict(parent='parent_value', entity_type_names=['entity_type_names_value'])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_delete_entity_types(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/agent}/entityTypes:batchDelete' % client.transport._host, args[1])

def test_batch_delete_entity_types_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_delete_entity_types(entity_type.BatchDeleteEntityTypesRequest(), parent='parent_value', entity_type_names=['entity_type_names_value'])

def test_batch_delete_entity_types_rest_error():
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [entity_type.BatchCreateEntitiesRequest, dict])
def test_batch_create_entities_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/agent/entityTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_create_entities(request)
    assert response.operation.name == 'operations/spam'

def test_batch_create_entities_rest_required_fields(request_type=entity_type.BatchCreateEntitiesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EntityTypesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_entities._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_entities._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_create_entities(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_create_entities_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_create_entities._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'entities'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_create_entities_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EntityTypesRestInterceptor())
    client = EntityTypesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EntityTypesRestInterceptor, 'post_batch_create_entities') as post, mock.patch.object(transports.EntityTypesRestInterceptor, 'pre_batch_create_entities') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = entity_type.BatchCreateEntitiesRequest.pb(entity_type.BatchCreateEntitiesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = entity_type.BatchCreateEntitiesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_create_entities(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_create_entities_rest_bad_request(transport: str='rest', request_type=entity_type.BatchCreateEntitiesRequest):
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/agent/entityTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_create_entities(request)

def test_batch_create_entities_rest_flattened():
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/agent/entityTypes/sample2'}
        mock_args = dict(parent='parent_value', entities=[entity_type.EntityType.Entity(value='value_value')], language_code='language_code_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_create_entities(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/agent/entityTypes/*}/entities:batchCreate' % client.transport._host, args[1])

def test_batch_create_entities_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_create_entities(entity_type.BatchCreateEntitiesRequest(), parent='parent_value', entities=[entity_type.EntityType.Entity(value='value_value')], language_code='language_code_value')

def test_batch_create_entities_rest_error():
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [entity_type.BatchUpdateEntitiesRequest, dict])
def test_batch_update_entities_rest(request_type):
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/agent/entityTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_update_entities(request)
    assert response.operation.name == 'operations/spam'

def test_batch_update_entities_rest_required_fields(request_type=entity_type.BatchUpdateEntitiesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EntityTypesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_update_entities._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_update_entities._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_update_entities(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_update_entities_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_update_entities._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'entities'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_update_entities_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EntityTypesRestInterceptor())
    client = EntityTypesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EntityTypesRestInterceptor, 'post_batch_update_entities') as post, mock.patch.object(transports.EntityTypesRestInterceptor, 'pre_batch_update_entities') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = entity_type.BatchUpdateEntitiesRequest.pb(entity_type.BatchUpdateEntitiesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = entity_type.BatchUpdateEntitiesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_update_entities(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_update_entities_rest_bad_request(transport: str='rest', request_type=entity_type.BatchUpdateEntitiesRequest):
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/agent/entityTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_update_entities(request)

def test_batch_update_entities_rest_flattened():
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/agent/entityTypes/sample2'}
        mock_args = dict(parent='parent_value', entities=[entity_type.EntityType.Entity(value='value_value')], language_code='language_code_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_update_entities(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/agent/entityTypes/*}/entities:batchUpdate' % client.transport._host, args[1])

def test_batch_update_entities_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_update_entities(entity_type.BatchUpdateEntitiesRequest(), parent='parent_value', entities=[entity_type.EntityType.Entity(value='value_value')], language_code='language_code_value')

def test_batch_update_entities_rest_error():
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [entity_type.BatchDeleteEntitiesRequest, dict])
def test_batch_delete_entities_rest(request_type):
    if False:
        while True:
            i = 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/agent/entityTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_delete_entities(request)
    assert response.operation.name == 'operations/spam'

def test_batch_delete_entities_rest_required_fields(request_type=entity_type.BatchDeleteEntitiesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.EntityTypesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['entity_values'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_delete_entities._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['entityValues'] = 'entity_values_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_delete_entities._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'entityValues' in jsonified_request
    assert jsonified_request['entityValues'] == 'entity_values_value'
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_delete_entities(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_delete_entities_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_delete_entities._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'entityValues'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_delete_entities_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.EntityTypesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EntityTypesRestInterceptor())
    client = EntityTypesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EntityTypesRestInterceptor, 'post_batch_delete_entities') as post, mock.patch.object(transports.EntityTypesRestInterceptor, 'pre_batch_delete_entities') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = entity_type.BatchDeleteEntitiesRequest.pb(entity_type.BatchDeleteEntitiesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = entity_type.BatchDeleteEntitiesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_delete_entities(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_delete_entities_rest_bad_request(transport: str='rest', request_type=entity_type.BatchDeleteEntitiesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/agent/entityTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_delete_entities(request)

def test_batch_delete_entities_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/agent/entityTypes/sample2'}
        mock_args = dict(parent='parent_value', entity_values=['entity_values_value'], language_code='language_code_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_delete_entities(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/agent/entityTypes/*}/entities:batchDelete' % client.transport._host, args[1])

def test_batch_delete_entities_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_delete_entities(entity_type.BatchDeleteEntitiesRequest(), parent='parent_value', entity_values=['entity_values_value'], language_code='language_code_value')

def test_batch_delete_entities_rest_error():
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.EntityTypesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.EntityTypesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EntityTypesClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.EntityTypesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = EntityTypesClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = EntityTypesClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.EntityTypesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EntityTypesClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.EntityTypesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = EntityTypesClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EntityTypesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.EntityTypesGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.EntityTypesGrpcTransport, transports.EntityTypesGrpcAsyncIOTransport, transports.EntityTypesRestTransport])
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
        print('Hello World!')
    transport = EntityTypesClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.EntityTypesGrpcTransport)

def test_entity_types_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.EntityTypesTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_entity_types_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.dialogflow_v2.services.entity_types.transports.EntityTypesTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.EntityTypesTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_entity_types', 'get_entity_type', 'create_entity_type', 'update_entity_type', 'delete_entity_type', 'batch_update_entity_types', 'batch_delete_entity_types', 'batch_create_entities', 'batch_update_entities', 'batch_delete_entities', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'list_operations')
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

def test_entity_types_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dialogflow_v2.services.entity_types.transports.EntityTypesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.EntityTypesTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id='octopus')

def test_entity_types_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dialogflow_v2.services.entity_types.transports.EntityTypesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.EntityTypesTransport()
        adc.assert_called_once()

def test_entity_types_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        EntityTypesClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.EntityTypesGrpcTransport, transports.EntityTypesGrpcAsyncIOTransport])
def test_entity_types_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.EntityTypesGrpcTransport, transports.EntityTypesGrpcAsyncIOTransport, transports.EntityTypesRestTransport])
def test_entity_types_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.EntityTypesGrpcTransport, grpc_helpers), (transports.EntityTypesGrpcAsyncIOTransport, grpc_helpers_async)])
def test_entity_types_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dialogflow.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), scopes=['1', '2'], default_host='dialogflow.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.EntityTypesGrpcTransport, transports.EntityTypesGrpcAsyncIOTransport])
def test_entity_types_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_entity_types_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.EntityTypesRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_entity_types_rest_lro_client():
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_entity_types_host_no_port(transport_name):
    if False:
        return 10
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dialogflow.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_entity_types_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dialogflow.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('dialogflow.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_entity_types_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = EntityTypesClient(credentials=creds1, transport=transport_name)
    client2 = EntityTypesClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_entity_types._session
    session2 = client2.transport.list_entity_types._session
    assert session1 != session2
    session1 = client1.transport.get_entity_type._session
    session2 = client2.transport.get_entity_type._session
    assert session1 != session2
    session1 = client1.transport.create_entity_type._session
    session2 = client2.transport.create_entity_type._session
    assert session1 != session2
    session1 = client1.transport.update_entity_type._session
    session2 = client2.transport.update_entity_type._session
    assert session1 != session2
    session1 = client1.transport.delete_entity_type._session
    session2 = client2.transport.delete_entity_type._session
    assert session1 != session2
    session1 = client1.transport.batch_update_entity_types._session
    session2 = client2.transport.batch_update_entity_types._session
    assert session1 != session2
    session1 = client1.transport.batch_delete_entity_types._session
    session2 = client2.transport.batch_delete_entity_types._session
    assert session1 != session2
    session1 = client1.transport.batch_create_entities._session
    session2 = client2.transport.batch_create_entities._session
    assert session1 != session2
    session1 = client1.transport.batch_update_entities._session
    session2 = client2.transport.batch_update_entities._session
    assert session1 != session2
    session1 = client1.transport.batch_delete_entities._session
    session2 = client2.transport.batch_delete_entities._session
    assert session1 != session2

def test_entity_types_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.EntityTypesGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_entity_types_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.EntityTypesGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.EntityTypesGrpcTransport, transports.EntityTypesGrpcAsyncIOTransport])
def test_entity_types_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.EntityTypesGrpcTransport, transports.EntityTypesGrpcAsyncIOTransport])
def test_entity_types_transport_channel_mtls_with_adc(transport_class):
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

def test_entity_types_grpc_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_entity_types_grpc_lro_async_client():
    if False:
        while True:
            i = 10
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_entity_type_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    entity_type = 'clam'
    expected = 'projects/{project}/agent/entityTypes/{entity_type}'.format(project=project, entity_type=entity_type)
    actual = EntityTypesClient.entity_type_path(project, entity_type)
    assert expected == actual

def test_parse_entity_type_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'whelk', 'entity_type': 'octopus'}
    path = EntityTypesClient.entity_type_path(**expected)
    actual = EntityTypesClient.parse_entity_type_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = EntityTypesClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'nudibranch'}
    path = EntityTypesClient.common_billing_account_path(**expected)
    actual = EntityTypesClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = EntityTypesClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'mussel'}
    path = EntityTypesClient.common_folder_path(**expected)
    actual = EntityTypesClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = EntityTypesClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'nautilus'}
    path = EntityTypesClient.common_organization_path(**expected)
    actual = EntityTypesClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = EntityTypesClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'abalone'}
    path = EntityTypesClient.common_project_path(**expected)
    actual = EntityTypesClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = EntityTypesClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = EntityTypesClient.common_location_path(**expected)
    actual = EntityTypesClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.EntityTypesTransport, '_prep_wrapped_messages') as prep:
        client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.EntityTypesTransport, '_prep_wrapped_messages') as prep:
        transport_class = EntityTypesClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_cancel_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.CancelOperationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/operations/sample2'}, request)
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
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/operations/sample2'}
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
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/operations/sample2'}, request)
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
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/operations/sample2'}
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
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_operations(request)

@pytest.mark.parametrize('request_type', [operations_pb2.ListOperationsRequest, dict])
def test_list_operations_rest(request_type):
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1'}
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

def test_cancel_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = EntityTypesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = EntityTypesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(EntityTypesClient, transports.EntityTypesGrpcTransport), (EntityTypesAsyncClient, transports.EntityTypesGrpcAsyncIOTransport)])
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