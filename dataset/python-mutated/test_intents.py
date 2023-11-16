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
from google.cloud.dialogflow_v2.services.intents import IntentsAsyncClient, IntentsClient, pagers, transports
from google.cloud.dialogflow_v2.types import context
from google.cloud.dialogflow_v2.types import intent
from google.cloud.dialogflow_v2.types import intent as gcd_intent

def client_cert_source_callback():
    if False:
        while True:
            i = 10
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
    assert IntentsClient._get_default_mtls_endpoint(None) is None
    assert IntentsClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert IntentsClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert IntentsClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert IntentsClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert IntentsClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(IntentsClient, 'grpc'), (IntentsAsyncClient, 'grpc_asyncio'), (IntentsClient, 'rest')])
def test_intents_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.IntentsGrpcTransport, 'grpc'), (transports.IntentsGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.IntentsRestTransport, 'rest')])
def test_intents_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(IntentsClient, 'grpc'), (IntentsAsyncClient, 'grpc_asyncio'), (IntentsClient, 'rest')])
def test_intents_client_from_service_account_file(client_class, transport_name):
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

def test_intents_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = IntentsClient.get_transport_class()
    available_transports = [transports.IntentsGrpcTransport, transports.IntentsRestTransport]
    assert transport in available_transports
    transport = IntentsClient.get_transport_class('grpc')
    assert transport == transports.IntentsGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(IntentsClient, transports.IntentsGrpcTransport, 'grpc'), (IntentsAsyncClient, transports.IntentsGrpcAsyncIOTransport, 'grpc_asyncio'), (IntentsClient, transports.IntentsRestTransport, 'rest')])
@mock.patch.object(IntentsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(IntentsClient))
@mock.patch.object(IntentsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(IntentsAsyncClient))
def test_intents_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(IntentsClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(IntentsClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(IntentsClient, transports.IntentsGrpcTransport, 'grpc', 'true'), (IntentsAsyncClient, transports.IntentsGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (IntentsClient, transports.IntentsGrpcTransport, 'grpc', 'false'), (IntentsAsyncClient, transports.IntentsGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (IntentsClient, transports.IntentsRestTransport, 'rest', 'true'), (IntentsClient, transports.IntentsRestTransport, 'rest', 'false')])
@mock.patch.object(IntentsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(IntentsClient))
@mock.patch.object(IntentsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(IntentsAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_intents_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [IntentsClient, IntentsAsyncClient])
@mock.patch.object(IntentsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(IntentsClient))
@mock.patch.object(IntentsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(IntentsAsyncClient))
def test_intents_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(IntentsClient, transports.IntentsGrpcTransport, 'grpc'), (IntentsAsyncClient, transports.IntentsGrpcAsyncIOTransport, 'grpc_asyncio'), (IntentsClient, transports.IntentsRestTransport, 'rest')])
def test_intents_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(IntentsClient, transports.IntentsGrpcTransport, 'grpc', grpc_helpers), (IntentsAsyncClient, transports.IntentsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (IntentsClient, transports.IntentsRestTransport, 'rest', None)])
def test_intents_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_intents_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.dialogflow_v2.services.intents.transports.IntentsGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = IntentsClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(IntentsClient, transports.IntentsGrpcTransport, 'grpc', grpc_helpers), (IntentsAsyncClient, transports.IntentsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_intents_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('dialogflow.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), scopes=None, default_host='dialogflow.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [intent.ListIntentsRequest, dict])
def test_list_intents(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_intents), '__call__') as call:
        call.return_value = intent.ListIntentsResponse(next_page_token='next_page_token_value')
        response = client.list_intents(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.ListIntentsRequest()
    assert isinstance(response, pagers.ListIntentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_intents_empty_call():
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_intents), '__call__') as call:
        client.list_intents()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.ListIntentsRequest()

@pytest.mark.asyncio
async def test_list_intents_async(transport: str='grpc_asyncio', request_type=intent.ListIntentsRequest):
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_intents), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(intent.ListIntentsResponse(next_page_token='next_page_token_value'))
        response = await client.list_intents(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.ListIntentsRequest()
    assert isinstance(response, pagers.ListIntentsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_intents_async_from_dict():
    await test_list_intents_async(request_type=dict)

def test_list_intents_field_headers():
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    request = intent.ListIntentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_intents), '__call__') as call:
        call.return_value = intent.ListIntentsResponse()
        client.list_intents(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_intents_field_headers_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = intent.ListIntentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_intents), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(intent.ListIntentsResponse())
        await client.list_intents(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_intents_flattened():
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_intents), '__call__') as call:
        call.return_value = intent.ListIntentsResponse()
        client.list_intents(parent='parent_value', language_code='language_code_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

def test_list_intents_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_intents(intent.ListIntentsRequest(), parent='parent_value', language_code='language_code_value')

@pytest.mark.asyncio
async def test_list_intents_flattened_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_intents), '__call__') as call:
        call.return_value = intent.ListIntentsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(intent.ListIntentsResponse())
        response = await client.list_intents(parent='parent_value', language_code='language_code_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_intents_flattened_error_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_intents(intent.ListIntentsRequest(), parent='parent_value', language_code='language_code_value')

def test_list_intents_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_intents), '__call__') as call:
        call.side_effect = (intent.ListIntentsResponse(intents=[intent.Intent(), intent.Intent(), intent.Intent()], next_page_token='abc'), intent.ListIntentsResponse(intents=[], next_page_token='def'), intent.ListIntentsResponse(intents=[intent.Intent()], next_page_token='ghi'), intent.ListIntentsResponse(intents=[intent.Intent(), intent.Intent()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_intents(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, intent.Intent) for i in results))

def test_list_intents_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_intents), '__call__') as call:
        call.side_effect = (intent.ListIntentsResponse(intents=[intent.Intent(), intent.Intent(), intent.Intent()], next_page_token='abc'), intent.ListIntentsResponse(intents=[], next_page_token='def'), intent.ListIntentsResponse(intents=[intent.Intent()], next_page_token='ghi'), intent.ListIntentsResponse(intents=[intent.Intent(), intent.Intent()]), RuntimeError)
        pages = list(client.list_intents(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_intents_async_pager():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_intents), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (intent.ListIntentsResponse(intents=[intent.Intent(), intent.Intent(), intent.Intent()], next_page_token='abc'), intent.ListIntentsResponse(intents=[], next_page_token='def'), intent.ListIntentsResponse(intents=[intent.Intent()], next_page_token='ghi'), intent.ListIntentsResponse(intents=[intent.Intent(), intent.Intent()]), RuntimeError)
        async_pager = await client.list_intents(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, intent.Intent) for i in responses))

@pytest.mark.asyncio
async def test_list_intents_async_pages():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_intents), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (intent.ListIntentsResponse(intents=[intent.Intent(), intent.Intent(), intent.Intent()], next_page_token='abc'), intent.ListIntentsResponse(intents=[], next_page_token='def'), intent.ListIntentsResponse(intents=[intent.Intent()], next_page_token='ghi'), intent.ListIntentsResponse(intents=[intent.Intent(), intent.Intent()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_intents(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [intent.GetIntentRequest, dict])
def test_get_intent(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_intent), '__call__') as call:
        call.return_value = intent.Intent(name='name_value', display_name='display_name_value', webhook_state=intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED, priority=898, is_fallback=True, ml_disabled=True, live_agent_handoff=True, end_interaction=True, input_context_names=['input_context_names_value'], events=['events_value'], action='action_value', reset_contexts=True, default_response_platforms=[intent.Intent.Message.Platform.FACEBOOK], root_followup_intent_name='root_followup_intent_name_value', parent_followup_intent_name='parent_followup_intent_name_value')
        response = client.get_intent(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.GetIntentRequest()
    assert isinstance(response, intent.Intent)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.webhook_state == intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED
    assert response.priority == 898
    assert response.is_fallback is True
    assert response.ml_disabled is True
    assert response.live_agent_handoff is True
    assert response.end_interaction is True
    assert response.input_context_names == ['input_context_names_value']
    assert response.events == ['events_value']
    assert response.action == 'action_value'
    assert response.reset_contexts is True
    assert response.default_response_platforms == [intent.Intent.Message.Platform.FACEBOOK]
    assert response.root_followup_intent_name == 'root_followup_intent_name_value'
    assert response.parent_followup_intent_name == 'parent_followup_intent_name_value'

def test_get_intent_empty_call():
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_intent), '__call__') as call:
        client.get_intent()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.GetIntentRequest()

@pytest.mark.asyncio
async def test_get_intent_async(transport: str='grpc_asyncio', request_type=intent.GetIntentRequest):
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_intent), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(intent.Intent(name='name_value', display_name='display_name_value', webhook_state=intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED, priority=898, is_fallback=True, ml_disabled=True, live_agent_handoff=True, end_interaction=True, input_context_names=['input_context_names_value'], events=['events_value'], action='action_value', reset_contexts=True, default_response_platforms=[intent.Intent.Message.Platform.FACEBOOK], root_followup_intent_name='root_followup_intent_name_value', parent_followup_intent_name='parent_followup_intent_name_value'))
        response = await client.get_intent(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.GetIntentRequest()
    assert isinstance(response, intent.Intent)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.webhook_state == intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED
    assert response.priority == 898
    assert response.is_fallback is True
    assert response.ml_disabled is True
    assert response.live_agent_handoff is True
    assert response.end_interaction is True
    assert response.input_context_names == ['input_context_names_value']
    assert response.events == ['events_value']
    assert response.action == 'action_value'
    assert response.reset_contexts is True
    assert response.default_response_platforms == [intent.Intent.Message.Platform.FACEBOOK]
    assert response.root_followup_intent_name == 'root_followup_intent_name_value'
    assert response.parent_followup_intent_name == 'parent_followup_intent_name_value'

@pytest.mark.asyncio
async def test_get_intent_async_from_dict():
    await test_get_intent_async(request_type=dict)

def test_get_intent_field_headers():
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    request = intent.GetIntentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_intent), '__call__') as call:
        call.return_value = intent.Intent()
        client.get_intent(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_intent_field_headers_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = intent.GetIntentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_intent), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(intent.Intent())
        await client.get_intent(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_intent_flattened():
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_intent), '__call__') as call:
        call.return_value = intent.Intent()
        client.get_intent(name='name_value', language_code='language_code_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

def test_get_intent_flattened_error():
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_intent(intent.GetIntentRequest(), name='name_value', language_code='language_code_value')

@pytest.mark.asyncio
async def test_get_intent_flattened_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_intent), '__call__') as call:
        call.return_value = intent.Intent()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(intent.Intent())
        response = await client.get_intent(name='name_value', language_code='language_code_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_intent_flattened_error_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_intent(intent.GetIntentRequest(), name='name_value', language_code='language_code_value')

@pytest.mark.parametrize('request_type', [gcd_intent.CreateIntentRequest, dict])
def test_create_intent(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_intent), '__call__') as call:
        call.return_value = gcd_intent.Intent(name='name_value', display_name='display_name_value', webhook_state=gcd_intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED, priority=898, is_fallback=True, ml_disabled=True, live_agent_handoff=True, end_interaction=True, input_context_names=['input_context_names_value'], events=['events_value'], action='action_value', reset_contexts=True, default_response_platforms=[gcd_intent.Intent.Message.Platform.FACEBOOK], root_followup_intent_name='root_followup_intent_name_value', parent_followup_intent_name='parent_followup_intent_name_value')
        response = client.create_intent(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_intent.CreateIntentRequest()
    assert isinstance(response, gcd_intent.Intent)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.webhook_state == gcd_intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED
    assert response.priority == 898
    assert response.is_fallback is True
    assert response.ml_disabled is True
    assert response.live_agent_handoff is True
    assert response.end_interaction is True
    assert response.input_context_names == ['input_context_names_value']
    assert response.events == ['events_value']
    assert response.action == 'action_value'
    assert response.reset_contexts is True
    assert response.default_response_platforms == [gcd_intent.Intent.Message.Platform.FACEBOOK]
    assert response.root_followup_intent_name == 'root_followup_intent_name_value'
    assert response.parent_followup_intent_name == 'parent_followup_intent_name_value'

def test_create_intent_empty_call():
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_intent), '__call__') as call:
        client.create_intent()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_intent.CreateIntentRequest()

@pytest.mark.asyncio
async def test_create_intent_async(transport: str='grpc_asyncio', request_type=gcd_intent.CreateIntentRequest):
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_intent), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_intent.Intent(name='name_value', display_name='display_name_value', webhook_state=gcd_intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED, priority=898, is_fallback=True, ml_disabled=True, live_agent_handoff=True, end_interaction=True, input_context_names=['input_context_names_value'], events=['events_value'], action='action_value', reset_contexts=True, default_response_platforms=[gcd_intent.Intent.Message.Platform.FACEBOOK], root_followup_intent_name='root_followup_intent_name_value', parent_followup_intent_name='parent_followup_intent_name_value'))
        response = await client.create_intent(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_intent.CreateIntentRequest()
    assert isinstance(response, gcd_intent.Intent)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.webhook_state == gcd_intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED
    assert response.priority == 898
    assert response.is_fallback is True
    assert response.ml_disabled is True
    assert response.live_agent_handoff is True
    assert response.end_interaction is True
    assert response.input_context_names == ['input_context_names_value']
    assert response.events == ['events_value']
    assert response.action == 'action_value'
    assert response.reset_contexts is True
    assert response.default_response_platforms == [gcd_intent.Intent.Message.Platform.FACEBOOK]
    assert response.root_followup_intent_name == 'root_followup_intent_name_value'
    assert response.parent_followup_intent_name == 'parent_followup_intent_name_value'

@pytest.mark.asyncio
async def test_create_intent_async_from_dict():
    await test_create_intent_async(request_type=dict)

def test_create_intent_field_headers():
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_intent.CreateIntentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_intent), '__call__') as call:
        call.return_value = gcd_intent.Intent()
        client.create_intent(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_intent_field_headers_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_intent.CreateIntentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_intent), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_intent.Intent())
        await client.create_intent(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_intent_flattened():
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_intent), '__call__') as call:
        call.return_value = gcd_intent.Intent()
        client.create_intent(parent='parent_value', intent=gcd_intent.Intent(name='name_value'), language_code='language_code_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].intent
        mock_val = gcd_intent.Intent(name='name_value')
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

def test_create_intent_flattened_error():
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_intent(gcd_intent.CreateIntentRequest(), parent='parent_value', intent=gcd_intent.Intent(name='name_value'), language_code='language_code_value')

@pytest.mark.asyncio
async def test_create_intent_flattened_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_intent), '__call__') as call:
        call.return_value = gcd_intent.Intent()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_intent.Intent())
        response = await client.create_intent(parent='parent_value', intent=gcd_intent.Intent(name='name_value'), language_code='language_code_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].intent
        mock_val = gcd_intent.Intent(name='name_value')
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_intent_flattened_error_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_intent(gcd_intent.CreateIntentRequest(), parent='parent_value', intent=gcd_intent.Intent(name='name_value'), language_code='language_code_value')

@pytest.mark.parametrize('request_type', [gcd_intent.UpdateIntentRequest, dict])
def test_update_intent(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_intent), '__call__') as call:
        call.return_value = gcd_intent.Intent(name='name_value', display_name='display_name_value', webhook_state=gcd_intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED, priority=898, is_fallback=True, ml_disabled=True, live_agent_handoff=True, end_interaction=True, input_context_names=['input_context_names_value'], events=['events_value'], action='action_value', reset_contexts=True, default_response_platforms=[gcd_intent.Intent.Message.Platform.FACEBOOK], root_followup_intent_name='root_followup_intent_name_value', parent_followup_intent_name='parent_followup_intent_name_value')
        response = client.update_intent(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_intent.UpdateIntentRequest()
    assert isinstance(response, gcd_intent.Intent)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.webhook_state == gcd_intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED
    assert response.priority == 898
    assert response.is_fallback is True
    assert response.ml_disabled is True
    assert response.live_agent_handoff is True
    assert response.end_interaction is True
    assert response.input_context_names == ['input_context_names_value']
    assert response.events == ['events_value']
    assert response.action == 'action_value'
    assert response.reset_contexts is True
    assert response.default_response_platforms == [gcd_intent.Intent.Message.Platform.FACEBOOK]
    assert response.root_followup_intent_name == 'root_followup_intent_name_value'
    assert response.parent_followup_intent_name == 'parent_followup_intent_name_value'

def test_update_intent_empty_call():
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_intent), '__call__') as call:
        client.update_intent()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_intent.UpdateIntentRequest()

@pytest.mark.asyncio
async def test_update_intent_async(transport: str='grpc_asyncio', request_type=gcd_intent.UpdateIntentRequest):
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_intent), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_intent.Intent(name='name_value', display_name='display_name_value', webhook_state=gcd_intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED, priority=898, is_fallback=True, ml_disabled=True, live_agent_handoff=True, end_interaction=True, input_context_names=['input_context_names_value'], events=['events_value'], action='action_value', reset_contexts=True, default_response_platforms=[gcd_intent.Intent.Message.Platform.FACEBOOK], root_followup_intent_name='root_followup_intent_name_value', parent_followup_intent_name='parent_followup_intent_name_value'))
        response = await client.update_intent(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_intent.UpdateIntentRequest()
    assert isinstance(response, gcd_intent.Intent)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.webhook_state == gcd_intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED
    assert response.priority == 898
    assert response.is_fallback is True
    assert response.ml_disabled is True
    assert response.live_agent_handoff is True
    assert response.end_interaction is True
    assert response.input_context_names == ['input_context_names_value']
    assert response.events == ['events_value']
    assert response.action == 'action_value'
    assert response.reset_contexts is True
    assert response.default_response_platforms == [gcd_intent.Intent.Message.Platform.FACEBOOK]
    assert response.root_followup_intent_name == 'root_followup_intent_name_value'
    assert response.parent_followup_intent_name == 'parent_followup_intent_name_value'

@pytest.mark.asyncio
async def test_update_intent_async_from_dict():
    await test_update_intent_async(request_type=dict)

def test_update_intent_field_headers():
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_intent.UpdateIntentRequest()
    request.intent.name = 'name_value'
    with mock.patch.object(type(client.transport.update_intent), '__call__') as call:
        call.return_value = gcd_intent.Intent()
        client.update_intent(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'intent.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_intent_field_headers_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_intent.UpdateIntentRequest()
    request.intent.name = 'name_value'
    with mock.patch.object(type(client.transport.update_intent), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_intent.Intent())
        await client.update_intent(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'intent.name=name_value') in kw['metadata']

def test_update_intent_flattened():
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_intent), '__call__') as call:
        call.return_value = gcd_intent.Intent()
        client.update_intent(intent=gcd_intent.Intent(name='name_value'), language_code='language_code_value', update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].intent
        mock_val = gcd_intent.Intent(name='name_value')
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_intent_flattened_error():
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_intent(gcd_intent.UpdateIntentRequest(), intent=gcd_intent.Intent(name='name_value'), language_code='language_code_value', update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_intent_flattened_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_intent), '__call__') as call:
        call.return_value = gcd_intent.Intent()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_intent.Intent())
        response = await client.update_intent(intent=gcd_intent.Intent(name='name_value'), language_code='language_code_value', update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].intent
        mock_val = gcd_intent.Intent(name='name_value')
        assert arg == mock_val
        arg = args[0].language_code
        mock_val = 'language_code_value'
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_intent_flattened_error_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_intent(gcd_intent.UpdateIntentRequest(), intent=gcd_intent.Intent(name='name_value'), language_code='language_code_value', update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [intent.DeleteIntentRequest, dict])
def test_delete_intent(request_type, transport: str='grpc'):
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_intent), '__call__') as call:
        call.return_value = None
        response = client.delete_intent(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.DeleteIntentRequest()
    assert response is None

def test_delete_intent_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_intent), '__call__') as call:
        client.delete_intent()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.DeleteIntentRequest()

@pytest.mark.asyncio
async def test_delete_intent_async(transport: str='grpc_asyncio', request_type=intent.DeleteIntentRequest):
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_intent), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_intent(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.DeleteIntentRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_intent_async_from_dict():
    await test_delete_intent_async(request_type=dict)

def test_delete_intent_field_headers():
    if False:
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    request = intent.DeleteIntentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_intent), '__call__') as call:
        call.return_value = None
        client.delete_intent(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_intent_field_headers_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = intent.DeleteIntentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_intent), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_intent(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_intent_flattened():
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_intent), '__call__') as call:
        call.return_value = None
        client.delete_intent(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_intent_flattened_error():
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_intent(intent.DeleteIntentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_intent_flattened_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_intent), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_intent(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_intent_flattened_error_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_intent(intent.DeleteIntentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [intent.BatchUpdateIntentsRequest, dict])
def test_batch_update_intents(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_update_intents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_update_intents(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.BatchUpdateIntentsRequest()
    assert isinstance(response, future.Future)

def test_batch_update_intents_empty_call():
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_update_intents), '__call__') as call:
        client.batch_update_intents()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.BatchUpdateIntentsRequest()

@pytest.mark.asyncio
async def test_batch_update_intents_async(transport: str='grpc_asyncio', request_type=intent.BatchUpdateIntentsRequest):
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_update_intents), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_update_intents(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.BatchUpdateIntentsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_update_intents_async_from_dict():
    await test_batch_update_intents_async(request_type=dict)

def test_batch_update_intents_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    request = intent.BatchUpdateIntentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_update_intents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_update_intents(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_update_intents_field_headers_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = intent.BatchUpdateIntentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_update_intents), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_update_intents(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_batch_update_intents_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_update_intents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_update_intents(parent='parent_value', intent_batch_uri='intent_batch_uri_value', intent_batch_inline=intent.IntentBatch(intents=[intent.Intent(name='name_value')]))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        assert args[0].intent_batch_inline == intent.IntentBatch(intents=[intent.Intent(name='name_value')])

def test_batch_update_intents_flattened_error():
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_update_intents(intent.BatchUpdateIntentsRequest(), parent='parent_value', intent_batch_uri='intent_batch_uri_value', intent_batch_inline=intent.IntentBatch(intents=[intent.Intent(name='name_value')]))

@pytest.mark.asyncio
async def test_batch_update_intents_flattened_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_update_intents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_update_intents(parent='parent_value', intent_batch_uri='intent_batch_uri_value', intent_batch_inline=intent.IntentBatch(intents=[intent.Intent(name='name_value')]))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        assert args[0].intent_batch_inline == intent.IntentBatch(intents=[intent.Intent(name='name_value')])

@pytest.mark.asyncio
async def test_batch_update_intents_flattened_error_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_update_intents(intent.BatchUpdateIntentsRequest(), parent='parent_value', intent_batch_uri='intent_batch_uri_value', intent_batch_inline=intent.IntentBatch(intents=[intent.Intent(name='name_value')]))

@pytest.mark.parametrize('request_type', [intent.BatchDeleteIntentsRequest, dict])
def test_batch_delete_intents(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_delete_intents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_delete_intents(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.BatchDeleteIntentsRequest()
    assert isinstance(response, future.Future)

def test_batch_delete_intents_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_delete_intents), '__call__') as call:
        client.batch_delete_intents()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.BatchDeleteIntentsRequest()

@pytest.mark.asyncio
async def test_batch_delete_intents_async(transport: str='grpc_asyncio', request_type=intent.BatchDeleteIntentsRequest):
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_delete_intents), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_delete_intents(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == intent.BatchDeleteIntentsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_delete_intents_async_from_dict():
    await test_batch_delete_intents_async(request_type=dict)

def test_batch_delete_intents_field_headers():
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    request = intent.BatchDeleteIntentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_delete_intents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_delete_intents(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_delete_intents_field_headers_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = intent.BatchDeleteIntentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_delete_intents), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_delete_intents(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_batch_delete_intents_flattened():
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_delete_intents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_delete_intents(parent='parent_value', intents=[intent.Intent(name='name_value')])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].intents
        mock_val = [intent.Intent(name='name_value')]
        assert arg == mock_val

def test_batch_delete_intents_flattened_error():
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_delete_intents(intent.BatchDeleteIntentsRequest(), parent='parent_value', intents=[intent.Intent(name='name_value')])

@pytest.mark.asyncio
async def test_batch_delete_intents_flattened_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_delete_intents), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_delete_intents(parent='parent_value', intents=[intent.Intent(name='name_value')])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].intents
        mock_val = [intent.Intent(name='name_value')]
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_delete_intents_flattened_error_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_delete_intents(intent.BatchDeleteIntentsRequest(), parent='parent_value', intents=[intent.Intent(name='name_value')])

@pytest.mark.parametrize('request_type', [intent.ListIntentsRequest, dict])
def test_list_intents_rest(request_type):
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/agent'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = intent.ListIntentsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = intent.ListIntentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_intents(request)
    assert isinstance(response, pagers.ListIntentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_intents_rest_required_fields(request_type=intent.ListIntentsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.IntentsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_intents._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_intents._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('intent_view', 'language_code', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = intent.ListIntentsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = intent.ListIntentsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_intents(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_intents_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.IntentsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_intents._get_unset_required_fields({})
    assert set(unset_fields) == set(('intentView', 'languageCode', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_intents_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.IntentsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IntentsRestInterceptor())
    client = IntentsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IntentsRestInterceptor, 'post_list_intents') as post, mock.patch.object(transports.IntentsRestInterceptor, 'pre_list_intents') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = intent.ListIntentsRequest.pb(intent.ListIntentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = intent.ListIntentsResponse.to_json(intent.ListIntentsResponse())
        request = intent.ListIntentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = intent.ListIntentsResponse()
        client.list_intents(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_intents_rest_bad_request(transport: str='rest', request_type=intent.ListIntentsRequest):
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/agent'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_intents(request)

def test_list_intents_rest_flattened():
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = intent.ListIntentsResponse()
        sample_request = {'parent': 'projects/sample1/agent'}
        mock_args = dict(parent='parent_value', language_code='language_code_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = intent.ListIntentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_intents(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/agent}/intents' % client.transport._host, args[1])

def test_list_intents_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_intents(intent.ListIntentsRequest(), parent='parent_value', language_code='language_code_value')

def test_list_intents_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (intent.ListIntentsResponse(intents=[intent.Intent(), intent.Intent(), intent.Intent()], next_page_token='abc'), intent.ListIntentsResponse(intents=[], next_page_token='def'), intent.ListIntentsResponse(intents=[intent.Intent()], next_page_token='ghi'), intent.ListIntentsResponse(intents=[intent.Intent(), intent.Intent()]))
        response = response + response
        response = tuple((intent.ListIntentsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/agent'}
        pager = client.list_intents(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, intent.Intent) for i in results))
        pages = list(client.list_intents(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [intent.GetIntentRequest, dict])
def test_get_intent_rest(request_type):
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/agent/intents/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = intent.Intent(name='name_value', display_name='display_name_value', webhook_state=intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED, priority=898, is_fallback=True, ml_disabled=True, live_agent_handoff=True, end_interaction=True, input_context_names=['input_context_names_value'], events=['events_value'], action='action_value', reset_contexts=True, default_response_platforms=[intent.Intent.Message.Platform.FACEBOOK], root_followup_intent_name='root_followup_intent_name_value', parent_followup_intent_name='parent_followup_intent_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = intent.Intent.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_intent(request)
    assert isinstance(response, intent.Intent)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.webhook_state == intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED
    assert response.priority == 898
    assert response.is_fallback is True
    assert response.ml_disabled is True
    assert response.live_agent_handoff is True
    assert response.end_interaction is True
    assert response.input_context_names == ['input_context_names_value']
    assert response.events == ['events_value']
    assert response.action == 'action_value'
    assert response.reset_contexts is True
    assert response.default_response_platforms == [intent.Intent.Message.Platform.FACEBOOK]
    assert response.root_followup_intent_name == 'root_followup_intent_name_value'
    assert response.parent_followup_intent_name == 'parent_followup_intent_name_value'

def test_get_intent_rest_required_fields(request_type=intent.GetIntentRequest):
    if False:
        return 10
    transport_class = transports.IntentsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_intent._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_intent._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('intent_view', 'language_code'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = intent.Intent()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = intent.Intent.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_intent(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_intent_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.IntentsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_intent._get_unset_required_fields({})
    assert set(unset_fields) == set(('intentView', 'languageCode')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_intent_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.IntentsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IntentsRestInterceptor())
    client = IntentsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IntentsRestInterceptor, 'post_get_intent') as post, mock.patch.object(transports.IntentsRestInterceptor, 'pre_get_intent') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = intent.GetIntentRequest.pb(intent.GetIntentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = intent.Intent.to_json(intent.Intent())
        request = intent.GetIntentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = intent.Intent()
        client.get_intent(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_intent_rest_bad_request(transport: str='rest', request_type=intent.GetIntentRequest):
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/agent/intents/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_intent(request)

def test_get_intent_rest_flattened():
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = intent.Intent()
        sample_request = {'name': 'projects/sample1/agent/intents/sample2'}
        mock_args = dict(name='name_value', language_code='language_code_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = intent.Intent.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_intent(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/agent/intents/*}' % client.transport._host, args[1])

def test_get_intent_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_intent(intent.GetIntentRequest(), name='name_value', language_code='language_code_value')

def test_get_intent_rest_error():
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcd_intent.CreateIntentRequest, dict])
def test_create_intent_rest(request_type):
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/agent'}
    request_init['intent'] = {'name': 'name_value', 'display_name': 'display_name_value', 'webhook_state': 1, 'priority': 898, 'is_fallback': True, 'ml_disabled': True, 'live_agent_handoff': True, 'end_interaction': True, 'input_context_names': ['input_context_names_value1', 'input_context_names_value2'], 'events': ['events_value1', 'events_value2'], 'training_phrases': [{'name': 'name_value', 'type_': 1, 'parts': [{'text': 'text_value', 'entity_type': 'entity_type_value', 'alias': 'alias_value', 'user_defined': True}], 'times_added_count': 1787}], 'action': 'action_value', 'output_contexts': [{'name': 'name_value', 'lifespan_count': 1498, 'parameters': {'fields': {}}}], 'reset_contexts': True, 'parameters': [{'name': 'name_value', 'display_name': 'display_name_value', 'value': 'value_value', 'default_value': 'default_value_value', 'entity_type_display_name': 'entity_type_display_name_value', 'mandatory': True, 'prompts': ['prompts_value1', 'prompts_value2'], 'is_list': True}], 'messages': [{'text': {'text': ['text_value1', 'text_value2']}, 'image': {'image_uri': 'image_uri_value', 'accessibility_text': 'accessibility_text_value'}, 'quick_replies': {'title': 'title_value', 'quick_replies': ['quick_replies_value1', 'quick_replies_value2']}, 'card': {'title': 'title_value', 'subtitle': 'subtitle_value', 'image_uri': 'image_uri_value', 'buttons': [{'text': 'text_value', 'postback': 'postback_value'}]}, 'payload': {}, 'simple_responses': {'simple_responses': [{'text_to_speech': 'text_to_speech_value', 'ssml': 'ssml_value', 'display_text': 'display_text_value'}]}, 'basic_card': {'title': 'title_value', 'subtitle': 'subtitle_value', 'formatted_text': 'formatted_text_value', 'image': {}, 'buttons': [{'title': 'title_value', 'open_uri_action': {'uri': 'uri_value'}}]}, 'suggestions': {'suggestions': [{'title': 'title_value'}]}, 'link_out_suggestion': {'destination_name': 'destination_name_value', 'uri': 'uri_value'}, 'list_select': {'title': 'title_value', 'items': [{'info': {'key': 'key_value', 'synonyms': ['synonyms_value1', 'synonyms_value2']}, 'title': 'title_value', 'description': 'description_value', 'image': {}}], 'subtitle': 'subtitle_value'}, 'carousel_select': {'items': [{'info': {}, 'title': 'title_value', 'description': 'description_value', 'image': {}}]}, 'browse_carousel_card': {'items': [{'open_uri_action': {'url': 'url_value', 'url_type_hint': 1}, 'title': 'title_value', 'description': 'description_value', 'image': {}, 'footer': 'footer_value'}], 'image_display_options': 1}, 'table_card': {'title': 'title_value', 'subtitle': 'subtitle_value', 'image': {}, 'column_properties': [{'header': 'header_value', 'horizontal_alignment': 1}], 'rows': [{'cells': [{'text': 'text_value'}], 'divider_after': True}], 'buttons': {}}, 'media_content': {'media_type': 1, 'media_objects': [{'name': 'name_value', 'description': 'description_value', 'large_image': {}, 'icon': {}, 'content_url': 'content_url_value'}]}, 'platform': 1}], 'default_response_platforms': [1], 'root_followup_intent_name': 'root_followup_intent_name_value', 'parent_followup_intent_name': 'parent_followup_intent_name_value', 'followup_intent_info': [{'followup_intent_name': 'followup_intent_name_value', 'parent_followup_intent_name': 'parent_followup_intent_name_value'}]}
    test_field = gcd_intent.CreateIntentRequest.meta.fields['intent']

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
    for (field, value) in request_init['intent'].items():
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
                for i in range(0, len(request_init['intent'][field])):
                    del request_init['intent'][field][i][subfield]
            else:
                del request_init['intent'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_intent.Intent(name='name_value', display_name='display_name_value', webhook_state=gcd_intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED, priority=898, is_fallback=True, ml_disabled=True, live_agent_handoff=True, end_interaction=True, input_context_names=['input_context_names_value'], events=['events_value'], action='action_value', reset_contexts=True, default_response_platforms=[gcd_intent.Intent.Message.Platform.FACEBOOK], root_followup_intent_name='root_followup_intent_name_value', parent_followup_intent_name='parent_followup_intent_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_intent.Intent.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_intent(request)
    assert isinstance(response, gcd_intent.Intent)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.webhook_state == gcd_intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED
    assert response.priority == 898
    assert response.is_fallback is True
    assert response.ml_disabled is True
    assert response.live_agent_handoff is True
    assert response.end_interaction is True
    assert response.input_context_names == ['input_context_names_value']
    assert response.events == ['events_value']
    assert response.action == 'action_value'
    assert response.reset_contexts is True
    assert response.default_response_platforms == [gcd_intent.Intent.Message.Platform.FACEBOOK]
    assert response.root_followup_intent_name == 'root_followup_intent_name_value'
    assert response.parent_followup_intent_name == 'parent_followup_intent_name_value'

def test_create_intent_rest_required_fields(request_type=gcd_intent.CreateIntentRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.IntentsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_intent._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_intent._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('intent_view', 'language_code'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcd_intent.Intent()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcd_intent.Intent.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_intent(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_intent_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.IntentsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_intent._get_unset_required_fields({})
    assert set(unset_fields) == set(('intentView', 'languageCode')) & set(('parent', 'intent'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_intent_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.IntentsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IntentsRestInterceptor())
    client = IntentsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IntentsRestInterceptor, 'post_create_intent') as post, mock.patch.object(transports.IntentsRestInterceptor, 'pre_create_intent') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcd_intent.CreateIntentRequest.pb(gcd_intent.CreateIntentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcd_intent.Intent.to_json(gcd_intent.Intent())
        request = gcd_intent.CreateIntentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcd_intent.Intent()
        client.create_intent(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_intent_rest_bad_request(transport: str='rest', request_type=gcd_intent.CreateIntentRequest):
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/agent'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_intent(request)

def test_create_intent_rest_flattened():
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_intent.Intent()
        sample_request = {'parent': 'projects/sample1/agent'}
        mock_args = dict(parent='parent_value', intent=gcd_intent.Intent(name='name_value'), language_code='language_code_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_intent.Intent.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_intent(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/agent}/intents' % client.transport._host, args[1])

def test_create_intent_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_intent(gcd_intent.CreateIntentRequest(), parent='parent_value', intent=gcd_intent.Intent(name='name_value'), language_code='language_code_value')

def test_create_intent_rest_error():
    if False:
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcd_intent.UpdateIntentRequest, dict])
def test_update_intent_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'intent': {'name': 'projects/sample1/agent/intents/sample2'}}
    request_init['intent'] = {'name': 'projects/sample1/agent/intents/sample2', 'display_name': 'display_name_value', 'webhook_state': 1, 'priority': 898, 'is_fallback': True, 'ml_disabled': True, 'live_agent_handoff': True, 'end_interaction': True, 'input_context_names': ['input_context_names_value1', 'input_context_names_value2'], 'events': ['events_value1', 'events_value2'], 'training_phrases': [{'name': 'name_value', 'type_': 1, 'parts': [{'text': 'text_value', 'entity_type': 'entity_type_value', 'alias': 'alias_value', 'user_defined': True}], 'times_added_count': 1787}], 'action': 'action_value', 'output_contexts': [{'name': 'name_value', 'lifespan_count': 1498, 'parameters': {'fields': {}}}], 'reset_contexts': True, 'parameters': [{'name': 'name_value', 'display_name': 'display_name_value', 'value': 'value_value', 'default_value': 'default_value_value', 'entity_type_display_name': 'entity_type_display_name_value', 'mandatory': True, 'prompts': ['prompts_value1', 'prompts_value2'], 'is_list': True}], 'messages': [{'text': {'text': ['text_value1', 'text_value2']}, 'image': {'image_uri': 'image_uri_value', 'accessibility_text': 'accessibility_text_value'}, 'quick_replies': {'title': 'title_value', 'quick_replies': ['quick_replies_value1', 'quick_replies_value2']}, 'card': {'title': 'title_value', 'subtitle': 'subtitle_value', 'image_uri': 'image_uri_value', 'buttons': [{'text': 'text_value', 'postback': 'postback_value'}]}, 'payload': {}, 'simple_responses': {'simple_responses': [{'text_to_speech': 'text_to_speech_value', 'ssml': 'ssml_value', 'display_text': 'display_text_value'}]}, 'basic_card': {'title': 'title_value', 'subtitle': 'subtitle_value', 'formatted_text': 'formatted_text_value', 'image': {}, 'buttons': [{'title': 'title_value', 'open_uri_action': {'uri': 'uri_value'}}]}, 'suggestions': {'suggestions': [{'title': 'title_value'}]}, 'link_out_suggestion': {'destination_name': 'destination_name_value', 'uri': 'uri_value'}, 'list_select': {'title': 'title_value', 'items': [{'info': {'key': 'key_value', 'synonyms': ['synonyms_value1', 'synonyms_value2']}, 'title': 'title_value', 'description': 'description_value', 'image': {}}], 'subtitle': 'subtitle_value'}, 'carousel_select': {'items': [{'info': {}, 'title': 'title_value', 'description': 'description_value', 'image': {}}]}, 'browse_carousel_card': {'items': [{'open_uri_action': {'url': 'url_value', 'url_type_hint': 1}, 'title': 'title_value', 'description': 'description_value', 'image': {}, 'footer': 'footer_value'}], 'image_display_options': 1}, 'table_card': {'title': 'title_value', 'subtitle': 'subtitle_value', 'image': {}, 'column_properties': [{'header': 'header_value', 'horizontal_alignment': 1}], 'rows': [{'cells': [{'text': 'text_value'}], 'divider_after': True}], 'buttons': {}}, 'media_content': {'media_type': 1, 'media_objects': [{'name': 'name_value', 'description': 'description_value', 'large_image': {}, 'icon': {}, 'content_url': 'content_url_value'}]}, 'platform': 1}], 'default_response_platforms': [1], 'root_followup_intent_name': 'root_followup_intent_name_value', 'parent_followup_intent_name': 'parent_followup_intent_name_value', 'followup_intent_info': [{'followup_intent_name': 'followup_intent_name_value', 'parent_followup_intent_name': 'parent_followup_intent_name_value'}]}
    test_field = gcd_intent.UpdateIntentRequest.meta.fields['intent']

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
    for (field, value) in request_init['intent'].items():
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
                for i in range(0, len(request_init['intent'][field])):
                    del request_init['intent'][field][i][subfield]
            else:
                del request_init['intent'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_intent.Intent(name='name_value', display_name='display_name_value', webhook_state=gcd_intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED, priority=898, is_fallback=True, ml_disabled=True, live_agent_handoff=True, end_interaction=True, input_context_names=['input_context_names_value'], events=['events_value'], action='action_value', reset_contexts=True, default_response_platforms=[gcd_intent.Intent.Message.Platform.FACEBOOK], root_followup_intent_name='root_followup_intent_name_value', parent_followup_intent_name='parent_followup_intent_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_intent.Intent.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_intent(request)
    assert isinstance(response, gcd_intent.Intent)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.webhook_state == gcd_intent.Intent.WebhookState.WEBHOOK_STATE_ENABLED
    assert response.priority == 898
    assert response.is_fallback is True
    assert response.ml_disabled is True
    assert response.live_agent_handoff is True
    assert response.end_interaction is True
    assert response.input_context_names == ['input_context_names_value']
    assert response.events == ['events_value']
    assert response.action == 'action_value'
    assert response.reset_contexts is True
    assert response.default_response_platforms == [gcd_intent.Intent.Message.Platform.FACEBOOK]
    assert response.root_followup_intent_name == 'root_followup_intent_name_value'
    assert response.parent_followup_intent_name == 'parent_followup_intent_name_value'

def test_update_intent_rest_required_fields(request_type=gcd_intent.UpdateIntentRequest):
    if False:
        return 10
    transport_class = transports.IntentsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_intent._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_intent._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('intent_view', 'language_code', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcd_intent.Intent()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcd_intent.Intent.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_intent(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_intent_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.IntentsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_intent._get_unset_required_fields({})
    assert set(unset_fields) == set(('intentView', 'languageCode', 'updateMask')) & set(('intent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_intent_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.IntentsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IntentsRestInterceptor())
    client = IntentsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IntentsRestInterceptor, 'post_update_intent') as post, mock.patch.object(transports.IntentsRestInterceptor, 'pre_update_intent') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcd_intent.UpdateIntentRequest.pb(gcd_intent.UpdateIntentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcd_intent.Intent.to_json(gcd_intent.Intent())
        request = gcd_intent.UpdateIntentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcd_intent.Intent()
        client.update_intent(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_intent_rest_bad_request(transport: str='rest', request_type=gcd_intent.UpdateIntentRequest):
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'intent': {'name': 'projects/sample1/agent/intents/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_intent(request)

def test_update_intent_rest_flattened():
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_intent.Intent()
        sample_request = {'intent': {'name': 'projects/sample1/agent/intents/sample2'}}
        mock_args = dict(intent=gcd_intent.Intent(name='name_value'), language_code='language_code_value', update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_intent.Intent.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_intent(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{intent.name=projects/*/agent/intents/*}' % client.transport._host, args[1])

def test_update_intent_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_intent(gcd_intent.UpdateIntentRequest(), intent=gcd_intent.Intent(name='name_value'), language_code='language_code_value', update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_intent_rest_error():
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [intent.DeleteIntentRequest, dict])
def test_delete_intent_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/agent/intents/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_intent(request)
    assert response is None

def test_delete_intent_rest_required_fields(request_type=intent.DeleteIntentRequest):
    if False:
        return 10
    transport_class = transports.IntentsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_intent._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_intent._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_intent(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_intent_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.IntentsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_intent._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_intent_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.IntentsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IntentsRestInterceptor())
    client = IntentsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.IntentsRestInterceptor, 'pre_delete_intent') as pre:
        pre.assert_not_called()
        pb_message = intent.DeleteIntentRequest.pb(intent.DeleteIntentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = intent.DeleteIntentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_intent(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_intent_rest_bad_request(transport: str='rest', request_type=intent.DeleteIntentRequest):
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/agent/intents/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_intent(request)

def test_delete_intent_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/agent/intents/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_intent(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/agent/intents/*}' % client.transport._host, args[1])

def test_delete_intent_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_intent(intent.DeleteIntentRequest(), name='name_value')

def test_delete_intent_rest_error():
    if False:
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [intent.BatchUpdateIntentsRequest, dict])
def test_batch_update_intents_rest(request_type):
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/agent'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_update_intents(request)
    assert response.operation.name == 'operations/spam'

def test_batch_update_intents_rest_required_fields(request_type=intent.BatchUpdateIntentsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.IntentsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_update_intents._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_update_intents._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_update_intents(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_update_intents_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.IntentsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_update_intents._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_update_intents_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.IntentsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IntentsRestInterceptor())
    client = IntentsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.IntentsRestInterceptor, 'post_batch_update_intents') as post, mock.patch.object(transports.IntentsRestInterceptor, 'pre_batch_update_intents') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = intent.BatchUpdateIntentsRequest.pb(intent.BatchUpdateIntentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = intent.BatchUpdateIntentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_update_intents(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_update_intents_rest_bad_request(transport: str='rest', request_type=intent.BatchUpdateIntentsRequest):
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/agent'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_update_intents(request)

def test_batch_update_intents_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/agent'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_update_intents(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/agent}/intents:batchUpdate' % client.transport._host, args[1])

def test_batch_update_intents_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_update_intents(intent.BatchUpdateIntentsRequest(), parent='parent_value', intent_batch_uri='intent_batch_uri_value', intent_batch_inline=intent.IntentBatch(intents=[intent.Intent(name='name_value')]))

def test_batch_update_intents_rest_error():
    if False:
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [intent.BatchDeleteIntentsRequest, dict])
def test_batch_delete_intents_rest(request_type):
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/agent'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_delete_intents(request)
    assert response.operation.name == 'operations/spam'

def test_batch_delete_intents_rest_required_fields(request_type=intent.BatchDeleteIntentsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.IntentsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_delete_intents._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_delete_intents._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_delete_intents(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_delete_intents_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.IntentsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_delete_intents._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'intents'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_delete_intents_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.IntentsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.IntentsRestInterceptor())
    client = IntentsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.IntentsRestInterceptor, 'post_batch_delete_intents') as post, mock.patch.object(transports.IntentsRestInterceptor, 'pre_batch_delete_intents') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = intent.BatchDeleteIntentsRequest.pb(intent.BatchDeleteIntentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = intent.BatchDeleteIntentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_delete_intents(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_delete_intents_rest_bad_request(transport: str='rest', request_type=intent.BatchDeleteIntentsRequest):
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/agent'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_delete_intents(request)

def test_batch_delete_intents_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/agent'}
        mock_args = dict(parent='parent_value', intents=[intent.Intent(name='name_value')])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_delete_intents(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/agent}/intents:batchDelete' % client.transport._host, args[1])

def test_batch_delete_intents_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_delete_intents(intent.BatchDeleteIntentsRequest(), parent='parent_value', intents=[intent.Intent(name='name_value')])

def test_batch_delete_intents_rest_error():
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.IntentsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.IntentsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = IntentsClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.IntentsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = IntentsClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = IntentsClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.IntentsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = IntentsClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.IntentsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = IntentsClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.IntentsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.IntentsGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.IntentsGrpcTransport, transports.IntentsGrpcAsyncIOTransport, transports.IntentsRestTransport])
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
        for i in range(10):
            print('nop')
    transport = IntentsClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.IntentsGrpcTransport)

def test_intents_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.IntentsTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_intents_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.dialogflow_v2.services.intents.transports.IntentsTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.IntentsTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_intents', 'get_intent', 'create_intent', 'update_intent', 'delete_intent', 'batch_update_intents', 'batch_delete_intents', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'list_operations')
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

def test_intents_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dialogflow_v2.services.intents.transports.IntentsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.IntentsTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id='octopus')

def test_intents_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dialogflow_v2.services.intents.transports.IntentsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.IntentsTransport()
        adc.assert_called_once()

def test_intents_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        IntentsClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.IntentsGrpcTransport, transports.IntentsGrpcAsyncIOTransport])
def test_intents_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.IntentsGrpcTransport, transports.IntentsGrpcAsyncIOTransport, transports.IntentsRestTransport])
def test_intents_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.IntentsGrpcTransport, grpc_helpers), (transports.IntentsGrpcAsyncIOTransport, grpc_helpers_async)])
def test_intents_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dialogflow.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), scopes=['1', '2'], default_host='dialogflow.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.IntentsGrpcTransport, transports.IntentsGrpcAsyncIOTransport])
def test_intents_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_intents_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.IntentsRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_intents_rest_lro_client():
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_intents_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dialogflow.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_intents_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dialogflow.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('dialogflow.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_intents_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = IntentsClient(credentials=creds1, transport=transport_name)
    client2 = IntentsClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_intents._session
    session2 = client2.transport.list_intents._session
    assert session1 != session2
    session1 = client1.transport.get_intent._session
    session2 = client2.transport.get_intent._session
    assert session1 != session2
    session1 = client1.transport.create_intent._session
    session2 = client2.transport.create_intent._session
    assert session1 != session2
    session1 = client1.transport.update_intent._session
    session2 = client2.transport.update_intent._session
    assert session1 != session2
    session1 = client1.transport.delete_intent._session
    session2 = client2.transport.delete_intent._session
    assert session1 != session2
    session1 = client1.transport.batch_update_intents._session
    session2 = client2.transport.batch_update_intents._session
    assert session1 != session2
    session1 = client1.transport.batch_delete_intents._session
    session2 = client2.transport.batch_delete_intents._session
    assert session1 != session2

def test_intents_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.IntentsGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_intents_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.IntentsGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.IntentsGrpcTransport, transports.IntentsGrpcAsyncIOTransport])
def test_intents_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.IntentsGrpcTransport, transports.IntentsGrpcAsyncIOTransport])
def test_intents_transport_channel_mtls_with_adc(transport_class):
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

def test_intents_grpc_lro_client():
    if False:
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_intents_grpc_lro_async_client():
    if False:
        print('Hello World!')
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_context_path():
    if False:
        print('Hello World!')
    project = 'squid'
    session = 'clam'
    context = 'whelk'
    expected = 'projects/{project}/agent/sessions/{session}/contexts/{context}'.format(project=project, session=session, context=context)
    actual = IntentsClient.context_path(project, session, context)
    assert expected == actual

def test_parse_context_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus', 'session': 'oyster', 'context': 'nudibranch'}
    path = IntentsClient.context_path(**expected)
    actual = IntentsClient.parse_context_path(path)
    assert expected == actual

def test_intent_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    intent = 'mussel'
    expected = 'projects/{project}/agent/intents/{intent}'.format(project=project, intent=intent)
    actual = IntentsClient.intent_path(project, intent)
    assert expected == actual

def test_parse_intent_path():
    if False:
        return 10
    expected = {'project': 'winkle', 'intent': 'nautilus'}
    path = IntentsClient.intent_path(**expected)
    actual = IntentsClient.parse_intent_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = IntentsClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'abalone'}
    path = IntentsClient.common_billing_account_path(**expected)
    actual = IntentsClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = IntentsClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'clam'}
    path = IntentsClient.common_folder_path(**expected)
    actual = IntentsClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = IntentsClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'octopus'}
    path = IntentsClient.common_organization_path(**expected)
    actual = IntentsClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = IntentsClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nudibranch'}
    path = IntentsClient.common_project_path(**expected)
    actual = IntentsClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = IntentsClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = IntentsClient.common_location_path(**expected)
    actual = IntentsClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.IntentsTransport, '_prep_wrapped_messages') as prep:
        client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.IntentsTransport, '_prep_wrapped_messages') as prep:
        transport_class = IntentsClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = IntentsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = IntentsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = IntentsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(IntentsClient, transports.IntentsGrpcTransport), (IntentsAsyncClient, transports.IntentsGrpcAsyncIOTransport)])
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