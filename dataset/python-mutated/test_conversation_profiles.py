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
from google.cloud.dialogflow_v2.services.conversation_profiles import ConversationProfilesAsyncClient, ConversationProfilesClient, pagers, transports
from google.cloud.dialogflow_v2.types import conversation_profile as gcd_conversation_profile
from google.cloud.dialogflow_v2.types import audio_config
from google.cloud.dialogflow_v2.types import conversation_profile
from google.cloud.dialogflow_v2.types import participant

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
    assert ConversationProfilesClient._get_default_mtls_endpoint(None) is None
    assert ConversationProfilesClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ConversationProfilesClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ConversationProfilesClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ConversationProfilesClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ConversationProfilesClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ConversationProfilesClient, 'grpc'), (ConversationProfilesAsyncClient, 'grpc_asyncio'), (ConversationProfilesClient, 'rest')])
def test_conversation_profiles_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ConversationProfilesGrpcTransport, 'grpc'), (transports.ConversationProfilesGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.ConversationProfilesRestTransport, 'rest')])
def test_conversation_profiles_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ConversationProfilesClient, 'grpc'), (ConversationProfilesAsyncClient, 'grpc_asyncio'), (ConversationProfilesClient, 'rest')])
def test_conversation_profiles_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

def test_conversation_profiles_client_get_transport_class():
    if False:
        return 10
    transport = ConversationProfilesClient.get_transport_class()
    available_transports = [transports.ConversationProfilesGrpcTransport, transports.ConversationProfilesRestTransport]
    assert transport in available_transports
    transport = ConversationProfilesClient.get_transport_class('grpc')
    assert transport == transports.ConversationProfilesGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ConversationProfilesClient, transports.ConversationProfilesGrpcTransport, 'grpc'), (ConversationProfilesAsyncClient, transports.ConversationProfilesGrpcAsyncIOTransport, 'grpc_asyncio'), (ConversationProfilesClient, transports.ConversationProfilesRestTransport, 'rest')])
@mock.patch.object(ConversationProfilesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConversationProfilesClient))
@mock.patch.object(ConversationProfilesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConversationProfilesAsyncClient))
def test_conversation_profiles_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(ConversationProfilesClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ConversationProfilesClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ConversationProfilesClient, transports.ConversationProfilesGrpcTransport, 'grpc', 'true'), (ConversationProfilesAsyncClient, transports.ConversationProfilesGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ConversationProfilesClient, transports.ConversationProfilesGrpcTransport, 'grpc', 'false'), (ConversationProfilesAsyncClient, transports.ConversationProfilesGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (ConversationProfilesClient, transports.ConversationProfilesRestTransport, 'rest', 'true'), (ConversationProfilesClient, transports.ConversationProfilesRestTransport, 'rest', 'false')])
@mock.patch.object(ConversationProfilesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConversationProfilesClient))
@mock.patch.object(ConversationProfilesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConversationProfilesAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_conversation_profiles_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
    if False:
        for i in range(10):
            print('nop')
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

@pytest.mark.parametrize('client_class', [ConversationProfilesClient, ConversationProfilesAsyncClient])
@mock.patch.object(ConversationProfilesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConversationProfilesClient))
@mock.patch.object(ConversationProfilesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConversationProfilesAsyncClient))
def test_conversation_profiles_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ConversationProfilesClient, transports.ConversationProfilesGrpcTransport, 'grpc'), (ConversationProfilesAsyncClient, transports.ConversationProfilesGrpcAsyncIOTransport, 'grpc_asyncio'), (ConversationProfilesClient, transports.ConversationProfilesRestTransport, 'rest')])
def test_conversation_profiles_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ConversationProfilesClient, transports.ConversationProfilesGrpcTransport, 'grpc', grpc_helpers), (ConversationProfilesAsyncClient, transports.ConversationProfilesGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (ConversationProfilesClient, transports.ConversationProfilesRestTransport, 'rest', None)])
def test_conversation_profiles_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_conversation_profiles_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.dialogflow_v2.services.conversation_profiles.transports.ConversationProfilesGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ConversationProfilesClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ConversationProfilesClient, transports.ConversationProfilesGrpcTransport, 'grpc', grpc_helpers), (ConversationProfilesAsyncClient, transports.ConversationProfilesGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_conversation_profiles_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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

@pytest.mark.parametrize('request_type', [conversation_profile.ListConversationProfilesRequest, dict])
def test_list_conversation_profiles(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_conversation_profiles), '__call__') as call:
        call.return_value = conversation_profile.ListConversationProfilesResponse(next_page_token='next_page_token_value')
        response = client.list_conversation_profiles(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_profile.ListConversationProfilesRequest()
    assert isinstance(response, pagers.ListConversationProfilesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_conversation_profiles_empty_call():
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_conversation_profiles), '__call__') as call:
        client.list_conversation_profiles()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_profile.ListConversationProfilesRequest()

@pytest.mark.asyncio
async def test_list_conversation_profiles_async(transport: str='grpc_asyncio', request_type=conversation_profile.ListConversationProfilesRequest):
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_conversation_profiles), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_profile.ListConversationProfilesResponse(next_page_token='next_page_token_value'))
        response = await client.list_conversation_profiles(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_profile.ListConversationProfilesRequest()
    assert isinstance(response, pagers.ListConversationProfilesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_conversation_profiles_async_from_dict():
    await test_list_conversation_profiles_async(request_type=dict)

def test_list_conversation_profiles_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_profile.ListConversationProfilesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_conversation_profiles), '__call__') as call:
        call.return_value = conversation_profile.ListConversationProfilesResponse()
        client.list_conversation_profiles(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_conversation_profiles_field_headers_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_profile.ListConversationProfilesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_conversation_profiles), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_profile.ListConversationProfilesResponse())
        await client.list_conversation_profiles(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_conversation_profiles_flattened():
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_conversation_profiles), '__call__') as call:
        call.return_value = conversation_profile.ListConversationProfilesResponse()
        client.list_conversation_profiles(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_conversation_profiles_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_conversation_profiles(conversation_profile.ListConversationProfilesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_conversation_profiles_flattened_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_conversation_profiles), '__call__') as call:
        call.return_value = conversation_profile.ListConversationProfilesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_profile.ListConversationProfilesResponse())
        response = await client.list_conversation_profiles(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_conversation_profiles_flattened_error_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_conversation_profiles(conversation_profile.ListConversationProfilesRequest(), parent='parent_value')

def test_list_conversation_profiles_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_conversation_profiles), '__call__') as call:
        call.side_effect = (conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile()], next_page_token='abc'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[], next_page_token='def'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile()], next_page_token='ghi'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_conversation_profiles(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, conversation_profile.ConversationProfile) for i in results))

def test_list_conversation_profiles_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_conversation_profiles), '__call__') as call:
        call.side_effect = (conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile()], next_page_token='abc'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[], next_page_token='def'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile()], next_page_token='ghi'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile()]), RuntimeError)
        pages = list(client.list_conversation_profiles(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_conversation_profiles_async_pager():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_conversation_profiles), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile()], next_page_token='abc'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[], next_page_token='def'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile()], next_page_token='ghi'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile()]), RuntimeError)
        async_pager = await client.list_conversation_profiles(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, conversation_profile.ConversationProfile) for i in responses))

@pytest.mark.asyncio
async def test_list_conversation_profiles_async_pages():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_conversation_profiles), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile()], next_page_token='abc'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[], next_page_token='def'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile()], next_page_token='ghi'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_conversation_profiles(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [conversation_profile.GetConversationProfileRequest, dict])
def test_get_conversation_profile(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_conversation_profile), '__call__') as call:
        call.return_value = conversation_profile.ConversationProfile(name='name_value', display_name='display_name_value', language_code='language_code_value', time_zone='time_zone_value', security_settings='security_settings_value')
        response = client.get_conversation_profile(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_profile.GetConversationProfileRequest()
    assert isinstance(response, conversation_profile.ConversationProfile)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.language_code == 'language_code_value'
    assert response.time_zone == 'time_zone_value'
    assert response.security_settings == 'security_settings_value'

def test_get_conversation_profile_empty_call():
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_conversation_profile), '__call__') as call:
        client.get_conversation_profile()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_profile.GetConversationProfileRequest()

@pytest.mark.asyncio
async def test_get_conversation_profile_async(transport: str='grpc_asyncio', request_type=conversation_profile.GetConversationProfileRequest):
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_conversation_profile), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_profile.ConversationProfile(name='name_value', display_name='display_name_value', language_code='language_code_value', time_zone='time_zone_value', security_settings='security_settings_value'))
        response = await client.get_conversation_profile(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_profile.GetConversationProfileRequest()
    assert isinstance(response, conversation_profile.ConversationProfile)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.language_code == 'language_code_value'
    assert response.time_zone == 'time_zone_value'
    assert response.security_settings == 'security_settings_value'

@pytest.mark.asyncio
async def test_get_conversation_profile_async_from_dict():
    await test_get_conversation_profile_async(request_type=dict)

def test_get_conversation_profile_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_profile.GetConversationProfileRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_conversation_profile), '__call__') as call:
        call.return_value = conversation_profile.ConversationProfile()
        client.get_conversation_profile(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_conversation_profile_field_headers_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_profile.GetConversationProfileRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_conversation_profile), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_profile.ConversationProfile())
        await client.get_conversation_profile(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_conversation_profile_flattened():
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_conversation_profile), '__call__') as call:
        call.return_value = conversation_profile.ConversationProfile()
        client.get_conversation_profile(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_conversation_profile_flattened_error():
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_conversation_profile(conversation_profile.GetConversationProfileRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_conversation_profile_flattened_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_conversation_profile), '__call__') as call:
        call.return_value = conversation_profile.ConversationProfile()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_profile.ConversationProfile())
        response = await client.get_conversation_profile(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_conversation_profile_flattened_error_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_conversation_profile(conversation_profile.GetConversationProfileRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcd_conversation_profile.CreateConversationProfileRequest, dict])
def test_create_conversation_profile(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_conversation_profile), '__call__') as call:
        call.return_value = gcd_conversation_profile.ConversationProfile(name='name_value', display_name='display_name_value', language_code='language_code_value', time_zone='time_zone_value', security_settings='security_settings_value')
        response = client.create_conversation_profile(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_profile.CreateConversationProfileRequest()
    assert isinstance(response, gcd_conversation_profile.ConversationProfile)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.language_code == 'language_code_value'
    assert response.time_zone == 'time_zone_value'
    assert response.security_settings == 'security_settings_value'

def test_create_conversation_profile_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_conversation_profile), '__call__') as call:
        client.create_conversation_profile()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_profile.CreateConversationProfileRequest()

@pytest.mark.asyncio
async def test_create_conversation_profile_async(transport: str='grpc_asyncio', request_type=gcd_conversation_profile.CreateConversationProfileRequest):
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_conversation_profile), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_conversation_profile.ConversationProfile(name='name_value', display_name='display_name_value', language_code='language_code_value', time_zone='time_zone_value', security_settings='security_settings_value'))
        response = await client.create_conversation_profile(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_profile.CreateConversationProfileRequest()
    assert isinstance(response, gcd_conversation_profile.ConversationProfile)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.language_code == 'language_code_value'
    assert response.time_zone == 'time_zone_value'
    assert response.security_settings == 'security_settings_value'

@pytest.mark.asyncio
async def test_create_conversation_profile_async_from_dict():
    await test_create_conversation_profile_async(request_type=dict)

def test_create_conversation_profile_field_headers():
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_conversation_profile.CreateConversationProfileRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_conversation_profile), '__call__') as call:
        call.return_value = gcd_conversation_profile.ConversationProfile()
        client.create_conversation_profile(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_conversation_profile_field_headers_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_conversation_profile.CreateConversationProfileRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_conversation_profile), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_conversation_profile.ConversationProfile())
        await client.create_conversation_profile(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_conversation_profile_flattened():
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_conversation_profile), '__call__') as call:
        call.return_value = gcd_conversation_profile.ConversationProfile()
        client.create_conversation_profile(parent='parent_value', conversation_profile=gcd_conversation_profile.ConversationProfile(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].conversation_profile
        mock_val = gcd_conversation_profile.ConversationProfile(name='name_value')
        assert arg == mock_val

def test_create_conversation_profile_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_conversation_profile(gcd_conversation_profile.CreateConversationProfileRequest(), parent='parent_value', conversation_profile=gcd_conversation_profile.ConversationProfile(name='name_value'))

@pytest.mark.asyncio
async def test_create_conversation_profile_flattened_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_conversation_profile), '__call__') as call:
        call.return_value = gcd_conversation_profile.ConversationProfile()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_conversation_profile.ConversationProfile())
        response = await client.create_conversation_profile(parent='parent_value', conversation_profile=gcd_conversation_profile.ConversationProfile(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].conversation_profile
        mock_val = gcd_conversation_profile.ConversationProfile(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_conversation_profile_flattened_error_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_conversation_profile(gcd_conversation_profile.CreateConversationProfileRequest(), parent='parent_value', conversation_profile=gcd_conversation_profile.ConversationProfile(name='name_value'))

@pytest.mark.parametrize('request_type', [gcd_conversation_profile.UpdateConversationProfileRequest, dict])
def test_update_conversation_profile(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_conversation_profile), '__call__') as call:
        call.return_value = gcd_conversation_profile.ConversationProfile(name='name_value', display_name='display_name_value', language_code='language_code_value', time_zone='time_zone_value', security_settings='security_settings_value')
        response = client.update_conversation_profile(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_profile.UpdateConversationProfileRequest()
    assert isinstance(response, gcd_conversation_profile.ConversationProfile)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.language_code == 'language_code_value'
    assert response.time_zone == 'time_zone_value'
    assert response.security_settings == 'security_settings_value'

def test_update_conversation_profile_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_conversation_profile), '__call__') as call:
        client.update_conversation_profile()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_profile.UpdateConversationProfileRequest()

@pytest.mark.asyncio
async def test_update_conversation_profile_async(transport: str='grpc_asyncio', request_type=gcd_conversation_profile.UpdateConversationProfileRequest):
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_conversation_profile), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_conversation_profile.ConversationProfile(name='name_value', display_name='display_name_value', language_code='language_code_value', time_zone='time_zone_value', security_settings='security_settings_value'))
        response = await client.update_conversation_profile(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_profile.UpdateConversationProfileRequest()
    assert isinstance(response, gcd_conversation_profile.ConversationProfile)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.language_code == 'language_code_value'
    assert response.time_zone == 'time_zone_value'
    assert response.security_settings == 'security_settings_value'

@pytest.mark.asyncio
async def test_update_conversation_profile_async_from_dict():
    await test_update_conversation_profile_async(request_type=dict)

def test_update_conversation_profile_field_headers():
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_conversation_profile.UpdateConversationProfileRequest()
    request.conversation_profile.name = 'name_value'
    with mock.patch.object(type(client.transport.update_conversation_profile), '__call__') as call:
        call.return_value = gcd_conversation_profile.ConversationProfile()
        client.update_conversation_profile(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'conversation_profile.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_conversation_profile_field_headers_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_conversation_profile.UpdateConversationProfileRequest()
    request.conversation_profile.name = 'name_value'
    with mock.patch.object(type(client.transport.update_conversation_profile), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_conversation_profile.ConversationProfile())
        await client.update_conversation_profile(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'conversation_profile.name=name_value') in kw['metadata']

def test_update_conversation_profile_flattened():
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_conversation_profile), '__call__') as call:
        call.return_value = gcd_conversation_profile.ConversationProfile()
        client.update_conversation_profile(conversation_profile=gcd_conversation_profile.ConversationProfile(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].conversation_profile
        mock_val = gcd_conversation_profile.ConversationProfile(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_conversation_profile_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_conversation_profile(gcd_conversation_profile.UpdateConversationProfileRequest(), conversation_profile=gcd_conversation_profile.ConversationProfile(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_conversation_profile_flattened_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_conversation_profile), '__call__') as call:
        call.return_value = gcd_conversation_profile.ConversationProfile()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_conversation_profile.ConversationProfile())
        response = await client.update_conversation_profile(conversation_profile=gcd_conversation_profile.ConversationProfile(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].conversation_profile
        mock_val = gcd_conversation_profile.ConversationProfile(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_conversation_profile_flattened_error_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_conversation_profile(gcd_conversation_profile.UpdateConversationProfileRequest(), conversation_profile=gcd_conversation_profile.ConversationProfile(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [conversation_profile.DeleteConversationProfileRequest, dict])
def test_delete_conversation_profile(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_conversation_profile), '__call__') as call:
        call.return_value = None
        response = client.delete_conversation_profile(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_profile.DeleteConversationProfileRequest()
    assert response is None

def test_delete_conversation_profile_empty_call():
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_conversation_profile), '__call__') as call:
        client.delete_conversation_profile()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_profile.DeleteConversationProfileRequest()

@pytest.mark.asyncio
async def test_delete_conversation_profile_async(transport: str='grpc_asyncio', request_type=conversation_profile.DeleteConversationProfileRequest):
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_conversation_profile), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_conversation_profile(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_profile.DeleteConversationProfileRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_conversation_profile_async_from_dict():
    await test_delete_conversation_profile_async(request_type=dict)

def test_delete_conversation_profile_field_headers():
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_profile.DeleteConversationProfileRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_conversation_profile), '__call__') as call:
        call.return_value = None
        client.delete_conversation_profile(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_conversation_profile_field_headers_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_profile.DeleteConversationProfileRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_conversation_profile), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_conversation_profile(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_conversation_profile_flattened():
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_conversation_profile), '__call__') as call:
        call.return_value = None
        client.delete_conversation_profile(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_conversation_profile_flattened_error():
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_conversation_profile(conversation_profile.DeleteConversationProfileRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_conversation_profile_flattened_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_conversation_profile), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_conversation_profile(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_conversation_profile_flattened_error_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_conversation_profile(conversation_profile.DeleteConversationProfileRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcd_conversation_profile.SetSuggestionFeatureConfigRequest, dict])
def test_set_suggestion_feature_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_suggestion_feature_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.set_suggestion_feature_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_profile.SetSuggestionFeatureConfigRequest()
    assert isinstance(response, future.Future)

def test_set_suggestion_feature_config_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_suggestion_feature_config), '__call__') as call:
        client.set_suggestion_feature_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_profile.SetSuggestionFeatureConfigRequest()

@pytest.mark.asyncio
async def test_set_suggestion_feature_config_async(transport: str='grpc_asyncio', request_type=gcd_conversation_profile.SetSuggestionFeatureConfigRequest):
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_suggestion_feature_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.set_suggestion_feature_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_profile.SetSuggestionFeatureConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_set_suggestion_feature_config_async_from_dict():
    await test_set_suggestion_feature_config_async(request_type=dict)

def test_set_suggestion_feature_config_field_headers():
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_conversation_profile.SetSuggestionFeatureConfigRequest()
    request.conversation_profile = 'conversation_profile_value'
    with mock.patch.object(type(client.transport.set_suggestion_feature_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.set_suggestion_feature_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'conversation_profile=conversation_profile_value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_suggestion_feature_config_field_headers_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_conversation_profile.SetSuggestionFeatureConfigRequest()
    request.conversation_profile = 'conversation_profile_value'
    with mock.patch.object(type(client.transport.set_suggestion_feature_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.set_suggestion_feature_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'conversation_profile=conversation_profile_value') in kw['metadata']

def test_set_suggestion_feature_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_suggestion_feature_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.set_suggestion_feature_config(conversation_profile='conversation_profile_value', participant_role=participant.Participant.Role.HUMAN_AGENT, suggestion_feature_config=gcd_conversation_profile.HumanAgentAssistantConfig.SuggestionFeatureConfig(suggestion_feature=gcd_conversation_profile.SuggestionFeature(type_=gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION)))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].conversation_profile
        mock_val = 'conversation_profile_value'
        assert arg == mock_val
        arg = args[0].participant_role
        mock_val = participant.Participant.Role.HUMAN_AGENT
        assert arg == mock_val
        arg = args[0].suggestion_feature_config
        mock_val = gcd_conversation_profile.HumanAgentAssistantConfig.SuggestionFeatureConfig(suggestion_feature=gcd_conversation_profile.SuggestionFeature(type_=gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION))
        assert arg == mock_val

def test_set_suggestion_feature_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.set_suggestion_feature_config(gcd_conversation_profile.SetSuggestionFeatureConfigRequest(), conversation_profile='conversation_profile_value', participant_role=participant.Participant.Role.HUMAN_AGENT, suggestion_feature_config=gcd_conversation_profile.HumanAgentAssistantConfig.SuggestionFeatureConfig(suggestion_feature=gcd_conversation_profile.SuggestionFeature(type_=gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION)))

@pytest.mark.asyncio
async def test_set_suggestion_feature_config_flattened_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_suggestion_feature_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.set_suggestion_feature_config(conversation_profile='conversation_profile_value', participant_role=participant.Participant.Role.HUMAN_AGENT, suggestion_feature_config=gcd_conversation_profile.HumanAgentAssistantConfig.SuggestionFeatureConfig(suggestion_feature=gcd_conversation_profile.SuggestionFeature(type_=gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION)))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].conversation_profile
        mock_val = 'conversation_profile_value'
        assert arg == mock_val
        arg = args[0].participant_role
        mock_val = participant.Participant.Role.HUMAN_AGENT
        assert arg == mock_val
        arg = args[0].suggestion_feature_config
        mock_val = gcd_conversation_profile.HumanAgentAssistantConfig.SuggestionFeatureConfig(suggestion_feature=gcd_conversation_profile.SuggestionFeature(type_=gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION))
        assert arg == mock_val

@pytest.mark.asyncio
async def test_set_suggestion_feature_config_flattened_error_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.set_suggestion_feature_config(gcd_conversation_profile.SetSuggestionFeatureConfigRequest(), conversation_profile='conversation_profile_value', participant_role=participant.Participant.Role.HUMAN_AGENT, suggestion_feature_config=gcd_conversation_profile.HumanAgentAssistantConfig.SuggestionFeatureConfig(suggestion_feature=gcd_conversation_profile.SuggestionFeature(type_=gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION)))

@pytest.mark.parametrize('request_type', [gcd_conversation_profile.ClearSuggestionFeatureConfigRequest, dict])
def test_clear_suggestion_feature_config(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.clear_suggestion_feature_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.clear_suggestion_feature_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_profile.ClearSuggestionFeatureConfigRequest()
    assert isinstance(response, future.Future)

def test_clear_suggestion_feature_config_empty_call():
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.clear_suggestion_feature_config), '__call__') as call:
        client.clear_suggestion_feature_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_profile.ClearSuggestionFeatureConfigRequest()

@pytest.mark.asyncio
async def test_clear_suggestion_feature_config_async(transport: str='grpc_asyncio', request_type=gcd_conversation_profile.ClearSuggestionFeatureConfigRequest):
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.clear_suggestion_feature_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.clear_suggestion_feature_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_profile.ClearSuggestionFeatureConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_clear_suggestion_feature_config_async_from_dict():
    await test_clear_suggestion_feature_config_async(request_type=dict)

def test_clear_suggestion_feature_config_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_conversation_profile.ClearSuggestionFeatureConfigRequest()
    request.conversation_profile = 'conversation_profile_value'
    with mock.patch.object(type(client.transport.clear_suggestion_feature_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.clear_suggestion_feature_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'conversation_profile=conversation_profile_value') in kw['metadata']

@pytest.mark.asyncio
async def test_clear_suggestion_feature_config_field_headers_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_conversation_profile.ClearSuggestionFeatureConfigRequest()
    request.conversation_profile = 'conversation_profile_value'
    with mock.patch.object(type(client.transport.clear_suggestion_feature_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.clear_suggestion_feature_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'conversation_profile=conversation_profile_value') in kw['metadata']

def test_clear_suggestion_feature_config_flattened():
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.clear_suggestion_feature_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.clear_suggestion_feature_config(conversation_profile='conversation_profile_value', participant_role=participant.Participant.Role.HUMAN_AGENT, suggestion_feature_type=gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].conversation_profile
        mock_val = 'conversation_profile_value'
        assert arg == mock_val
        arg = args[0].participant_role
        mock_val = participant.Participant.Role.HUMAN_AGENT
        assert arg == mock_val
        arg = args[0].suggestion_feature_type
        mock_val = gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION
        assert arg == mock_val

def test_clear_suggestion_feature_config_flattened_error():
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.clear_suggestion_feature_config(gcd_conversation_profile.ClearSuggestionFeatureConfigRequest(), conversation_profile='conversation_profile_value', participant_role=participant.Participant.Role.HUMAN_AGENT, suggestion_feature_type=gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION)

@pytest.mark.asyncio
async def test_clear_suggestion_feature_config_flattened_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.clear_suggestion_feature_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.clear_suggestion_feature_config(conversation_profile='conversation_profile_value', participant_role=participant.Participant.Role.HUMAN_AGENT, suggestion_feature_type=gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].conversation_profile
        mock_val = 'conversation_profile_value'
        assert arg == mock_val
        arg = args[0].participant_role
        mock_val = participant.Participant.Role.HUMAN_AGENT
        assert arg == mock_val
        arg = args[0].suggestion_feature_type
        mock_val = gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION
        assert arg == mock_val

@pytest.mark.asyncio
async def test_clear_suggestion_feature_config_flattened_error_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.clear_suggestion_feature_config(gcd_conversation_profile.ClearSuggestionFeatureConfigRequest(), conversation_profile='conversation_profile_value', participant_role=participant.Participant.Role.HUMAN_AGENT, suggestion_feature_type=gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION)

@pytest.mark.parametrize('request_type', [conversation_profile.ListConversationProfilesRequest, dict])
def test_list_conversation_profiles_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = conversation_profile.ListConversationProfilesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = conversation_profile.ListConversationProfilesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_conversation_profiles(request)
    assert isinstance(response, pagers.ListConversationProfilesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_conversation_profiles_rest_required_fields(request_type=conversation_profile.ListConversationProfilesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ConversationProfilesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_conversation_profiles._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_conversation_profiles._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = conversation_profile.ListConversationProfilesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = conversation_profile.ListConversationProfilesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_conversation_profiles(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_conversation_profiles_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ConversationProfilesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_conversation_profiles._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_conversation_profiles_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ConversationProfilesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationProfilesRestInterceptor())
    client = ConversationProfilesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConversationProfilesRestInterceptor, 'post_list_conversation_profiles') as post, mock.patch.object(transports.ConversationProfilesRestInterceptor, 'pre_list_conversation_profiles') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = conversation_profile.ListConversationProfilesRequest.pb(conversation_profile.ListConversationProfilesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = conversation_profile.ListConversationProfilesResponse.to_json(conversation_profile.ListConversationProfilesResponse())
        request = conversation_profile.ListConversationProfilesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = conversation_profile.ListConversationProfilesResponse()
        client.list_conversation_profiles(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_conversation_profiles_rest_bad_request(transport: str='rest', request_type=conversation_profile.ListConversationProfilesRequest):
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_conversation_profiles(request)

def test_list_conversation_profiles_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = conversation_profile.ListConversationProfilesResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = conversation_profile.ListConversationProfilesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_conversation_profiles(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*}/conversationProfiles' % client.transport._host, args[1])

def test_list_conversation_profiles_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_conversation_profiles(conversation_profile.ListConversationProfilesRequest(), parent='parent_value')

def test_list_conversation_profiles_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile()], next_page_token='abc'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[], next_page_token='def'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile()], next_page_token='ghi'), conversation_profile.ListConversationProfilesResponse(conversation_profiles=[conversation_profile.ConversationProfile(), conversation_profile.ConversationProfile()]))
        response = response + response
        response = tuple((conversation_profile.ListConversationProfilesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_conversation_profiles(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, conversation_profile.ConversationProfile) for i in results))
        pages = list(client.list_conversation_profiles(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [conversation_profile.GetConversationProfileRequest, dict])
def test_get_conversation_profile_rest(request_type):
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/conversationProfiles/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = conversation_profile.ConversationProfile(name='name_value', display_name='display_name_value', language_code='language_code_value', time_zone='time_zone_value', security_settings='security_settings_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = conversation_profile.ConversationProfile.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_conversation_profile(request)
    assert isinstance(response, conversation_profile.ConversationProfile)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.language_code == 'language_code_value'
    assert response.time_zone == 'time_zone_value'
    assert response.security_settings == 'security_settings_value'

def test_get_conversation_profile_rest_required_fields(request_type=conversation_profile.GetConversationProfileRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ConversationProfilesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_conversation_profile._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_conversation_profile._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = conversation_profile.ConversationProfile()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = conversation_profile.ConversationProfile.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_conversation_profile(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_conversation_profile_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ConversationProfilesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_conversation_profile._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_conversation_profile_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ConversationProfilesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationProfilesRestInterceptor())
    client = ConversationProfilesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConversationProfilesRestInterceptor, 'post_get_conversation_profile') as post, mock.patch.object(transports.ConversationProfilesRestInterceptor, 'pre_get_conversation_profile') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = conversation_profile.GetConversationProfileRequest.pb(conversation_profile.GetConversationProfileRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = conversation_profile.ConversationProfile.to_json(conversation_profile.ConversationProfile())
        request = conversation_profile.GetConversationProfileRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = conversation_profile.ConversationProfile()
        client.get_conversation_profile(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_conversation_profile_rest_bad_request(transport: str='rest', request_type=conversation_profile.GetConversationProfileRequest):
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/conversationProfiles/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_conversation_profile(request)

def test_get_conversation_profile_rest_flattened():
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = conversation_profile.ConversationProfile()
        sample_request = {'name': 'projects/sample1/conversationProfiles/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = conversation_profile.ConversationProfile.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_conversation_profile(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/conversationProfiles/*}' % client.transport._host, args[1])

def test_get_conversation_profile_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_conversation_profile(conversation_profile.GetConversationProfileRequest(), name='name_value')

def test_get_conversation_profile_rest_error():
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcd_conversation_profile.CreateConversationProfileRequest, dict])
def test_create_conversation_profile_rest(request_type):
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request_init['conversation_profile'] = {'name': 'name_value', 'display_name': 'display_name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'automated_agent_config': {'agent': 'agent_value', 'session_ttl': {'seconds': 751, 'nanos': 543}}, 'human_agent_assistant_config': {'notification_config': {'topic': 'topic_value', 'message_format': 1}, 'human_agent_suggestion_config': {'feature_configs': [{'suggestion_feature': {'type_': 1}, 'enable_event_based_suggestion': True, 'disable_agent_query_logging': True, 'suggestion_trigger_settings': {'no_smalltalk': True, 'only_end_user': True}, 'query_config': {'knowledge_base_query_source': {'knowledge_bases': ['knowledge_bases_value1', 'knowledge_bases_value2']}, 'document_query_source': {'documents': ['documents_value1', 'documents_value2']}, 'dialogflow_query_source': {'agent': 'agent_value', 'human_agent_side_config': {'agent': 'agent_value'}}, 'max_results': 1207, 'confidence_threshold': 0.2106, 'context_filter_settings': {'drop_handoff_messages': True, 'drop_virtual_agent_messages': True, 'drop_ivr_messages': True}}, 'conversation_model_config': {'model': 'model_value', 'baseline_model_version': 'baseline_model_version_value'}, 'conversation_process_config': {'recent_sentences_count': 2352}}], 'group_suggestion_responses': True}, 'end_user_suggestion_config': {}, 'message_analysis_config': {'enable_entity_extraction': True, 'enable_sentiment_analysis': True}}, 'human_agent_handoff_config': {'live_person_config': {'account_number': 'account_number_value'}, 'salesforce_live_agent_config': {'organization_id': 'organization_id_value', 'deployment_id': 'deployment_id_value', 'button_id': 'button_id_value', 'endpoint_domain': 'endpoint_domain_value'}}, 'notification_config': {}, 'logging_config': {'enable_stackdriver_logging': True}, 'new_message_event_notification_config': {}, 'stt_config': {'speech_model_variant': 1, 'model': 'model_value', 'use_timeout_based_endpointing': True}, 'language_code': 'language_code_value', 'time_zone': 'time_zone_value', 'security_settings': 'security_settings_value', 'tts_config': {'speaking_rate': 0.1373, 'pitch': 0.536, 'volume_gain_db': 0.1467, 'effects_profile_id': ['effects_profile_id_value1', 'effects_profile_id_value2'], 'voice': {'name': 'name_value', 'ssml_gender': 1}}}
    test_field = gcd_conversation_profile.CreateConversationProfileRequest.meta.fields['conversation_profile']

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
    for (field, value) in request_init['conversation_profile'].items():
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
                for i in range(0, len(request_init['conversation_profile'][field])):
                    del request_init['conversation_profile'][field][i][subfield]
            else:
                del request_init['conversation_profile'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_conversation_profile.ConversationProfile(name='name_value', display_name='display_name_value', language_code='language_code_value', time_zone='time_zone_value', security_settings='security_settings_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_conversation_profile.ConversationProfile.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_conversation_profile(request)
    assert isinstance(response, gcd_conversation_profile.ConversationProfile)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.language_code == 'language_code_value'
    assert response.time_zone == 'time_zone_value'
    assert response.security_settings == 'security_settings_value'

def test_create_conversation_profile_rest_required_fields(request_type=gcd_conversation_profile.CreateConversationProfileRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ConversationProfilesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_conversation_profile._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_conversation_profile._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcd_conversation_profile.ConversationProfile()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcd_conversation_profile.ConversationProfile.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_conversation_profile(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_conversation_profile_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ConversationProfilesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_conversation_profile._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'conversationProfile'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_conversation_profile_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ConversationProfilesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationProfilesRestInterceptor())
    client = ConversationProfilesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConversationProfilesRestInterceptor, 'post_create_conversation_profile') as post, mock.patch.object(transports.ConversationProfilesRestInterceptor, 'pre_create_conversation_profile') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcd_conversation_profile.CreateConversationProfileRequest.pb(gcd_conversation_profile.CreateConversationProfileRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcd_conversation_profile.ConversationProfile.to_json(gcd_conversation_profile.ConversationProfile())
        request = gcd_conversation_profile.CreateConversationProfileRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcd_conversation_profile.ConversationProfile()
        client.create_conversation_profile(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_conversation_profile_rest_bad_request(transport: str='rest', request_type=gcd_conversation_profile.CreateConversationProfileRequest):
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_conversation_profile(request)

def test_create_conversation_profile_rest_flattened():
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_conversation_profile.ConversationProfile()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', conversation_profile=gcd_conversation_profile.ConversationProfile(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_conversation_profile.ConversationProfile.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_conversation_profile(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*}/conversationProfiles' % client.transport._host, args[1])

def test_create_conversation_profile_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_conversation_profile(gcd_conversation_profile.CreateConversationProfileRequest(), parent='parent_value', conversation_profile=gcd_conversation_profile.ConversationProfile(name='name_value'))

def test_create_conversation_profile_rest_error():
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcd_conversation_profile.UpdateConversationProfileRequest, dict])
def test_update_conversation_profile_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'conversation_profile': {'name': 'projects/sample1/conversationProfiles/sample2'}}
    request_init['conversation_profile'] = {'name': 'projects/sample1/conversationProfiles/sample2', 'display_name': 'display_name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'automated_agent_config': {'agent': 'agent_value', 'session_ttl': {'seconds': 751, 'nanos': 543}}, 'human_agent_assistant_config': {'notification_config': {'topic': 'topic_value', 'message_format': 1}, 'human_agent_suggestion_config': {'feature_configs': [{'suggestion_feature': {'type_': 1}, 'enable_event_based_suggestion': True, 'disable_agent_query_logging': True, 'suggestion_trigger_settings': {'no_smalltalk': True, 'only_end_user': True}, 'query_config': {'knowledge_base_query_source': {'knowledge_bases': ['knowledge_bases_value1', 'knowledge_bases_value2']}, 'document_query_source': {'documents': ['documents_value1', 'documents_value2']}, 'dialogflow_query_source': {'agent': 'agent_value', 'human_agent_side_config': {'agent': 'agent_value'}}, 'max_results': 1207, 'confidence_threshold': 0.2106, 'context_filter_settings': {'drop_handoff_messages': True, 'drop_virtual_agent_messages': True, 'drop_ivr_messages': True}}, 'conversation_model_config': {'model': 'model_value', 'baseline_model_version': 'baseline_model_version_value'}, 'conversation_process_config': {'recent_sentences_count': 2352}}], 'group_suggestion_responses': True}, 'end_user_suggestion_config': {}, 'message_analysis_config': {'enable_entity_extraction': True, 'enable_sentiment_analysis': True}}, 'human_agent_handoff_config': {'live_person_config': {'account_number': 'account_number_value'}, 'salesforce_live_agent_config': {'organization_id': 'organization_id_value', 'deployment_id': 'deployment_id_value', 'button_id': 'button_id_value', 'endpoint_domain': 'endpoint_domain_value'}}, 'notification_config': {}, 'logging_config': {'enable_stackdriver_logging': True}, 'new_message_event_notification_config': {}, 'stt_config': {'speech_model_variant': 1, 'model': 'model_value', 'use_timeout_based_endpointing': True}, 'language_code': 'language_code_value', 'time_zone': 'time_zone_value', 'security_settings': 'security_settings_value', 'tts_config': {'speaking_rate': 0.1373, 'pitch': 0.536, 'volume_gain_db': 0.1467, 'effects_profile_id': ['effects_profile_id_value1', 'effects_profile_id_value2'], 'voice': {'name': 'name_value', 'ssml_gender': 1}}}
    test_field = gcd_conversation_profile.UpdateConversationProfileRequest.meta.fields['conversation_profile']

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
    for (field, value) in request_init['conversation_profile'].items():
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
                for i in range(0, len(request_init['conversation_profile'][field])):
                    del request_init['conversation_profile'][field][i][subfield]
            else:
                del request_init['conversation_profile'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_conversation_profile.ConversationProfile(name='name_value', display_name='display_name_value', language_code='language_code_value', time_zone='time_zone_value', security_settings='security_settings_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_conversation_profile.ConversationProfile.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_conversation_profile(request)
    assert isinstance(response, gcd_conversation_profile.ConversationProfile)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.language_code == 'language_code_value'
    assert response.time_zone == 'time_zone_value'
    assert response.security_settings == 'security_settings_value'

def test_update_conversation_profile_rest_required_fields(request_type=gcd_conversation_profile.UpdateConversationProfileRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ConversationProfilesRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_conversation_profile._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_conversation_profile._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcd_conversation_profile.ConversationProfile()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcd_conversation_profile.ConversationProfile.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_conversation_profile(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_conversation_profile_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ConversationProfilesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_conversation_profile._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('conversationProfile', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_conversation_profile_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ConversationProfilesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationProfilesRestInterceptor())
    client = ConversationProfilesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConversationProfilesRestInterceptor, 'post_update_conversation_profile') as post, mock.patch.object(transports.ConversationProfilesRestInterceptor, 'pre_update_conversation_profile') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcd_conversation_profile.UpdateConversationProfileRequest.pb(gcd_conversation_profile.UpdateConversationProfileRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcd_conversation_profile.ConversationProfile.to_json(gcd_conversation_profile.ConversationProfile())
        request = gcd_conversation_profile.UpdateConversationProfileRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcd_conversation_profile.ConversationProfile()
        client.update_conversation_profile(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_conversation_profile_rest_bad_request(transport: str='rest', request_type=gcd_conversation_profile.UpdateConversationProfileRequest):
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'conversation_profile': {'name': 'projects/sample1/conversationProfiles/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_conversation_profile(request)

def test_update_conversation_profile_rest_flattened():
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_conversation_profile.ConversationProfile()
        sample_request = {'conversation_profile': {'name': 'projects/sample1/conversationProfiles/sample2'}}
        mock_args = dict(conversation_profile=gcd_conversation_profile.ConversationProfile(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_conversation_profile.ConversationProfile.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_conversation_profile(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{conversation_profile.name=projects/*/conversationProfiles/*}' % client.transport._host, args[1])

def test_update_conversation_profile_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_conversation_profile(gcd_conversation_profile.UpdateConversationProfileRequest(), conversation_profile=gcd_conversation_profile.ConversationProfile(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_conversation_profile_rest_error():
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [conversation_profile.DeleteConversationProfileRequest, dict])
def test_delete_conversation_profile_rest(request_type):
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/conversationProfiles/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_conversation_profile(request)
    assert response is None

def test_delete_conversation_profile_rest_required_fields(request_type=conversation_profile.DeleteConversationProfileRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ConversationProfilesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_conversation_profile._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_conversation_profile._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_conversation_profile(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_conversation_profile_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ConversationProfilesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_conversation_profile._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_conversation_profile_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ConversationProfilesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationProfilesRestInterceptor())
    client = ConversationProfilesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConversationProfilesRestInterceptor, 'pre_delete_conversation_profile') as pre:
        pre.assert_not_called()
        pb_message = conversation_profile.DeleteConversationProfileRequest.pb(conversation_profile.DeleteConversationProfileRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = conversation_profile.DeleteConversationProfileRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_conversation_profile(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_conversation_profile_rest_bad_request(transport: str='rest', request_type=conversation_profile.DeleteConversationProfileRequest):
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/conversationProfiles/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_conversation_profile(request)

def test_delete_conversation_profile_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/conversationProfiles/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_conversation_profile(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/conversationProfiles/*}' % client.transport._host, args[1])

def test_delete_conversation_profile_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_conversation_profile(conversation_profile.DeleteConversationProfileRequest(), name='name_value')

def test_delete_conversation_profile_rest_error():
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcd_conversation_profile.SetSuggestionFeatureConfigRequest, dict])
def test_set_suggestion_feature_config_rest(request_type):
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'conversation_profile': 'projects/sample1/conversationProfiles/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_suggestion_feature_config(request)
    assert response.operation.name == 'operations/spam'

def test_set_suggestion_feature_config_rest_required_fields(request_type=gcd_conversation_profile.SetSuggestionFeatureConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ConversationProfilesRestTransport
    request_init = {}
    request_init['conversation_profile'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_suggestion_feature_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['conversationProfile'] = 'conversation_profile_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_suggestion_feature_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'conversationProfile' in jsonified_request
    assert jsonified_request['conversationProfile'] == 'conversation_profile_value'
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_suggestion_feature_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_suggestion_feature_config_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ConversationProfilesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_suggestion_feature_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('conversationProfile', 'participantRole', 'suggestionFeatureConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_suggestion_feature_config_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ConversationProfilesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationProfilesRestInterceptor())
    client = ConversationProfilesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ConversationProfilesRestInterceptor, 'post_set_suggestion_feature_config') as post, mock.patch.object(transports.ConversationProfilesRestInterceptor, 'pre_set_suggestion_feature_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcd_conversation_profile.SetSuggestionFeatureConfigRequest.pb(gcd_conversation_profile.SetSuggestionFeatureConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcd_conversation_profile.SetSuggestionFeatureConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.set_suggestion_feature_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_suggestion_feature_config_rest_bad_request(transport: str='rest', request_type=gcd_conversation_profile.SetSuggestionFeatureConfigRequest):
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'conversation_profile': 'projects/sample1/conversationProfiles/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_suggestion_feature_config(request)

def test_set_suggestion_feature_config_rest_flattened():
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'conversation_profile': 'projects/sample1/conversationProfiles/sample2'}
        mock_args = dict(conversation_profile='conversation_profile_value', participant_role=participant.Participant.Role.HUMAN_AGENT, suggestion_feature_config=gcd_conversation_profile.HumanAgentAssistantConfig.SuggestionFeatureConfig(suggestion_feature=gcd_conversation_profile.SuggestionFeature(type_=gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION)))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_suggestion_feature_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{conversation_profile=projects/*/conversationProfiles/*}:setSuggestionFeatureConfig' % client.transport._host, args[1])

def test_set_suggestion_feature_config_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_suggestion_feature_config(gcd_conversation_profile.SetSuggestionFeatureConfigRequest(), conversation_profile='conversation_profile_value', participant_role=participant.Participant.Role.HUMAN_AGENT, suggestion_feature_config=gcd_conversation_profile.HumanAgentAssistantConfig.SuggestionFeatureConfig(suggestion_feature=gcd_conversation_profile.SuggestionFeature(type_=gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION)))

def test_set_suggestion_feature_config_rest_error():
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcd_conversation_profile.ClearSuggestionFeatureConfigRequest, dict])
def test_clear_suggestion_feature_config_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'conversation_profile': 'projects/sample1/conversationProfiles/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.clear_suggestion_feature_config(request)
    assert response.operation.name == 'operations/spam'

def test_clear_suggestion_feature_config_rest_required_fields(request_type=gcd_conversation_profile.ClearSuggestionFeatureConfigRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ConversationProfilesRestTransport
    request_init = {}
    request_init['conversation_profile'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).clear_suggestion_feature_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['conversationProfile'] = 'conversation_profile_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).clear_suggestion_feature_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'conversationProfile' in jsonified_request
    assert jsonified_request['conversationProfile'] == 'conversation_profile_value'
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.clear_suggestion_feature_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_clear_suggestion_feature_config_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ConversationProfilesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.clear_suggestion_feature_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('conversationProfile', 'participantRole', 'suggestionFeatureType'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_clear_suggestion_feature_config_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ConversationProfilesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationProfilesRestInterceptor())
    client = ConversationProfilesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ConversationProfilesRestInterceptor, 'post_clear_suggestion_feature_config') as post, mock.patch.object(transports.ConversationProfilesRestInterceptor, 'pre_clear_suggestion_feature_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcd_conversation_profile.ClearSuggestionFeatureConfigRequest.pb(gcd_conversation_profile.ClearSuggestionFeatureConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcd_conversation_profile.ClearSuggestionFeatureConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.clear_suggestion_feature_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_clear_suggestion_feature_config_rest_bad_request(transport: str='rest', request_type=gcd_conversation_profile.ClearSuggestionFeatureConfigRequest):
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'conversation_profile': 'projects/sample1/conversationProfiles/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.clear_suggestion_feature_config(request)

def test_clear_suggestion_feature_config_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'conversation_profile': 'projects/sample1/conversationProfiles/sample2'}
        mock_args = dict(conversation_profile='conversation_profile_value', participant_role=participant.Participant.Role.HUMAN_AGENT, suggestion_feature_type=gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.clear_suggestion_feature_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{conversation_profile=projects/*/conversationProfiles/*}:clearSuggestionFeatureConfig' % client.transport._host, args[1])

def test_clear_suggestion_feature_config_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.clear_suggestion_feature_config(gcd_conversation_profile.ClearSuggestionFeatureConfigRequest(), conversation_profile='conversation_profile_value', participant_role=participant.Participant.Role.HUMAN_AGENT, suggestion_feature_type=gcd_conversation_profile.SuggestionFeature.Type.ARTICLE_SUGGESTION)

def test_clear_suggestion_feature_config_rest_error():
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.ConversationProfilesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ConversationProfilesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ConversationProfilesClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ConversationProfilesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ConversationProfilesClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ConversationProfilesClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ConversationProfilesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ConversationProfilesClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.ConversationProfilesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ConversationProfilesClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.ConversationProfilesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ConversationProfilesGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ConversationProfilesGrpcTransport, transports.ConversationProfilesGrpcAsyncIOTransport, transports.ConversationProfilesRestTransport])
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
        i = 10
        return i + 15
    transport = ConversationProfilesClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ConversationProfilesGrpcTransport)

def test_conversation_profiles_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ConversationProfilesTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_conversation_profiles_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.dialogflow_v2.services.conversation_profiles.transports.ConversationProfilesTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ConversationProfilesTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_conversation_profiles', 'get_conversation_profile', 'create_conversation_profile', 'update_conversation_profile', 'delete_conversation_profile', 'set_suggestion_feature_config', 'clear_suggestion_feature_config', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'list_operations')
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

def test_conversation_profiles_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dialogflow_v2.services.conversation_profiles.transports.ConversationProfilesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ConversationProfilesTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id='octopus')

def test_conversation_profiles_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dialogflow_v2.services.conversation_profiles.transports.ConversationProfilesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ConversationProfilesTransport()
        adc.assert_called_once()

def test_conversation_profiles_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ConversationProfilesClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ConversationProfilesGrpcTransport, transports.ConversationProfilesGrpcAsyncIOTransport])
def test_conversation_profiles_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ConversationProfilesGrpcTransport, transports.ConversationProfilesGrpcAsyncIOTransport, transports.ConversationProfilesRestTransport])
def test_conversation_profiles_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ConversationProfilesGrpcTransport, grpc_helpers), (transports.ConversationProfilesGrpcAsyncIOTransport, grpc_helpers_async)])
def test_conversation_profiles_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dialogflow.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), scopes=['1', '2'], default_host='dialogflow.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ConversationProfilesGrpcTransport, transports.ConversationProfilesGrpcAsyncIOTransport])
def test_conversation_profiles_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_conversation_profiles_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.ConversationProfilesRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_conversation_profiles_rest_lro_client():
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_conversation_profiles_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dialogflow.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_conversation_profiles_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dialogflow.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('dialogflow.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_conversation_profiles_client_transport_session_collision(transport_name):
    if False:
        while True:
            i = 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = ConversationProfilesClient(credentials=creds1, transport=transport_name)
    client2 = ConversationProfilesClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_conversation_profiles._session
    session2 = client2.transport.list_conversation_profiles._session
    assert session1 != session2
    session1 = client1.transport.get_conversation_profile._session
    session2 = client2.transport.get_conversation_profile._session
    assert session1 != session2
    session1 = client1.transport.create_conversation_profile._session
    session2 = client2.transport.create_conversation_profile._session
    assert session1 != session2
    session1 = client1.transport.update_conversation_profile._session
    session2 = client2.transport.update_conversation_profile._session
    assert session1 != session2
    session1 = client1.transport.delete_conversation_profile._session
    session2 = client2.transport.delete_conversation_profile._session
    assert session1 != session2
    session1 = client1.transport.set_suggestion_feature_config._session
    session2 = client2.transport.set_suggestion_feature_config._session
    assert session1 != session2
    session1 = client1.transport.clear_suggestion_feature_config._session
    session2 = client2.transport.clear_suggestion_feature_config._session
    assert session1 != session2

def test_conversation_profiles_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ConversationProfilesGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_conversation_profiles_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ConversationProfilesGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ConversationProfilesGrpcTransport, transports.ConversationProfilesGrpcAsyncIOTransport])
def test_conversation_profiles_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.ConversationProfilesGrpcTransport, transports.ConversationProfilesGrpcAsyncIOTransport])
def test_conversation_profiles_transport_channel_mtls_with_adc(transport_class):
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

def test_conversation_profiles_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_conversation_profiles_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_agent_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    expected = 'projects/{project}/agent'.format(project=project)
    actual = ConversationProfilesClient.agent_path(project)
    assert expected == actual

def test_parse_agent_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'clam'}
    path = ConversationProfilesClient.agent_path(**expected)
    actual = ConversationProfilesClient.parse_agent_path(path)
    assert expected == actual

def test_conversation_model_path():
    if False:
        return 10
    project = 'whelk'
    location = 'octopus'
    conversation_model = 'oyster'
    expected = 'projects/{project}/locations/{location}/conversationModels/{conversation_model}'.format(project=project, location=location, conversation_model=conversation_model)
    actual = ConversationProfilesClient.conversation_model_path(project, location, conversation_model)
    assert expected == actual

def test_parse_conversation_model_path():
    if False:
        return 10
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'conversation_model': 'mussel'}
    path = ConversationProfilesClient.conversation_model_path(**expected)
    actual = ConversationProfilesClient.parse_conversation_model_path(path)
    assert expected == actual

def test_conversation_profile_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    conversation_profile = 'nautilus'
    expected = 'projects/{project}/conversationProfiles/{conversation_profile}'.format(project=project, conversation_profile=conversation_profile)
    actual = ConversationProfilesClient.conversation_profile_path(project, conversation_profile)
    assert expected == actual

def test_parse_conversation_profile_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'scallop', 'conversation_profile': 'abalone'}
    path = ConversationProfilesClient.conversation_profile_path(**expected)
    actual = ConversationProfilesClient.parse_conversation_profile_path(path)
    assert expected == actual

def test_cx_security_settings_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    security_settings = 'whelk'
    expected = 'projects/{project}/locations/{location}/securitySettings/{security_settings}'.format(project=project, location=location, security_settings=security_settings)
    actual = ConversationProfilesClient.cx_security_settings_path(project, location, security_settings)
    assert expected == actual

def test_parse_cx_security_settings_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'location': 'oyster', 'security_settings': 'nudibranch'}
    path = ConversationProfilesClient.cx_security_settings_path(**expected)
    actual = ConversationProfilesClient.parse_cx_security_settings_path(path)
    assert expected == actual

def test_document_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    knowledge_base = 'mussel'
    document = 'winkle'
    expected = 'projects/{project}/knowledgeBases/{knowledge_base}/documents/{document}'.format(project=project, knowledge_base=knowledge_base, document=document)
    actual = ConversationProfilesClient.document_path(project, knowledge_base, document)
    assert expected == actual

def test_parse_document_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nautilus', 'knowledge_base': 'scallop', 'document': 'abalone'}
    path = ConversationProfilesClient.document_path(**expected)
    actual = ConversationProfilesClient.parse_document_path(path)
    assert expected == actual

def test_knowledge_base_path():
    if False:
        return 10
    project = 'squid'
    knowledge_base = 'clam'
    expected = 'projects/{project}/knowledgeBases/{knowledge_base}'.format(project=project, knowledge_base=knowledge_base)
    actual = ConversationProfilesClient.knowledge_base_path(project, knowledge_base)
    assert expected == actual

def test_parse_knowledge_base_path():
    if False:
        return 10
    expected = {'project': 'whelk', 'knowledge_base': 'octopus'}
    path = ConversationProfilesClient.knowledge_base_path(**expected)
    actual = ConversationProfilesClient.parse_knowledge_base_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ConversationProfilesClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'nudibranch'}
    path = ConversationProfilesClient.common_billing_account_path(**expected)
    actual = ConversationProfilesClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ConversationProfilesClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'mussel'}
    path = ConversationProfilesClient.common_folder_path(**expected)
    actual = ConversationProfilesClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ConversationProfilesClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'nautilus'}
    path = ConversationProfilesClient.common_organization_path(**expected)
    actual = ConversationProfilesClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = ConversationProfilesClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'abalone'}
    path = ConversationProfilesClient.common_project_path(**expected)
    actual = ConversationProfilesClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ConversationProfilesClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = ConversationProfilesClient.common_location_path(**expected)
    actual = ConversationProfilesClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ConversationProfilesTransport, '_prep_wrapped_messages') as prep:
        client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ConversationProfilesTransport, '_prep_wrapped_messages') as prep:
        transport_class = ConversationProfilesClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = ConversationProfilesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        i = 10
        return i + 15
    transports = ['rest', 'grpc']
    for transport in transports:
        client = ConversationProfilesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ConversationProfilesClient, transports.ConversationProfilesGrpcTransport), (ConversationProfilesAsyncClient, transports.ConversationProfilesGrpcAsyncIOTransport)])
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