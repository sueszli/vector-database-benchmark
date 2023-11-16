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
from google.cloud.contact_center_insights_v1.services.contact_center_insights import ContactCenterInsightsAsyncClient, ContactCenterInsightsClient, pagers, transports
from google.cloud.contact_center_insights_v1.types import contact_center_insights, resources

def client_cert_source_callback():
    if False:
        print('Hello World!')
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
    assert ContactCenterInsightsClient._get_default_mtls_endpoint(None) is None
    assert ContactCenterInsightsClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ContactCenterInsightsClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ContactCenterInsightsClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ContactCenterInsightsClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ContactCenterInsightsClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ContactCenterInsightsClient, 'grpc'), (ContactCenterInsightsAsyncClient, 'grpc_asyncio'), (ContactCenterInsightsClient, 'rest')])
def test_contact_center_insights_client_from_service_account_info(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('contactcenterinsights.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://contactcenterinsights.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ContactCenterInsightsGrpcTransport, 'grpc'), (transports.ContactCenterInsightsGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.ContactCenterInsightsRestTransport, 'rest')])
def test_contact_center_insights_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ContactCenterInsightsClient, 'grpc'), (ContactCenterInsightsAsyncClient, 'grpc_asyncio'), (ContactCenterInsightsClient, 'rest')])
def test_contact_center_insights_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('contactcenterinsights.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://contactcenterinsights.googleapis.com')

def test_contact_center_insights_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = ContactCenterInsightsClient.get_transport_class()
    available_transports = [transports.ContactCenterInsightsGrpcTransport, transports.ContactCenterInsightsRestTransport]
    assert transport in available_transports
    transport = ContactCenterInsightsClient.get_transport_class('grpc')
    assert transport == transports.ContactCenterInsightsGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ContactCenterInsightsClient, transports.ContactCenterInsightsGrpcTransport, 'grpc'), (ContactCenterInsightsAsyncClient, transports.ContactCenterInsightsGrpcAsyncIOTransport, 'grpc_asyncio'), (ContactCenterInsightsClient, transports.ContactCenterInsightsRestTransport, 'rest')])
@mock.patch.object(ContactCenterInsightsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ContactCenterInsightsClient))
@mock.patch.object(ContactCenterInsightsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ContactCenterInsightsAsyncClient))
def test_contact_center_insights_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(ContactCenterInsightsClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ContactCenterInsightsClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ContactCenterInsightsClient, transports.ContactCenterInsightsGrpcTransport, 'grpc', 'true'), (ContactCenterInsightsAsyncClient, transports.ContactCenterInsightsGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ContactCenterInsightsClient, transports.ContactCenterInsightsGrpcTransport, 'grpc', 'false'), (ContactCenterInsightsAsyncClient, transports.ContactCenterInsightsGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (ContactCenterInsightsClient, transports.ContactCenterInsightsRestTransport, 'rest', 'true'), (ContactCenterInsightsClient, transports.ContactCenterInsightsRestTransport, 'rest', 'false')])
@mock.patch.object(ContactCenterInsightsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ContactCenterInsightsClient))
@mock.patch.object(ContactCenterInsightsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ContactCenterInsightsAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_contact_center_insights_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [ContactCenterInsightsClient, ContactCenterInsightsAsyncClient])
@mock.patch.object(ContactCenterInsightsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ContactCenterInsightsClient))
@mock.patch.object(ContactCenterInsightsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ContactCenterInsightsAsyncClient))
def test_contact_center_insights_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ContactCenterInsightsClient, transports.ContactCenterInsightsGrpcTransport, 'grpc'), (ContactCenterInsightsAsyncClient, transports.ContactCenterInsightsGrpcAsyncIOTransport, 'grpc_asyncio'), (ContactCenterInsightsClient, transports.ContactCenterInsightsRestTransport, 'rest')])
def test_contact_center_insights_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ContactCenterInsightsClient, transports.ContactCenterInsightsGrpcTransport, 'grpc', grpc_helpers), (ContactCenterInsightsAsyncClient, transports.ContactCenterInsightsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (ContactCenterInsightsClient, transports.ContactCenterInsightsRestTransport, 'rest', None)])
def test_contact_center_insights_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_contact_center_insights_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.contact_center_insights_v1.services.contact_center_insights.transports.ContactCenterInsightsGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ContactCenterInsightsClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ContactCenterInsightsClient, transports.ContactCenterInsightsGrpcTransport, 'grpc', grpc_helpers), (ContactCenterInsightsAsyncClient, transports.ContactCenterInsightsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_contact_center_insights_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('contactcenterinsights.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='contactcenterinsights.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [contact_center_insights.CreateConversationRequest, dict])
def test_create_conversation(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_conversation), '__call__') as call:
        call.return_value = resources.Conversation(name='name_value', language_code='language_code_value', agent_id='agent_id_value', medium=resources.Conversation.Medium.PHONE_CALL, turn_count=1105, obfuscated_user_id='obfuscated_user_id_value')
        response = client.create_conversation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreateConversationRequest()
    assert isinstance(response, resources.Conversation)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'
    assert response.agent_id == 'agent_id_value'
    assert response.medium == resources.Conversation.Medium.PHONE_CALL
    assert response.turn_count == 1105
    assert response.obfuscated_user_id == 'obfuscated_user_id_value'

def test_create_conversation_empty_call():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_conversation), '__call__') as call:
        client.create_conversation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreateConversationRequest()

@pytest.mark.asyncio
async def test_create_conversation_async(transport: str='grpc_asyncio', request_type=contact_center_insights.CreateConversationRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_conversation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Conversation(name='name_value', language_code='language_code_value', agent_id='agent_id_value', medium=resources.Conversation.Medium.PHONE_CALL, turn_count=1105, obfuscated_user_id='obfuscated_user_id_value'))
        response = await client.create_conversation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreateConversationRequest()
    assert isinstance(response, resources.Conversation)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'
    assert response.agent_id == 'agent_id_value'
    assert response.medium == resources.Conversation.Medium.PHONE_CALL
    assert response.turn_count == 1105
    assert response.obfuscated_user_id == 'obfuscated_user_id_value'

@pytest.mark.asyncio
async def test_create_conversation_async_from_dict():
    await test_create_conversation_async(request_type=dict)

def test_create_conversation_field_headers():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.CreateConversationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_conversation), '__call__') as call:
        call.return_value = resources.Conversation()
        client.create_conversation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_conversation_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.CreateConversationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_conversation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Conversation())
        await client.create_conversation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_conversation_flattened():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_conversation), '__call__') as call:
        call.return_value = resources.Conversation()
        client.create_conversation(parent='parent_value', conversation=resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706)), conversation_id='conversation_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].conversation
        mock_val = resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706))
        assert arg == mock_val
        arg = args[0].conversation_id
        mock_val = 'conversation_id_value'
        assert arg == mock_val

def test_create_conversation_flattened_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_conversation(contact_center_insights.CreateConversationRequest(), parent='parent_value', conversation=resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706)), conversation_id='conversation_id_value')

@pytest.mark.asyncio
async def test_create_conversation_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_conversation), '__call__') as call:
        call.return_value = resources.Conversation()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Conversation())
        response = await client.create_conversation(parent='parent_value', conversation=resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706)), conversation_id='conversation_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].conversation
        mock_val = resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706))
        assert arg == mock_val
        arg = args[0].conversation_id
        mock_val = 'conversation_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_conversation_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_conversation(contact_center_insights.CreateConversationRequest(), parent='parent_value', conversation=resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706)), conversation_id='conversation_id_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.UploadConversationRequest, dict])
def test_upload_conversation(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.upload_conversation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.upload_conversation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UploadConversationRequest()
    assert isinstance(response, future.Future)

def test_upload_conversation_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.upload_conversation), '__call__') as call:
        client.upload_conversation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UploadConversationRequest()

@pytest.mark.asyncio
async def test_upload_conversation_async(transport: str='grpc_asyncio', request_type=contact_center_insights.UploadConversationRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.upload_conversation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.upload_conversation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UploadConversationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_upload_conversation_async_from_dict():
    await test_upload_conversation_async(request_type=dict)

def test_upload_conversation_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UploadConversationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.upload_conversation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.upload_conversation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_upload_conversation_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UploadConversationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.upload_conversation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.upload_conversation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [contact_center_insights.UpdateConversationRequest, dict])
def test_update_conversation(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_conversation), '__call__') as call:
        call.return_value = resources.Conversation(name='name_value', language_code='language_code_value', agent_id='agent_id_value', medium=resources.Conversation.Medium.PHONE_CALL, turn_count=1105, obfuscated_user_id='obfuscated_user_id_value')
        response = client.update_conversation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateConversationRequest()
    assert isinstance(response, resources.Conversation)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'
    assert response.agent_id == 'agent_id_value'
    assert response.medium == resources.Conversation.Medium.PHONE_CALL
    assert response.turn_count == 1105
    assert response.obfuscated_user_id == 'obfuscated_user_id_value'

def test_update_conversation_empty_call():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_conversation), '__call__') as call:
        client.update_conversation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateConversationRequest()

@pytest.mark.asyncio
async def test_update_conversation_async(transport: str='grpc_asyncio', request_type=contact_center_insights.UpdateConversationRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_conversation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Conversation(name='name_value', language_code='language_code_value', agent_id='agent_id_value', medium=resources.Conversation.Medium.PHONE_CALL, turn_count=1105, obfuscated_user_id='obfuscated_user_id_value'))
        response = await client.update_conversation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateConversationRequest()
    assert isinstance(response, resources.Conversation)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'
    assert response.agent_id == 'agent_id_value'
    assert response.medium == resources.Conversation.Medium.PHONE_CALL
    assert response.turn_count == 1105
    assert response.obfuscated_user_id == 'obfuscated_user_id_value'

@pytest.mark.asyncio
async def test_update_conversation_async_from_dict():
    await test_update_conversation_async(request_type=dict)

def test_update_conversation_field_headers():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UpdateConversationRequest()
    request.conversation.name = 'name_value'
    with mock.patch.object(type(client.transport.update_conversation), '__call__') as call:
        call.return_value = resources.Conversation()
        client.update_conversation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'conversation.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_conversation_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UpdateConversationRequest()
    request.conversation.name = 'name_value'
    with mock.patch.object(type(client.transport.update_conversation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Conversation())
        await client.update_conversation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'conversation.name=name_value') in kw['metadata']

def test_update_conversation_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_conversation), '__call__') as call:
        call.return_value = resources.Conversation()
        client.update_conversation(conversation=resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706)), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].conversation
        mock_val = resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_conversation_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_conversation(contact_center_insights.UpdateConversationRequest(), conversation=resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706)), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_conversation_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_conversation), '__call__') as call:
        call.return_value = resources.Conversation()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Conversation())
        response = await client.update_conversation(conversation=resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706)), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].conversation
        mock_val = resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_conversation_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_conversation(contact_center_insights.UpdateConversationRequest(), conversation=resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706)), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [contact_center_insights.GetConversationRequest, dict])
def test_get_conversation(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_conversation), '__call__') as call:
        call.return_value = resources.Conversation(name='name_value', language_code='language_code_value', agent_id='agent_id_value', medium=resources.Conversation.Medium.PHONE_CALL, turn_count=1105, obfuscated_user_id='obfuscated_user_id_value')
        response = client.get_conversation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetConversationRequest()
    assert isinstance(response, resources.Conversation)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'
    assert response.agent_id == 'agent_id_value'
    assert response.medium == resources.Conversation.Medium.PHONE_CALL
    assert response.turn_count == 1105
    assert response.obfuscated_user_id == 'obfuscated_user_id_value'

def test_get_conversation_empty_call():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_conversation), '__call__') as call:
        client.get_conversation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetConversationRequest()

@pytest.mark.asyncio
async def test_get_conversation_async(transport: str='grpc_asyncio', request_type=contact_center_insights.GetConversationRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_conversation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Conversation(name='name_value', language_code='language_code_value', agent_id='agent_id_value', medium=resources.Conversation.Medium.PHONE_CALL, turn_count=1105, obfuscated_user_id='obfuscated_user_id_value'))
        response = await client.get_conversation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetConversationRequest()
    assert isinstance(response, resources.Conversation)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'
    assert response.agent_id == 'agent_id_value'
    assert response.medium == resources.Conversation.Medium.PHONE_CALL
    assert response.turn_count == 1105
    assert response.obfuscated_user_id == 'obfuscated_user_id_value'

@pytest.mark.asyncio
async def test_get_conversation_async_from_dict():
    await test_get_conversation_async(request_type=dict)

def test_get_conversation_field_headers():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.GetConversationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_conversation), '__call__') as call:
        call.return_value = resources.Conversation()
        client.get_conversation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_conversation_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.GetConversationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_conversation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Conversation())
        await client.get_conversation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_conversation_flattened():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_conversation), '__call__') as call:
        call.return_value = resources.Conversation()
        client.get_conversation(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_conversation_flattened_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_conversation(contact_center_insights.GetConversationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_conversation_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_conversation), '__call__') as call:
        call.return_value = resources.Conversation()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Conversation())
        response = await client.get_conversation(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_conversation_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_conversation(contact_center_insights.GetConversationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.ListConversationsRequest, dict])
def test_list_conversations(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_conversations), '__call__') as call:
        call.return_value = contact_center_insights.ListConversationsResponse(next_page_token='next_page_token_value')
        response = client.list_conversations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListConversationsRequest()
    assert isinstance(response, pagers.ListConversationsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_conversations_empty_call():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_conversations), '__call__') as call:
        client.list_conversations()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListConversationsRequest()

@pytest.mark.asyncio
async def test_list_conversations_async(transport: str='grpc_asyncio', request_type=contact_center_insights.ListConversationsRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_conversations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListConversationsResponse(next_page_token='next_page_token_value'))
        response = await client.list_conversations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListConversationsRequest()
    assert isinstance(response, pagers.ListConversationsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_conversations_async_from_dict():
    await test_list_conversations_async(request_type=dict)

def test_list_conversations_field_headers():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.ListConversationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_conversations), '__call__') as call:
        call.return_value = contact_center_insights.ListConversationsResponse()
        client.list_conversations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_conversations_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.ListConversationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_conversations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListConversationsResponse())
        await client.list_conversations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_conversations_flattened():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_conversations), '__call__') as call:
        call.return_value = contact_center_insights.ListConversationsResponse()
        client.list_conversations(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_conversations_flattened_error():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_conversations(contact_center_insights.ListConversationsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_conversations_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_conversations), '__call__') as call:
        call.return_value = contact_center_insights.ListConversationsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListConversationsResponse())
        response = await client.list_conversations(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_conversations_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_conversations(contact_center_insights.ListConversationsRequest(), parent='parent_value')

def test_list_conversations_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_conversations), '__call__') as call:
        call.side_effect = (contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation(), resources.Conversation(), resources.Conversation()], next_page_token='abc'), contact_center_insights.ListConversationsResponse(conversations=[], next_page_token='def'), contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation()], next_page_token='ghi'), contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation(), resources.Conversation()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_conversations(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Conversation) for i in results))

def test_list_conversations_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_conversations), '__call__') as call:
        call.side_effect = (contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation(), resources.Conversation(), resources.Conversation()], next_page_token='abc'), contact_center_insights.ListConversationsResponse(conversations=[], next_page_token='def'), contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation()], next_page_token='ghi'), contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation(), resources.Conversation()]), RuntimeError)
        pages = list(client.list_conversations(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_conversations_async_pager():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_conversations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation(), resources.Conversation(), resources.Conversation()], next_page_token='abc'), contact_center_insights.ListConversationsResponse(conversations=[], next_page_token='def'), contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation()], next_page_token='ghi'), contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation(), resources.Conversation()]), RuntimeError)
        async_pager = await client.list_conversations(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Conversation) for i in responses))

@pytest.mark.asyncio
async def test_list_conversations_async_pages():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_conversations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation(), resources.Conversation(), resources.Conversation()], next_page_token='abc'), contact_center_insights.ListConversationsResponse(conversations=[], next_page_token='def'), contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation()], next_page_token='ghi'), contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation(), resources.Conversation()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_conversations(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [contact_center_insights.DeleteConversationRequest, dict])
def test_delete_conversation(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_conversation), '__call__') as call:
        call.return_value = None
        response = client.delete_conversation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteConversationRequest()
    assert response is None

def test_delete_conversation_empty_call():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_conversation), '__call__') as call:
        client.delete_conversation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteConversationRequest()

@pytest.mark.asyncio
async def test_delete_conversation_async(transport: str='grpc_asyncio', request_type=contact_center_insights.DeleteConversationRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_conversation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_conversation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteConversationRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_conversation_async_from_dict():
    await test_delete_conversation_async(request_type=dict)

def test_delete_conversation_field_headers():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.DeleteConversationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_conversation), '__call__') as call:
        call.return_value = None
        client.delete_conversation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_conversation_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.DeleteConversationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_conversation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_conversation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_conversation_flattened():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_conversation), '__call__') as call:
        call.return_value = None
        client.delete_conversation(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_conversation_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_conversation(contact_center_insights.DeleteConversationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_conversation_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_conversation), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_conversation(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_conversation_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_conversation(contact_center_insights.DeleteConversationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.CreateAnalysisRequest, dict])
def test_create_analysis(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_analysis), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_analysis(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreateAnalysisRequest()
    assert isinstance(response, future.Future)

def test_create_analysis_empty_call():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_analysis), '__call__') as call:
        client.create_analysis()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreateAnalysisRequest()

@pytest.mark.asyncio
async def test_create_analysis_async(transport: str='grpc_asyncio', request_type=contact_center_insights.CreateAnalysisRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_analysis), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_analysis(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreateAnalysisRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_analysis_async_from_dict():
    await test_create_analysis_async(request_type=dict)

def test_create_analysis_field_headers():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.CreateAnalysisRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_analysis), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_analysis(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_analysis_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.CreateAnalysisRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_analysis), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_analysis(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_analysis_flattened():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_analysis), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_analysis(parent='parent_value', analysis=resources.Analysis(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].analysis
        mock_val = resources.Analysis(name='name_value')
        assert arg == mock_val

def test_create_analysis_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_analysis(contact_center_insights.CreateAnalysisRequest(), parent='parent_value', analysis=resources.Analysis(name='name_value'))

@pytest.mark.asyncio
async def test_create_analysis_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_analysis), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_analysis(parent='parent_value', analysis=resources.Analysis(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].analysis
        mock_val = resources.Analysis(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_analysis_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_analysis(contact_center_insights.CreateAnalysisRequest(), parent='parent_value', analysis=resources.Analysis(name='name_value'))

@pytest.mark.parametrize('request_type', [contact_center_insights.GetAnalysisRequest, dict])
def test_get_analysis(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_analysis), '__call__') as call:
        call.return_value = resources.Analysis(name='name_value')
        response = client.get_analysis(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetAnalysisRequest()
    assert isinstance(response, resources.Analysis)
    assert response.name == 'name_value'

def test_get_analysis_empty_call():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_analysis), '__call__') as call:
        client.get_analysis()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetAnalysisRequest()

@pytest.mark.asyncio
async def test_get_analysis_async(transport: str='grpc_asyncio', request_type=contact_center_insights.GetAnalysisRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_analysis), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Analysis(name='name_value'))
        response = await client.get_analysis(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetAnalysisRequest()
    assert isinstance(response, resources.Analysis)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_analysis_async_from_dict():
    await test_get_analysis_async(request_type=dict)

def test_get_analysis_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.GetAnalysisRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_analysis), '__call__') as call:
        call.return_value = resources.Analysis()
        client.get_analysis(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_analysis_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.GetAnalysisRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_analysis), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Analysis())
        await client.get_analysis(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_analysis_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_analysis), '__call__') as call:
        call.return_value = resources.Analysis()
        client.get_analysis(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_analysis_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_analysis(contact_center_insights.GetAnalysisRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_analysis_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_analysis), '__call__') as call:
        call.return_value = resources.Analysis()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Analysis())
        response = await client.get_analysis(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_analysis_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_analysis(contact_center_insights.GetAnalysisRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.ListAnalysesRequest, dict])
def test_list_analyses(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_analyses), '__call__') as call:
        call.return_value = contact_center_insights.ListAnalysesResponse(next_page_token='next_page_token_value')
        response = client.list_analyses(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListAnalysesRequest()
    assert isinstance(response, pagers.ListAnalysesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_analyses_empty_call():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_analyses), '__call__') as call:
        client.list_analyses()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListAnalysesRequest()

@pytest.mark.asyncio
async def test_list_analyses_async(transport: str='grpc_asyncio', request_type=contact_center_insights.ListAnalysesRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_analyses), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListAnalysesResponse(next_page_token='next_page_token_value'))
        response = await client.list_analyses(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListAnalysesRequest()
    assert isinstance(response, pagers.ListAnalysesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_analyses_async_from_dict():
    await test_list_analyses_async(request_type=dict)

def test_list_analyses_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.ListAnalysesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_analyses), '__call__') as call:
        call.return_value = contact_center_insights.ListAnalysesResponse()
        client.list_analyses(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_analyses_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.ListAnalysesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_analyses), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListAnalysesResponse())
        await client.list_analyses(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_analyses_flattened():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_analyses), '__call__') as call:
        call.return_value = contact_center_insights.ListAnalysesResponse()
        client.list_analyses(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_analyses_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_analyses(contact_center_insights.ListAnalysesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_analyses_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_analyses), '__call__') as call:
        call.return_value = contact_center_insights.ListAnalysesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListAnalysesResponse())
        response = await client.list_analyses(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_analyses_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_analyses(contact_center_insights.ListAnalysesRequest(), parent='parent_value')

def test_list_analyses_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_analyses), '__call__') as call:
        call.side_effect = (contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis(), resources.Analysis(), resources.Analysis()], next_page_token='abc'), contact_center_insights.ListAnalysesResponse(analyses=[], next_page_token='def'), contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis()], next_page_token='ghi'), contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis(), resources.Analysis()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_analyses(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Analysis) for i in results))

def test_list_analyses_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_analyses), '__call__') as call:
        call.side_effect = (contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis(), resources.Analysis(), resources.Analysis()], next_page_token='abc'), contact_center_insights.ListAnalysesResponse(analyses=[], next_page_token='def'), contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis()], next_page_token='ghi'), contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis(), resources.Analysis()]), RuntimeError)
        pages = list(client.list_analyses(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_analyses_async_pager():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_analyses), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis(), resources.Analysis(), resources.Analysis()], next_page_token='abc'), contact_center_insights.ListAnalysesResponse(analyses=[], next_page_token='def'), contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis()], next_page_token='ghi'), contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis(), resources.Analysis()]), RuntimeError)
        async_pager = await client.list_analyses(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Analysis) for i in responses))

@pytest.mark.asyncio
async def test_list_analyses_async_pages():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_analyses), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis(), resources.Analysis(), resources.Analysis()], next_page_token='abc'), contact_center_insights.ListAnalysesResponse(analyses=[], next_page_token='def'), contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis()], next_page_token='ghi'), contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis(), resources.Analysis()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_analyses(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [contact_center_insights.DeleteAnalysisRequest, dict])
def test_delete_analysis(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_analysis), '__call__') as call:
        call.return_value = None
        response = client.delete_analysis(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteAnalysisRequest()
    assert response is None

def test_delete_analysis_empty_call():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_analysis), '__call__') as call:
        client.delete_analysis()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteAnalysisRequest()

@pytest.mark.asyncio
async def test_delete_analysis_async(transport: str='grpc_asyncio', request_type=contact_center_insights.DeleteAnalysisRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_analysis), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_analysis(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteAnalysisRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_analysis_async_from_dict():
    await test_delete_analysis_async(request_type=dict)

def test_delete_analysis_field_headers():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.DeleteAnalysisRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_analysis), '__call__') as call:
        call.return_value = None
        client.delete_analysis(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_analysis_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.DeleteAnalysisRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_analysis), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_analysis(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_analysis_flattened():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_analysis), '__call__') as call:
        call.return_value = None
        client.delete_analysis(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_analysis_flattened_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_analysis(contact_center_insights.DeleteAnalysisRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_analysis_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_analysis), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_analysis(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_analysis_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_analysis(contact_center_insights.DeleteAnalysisRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.BulkAnalyzeConversationsRequest, dict])
def test_bulk_analyze_conversations(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.bulk_analyze_conversations), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.bulk_analyze_conversations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.BulkAnalyzeConversationsRequest()
    assert isinstance(response, future.Future)

def test_bulk_analyze_conversations_empty_call():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.bulk_analyze_conversations), '__call__') as call:
        client.bulk_analyze_conversations()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.BulkAnalyzeConversationsRequest()

@pytest.mark.asyncio
async def test_bulk_analyze_conversations_async(transport: str='grpc_asyncio', request_type=contact_center_insights.BulkAnalyzeConversationsRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.bulk_analyze_conversations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.bulk_analyze_conversations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.BulkAnalyzeConversationsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_bulk_analyze_conversations_async_from_dict():
    await test_bulk_analyze_conversations_async(request_type=dict)

def test_bulk_analyze_conversations_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.BulkAnalyzeConversationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.bulk_analyze_conversations), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.bulk_analyze_conversations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_bulk_analyze_conversations_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.BulkAnalyzeConversationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.bulk_analyze_conversations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.bulk_analyze_conversations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_bulk_analyze_conversations_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.bulk_analyze_conversations), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.bulk_analyze_conversations(parent='parent_value', filter='filter_value', analysis_percentage=0.20170000000000002)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val
        assert math.isclose(args[0].analysis_percentage, 0.20170000000000002, rel_tol=1e-06)

def test_bulk_analyze_conversations_flattened_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.bulk_analyze_conversations(contact_center_insights.BulkAnalyzeConversationsRequest(), parent='parent_value', filter='filter_value', analysis_percentage=0.20170000000000002)

@pytest.mark.asyncio
async def test_bulk_analyze_conversations_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.bulk_analyze_conversations), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.bulk_analyze_conversations(parent='parent_value', filter='filter_value', analysis_percentage=0.20170000000000002)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val
        assert math.isclose(args[0].analysis_percentage, 0.20170000000000002, rel_tol=1e-06)

@pytest.mark.asyncio
async def test_bulk_analyze_conversations_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.bulk_analyze_conversations(contact_center_insights.BulkAnalyzeConversationsRequest(), parent='parent_value', filter='filter_value', analysis_percentage=0.20170000000000002)

@pytest.mark.parametrize('request_type', [contact_center_insights.BulkDeleteConversationsRequest, dict])
def test_bulk_delete_conversations(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.bulk_delete_conversations), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.bulk_delete_conversations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.BulkDeleteConversationsRequest()
    assert isinstance(response, future.Future)

def test_bulk_delete_conversations_empty_call():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.bulk_delete_conversations), '__call__') as call:
        client.bulk_delete_conversations()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.BulkDeleteConversationsRequest()

@pytest.mark.asyncio
async def test_bulk_delete_conversations_async(transport: str='grpc_asyncio', request_type=contact_center_insights.BulkDeleteConversationsRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.bulk_delete_conversations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.bulk_delete_conversations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.BulkDeleteConversationsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_bulk_delete_conversations_async_from_dict():
    await test_bulk_delete_conversations_async(request_type=dict)

def test_bulk_delete_conversations_field_headers():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.BulkDeleteConversationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.bulk_delete_conversations), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.bulk_delete_conversations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_bulk_delete_conversations_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.BulkDeleteConversationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.bulk_delete_conversations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.bulk_delete_conversations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_bulk_delete_conversations_flattened():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.bulk_delete_conversations), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.bulk_delete_conversations(parent='parent_value', filter='filter_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

def test_bulk_delete_conversations_flattened_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.bulk_delete_conversations(contact_center_insights.BulkDeleteConversationsRequest(), parent='parent_value', filter='filter_value')

@pytest.mark.asyncio
async def test_bulk_delete_conversations_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.bulk_delete_conversations), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.bulk_delete_conversations(parent='parent_value', filter='filter_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_bulk_delete_conversations_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.bulk_delete_conversations(contact_center_insights.BulkDeleteConversationsRequest(), parent='parent_value', filter='filter_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.IngestConversationsRequest, dict])
def test_ingest_conversations(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.ingest_conversations), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.ingest_conversations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.IngestConversationsRequest()
    assert isinstance(response, future.Future)

def test_ingest_conversations_empty_call():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.ingest_conversations), '__call__') as call:
        client.ingest_conversations()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.IngestConversationsRequest()

@pytest.mark.asyncio
async def test_ingest_conversations_async(transport: str='grpc_asyncio', request_type=contact_center_insights.IngestConversationsRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.ingest_conversations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.ingest_conversations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.IngestConversationsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_ingest_conversations_async_from_dict():
    await test_ingest_conversations_async(request_type=dict)

def test_ingest_conversations_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.IngestConversationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.ingest_conversations), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.ingest_conversations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_ingest_conversations_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.IngestConversationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.ingest_conversations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.ingest_conversations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_ingest_conversations_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.ingest_conversations), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.ingest_conversations(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_ingest_conversations_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.ingest_conversations(contact_center_insights.IngestConversationsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_ingest_conversations_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.ingest_conversations), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.ingest_conversations(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_ingest_conversations_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.ingest_conversations(contact_center_insights.IngestConversationsRequest(), parent='parent_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.ExportInsightsDataRequest, dict])
def test_export_insights_data(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.export_insights_data), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.export_insights_data(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ExportInsightsDataRequest()
    assert isinstance(response, future.Future)

def test_export_insights_data_empty_call():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.export_insights_data), '__call__') as call:
        client.export_insights_data()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ExportInsightsDataRequest()

@pytest.mark.asyncio
async def test_export_insights_data_async(transport: str='grpc_asyncio', request_type=contact_center_insights.ExportInsightsDataRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.export_insights_data), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.export_insights_data(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ExportInsightsDataRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_export_insights_data_async_from_dict():
    await test_export_insights_data_async(request_type=dict)

def test_export_insights_data_field_headers():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.ExportInsightsDataRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.export_insights_data), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.export_insights_data(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_export_insights_data_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.ExportInsightsDataRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.export_insights_data), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.export_insights_data(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_export_insights_data_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.export_insights_data), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.export_insights_data(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_export_insights_data_flattened_error():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.export_insights_data(contact_center_insights.ExportInsightsDataRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_export_insights_data_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.export_insights_data), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.export_insights_data(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_export_insights_data_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.export_insights_data(contact_center_insights.ExportInsightsDataRequest(), parent='parent_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.CreateIssueModelRequest, dict])
def test_create_issue_model(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_issue_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreateIssueModelRequest()
    assert isinstance(response, future.Future)

def test_create_issue_model_empty_call():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_issue_model), '__call__') as call:
        client.create_issue_model()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreateIssueModelRequest()

@pytest.mark.asyncio
async def test_create_issue_model_async(transport: str='grpc_asyncio', request_type=contact_center_insights.CreateIssueModelRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_issue_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_issue_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreateIssueModelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_issue_model_async_from_dict():
    await test_create_issue_model_async(request_type=dict)

def test_create_issue_model_field_headers():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.CreateIssueModelRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_issue_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_issue_model_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.CreateIssueModelRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_issue_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_issue_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_issue_model_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_issue_model(parent='parent_value', issue_model=resources.IssueModel(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].issue_model
        mock_val = resources.IssueModel(name='name_value')
        assert arg == mock_val

def test_create_issue_model_flattened_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_issue_model(contact_center_insights.CreateIssueModelRequest(), parent='parent_value', issue_model=resources.IssueModel(name='name_value'))

@pytest.mark.asyncio
async def test_create_issue_model_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_issue_model(parent='parent_value', issue_model=resources.IssueModel(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].issue_model
        mock_val = resources.IssueModel(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_issue_model_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_issue_model(contact_center_insights.CreateIssueModelRequest(), parent='parent_value', issue_model=resources.IssueModel(name='name_value'))

@pytest.mark.parametrize('request_type', [contact_center_insights.UpdateIssueModelRequest, dict])
def test_update_issue_model(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_issue_model), '__call__') as call:
        call.return_value = resources.IssueModel(name='name_value', display_name='display_name_value', issue_count=1201, state=resources.IssueModel.State.UNDEPLOYED, model_type=resources.IssueModel.ModelType.TYPE_V1, language_code='language_code_value')
        response = client.update_issue_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateIssueModelRequest()
    assert isinstance(response, resources.IssueModel)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.issue_count == 1201
    assert response.state == resources.IssueModel.State.UNDEPLOYED
    assert response.model_type == resources.IssueModel.ModelType.TYPE_V1
    assert response.language_code == 'language_code_value'

def test_update_issue_model_empty_call():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_issue_model), '__call__') as call:
        client.update_issue_model()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateIssueModelRequest()

@pytest.mark.asyncio
async def test_update_issue_model_async(transport: str='grpc_asyncio', request_type=contact_center_insights.UpdateIssueModelRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_issue_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.IssueModel(name='name_value', display_name='display_name_value', issue_count=1201, state=resources.IssueModel.State.UNDEPLOYED, model_type=resources.IssueModel.ModelType.TYPE_V1, language_code='language_code_value'))
        response = await client.update_issue_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateIssueModelRequest()
    assert isinstance(response, resources.IssueModel)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.issue_count == 1201
    assert response.state == resources.IssueModel.State.UNDEPLOYED
    assert response.model_type == resources.IssueModel.ModelType.TYPE_V1
    assert response.language_code == 'language_code_value'

@pytest.mark.asyncio
async def test_update_issue_model_async_from_dict():
    await test_update_issue_model_async(request_type=dict)

def test_update_issue_model_field_headers():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UpdateIssueModelRequest()
    request.issue_model.name = 'name_value'
    with mock.patch.object(type(client.transport.update_issue_model), '__call__') as call:
        call.return_value = resources.IssueModel()
        client.update_issue_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'issue_model.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_issue_model_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UpdateIssueModelRequest()
    request.issue_model.name = 'name_value'
    with mock.patch.object(type(client.transport.update_issue_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.IssueModel())
        await client.update_issue_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'issue_model.name=name_value') in kw['metadata']

def test_update_issue_model_flattened():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_issue_model), '__call__') as call:
        call.return_value = resources.IssueModel()
        client.update_issue_model(issue_model=resources.IssueModel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].issue_model
        mock_val = resources.IssueModel(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_issue_model_flattened_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_issue_model(contact_center_insights.UpdateIssueModelRequest(), issue_model=resources.IssueModel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_issue_model_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_issue_model), '__call__') as call:
        call.return_value = resources.IssueModel()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.IssueModel())
        response = await client.update_issue_model(issue_model=resources.IssueModel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].issue_model
        mock_val = resources.IssueModel(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_issue_model_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_issue_model(contact_center_insights.UpdateIssueModelRequest(), issue_model=resources.IssueModel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [contact_center_insights.GetIssueModelRequest, dict])
def test_get_issue_model(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_issue_model), '__call__') as call:
        call.return_value = resources.IssueModel(name='name_value', display_name='display_name_value', issue_count=1201, state=resources.IssueModel.State.UNDEPLOYED, model_type=resources.IssueModel.ModelType.TYPE_V1, language_code='language_code_value')
        response = client.get_issue_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetIssueModelRequest()
    assert isinstance(response, resources.IssueModel)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.issue_count == 1201
    assert response.state == resources.IssueModel.State.UNDEPLOYED
    assert response.model_type == resources.IssueModel.ModelType.TYPE_V1
    assert response.language_code == 'language_code_value'

def test_get_issue_model_empty_call():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_issue_model), '__call__') as call:
        client.get_issue_model()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetIssueModelRequest()

@pytest.mark.asyncio
async def test_get_issue_model_async(transport: str='grpc_asyncio', request_type=contact_center_insights.GetIssueModelRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_issue_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.IssueModel(name='name_value', display_name='display_name_value', issue_count=1201, state=resources.IssueModel.State.UNDEPLOYED, model_type=resources.IssueModel.ModelType.TYPE_V1, language_code='language_code_value'))
        response = await client.get_issue_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetIssueModelRequest()
    assert isinstance(response, resources.IssueModel)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.issue_count == 1201
    assert response.state == resources.IssueModel.State.UNDEPLOYED
    assert response.model_type == resources.IssueModel.ModelType.TYPE_V1
    assert response.language_code == 'language_code_value'

@pytest.mark.asyncio
async def test_get_issue_model_async_from_dict():
    await test_get_issue_model_async(request_type=dict)

def test_get_issue_model_field_headers():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.GetIssueModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_issue_model), '__call__') as call:
        call.return_value = resources.IssueModel()
        client.get_issue_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_issue_model_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.GetIssueModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_issue_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.IssueModel())
        await client.get_issue_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_issue_model_flattened():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_issue_model), '__call__') as call:
        call.return_value = resources.IssueModel()
        client.get_issue_model(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_issue_model_flattened_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_issue_model(contact_center_insights.GetIssueModelRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_issue_model_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_issue_model), '__call__') as call:
        call.return_value = resources.IssueModel()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.IssueModel())
        response = await client.get_issue_model(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_issue_model_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_issue_model(contact_center_insights.GetIssueModelRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.ListIssueModelsRequest, dict])
def test_list_issue_models(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_issue_models), '__call__') as call:
        call.return_value = contact_center_insights.ListIssueModelsResponse()
        response = client.list_issue_models(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListIssueModelsRequest()
    assert isinstance(response, contact_center_insights.ListIssueModelsResponse)

def test_list_issue_models_empty_call():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_issue_models), '__call__') as call:
        client.list_issue_models()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListIssueModelsRequest()

@pytest.mark.asyncio
async def test_list_issue_models_async(transport: str='grpc_asyncio', request_type=contact_center_insights.ListIssueModelsRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_issue_models), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListIssueModelsResponse())
        response = await client.list_issue_models(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListIssueModelsRequest()
    assert isinstance(response, contact_center_insights.ListIssueModelsResponse)

@pytest.mark.asyncio
async def test_list_issue_models_async_from_dict():
    await test_list_issue_models_async(request_type=dict)

def test_list_issue_models_field_headers():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.ListIssueModelsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_issue_models), '__call__') as call:
        call.return_value = contact_center_insights.ListIssueModelsResponse()
        client.list_issue_models(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_issue_models_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.ListIssueModelsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_issue_models), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListIssueModelsResponse())
        await client.list_issue_models(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_issue_models_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_issue_models), '__call__') as call:
        call.return_value = contact_center_insights.ListIssueModelsResponse()
        client.list_issue_models(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_issue_models_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_issue_models(contact_center_insights.ListIssueModelsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_issue_models_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_issue_models), '__call__') as call:
        call.return_value = contact_center_insights.ListIssueModelsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListIssueModelsResponse())
        response = await client.list_issue_models(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_issue_models_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_issue_models(contact_center_insights.ListIssueModelsRequest(), parent='parent_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.DeleteIssueModelRequest, dict])
def test_delete_issue_model(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_issue_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteIssueModelRequest()
    assert isinstance(response, future.Future)

def test_delete_issue_model_empty_call():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_issue_model), '__call__') as call:
        client.delete_issue_model()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteIssueModelRequest()

@pytest.mark.asyncio
async def test_delete_issue_model_async(transport: str='grpc_asyncio', request_type=contact_center_insights.DeleteIssueModelRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_issue_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_issue_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteIssueModelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_issue_model_async_from_dict():
    await test_delete_issue_model_async(request_type=dict)

def test_delete_issue_model_field_headers():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.DeleteIssueModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_issue_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_issue_model_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.DeleteIssueModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_issue_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_issue_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_issue_model_flattened():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_issue_model(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_issue_model_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_issue_model(contact_center_insights.DeleteIssueModelRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_issue_model_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_issue_model(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_issue_model_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_issue_model(contact_center_insights.DeleteIssueModelRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.DeployIssueModelRequest, dict])
def test_deploy_issue_model(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.deploy_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.deploy_issue_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeployIssueModelRequest()
    assert isinstance(response, future.Future)

def test_deploy_issue_model_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.deploy_issue_model), '__call__') as call:
        client.deploy_issue_model()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeployIssueModelRequest()

@pytest.mark.asyncio
async def test_deploy_issue_model_async(transport: str='grpc_asyncio', request_type=contact_center_insights.DeployIssueModelRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.deploy_issue_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.deploy_issue_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeployIssueModelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_deploy_issue_model_async_from_dict():
    await test_deploy_issue_model_async(request_type=dict)

def test_deploy_issue_model_field_headers():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.DeployIssueModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.deploy_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.deploy_issue_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_deploy_issue_model_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.DeployIssueModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.deploy_issue_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.deploy_issue_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_deploy_issue_model_flattened():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.deploy_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.deploy_issue_model(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_deploy_issue_model_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.deploy_issue_model(contact_center_insights.DeployIssueModelRequest(), name='name_value')

@pytest.mark.asyncio
async def test_deploy_issue_model_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.deploy_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.deploy_issue_model(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_deploy_issue_model_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.deploy_issue_model(contact_center_insights.DeployIssueModelRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.UndeployIssueModelRequest, dict])
def test_undeploy_issue_model(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.undeploy_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.undeploy_issue_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UndeployIssueModelRequest()
    assert isinstance(response, future.Future)

def test_undeploy_issue_model_empty_call():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.undeploy_issue_model), '__call__') as call:
        client.undeploy_issue_model()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UndeployIssueModelRequest()

@pytest.mark.asyncio
async def test_undeploy_issue_model_async(transport: str='grpc_asyncio', request_type=contact_center_insights.UndeployIssueModelRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.undeploy_issue_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.undeploy_issue_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UndeployIssueModelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_undeploy_issue_model_async_from_dict():
    await test_undeploy_issue_model_async(request_type=dict)

def test_undeploy_issue_model_field_headers():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UndeployIssueModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.undeploy_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.undeploy_issue_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_undeploy_issue_model_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UndeployIssueModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.undeploy_issue_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.undeploy_issue_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_undeploy_issue_model_flattened():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.undeploy_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.undeploy_issue_model(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_undeploy_issue_model_flattened_error():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.undeploy_issue_model(contact_center_insights.UndeployIssueModelRequest(), name='name_value')

@pytest.mark.asyncio
async def test_undeploy_issue_model_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.undeploy_issue_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.undeploy_issue_model(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_undeploy_issue_model_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.undeploy_issue_model(contact_center_insights.UndeployIssueModelRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.GetIssueRequest, dict])
def test_get_issue(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_issue), '__call__') as call:
        call.return_value = resources.Issue(name='name_value', display_name='display_name_value', sample_utterances=['sample_utterances_value'])
        response = client.get_issue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetIssueRequest()
    assert isinstance(response, resources.Issue)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.sample_utterances == ['sample_utterances_value']

def test_get_issue_empty_call():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_issue), '__call__') as call:
        client.get_issue()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetIssueRequest()

@pytest.mark.asyncio
async def test_get_issue_async(transport: str='grpc_asyncio', request_type=contact_center_insights.GetIssueRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_issue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Issue(name='name_value', display_name='display_name_value', sample_utterances=['sample_utterances_value']))
        response = await client.get_issue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetIssueRequest()
    assert isinstance(response, resources.Issue)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.sample_utterances == ['sample_utterances_value']

@pytest.mark.asyncio
async def test_get_issue_async_from_dict():
    await test_get_issue_async(request_type=dict)

def test_get_issue_field_headers():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.GetIssueRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_issue), '__call__') as call:
        call.return_value = resources.Issue()
        client.get_issue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_issue_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.GetIssueRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_issue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Issue())
        await client.get_issue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_issue_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_issue), '__call__') as call:
        call.return_value = resources.Issue()
        client.get_issue(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_issue_flattened_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_issue(contact_center_insights.GetIssueRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_issue_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_issue), '__call__') as call:
        call.return_value = resources.Issue()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Issue())
        response = await client.get_issue(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_issue_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_issue(contact_center_insights.GetIssueRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.ListIssuesRequest, dict])
def test_list_issues(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_issues), '__call__') as call:
        call.return_value = contact_center_insights.ListIssuesResponse()
        response = client.list_issues(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListIssuesRequest()
    assert isinstance(response, contact_center_insights.ListIssuesResponse)

def test_list_issues_empty_call():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_issues), '__call__') as call:
        client.list_issues()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListIssuesRequest()

@pytest.mark.asyncio
async def test_list_issues_async(transport: str='grpc_asyncio', request_type=contact_center_insights.ListIssuesRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_issues), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListIssuesResponse())
        response = await client.list_issues(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListIssuesRequest()
    assert isinstance(response, contact_center_insights.ListIssuesResponse)

@pytest.mark.asyncio
async def test_list_issues_async_from_dict():
    await test_list_issues_async(request_type=dict)

def test_list_issues_field_headers():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.ListIssuesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_issues), '__call__') as call:
        call.return_value = contact_center_insights.ListIssuesResponse()
        client.list_issues(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_issues_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.ListIssuesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_issues), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListIssuesResponse())
        await client.list_issues(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_issues_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_issues), '__call__') as call:
        call.return_value = contact_center_insights.ListIssuesResponse()
        client.list_issues(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_issues_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_issues(contact_center_insights.ListIssuesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_issues_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_issues), '__call__') as call:
        call.return_value = contact_center_insights.ListIssuesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListIssuesResponse())
        response = await client.list_issues(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_issues_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_issues(contact_center_insights.ListIssuesRequest(), parent='parent_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.UpdateIssueRequest, dict])
def test_update_issue(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_issue), '__call__') as call:
        call.return_value = resources.Issue(name='name_value', display_name='display_name_value', sample_utterances=['sample_utterances_value'])
        response = client.update_issue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateIssueRequest()
    assert isinstance(response, resources.Issue)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.sample_utterances == ['sample_utterances_value']

def test_update_issue_empty_call():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_issue), '__call__') as call:
        client.update_issue()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateIssueRequest()

@pytest.mark.asyncio
async def test_update_issue_async(transport: str='grpc_asyncio', request_type=contact_center_insights.UpdateIssueRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_issue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Issue(name='name_value', display_name='display_name_value', sample_utterances=['sample_utterances_value']))
        response = await client.update_issue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateIssueRequest()
    assert isinstance(response, resources.Issue)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.sample_utterances == ['sample_utterances_value']

@pytest.mark.asyncio
async def test_update_issue_async_from_dict():
    await test_update_issue_async(request_type=dict)

def test_update_issue_field_headers():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UpdateIssueRequest()
    request.issue.name = 'name_value'
    with mock.patch.object(type(client.transport.update_issue), '__call__') as call:
        call.return_value = resources.Issue()
        client.update_issue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'issue.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_issue_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UpdateIssueRequest()
    request.issue.name = 'name_value'
    with mock.patch.object(type(client.transport.update_issue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Issue())
        await client.update_issue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'issue.name=name_value') in kw['metadata']

def test_update_issue_flattened():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_issue), '__call__') as call:
        call.return_value = resources.Issue()
        client.update_issue(issue=resources.Issue(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].issue
        mock_val = resources.Issue(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_issue_flattened_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_issue(contact_center_insights.UpdateIssueRequest(), issue=resources.Issue(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_issue_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_issue), '__call__') as call:
        call.return_value = resources.Issue()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Issue())
        response = await client.update_issue(issue=resources.Issue(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].issue
        mock_val = resources.Issue(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_issue_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_issue(contact_center_insights.UpdateIssueRequest(), issue=resources.Issue(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [contact_center_insights.DeleteIssueRequest, dict])
def test_delete_issue(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_issue), '__call__') as call:
        call.return_value = None
        response = client.delete_issue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteIssueRequest()
    assert response is None

def test_delete_issue_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_issue), '__call__') as call:
        client.delete_issue()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteIssueRequest()

@pytest.mark.asyncio
async def test_delete_issue_async(transport: str='grpc_asyncio', request_type=contact_center_insights.DeleteIssueRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_issue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_issue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteIssueRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_issue_async_from_dict():
    await test_delete_issue_async(request_type=dict)

def test_delete_issue_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.DeleteIssueRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_issue), '__call__') as call:
        call.return_value = None
        client.delete_issue(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_issue_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.DeleteIssueRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_issue), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_issue(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_issue_flattened():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_issue), '__call__') as call:
        call.return_value = None
        client.delete_issue(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_issue_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_issue(contact_center_insights.DeleteIssueRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_issue_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_issue), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_issue(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_issue_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_issue(contact_center_insights.DeleteIssueRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.CalculateIssueModelStatsRequest, dict])
def test_calculate_issue_model_stats(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.calculate_issue_model_stats), '__call__') as call:
        call.return_value = contact_center_insights.CalculateIssueModelStatsResponse()
        response = client.calculate_issue_model_stats(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CalculateIssueModelStatsRequest()
    assert isinstance(response, contact_center_insights.CalculateIssueModelStatsResponse)

def test_calculate_issue_model_stats_empty_call():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.calculate_issue_model_stats), '__call__') as call:
        client.calculate_issue_model_stats()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CalculateIssueModelStatsRequest()

@pytest.mark.asyncio
async def test_calculate_issue_model_stats_async(transport: str='grpc_asyncio', request_type=contact_center_insights.CalculateIssueModelStatsRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.calculate_issue_model_stats), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.CalculateIssueModelStatsResponse())
        response = await client.calculate_issue_model_stats(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CalculateIssueModelStatsRequest()
    assert isinstance(response, contact_center_insights.CalculateIssueModelStatsResponse)

@pytest.mark.asyncio
async def test_calculate_issue_model_stats_async_from_dict():
    await test_calculate_issue_model_stats_async(request_type=dict)

def test_calculate_issue_model_stats_field_headers():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.CalculateIssueModelStatsRequest()
    request.issue_model = 'issue_model_value'
    with mock.patch.object(type(client.transport.calculate_issue_model_stats), '__call__') as call:
        call.return_value = contact_center_insights.CalculateIssueModelStatsResponse()
        client.calculate_issue_model_stats(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'issue_model=issue_model_value') in kw['metadata']

@pytest.mark.asyncio
async def test_calculate_issue_model_stats_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.CalculateIssueModelStatsRequest()
    request.issue_model = 'issue_model_value'
    with mock.patch.object(type(client.transport.calculate_issue_model_stats), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.CalculateIssueModelStatsResponse())
        await client.calculate_issue_model_stats(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'issue_model=issue_model_value') in kw['metadata']

def test_calculate_issue_model_stats_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.calculate_issue_model_stats), '__call__') as call:
        call.return_value = contact_center_insights.CalculateIssueModelStatsResponse()
        client.calculate_issue_model_stats(issue_model='issue_model_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].issue_model
        mock_val = 'issue_model_value'
        assert arg == mock_val

def test_calculate_issue_model_stats_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.calculate_issue_model_stats(contact_center_insights.CalculateIssueModelStatsRequest(), issue_model='issue_model_value')

@pytest.mark.asyncio
async def test_calculate_issue_model_stats_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.calculate_issue_model_stats), '__call__') as call:
        call.return_value = contact_center_insights.CalculateIssueModelStatsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.CalculateIssueModelStatsResponse())
        response = await client.calculate_issue_model_stats(issue_model='issue_model_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].issue_model
        mock_val = 'issue_model_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_calculate_issue_model_stats_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.calculate_issue_model_stats(contact_center_insights.CalculateIssueModelStatsRequest(), issue_model='issue_model_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.CreatePhraseMatcherRequest, dict])
def test_create_phrase_matcher(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_phrase_matcher), '__call__') as call:
        call.return_value = resources.PhraseMatcher(name='name_value', revision_id='revision_id_value', version_tag='version_tag_value', display_name='display_name_value', type_=resources.PhraseMatcher.PhraseMatcherType.ALL_OF, active=True, role_match=resources.ConversationParticipant.Role.HUMAN_AGENT)
        response = client.create_phrase_matcher(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreatePhraseMatcherRequest()
    assert isinstance(response, resources.PhraseMatcher)
    assert response.name == 'name_value'
    assert response.revision_id == 'revision_id_value'
    assert response.version_tag == 'version_tag_value'
    assert response.display_name == 'display_name_value'
    assert response.type_ == resources.PhraseMatcher.PhraseMatcherType.ALL_OF
    assert response.active is True
    assert response.role_match == resources.ConversationParticipant.Role.HUMAN_AGENT

def test_create_phrase_matcher_empty_call():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_phrase_matcher), '__call__') as call:
        client.create_phrase_matcher()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreatePhraseMatcherRequest()

@pytest.mark.asyncio
async def test_create_phrase_matcher_async(transport: str='grpc_asyncio', request_type=contact_center_insights.CreatePhraseMatcherRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_phrase_matcher), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.PhraseMatcher(name='name_value', revision_id='revision_id_value', version_tag='version_tag_value', display_name='display_name_value', type_=resources.PhraseMatcher.PhraseMatcherType.ALL_OF, active=True, role_match=resources.ConversationParticipant.Role.HUMAN_AGENT))
        response = await client.create_phrase_matcher(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreatePhraseMatcherRequest()
    assert isinstance(response, resources.PhraseMatcher)
    assert response.name == 'name_value'
    assert response.revision_id == 'revision_id_value'
    assert response.version_tag == 'version_tag_value'
    assert response.display_name == 'display_name_value'
    assert response.type_ == resources.PhraseMatcher.PhraseMatcherType.ALL_OF
    assert response.active is True
    assert response.role_match == resources.ConversationParticipant.Role.HUMAN_AGENT

@pytest.mark.asyncio
async def test_create_phrase_matcher_async_from_dict():
    await test_create_phrase_matcher_async(request_type=dict)

def test_create_phrase_matcher_field_headers():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.CreatePhraseMatcherRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_phrase_matcher), '__call__') as call:
        call.return_value = resources.PhraseMatcher()
        client.create_phrase_matcher(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_phrase_matcher_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.CreatePhraseMatcherRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_phrase_matcher), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.PhraseMatcher())
        await client.create_phrase_matcher(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_phrase_matcher_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_phrase_matcher), '__call__') as call:
        call.return_value = resources.PhraseMatcher()
        client.create_phrase_matcher(parent='parent_value', phrase_matcher=resources.PhraseMatcher(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].phrase_matcher
        mock_val = resources.PhraseMatcher(name='name_value')
        assert arg == mock_val

def test_create_phrase_matcher_flattened_error():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_phrase_matcher(contact_center_insights.CreatePhraseMatcherRequest(), parent='parent_value', phrase_matcher=resources.PhraseMatcher(name='name_value'))

@pytest.mark.asyncio
async def test_create_phrase_matcher_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_phrase_matcher), '__call__') as call:
        call.return_value = resources.PhraseMatcher()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.PhraseMatcher())
        response = await client.create_phrase_matcher(parent='parent_value', phrase_matcher=resources.PhraseMatcher(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].phrase_matcher
        mock_val = resources.PhraseMatcher(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_phrase_matcher_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_phrase_matcher(contact_center_insights.CreatePhraseMatcherRequest(), parent='parent_value', phrase_matcher=resources.PhraseMatcher(name='name_value'))

@pytest.mark.parametrize('request_type', [contact_center_insights.GetPhraseMatcherRequest, dict])
def test_get_phrase_matcher(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_phrase_matcher), '__call__') as call:
        call.return_value = resources.PhraseMatcher(name='name_value', revision_id='revision_id_value', version_tag='version_tag_value', display_name='display_name_value', type_=resources.PhraseMatcher.PhraseMatcherType.ALL_OF, active=True, role_match=resources.ConversationParticipant.Role.HUMAN_AGENT)
        response = client.get_phrase_matcher(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetPhraseMatcherRequest()
    assert isinstance(response, resources.PhraseMatcher)
    assert response.name == 'name_value'
    assert response.revision_id == 'revision_id_value'
    assert response.version_tag == 'version_tag_value'
    assert response.display_name == 'display_name_value'
    assert response.type_ == resources.PhraseMatcher.PhraseMatcherType.ALL_OF
    assert response.active is True
    assert response.role_match == resources.ConversationParticipant.Role.HUMAN_AGENT

def test_get_phrase_matcher_empty_call():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_phrase_matcher), '__call__') as call:
        client.get_phrase_matcher()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetPhraseMatcherRequest()

@pytest.mark.asyncio
async def test_get_phrase_matcher_async(transport: str='grpc_asyncio', request_type=contact_center_insights.GetPhraseMatcherRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_phrase_matcher), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.PhraseMatcher(name='name_value', revision_id='revision_id_value', version_tag='version_tag_value', display_name='display_name_value', type_=resources.PhraseMatcher.PhraseMatcherType.ALL_OF, active=True, role_match=resources.ConversationParticipant.Role.HUMAN_AGENT))
        response = await client.get_phrase_matcher(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetPhraseMatcherRequest()
    assert isinstance(response, resources.PhraseMatcher)
    assert response.name == 'name_value'
    assert response.revision_id == 'revision_id_value'
    assert response.version_tag == 'version_tag_value'
    assert response.display_name == 'display_name_value'
    assert response.type_ == resources.PhraseMatcher.PhraseMatcherType.ALL_OF
    assert response.active is True
    assert response.role_match == resources.ConversationParticipant.Role.HUMAN_AGENT

@pytest.mark.asyncio
async def test_get_phrase_matcher_async_from_dict():
    await test_get_phrase_matcher_async(request_type=dict)

def test_get_phrase_matcher_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.GetPhraseMatcherRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_phrase_matcher), '__call__') as call:
        call.return_value = resources.PhraseMatcher()
        client.get_phrase_matcher(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_phrase_matcher_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.GetPhraseMatcherRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_phrase_matcher), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.PhraseMatcher())
        await client.get_phrase_matcher(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_phrase_matcher_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_phrase_matcher), '__call__') as call:
        call.return_value = resources.PhraseMatcher()
        client.get_phrase_matcher(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_phrase_matcher_flattened_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_phrase_matcher(contact_center_insights.GetPhraseMatcherRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_phrase_matcher_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_phrase_matcher), '__call__') as call:
        call.return_value = resources.PhraseMatcher()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.PhraseMatcher())
        response = await client.get_phrase_matcher(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_phrase_matcher_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_phrase_matcher(contact_center_insights.GetPhraseMatcherRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.ListPhraseMatchersRequest, dict])
def test_list_phrase_matchers(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_phrase_matchers), '__call__') as call:
        call.return_value = contact_center_insights.ListPhraseMatchersResponse(next_page_token='next_page_token_value')
        response = client.list_phrase_matchers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListPhraseMatchersRequest()
    assert isinstance(response, pagers.ListPhraseMatchersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_phrase_matchers_empty_call():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_phrase_matchers), '__call__') as call:
        client.list_phrase_matchers()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListPhraseMatchersRequest()

@pytest.mark.asyncio
async def test_list_phrase_matchers_async(transport: str='grpc_asyncio', request_type=contact_center_insights.ListPhraseMatchersRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_phrase_matchers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListPhraseMatchersResponse(next_page_token='next_page_token_value'))
        response = await client.list_phrase_matchers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListPhraseMatchersRequest()
    assert isinstance(response, pagers.ListPhraseMatchersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_phrase_matchers_async_from_dict():
    await test_list_phrase_matchers_async(request_type=dict)

def test_list_phrase_matchers_field_headers():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.ListPhraseMatchersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_phrase_matchers), '__call__') as call:
        call.return_value = contact_center_insights.ListPhraseMatchersResponse()
        client.list_phrase_matchers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_phrase_matchers_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.ListPhraseMatchersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_phrase_matchers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListPhraseMatchersResponse())
        await client.list_phrase_matchers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_phrase_matchers_flattened():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_phrase_matchers), '__call__') as call:
        call.return_value = contact_center_insights.ListPhraseMatchersResponse()
        client.list_phrase_matchers(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_phrase_matchers_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_phrase_matchers(contact_center_insights.ListPhraseMatchersRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_phrase_matchers_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_phrase_matchers), '__call__') as call:
        call.return_value = contact_center_insights.ListPhraseMatchersResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListPhraseMatchersResponse())
        response = await client.list_phrase_matchers(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_phrase_matchers_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_phrase_matchers(contact_center_insights.ListPhraseMatchersRequest(), parent='parent_value')

def test_list_phrase_matchers_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_phrase_matchers), '__call__') as call:
        call.side_effect = (contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher(), resources.PhraseMatcher(), resources.PhraseMatcher()], next_page_token='abc'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[], next_page_token='def'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher()], next_page_token='ghi'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher(), resources.PhraseMatcher()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_phrase_matchers(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.PhraseMatcher) for i in results))

def test_list_phrase_matchers_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_phrase_matchers), '__call__') as call:
        call.side_effect = (contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher(), resources.PhraseMatcher(), resources.PhraseMatcher()], next_page_token='abc'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[], next_page_token='def'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher()], next_page_token='ghi'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher(), resources.PhraseMatcher()]), RuntimeError)
        pages = list(client.list_phrase_matchers(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_phrase_matchers_async_pager():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_phrase_matchers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher(), resources.PhraseMatcher(), resources.PhraseMatcher()], next_page_token='abc'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[], next_page_token='def'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher()], next_page_token='ghi'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher(), resources.PhraseMatcher()]), RuntimeError)
        async_pager = await client.list_phrase_matchers(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.PhraseMatcher) for i in responses))

@pytest.mark.asyncio
async def test_list_phrase_matchers_async_pages():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_phrase_matchers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher(), resources.PhraseMatcher(), resources.PhraseMatcher()], next_page_token='abc'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[], next_page_token='def'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher()], next_page_token='ghi'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher(), resources.PhraseMatcher()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_phrase_matchers(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [contact_center_insights.DeletePhraseMatcherRequest, dict])
def test_delete_phrase_matcher(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_phrase_matcher), '__call__') as call:
        call.return_value = None
        response = client.delete_phrase_matcher(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeletePhraseMatcherRequest()
    assert response is None

def test_delete_phrase_matcher_empty_call():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_phrase_matcher), '__call__') as call:
        client.delete_phrase_matcher()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeletePhraseMatcherRequest()

@pytest.mark.asyncio
async def test_delete_phrase_matcher_async(transport: str='grpc_asyncio', request_type=contact_center_insights.DeletePhraseMatcherRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_phrase_matcher), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_phrase_matcher(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeletePhraseMatcherRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_phrase_matcher_async_from_dict():
    await test_delete_phrase_matcher_async(request_type=dict)

def test_delete_phrase_matcher_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.DeletePhraseMatcherRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_phrase_matcher), '__call__') as call:
        call.return_value = None
        client.delete_phrase_matcher(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_phrase_matcher_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.DeletePhraseMatcherRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_phrase_matcher), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_phrase_matcher(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_phrase_matcher_flattened():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_phrase_matcher), '__call__') as call:
        call.return_value = None
        client.delete_phrase_matcher(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_phrase_matcher_flattened_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_phrase_matcher(contact_center_insights.DeletePhraseMatcherRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_phrase_matcher_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_phrase_matcher), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_phrase_matcher(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_phrase_matcher_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_phrase_matcher(contact_center_insights.DeletePhraseMatcherRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.UpdatePhraseMatcherRequest, dict])
def test_update_phrase_matcher(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_phrase_matcher), '__call__') as call:
        call.return_value = resources.PhraseMatcher(name='name_value', revision_id='revision_id_value', version_tag='version_tag_value', display_name='display_name_value', type_=resources.PhraseMatcher.PhraseMatcherType.ALL_OF, active=True, role_match=resources.ConversationParticipant.Role.HUMAN_AGENT)
        response = client.update_phrase_matcher(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdatePhraseMatcherRequest()
    assert isinstance(response, resources.PhraseMatcher)
    assert response.name == 'name_value'
    assert response.revision_id == 'revision_id_value'
    assert response.version_tag == 'version_tag_value'
    assert response.display_name == 'display_name_value'
    assert response.type_ == resources.PhraseMatcher.PhraseMatcherType.ALL_OF
    assert response.active is True
    assert response.role_match == resources.ConversationParticipant.Role.HUMAN_AGENT

def test_update_phrase_matcher_empty_call():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_phrase_matcher), '__call__') as call:
        client.update_phrase_matcher()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdatePhraseMatcherRequest()

@pytest.mark.asyncio
async def test_update_phrase_matcher_async(transport: str='grpc_asyncio', request_type=contact_center_insights.UpdatePhraseMatcherRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_phrase_matcher), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.PhraseMatcher(name='name_value', revision_id='revision_id_value', version_tag='version_tag_value', display_name='display_name_value', type_=resources.PhraseMatcher.PhraseMatcherType.ALL_OF, active=True, role_match=resources.ConversationParticipant.Role.HUMAN_AGENT))
        response = await client.update_phrase_matcher(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdatePhraseMatcherRequest()
    assert isinstance(response, resources.PhraseMatcher)
    assert response.name == 'name_value'
    assert response.revision_id == 'revision_id_value'
    assert response.version_tag == 'version_tag_value'
    assert response.display_name == 'display_name_value'
    assert response.type_ == resources.PhraseMatcher.PhraseMatcherType.ALL_OF
    assert response.active is True
    assert response.role_match == resources.ConversationParticipant.Role.HUMAN_AGENT

@pytest.mark.asyncio
async def test_update_phrase_matcher_async_from_dict():
    await test_update_phrase_matcher_async(request_type=dict)

def test_update_phrase_matcher_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UpdatePhraseMatcherRequest()
    request.phrase_matcher.name = 'name_value'
    with mock.patch.object(type(client.transport.update_phrase_matcher), '__call__') as call:
        call.return_value = resources.PhraseMatcher()
        client.update_phrase_matcher(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'phrase_matcher.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_phrase_matcher_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UpdatePhraseMatcherRequest()
    request.phrase_matcher.name = 'name_value'
    with mock.patch.object(type(client.transport.update_phrase_matcher), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.PhraseMatcher())
        await client.update_phrase_matcher(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'phrase_matcher.name=name_value') in kw['metadata']

def test_update_phrase_matcher_flattened():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_phrase_matcher), '__call__') as call:
        call.return_value = resources.PhraseMatcher()
        client.update_phrase_matcher(phrase_matcher=resources.PhraseMatcher(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].phrase_matcher
        mock_val = resources.PhraseMatcher(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_phrase_matcher_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_phrase_matcher(contact_center_insights.UpdatePhraseMatcherRequest(), phrase_matcher=resources.PhraseMatcher(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_phrase_matcher_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_phrase_matcher), '__call__') as call:
        call.return_value = resources.PhraseMatcher()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.PhraseMatcher())
        response = await client.update_phrase_matcher(phrase_matcher=resources.PhraseMatcher(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].phrase_matcher
        mock_val = resources.PhraseMatcher(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_phrase_matcher_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_phrase_matcher(contact_center_insights.UpdatePhraseMatcherRequest(), phrase_matcher=resources.PhraseMatcher(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [contact_center_insights.CalculateStatsRequest, dict])
def test_calculate_stats(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.calculate_stats), '__call__') as call:
        call.return_value = contact_center_insights.CalculateStatsResponse(average_turn_count=1931, conversation_count=1955)
        response = client.calculate_stats(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CalculateStatsRequest()
    assert isinstance(response, contact_center_insights.CalculateStatsResponse)
    assert response.average_turn_count == 1931
    assert response.conversation_count == 1955

def test_calculate_stats_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.calculate_stats), '__call__') as call:
        client.calculate_stats()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CalculateStatsRequest()

@pytest.mark.asyncio
async def test_calculate_stats_async(transport: str='grpc_asyncio', request_type=contact_center_insights.CalculateStatsRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.calculate_stats), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.CalculateStatsResponse(average_turn_count=1931, conversation_count=1955))
        response = await client.calculate_stats(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CalculateStatsRequest()
    assert isinstance(response, contact_center_insights.CalculateStatsResponse)
    assert response.average_turn_count == 1931
    assert response.conversation_count == 1955

@pytest.mark.asyncio
async def test_calculate_stats_async_from_dict():
    await test_calculate_stats_async(request_type=dict)

def test_calculate_stats_field_headers():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.CalculateStatsRequest()
    request.location = 'location_value'
    with mock.patch.object(type(client.transport.calculate_stats), '__call__') as call:
        call.return_value = contact_center_insights.CalculateStatsResponse()
        client.calculate_stats(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'location=location_value') in kw['metadata']

@pytest.mark.asyncio
async def test_calculate_stats_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.CalculateStatsRequest()
    request.location = 'location_value'
    with mock.patch.object(type(client.transport.calculate_stats), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.CalculateStatsResponse())
        await client.calculate_stats(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'location=location_value') in kw['metadata']

def test_calculate_stats_flattened():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.calculate_stats), '__call__') as call:
        call.return_value = contact_center_insights.CalculateStatsResponse()
        client.calculate_stats(location='location_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].location
        mock_val = 'location_value'
        assert arg == mock_val

def test_calculate_stats_flattened_error():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.calculate_stats(contact_center_insights.CalculateStatsRequest(), location='location_value')

@pytest.mark.asyncio
async def test_calculate_stats_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.calculate_stats), '__call__') as call:
        call.return_value = contact_center_insights.CalculateStatsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.CalculateStatsResponse())
        response = await client.calculate_stats(location='location_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].location
        mock_val = 'location_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_calculate_stats_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.calculate_stats(contact_center_insights.CalculateStatsRequest(), location='location_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.GetSettingsRequest, dict])
def test_get_settings(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_settings), '__call__') as call:
        call.return_value = resources.Settings(name='name_value', language_code='language_code_value')
        response = client.get_settings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetSettingsRequest()
    assert isinstance(response, resources.Settings)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'

def test_get_settings_empty_call():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_settings), '__call__') as call:
        client.get_settings()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetSettingsRequest()

@pytest.mark.asyncio
async def test_get_settings_async(transport: str='grpc_asyncio', request_type=contact_center_insights.GetSettingsRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_settings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Settings(name='name_value', language_code='language_code_value'))
        response = await client.get_settings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetSettingsRequest()
    assert isinstance(response, resources.Settings)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'

@pytest.mark.asyncio
async def test_get_settings_async_from_dict():
    await test_get_settings_async(request_type=dict)

def test_get_settings_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.GetSettingsRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_settings), '__call__') as call:
        call.return_value = resources.Settings()
        client.get_settings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_settings_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.GetSettingsRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_settings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Settings())
        await client.get_settings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_settings_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_settings), '__call__') as call:
        call.return_value = resources.Settings()
        client.get_settings(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_settings_flattened_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_settings(contact_center_insights.GetSettingsRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_settings_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_settings), '__call__') as call:
        call.return_value = resources.Settings()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Settings())
        response = await client.get_settings(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_settings_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_settings(contact_center_insights.GetSettingsRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.UpdateSettingsRequest, dict])
def test_update_settings(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_settings), '__call__') as call:
        call.return_value = resources.Settings(name='name_value', language_code='language_code_value')
        response = client.update_settings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateSettingsRequest()
    assert isinstance(response, resources.Settings)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'

def test_update_settings_empty_call():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_settings), '__call__') as call:
        client.update_settings()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateSettingsRequest()

@pytest.mark.asyncio
async def test_update_settings_async(transport: str='grpc_asyncio', request_type=contact_center_insights.UpdateSettingsRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_settings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Settings(name='name_value', language_code='language_code_value'))
        response = await client.update_settings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateSettingsRequest()
    assert isinstance(response, resources.Settings)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'

@pytest.mark.asyncio
async def test_update_settings_async_from_dict():
    await test_update_settings_async(request_type=dict)

def test_update_settings_field_headers():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UpdateSettingsRequest()
    request.settings.name = 'name_value'
    with mock.patch.object(type(client.transport.update_settings), '__call__') as call:
        call.return_value = resources.Settings()
        client.update_settings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'settings.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_settings_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UpdateSettingsRequest()
    request.settings.name = 'name_value'
    with mock.patch.object(type(client.transport.update_settings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Settings())
        await client.update_settings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'settings.name=name_value') in kw['metadata']

def test_update_settings_flattened():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_settings), '__call__') as call:
        call.return_value = resources.Settings()
        client.update_settings(settings=resources.Settings(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].settings
        mock_val = resources.Settings(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_settings_flattened_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_settings(contact_center_insights.UpdateSettingsRequest(), settings=resources.Settings(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_settings_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_settings), '__call__') as call:
        call.return_value = resources.Settings()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Settings())
        response = await client.update_settings(settings=resources.Settings(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].settings
        mock_val = resources.Settings(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_settings_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_settings(contact_center_insights.UpdateSettingsRequest(), settings=resources.Settings(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [contact_center_insights.CreateViewRequest, dict])
def test_create_view(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_view), '__call__') as call:
        call.return_value = resources.View(name='name_value', display_name='display_name_value', value='value_value')
        response = client.create_view(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreateViewRequest()
    assert isinstance(response, resources.View)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.value == 'value_value'

def test_create_view_empty_call():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_view), '__call__') as call:
        client.create_view()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreateViewRequest()

@pytest.mark.asyncio
async def test_create_view_async(transport: str='grpc_asyncio', request_type=contact_center_insights.CreateViewRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_view), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.View(name='name_value', display_name='display_name_value', value='value_value'))
        response = await client.create_view(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.CreateViewRequest()
    assert isinstance(response, resources.View)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.value == 'value_value'

@pytest.mark.asyncio
async def test_create_view_async_from_dict():
    await test_create_view_async(request_type=dict)

def test_create_view_field_headers():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.CreateViewRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_view), '__call__') as call:
        call.return_value = resources.View()
        client.create_view(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_view_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.CreateViewRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_view), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.View())
        await client.create_view(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_view_flattened():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_view), '__call__') as call:
        call.return_value = resources.View()
        client.create_view(parent='parent_value', view=resources.View(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].view
        mock_val = resources.View(name='name_value')
        assert arg == mock_val

def test_create_view_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_view(contact_center_insights.CreateViewRequest(), parent='parent_value', view=resources.View(name='name_value'))

@pytest.mark.asyncio
async def test_create_view_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_view), '__call__') as call:
        call.return_value = resources.View()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.View())
        response = await client.create_view(parent='parent_value', view=resources.View(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].view
        mock_val = resources.View(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_view_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_view(contact_center_insights.CreateViewRequest(), parent='parent_value', view=resources.View(name='name_value'))

@pytest.mark.parametrize('request_type', [contact_center_insights.GetViewRequest, dict])
def test_get_view(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_view), '__call__') as call:
        call.return_value = resources.View(name='name_value', display_name='display_name_value', value='value_value')
        response = client.get_view(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetViewRequest()
    assert isinstance(response, resources.View)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.value == 'value_value'

def test_get_view_empty_call():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_view), '__call__') as call:
        client.get_view()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetViewRequest()

@pytest.mark.asyncio
async def test_get_view_async(transport: str='grpc_asyncio', request_type=contact_center_insights.GetViewRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_view), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.View(name='name_value', display_name='display_name_value', value='value_value'))
        response = await client.get_view(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.GetViewRequest()
    assert isinstance(response, resources.View)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.value == 'value_value'

@pytest.mark.asyncio
async def test_get_view_async_from_dict():
    await test_get_view_async(request_type=dict)

def test_get_view_field_headers():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.GetViewRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_view), '__call__') as call:
        call.return_value = resources.View()
        client.get_view(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_view_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.GetViewRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_view), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.View())
        await client.get_view(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_view_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_view), '__call__') as call:
        call.return_value = resources.View()
        client.get_view(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_view_flattened_error():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_view(contact_center_insights.GetViewRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_view_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_view), '__call__') as call:
        call.return_value = resources.View()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.View())
        response = await client.get_view(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_view_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_view(contact_center_insights.GetViewRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.ListViewsRequest, dict])
def test_list_views(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_views), '__call__') as call:
        call.return_value = contact_center_insights.ListViewsResponse(next_page_token='next_page_token_value')
        response = client.list_views(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListViewsRequest()
    assert isinstance(response, pagers.ListViewsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_views_empty_call():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_views), '__call__') as call:
        client.list_views()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListViewsRequest()

@pytest.mark.asyncio
async def test_list_views_async(transport: str='grpc_asyncio', request_type=contact_center_insights.ListViewsRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_views), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListViewsResponse(next_page_token='next_page_token_value'))
        response = await client.list_views(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.ListViewsRequest()
    assert isinstance(response, pagers.ListViewsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_views_async_from_dict():
    await test_list_views_async(request_type=dict)

def test_list_views_field_headers():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.ListViewsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_views), '__call__') as call:
        call.return_value = contact_center_insights.ListViewsResponse()
        client.list_views(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_views_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.ListViewsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_views), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListViewsResponse())
        await client.list_views(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_views_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_views), '__call__') as call:
        call.return_value = contact_center_insights.ListViewsResponse()
        client.list_views(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_views_flattened_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_views(contact_center_insights.ListViewsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_views_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_views), '__call__') as call:
        call.return_value = contact_center_insights.ListViewsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(contact_center_insights.ListViewsResponse())
        response = await client.list_views(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_views_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_views(contact_center_insights.ListViewsRequest(), parent='parent_value')

def test_list_views_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_views), '__call__') as call:
        call.side_effect = (contact_center_insights.ListViewsResponse(views=[resources.View(), resources.View(), resources.View()], next_page_token='abc'), contact_center_insights.ListViewsResponse(views=[], next_page_token='def'), contact_center_insights.ListViewsResponse(views=[resources.View()], next_page_token='ghi'), contact_center_insights.ListViewsResponse(views=[resources.View(), resources.View()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_views(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.View) for i in results))

def test_list_views_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_views), '__call__') as call:
        call.side_effect = (contact_center_insights.ListViewsResponse(views=[resources.View(), resources.View(), resources.View()], next_page_token='abc'), contact_center_insights.ListViewsResponse(views=[], next_page_token='def'), contact_center_insights.ListViewsResponse(views=[resources.View()], next_page_token='ghi'), contact_center_insights.ListViewsResponse(views=[resources.View(), resources.View()]), RuntimeError)
        pages = list(client.list_views(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_views_async_pager():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_views), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (contact_center_insights.ListViewsResponse(views=[resources.View(), resources.View(), resources.View()], next_page_token='abc'), contact_center_insights.ListViewsResponse(views=[], next_page_token='def'), contact_center_insights.ListViewsResponse(views=[resources.View()], next_page_token='ghi'), contact_center_insights.ListViewsResponse(views=[resources.View(), resources.View()]), RuntimeError)
        async_pager = await client.list_views(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.View) for i in responses))

@pytest.mark.asyncio
async def test_list_views_async_pages():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_views), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (contact_center_insights.ListViewsResponse(views=[resources.View(), resources.View(), resources.View()], next_page_token='abc'), contact_center_insights.ListViewsResponse(views=[], next_page_token='def'), contact_center_insights.ListViewsResponse(views=[resources.View()], next_page_token='ghi'), contact_center_insights.ListViewsResponse(views=[resources.View(), resources.View()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_views(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [contact_center_insights.UpdateViewRequest, dict])
def test_update_view(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_view), '__call__') as call:
        call.return_value = resources.View(name='name_value', display_name='display_name_value', value='value_value')
        response = client.update_view(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateViewRequest()
    assert isinstance(response, resources.View)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.value == 'value_value'

def test_update_view_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_view), '__call__') as call:
        client.update_view()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateViewRequest()

@pytest.mark.asyncio
async def test_update_view_async(transport: str='grpc_asyncio', request_type=contact_center_insights.UpdateViewRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_view), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.View(name='name_value', display_name='display_name_value', value='value_value'))
        response = await client.update_view(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.UpdateViewRequest()
    assert isinstance(response, resources.View)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.value == 'value_value'

@pytest.mark.asyncio
async def test_update_view_async_from_dict():
    await test_update_view_async(request_type=dict)

def test_update_view_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UpdateViewRequest()
    request.view.name = 'name_value'
    with mock.patch.object(type(client.transport.update_view), '__call__') as call:
        call.return_value = resources.View()
        client.update_view(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'view.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_view_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.UpdateViewRequest()
    request.view.name = 'name_value'
    with mock.patch.object(type(client.transport.update_view), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.View())
        await client.update_view(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'view.name=name_value') in kw['metadata']

def test_update_view_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_view), '__call__') as call:
        call.return_value = resources.View()
        client.update_view(view=resources.View(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].view
        mock_val = resources.View(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_view_flattened_error():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_view(contact_center_insights.UpdateViewRequest(), view=resources.View(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_view_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_view), '__call__') as call:
        call.return_value = resources.View()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.View())
        response = await client.update_view(view=resources.View(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].view
        mock_val = resources.View(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_view_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_view(contact_center_insights.UpdateViewRequest(), view=resources.View(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [contact_center_insights.DeleteViewRequest, dict])
def test_delete_view(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_view), '__call__') as call:
        call.return_value = None
        response = client.delete_view(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteViewRequest()
    assert response is None

def test_delete_view_empty_call():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_view), '__call__') as call:
        client.delete_view()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteViewRequest()

@pytest.mark.asyncio
async def test_delete_view_async(transport: str='grpc_asyncio', request_type=contact_center_insights.DeleteViewRequest):
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_view), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_view(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == contact_center_insights.DeleteViewRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_view_async_from_dict():
    await test_delete_view_async(request_type=dict)

def test_delete_view_field_headers():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.DeleteViewRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_view), '__call__') as call:
        call.return_value = None
        client.delete_view(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_view_field_headers_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = contact_center_insights.DeleteViewRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_view), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_view(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_view_flattened():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_view), '__call__') as call:
        call.return_value = None
        client.delete_view(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_view_flattened_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_view(contact_center_insights.DeleteViewRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_view_flattened_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_view), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_view(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_view_flattened_error_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_view(contact_center_insights.DeleteViewRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [contact_center_insights.CreateConversationRequest, dict])
def test_create_conversation_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['conversation'] = {'call_metadata': {'customer_channel': 1706, 'agent_channel': 1351}, 'expire_time': {'seconds': 751, 'nanos': 543}, 'ttl': {'seconds': 751, 'nanos': 543}, 'name': 'name_value', 'data_source': {'gcs_source': {'audio_uri': 'audio_uri_value', 'transcript_uri': 'transcript_uri_value'}, 'dialogflow_source': {'dialogflow_conversation': 'dialogflow_conversation_value', 'audio_uri': 'audio_uri_value'}}, 'create_time': {}, 'update_time': {}, 'start_time': {}, 'language_code': 'language_code_value', 'agent_id': 'agent_id_value', 'labels': {}, 'transcript': {'transcript_segments': [{'message_time': {}, 'text': 'text_value', 'confidence': 0.1038, 'words': [{'start_offset': {}, 'end_offset': {}, 'word': 'word_value', 'confidence': 0.1038}], 'language_code': 'language_code_value', 'channel_tag': 1140, 'segment_participant': {'dialogflow_participant_name': 'dialogflow_participant_name_value', 'user_id': 'user_id_value', 'dialogflow_participant': 'dialogflow_participant_value', 'obfuscated_external_user_id': 'obfuscated_external_user_id_value', 'role': 1}, 'dialogflow_segment_metadata': {'smart_reply_allowlist_covered': True}, 'sentiment': {'magnitude': 0.9580000000000001, 'score': 0.54}}]}, 'medium': 1, 'duration': {}, 'turn_count': 1105, 'latest_analysis': {'name': 'name_value', 'request_time': {}, 'create_time': {}, 'analysis_result': {'call_analysis_metadata': {'annotations': [{'interruption_data': {}, 'sentiment_data': {}, 'silence_data': {}, 'hold_data': {}, 'entity_mention_data': {'entity_unique_id': 'entity_unique_id_value', 'type_': 1, 'sentiment': {}}, 'intent_match_data': {'intent_unique_id': 'intent_unique_id_value'}, 'phrase_match_data': {'phrase_matcher': 'phrase_matcher_value', 'display_name': 'display_name_value'}, 'issue_match_data': {'issue_assignment': {'issue': 'issue_value', 'score': 0.54, 'display_name': 'display_name_value'}}, 'channel_tag': 1140, 'annotation_start_boundary': {'word_index': 1075, 'transcript_index': 1729}, 'annotation_end_boundary': {}}], 'entities': {}, 'sentiments': [{'channel_tag': 1140, 'sentiment_data': {}}], 'intents': {}, 'phrase_matchers': {}, 'issue_model_result': {'issue_model': 'issue_model_value', 'issues': {}}}, 'end_time': {}}, 'annotator_selector': {'run_interruption_annotator': True, 'run_silence_annotator': True, 'run_phrase_matcher_annotator': True, 'phrase_matchers': ['phrase_matchers_value1', 'phrase_matchers_value2'], 'run_sentiment_annotator': True, 'run_entity_annotator': True, 'run_intent_annotator': True, 'run_issue_model_annotator': True, 'issue_models': ['issue_models_value1', 'issue_models_value2'], 'run_summarization_annotator': True, 'summarization_config': {'conversation_profile': 'conversation_profile_value', 'summarization_model': 1}}}, 'latest_summary': {'text': 'text_value', 'text_sections': {}, 'confidence': 0.1038, 'metadata': {}, 'answer_record': 'answer_record_value', 'conversation_model': 'conversation_model_value'}, 'runtime_annotations': [{'article_suggestion': {'title': 'title_value', 'uri': 'uri_value', 'confidence_score': 0.1673, 'metadata': {}, 'query_record': 'query_record_value', 'source': 'source_value'}, 'faq_answer': {'answer': 'answer_value', 'confidence_score': 0.1673, 'question': 'question_value', 'metadata': {}, 'query_record': 'query_record_value', 'source': 'source_value'}, 'smart_reply': {'reply': 'reply_value', 'confidence_score': 0.1673, 'metadata': {}, 'query_record': 'query_record_value'}, 'smart_compose_suggestion': {'suggestion': 'suggestion_value', 'confidence_score': 0.1673, 'metadata': {}, 'query_record': 'query_record_value'}, 'dialogflow_interaction': {'dialogflow_intent_id': 'dialogflow_intent_id_value', 'confidence': 0.1038}, 'conversation_summarization_suggestion': {}, 'annotation_id': 'annotation_id_value', 'create_time': {}, 'start_boundary': {}, 'end_boundary': {}, 'answer_feedback': {'correctness_level': 1, 'clicked': True, 'displayed': True}}], 'dialogflow_intents': {}, 'obfuscated_user_id': 'obfuscated_user_id_value'}
    test_field = contact_center_insights.CreateConversationRequest.meta.fields['conversation']

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
    for (field, value) in request_init['conversation'].items():
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
                for i in range(0, len(request_init['conversation'][field])):
                    del request_init['conversation'][field][i][subfield]
            else:
                del request_init['conversation'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Conversation(name='name_value', language_code='language_code_value', agent_id='agent_id_value', medium=resources.Conversation.Medium.PHONE_CALL, turn_count=1105, obfuscated_user_id='obfuscated_user_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Conversation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_conversation(request)
    assert isinstance(response, resources.Conversation)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'
    assert response.agent_id == 'agent_id_value'
    assert response.medium == resources.Conversation.Medium.PHONE_CALL
    assert response.turn_count == 1105
    assert response.obfuscated_user_id == 'obfuscated_user_id_value'

def test_create_conversation_rest_required_fields(request_type=contact_center_insights.CreateConversationRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_conversation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_conversation._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('conversation_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Conversation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Conversation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_conversation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_conversation_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_conversation._get_unset_required_fields({})
    assert set(unset_fields) == set(('conversationId',)) & set(('parent', 'conversation'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_conversation_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_create_conversation') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_create_conversation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.CreateConversationRequest.pb(contact_center_insights.CreateConversationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Conversation.to_json(resources.Conversation())
        request = contact_center_insights.CreateConversationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Conversation()
        client.create_conversation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_conversation_rest_bad_request(transport: str='rest', request_type=contact_center_insights.CreateConversationRequest):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_conversation(request)

def test_create_conversation_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Conversation()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', conversation=resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706)), conversation_id='conversation_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Conversation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_conversation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/conversations' % client.transport._host, args[1])

def test_create_conversation_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_conversation(contact_center_insights.CreateConversationRequest(), parent='parent_value', conversation=resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706)), conversation_id='conversation_id_value')

def test_create_conversation_rest_error():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.UploadConversationRequest, dict])
def test_upload_conversation_rest(request_type):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.upload_conversation(request)
    assert response.operation.name == 'operations/spam'

def test_upload_conversation_rest_required_fields(request_type=contact_center_insights.UploadConversationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).upload_conversation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).upload_conversation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.upload_conversation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_upload_conversation_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.upload_conversation._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'conversation'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_upload_conversation_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_upload_conversation') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_upload_conversation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.UploadConversationRequest.pb(contact_center_insights.UploadConversationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = contact_center_insights.UploadConversationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.upload_conversation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_upload_conversation_rest_bad_request(transport: str='rest', request_type=contact_center_insights.UploadConversationRequest):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.upload_conversation(request)

def test_upload_conversation_rest_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.UpdateConversationRequest, dict])
def test_update_conversation_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'conversation': {'name': 'projects/sample1/locations/sample2/conversations/sample3'}}
    request_init['conversation'] = {'call_metadata': {'customer_channel': 1706, 'agent_channel': 1351}, 'expire_time': {'seconds': 751, 'nanos': 543}, 'ttl': {'seconds': 751, 'nanos': 543}, 'name': 'projects/sample1/locations/sample2/conversations/sample3', 'data_source': {'gcs_source': {'audio_uri': 'audio_uri_value', 'transcript_uri': 'transcript_uri_value'}, 'dialogflow_source': {'dialogflow_conversation': 'dialogflow_conversation_value', 'audio_uri': 'audio_uri_value'}}, 'create_time': {}, 'update_time': {}, 'start_time': {}, 'language_code': 'language_code_value', 'agent_id': 'agent_id_value', 'labels': {}, 'transcript': {'transcript_segments': [{'message_time': {}, 'text': 'text_value', 'confidence': 0.1038, 'words': [{'start_offset': {}, 'end_offset': {}, 'word': 'word_value', 'confidence': 0.1038}], 'language_code': 'language_code_value', 'channel_tag': 1140, 'segment_participant': {'dialogflow_participant_name': 'dialogflow_participant_name_value', 'user_id': 'user_id_value', 'dialogflow_participant': 'dialogflow_participant_value', 'obfuscated_external_user_id': 'obfuscated_external_user_id_value', 'role': 1}, 'dialogflow_segment_metadata': {'smart_reply_allowlist_covered': True}, 'sentiment': {'magnitude': 0.9580000000000001, 'score': 0.54}}]}, 'medium': 1, 'duration': {}, 'turn_count': 1105, 'latest_analysis': {'name': 'name_value', 'request_time': {}, 'create_time': {}, 'analysis_result': {'call_analysis_metadata': {'annotations': [{'interruption_data': {}, 'sentiment_data': {}, 'silence_data': {}, 'hold_data': {}, 'entity_mention_data': {'entity_unique_id': 'entity_unique_id_value', 'type_': 1, 'sentiment': {}}, 'intent_match_data': {'intent_unique_id': 'intent_unique_id_value'}, 'phrase_match_data': {'phrase_matcher': 'phrase_matcher_value', 'display_name': 'display_name_value'}, 'issue_match_data': {'issue_assignment': {'issue': 'issue_value', 'score': 0.54, 'display_name': 'display_name_value'}}, 'channel_tag': 1140, 'annotation_start_boundary': {'word_index': 1075, 'transcript_index': 1729}, 'annotation_end_boundary': {}}], 'entities': {}, 'sentiments': [{'channel_tag': 1140, 'sentiment_data': {}}], 'intents': {}, 'phrase_matchers': {}, 'issue_model_result': {'issue_model': 'issue_model_value', 'issues': {}}}, 'end_time': {}}, 'annotator_selector': {'run_interruption_annotator': True, 'run_silence_annotator': True, 'run_phrase_matcher_annotator': True, 'phrase_matchers': ['phrase_matchers_value1', 'phrase_matchers_value2'], 'run_sentiment_annotator': True, 'run_entity_annotator': True, 'run_intent_annotator': True, 'run_issue_model_annotator': True, 'issue_models': ['issue_models_value1', 'issue_models_value2'], 'run_summarization_annotator': True, 'summarization_config': {'conversation_profile': 'conversation_profile_value', 'summarization_model': 1}}}, 'latest_summary': {'text': 'text_value', 'text_sections': {}, 'confidence': 0.1038, 'metadata': {}, 'answer_record': 'answer_record_value', 'conversation_model': 'conversation_model_value'}, 'runtime_annotations': [{'article_suggestion': {'title': 'title_value', 'uri': 'uri_value', 'confidence_score': 0.1673, 'metadata': {}, 'query_record': 'query_record_value', 'source': 'source_value'}, 'faq_answer': {'answer': 'answer_value', 'confidence_score': 0.1673, 'question': 'question_value', 'metadata': {}, 'query_record': 'query_record_value', 'source': 'source_value'}, 'smart_reply': {'reply': 'reply_value', 'confidence_score': 0.1673, 'metadata': {}, 'query_record': 'query_record_value'}, 'smart_compose_suggestion': {'suggestion': 'suggestion_value', 'confidence_score': 0.1673, 'metadata': {}, 'query_record': 'query_record_value'}, 'dialogflow_interaction': {'dialogflow_intent_id': 'dialogflow_intent_id_value', 'confidence': 0.1038}, 'conversation_summarization_suggestion': {}, 'annotation_id': 'annotation_id_value', 'create_time': {}, 'start_boundary': {}, 'end_boundary': {}, 'answer_feedback': {'correctness_level': 1, 'clicked': True, 'displayed': True}}], 'dialogflow_intents': {}, 'obfuscated_user_id': 'obfuscated_user_id_value'}
    test_field = contact_center_insights.UpdateConversationRequest.meta.fields['conversation']

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
    for (field, value) in request_init['conversation'].items():
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
                for i in range(0, len(request_init['conversation'][field])):
                    del request_init['conversation'][field][i][subfield]
            else:
                del request_init['conversation'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Conversation(name='name_value', language_code='language_code_value', agent_id='agent_id_value', medium=resources.Conversation.Medium.PHONE_CALL, turn_count=1105, obfuscated_user_id='obfuscated_user_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Conversation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_conversation(request)
    assert isinstance(response, resources.Conversation)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'
    assert response.agent_id == 'agent_id_value'
    assert response.medium == resources.Conversation.Medium.PHONE_CALL
    assert response.turn_count == 1105
    assert response.obfuscated_user_id == 'obfuscated_user_id_value'

def test_update_conversation_rest_required_fields(request_type=contact_center_insights.UpdateConversationRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_conversation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_conversation._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Conversation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Conversation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_conversation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_conversation_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_conversation._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('conversation',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_conversation_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_update_conversation') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_update_conversation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.UpdateConversationRequest.pb(contact_center_insights.UpdateConversationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Conversation.to_json(resources.Conversation())
        request = contact_center_insights.UpdateConversationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Conversation()
        client.update_conversation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_conversation_rest_bad_request(transport: str='rest', request_type=contact_center_insights.UpdateConversationRequest):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'conversation': {'name': 'projects/sample1/locations/sample2/conversations/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_conversation(request)

def test_update_conversation_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Conversation()
        sample_request = {'conversation': {'name': 'projects/sample1/locations/sample2/conversations/sample3'}}
        mock_args = dict(conversation=resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706)), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Conversation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_conversation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{conversation.name=projects/*/locations/*/conversations/*}' % client.transport._host, args[1])

def test_update_conversation_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_conversation(contact_center_insights.UpdateConversationRequest(), conversation=resources.Conversation(call_metadata=resources.Conversation.CallMetadata(customer_channel=1706)), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_conversation_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.GetConversationRequest, dict])
def test_get_conversation_rest(request_type):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/conversations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Conversation(name='name_value', language_code='language_code_value', agent_id='agent_id_value', medium=resources.Conversation.Medium.PHONE_CALL, turn_count=1105, obfuscated_user_id='obfuscated_user_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Conversation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_conversation(request)
    assert isinstance(response, resources.Conversation)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'
    assert response.agent_id == 'agent_id_value'
    assert response.medium == resources.Conversation.Medium.PHONE_CALL
    assert response.turn_count == 1105
    assert response.obfuscated_user_id == 'obfuscated_user_id_value'

def test_get_conversation_rest_required_fields(request_type=contact_center_insights.GetConversationRequest):
    if False:
        return 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_conversation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_conversation._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('view',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Conversation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Conversation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_conversation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_conversation_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_conversation._get_unset_required_fields({})
    assert set(unset_fields) == set(('view',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_conversation_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_get_conversation') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_get_conversation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.GetConversationRequest.pb(contact_center_insights.GetConversationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Conversation.to_json(resources.Conversation())
        request = contact_center_insights.GetConversationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Conversation()
        client.get_conversation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_conversation_rest_bad_request(transport: str='rest', request_type=contact_center_insights.GetConversationRequest):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/conversations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_conversation(request)

def test_get_conversation_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Conversation()
        sample_request = {'name': 'projects/sample1/locations/sample2/conversations/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Conversation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_conversation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/conversations/*}' % client.transport._host, args[1])

def test_get_conversation_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_conversation(contact_center_insights.GetConversationRequest(), name='name_value')

def test_get_conversation_rest_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.ListConversationsRequest, dict])
def test_list_conversations_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.ListConversationsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.ListConversationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_conversations(request)
    assert isinstance(response, pagers.ListConversationsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_conversations_rest_required_fields(request_type=contact_center_insights.ListConversationsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_conversations._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_conversations._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token', 'view'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = contact_center_insights.ListConversationsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = contact_center_insights.ListConversationsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_conversations(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_conversations_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_conversations._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken', 'view')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_conversations_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_list_conversations') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_list_conversations') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.ListConversationsRequest.pb(contact_center_insights.ListConversationsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = contact_center_insights.ListConversationsResponse.to_json(contact_center_insights.ListConversationsResponse())
        request = contact_center_insights.ListConversationsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = contact_center_insights.ListConversationsResponse()
        client.list_conversations(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_conversations_rest_bad_request(transport: str='rest', request_type=contact_center_insights.ListConversationsRequest):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_conversations(request)

def test_list_conversations_rest_flattened():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.ListConversationsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.ListConversationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_conversations(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/conversations' % client.transport._host, args[1])

def test_list_conversations_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_conversations(contact_center_insights.ListConversationsRequest(), parent='parent_value')

def test_list_conversations_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation(), resources.Conversation(), resources.Conversation()], next_page_token='abc'), contact_center_insights.ListConversationsResponse(conversations=[], next_page_token='def'), contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation()], next_page_token='ghi'), contact_center_insights.ListConversationsResponse(conversations=[resources.Conversation(), resources.Conversation()]))
        response = response + response
        response = tuple((contact_center_insights.ListConversationsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_conversations(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Conversation) for i in results))
        pages = list(client.list_conversations(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [contact_center_insights.DeleteConversationRequest, dict])
def test_delete_conversation_rest(request_type):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/conversations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_conversation(request)
    assert response is None

def test_delete_conversation_rest_required_fields(request_type=contact_center_insights.DeleteConversationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_conversation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_conversation._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('force',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_conversation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_conversation_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_conversation._get_unset_required_fields({})
    assert set(unset_fields) == set(('force',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_conversation_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_delete_conversation') as pre:
        pre.assert_not_called()
        pb_message = contact_center_insights.DeleteConversationRequest.pb(contact_center_insights.DeleteConversationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = contact_center_insights.DeleteConversationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_conversation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_conversation_rest_bad_request(transport: str='rest', request_type=contact_center_insights.DeleteConversationRequest):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/conversations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_conversation(request)

def test_delete_conversation_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/conversations/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_conversation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/conversations/*}' % client.transport._host, args[1])

def test_delete_conversation_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_conversation(contact_center_insights.DeleteConversationRequest(), name='name_value')

def test_delete_conversation_rest_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.CreateAnalysisRequest, dict])
def test_create_analysis_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/conversations/sample3'}
    request_init['analysis'] = {'name': 'name_value', 'request_time': {'seconds': 751, 'nanos': 543}, 'create_time': {}, 'analysis_result': {'call_analysis_metadata': {'annotations': [{'interruption_data': {}, 'sentiment_data': {'magnitude': 0.9580000000000001, 'score': 0.54}, 'silence_data': {}, 'hold_data': {}, 'entity_mention_data': {'entity_unique_id': 'entity_unique_id_value', 'type_': 1, 'sentiment': {}}, 'intent_match_data': {'intent_unique_id': 'intent_unique_id_value'}, 'phrase_match_data': {'phrase_matcher': 'phrase_matcher_value', 'display_name': 'display_name_value'}, 'issue_match_data': {'issue_assignment': {'issue': 'issue_value', 'score': 0.54, 'display_name': 'display_name_value'}}, 'channel_tag': 1140, 'annotation_start_boundary': {'word_index': 1075, 'transcript_index': 1729}, 'annotation_end_boundary': {}}], 'entities': {}, 'sentiments': [{'channel_tag': 1140, 'sentiment_data': {}}], 'intents': {}, 'phrase_matchers': {}, 'issue_model_result': {'issue_model': 'issue_model_value', 'issues': {}}}, 'end_time': {}}, 'annotator_selector': {'run_interruption_annotator': True, 'run_silence_annotator': True, 'run_phrase_matcher_annotator': True, 'phrase_matchers': ['phrase_matchers_value1', 'phrase_matchers_value2'], 'run_sentiment_annotator': True, 'run_entity_annotator': True, 'run_intent_annotator': True, 'run_issue_model_annotator': True, 'issue_models': ['issue_models_value1', 'issue_models_value2'], 'run_summarization_annotator': True, 'summarization_config': {'conversation_profile': 'conversation_profile_value', 'summarization_model': 1}}}
    test_field = contact_center_insights.CreateAnalysisRequest.meta.fields['analysis']

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
    for (field, value) in request_init['analysis'].items():
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
                for i in range(0, len(request_init['analysis'][field])):
                    del request_init['analysis'][field][i][subfield]
            else:
                del request_init['analysis'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_analysis(request)
    assert response.operation.name == 'operations/spam'

def test_create_analysis_rest_required_fields(request_type=contact_center_insights.CreateAnalysisRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_analysis._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_analysis._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_analysis(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_analysis_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_analysis._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'analysis'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_analysis_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_create_analysis') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_create_analysis') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.CreateAnalysisRequest.pb(contact_center_insights.CreateAnalysisRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = contact_center_insights.CreateAnalysisRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_analysis(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_analysis_rest_bad_request(transport: str='rest', request_type=contact_center_insights.CreateAnalysisRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/conversations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_analysis(request)

def test_create_analysis_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/conversations/sample3'}
        mock_args = dict(parent='parent_value', analysis=resources.Analysis(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_analysis(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/conversations/*}/analyses' % client.transport._host, args[1])

def test_create_analysis_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_analysis(contact_center_insights.CreateAnalysisRequest(), parent='parent_value', analysis=resources.Analysis(name='name_value'))

def test_create_analysis_rest_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.GetAnalysisRequest, dict])
def test_get_analysis_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/conversations/sample3/analyses/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Analysis(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Analysis.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_analysis(request)
    assert isinstance(response, resources.Analysis)
    assert response.name == 'name_value'

def test_get_analysis_rest_required_fields(request_type=contact_center_insights.GetAnalysisRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_analysis._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_analysis._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Analysis()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Analysis.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_analysis(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_analysis_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_analysis._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_analysis_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_get_analysis') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_get_analysis') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.GetAnalysisRequest.pb(contact_center_insights.GetAnalysisRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Analysis.to_json(resources.Analysis())
        request = contact_center_insights.GetAnalysisRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Analysis()
        client.get_analysis(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_analysis_rest_bad_request(transport: str='rest', request_type=contact_center_insights.GetAnalysisRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/conversations/sample3/analyses/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_analysis(request)

def test_get_analysis_rest_flattened():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Analysis()
        sample_request = {'name': 'projects/sample1/locations/sample2/conversations/sample3/analyses/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Analysis.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_analysis(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/conversations/*/analyses/*}' % client.transport._host, args[1])

def test_get_analysis_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_analysis(contact_center_insights.GetAnalysisRequest(), name='name_value')

def test_get_analysis_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.ListAnalysesRequest, dict])
def test_list_analyses_rest(request_type):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/conversations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.ListAnalysesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.ListAnalysesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_analyses(request)
    assert isinstance(response, pagers.ListAnalysesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_analyses_rest_required_fields(request_type=contact_center_insights.ListAnalysesRequest):
    if False:
        return 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_analyses._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_analyses._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = contact_center_insights.ListAnalysesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = contact_center_insights.ListAnalysesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_analyses(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_analyses_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_analyses._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_analyses_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_list_analyses') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_list_analyses') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.ListAnalysesRequest.pb(contact_center_insights.ListAnalysesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = contact_center_insights.ListAnalysesResponse.to_json(contact_center_insights.ListAnalysesResponse())
        request = contact_center_insights.ListAnalysesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = contact_center_insights.ListAnalysesResponse()
        client.list_analyses(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_analyses_rest_bad_request(transport: str='rest', request_type=contact_center_insights.ListAnalysesRequest):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/conversations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_analyses(request)

def test_list_analyses_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.ListAnalysesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/conversations/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.ListAnalysesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_analyses(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/conversations/*}/analyses' % client.transport._host, args[1])

def test_list_analyses_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_analyses(contact_center_insights.ListAnalysesRequest(), parent='parent_value')

def test_list_analyses_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis(), resources.Analysis(), resources.Analysis()], next_page_token='abc'), contact_center_insights.ListAnalysesResponse(analyses=[], next_page_token='def'), contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis()], next_page_token='ghi'), contact_center_insights.ListAnalysesResponse(analyses=[resources.Analysis(), resources.Analysis()]))
        response = response + response
        response = tuple((contact_center_insights.ListAnalysesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/conversations/sample3'}
        pager = client.list_analyses(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Analysis) for i in results))
        pages = list(client.list_analyses(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [contact_center_insights.DeleteAnalysisRequest, dict])
def test_delete_analysis_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/conversations/sample3/analyses/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_analysis(request)
    assert response is None

def test_delete_analysis_rest_required_fields(request_type=contact_center_insights.DeleteAnalysisRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_analysis._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_analysis._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_analysis(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_analysis_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_analysis._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_analysis_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_delete_analysis') as pre:
        pre.assert_not_called()
        pb_message = contact_center_insights.DeleteAnalysisRequest.pb(contact_center_insights.DeleteAnalysisRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = contact_center_insights.DeleteAnalysisRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_analysis(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_analysis_rest_bad_request(transport: str='rest', request_type=contact_center_insights.DeleteAnalysisRequest):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/conversations/sample3/analyses/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_analysis(request)

def test_delete_analysis_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/conversations/sample3/analyses/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_analysis(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/conversations/*/analyses/*}' % client.transport._host, args[1])

def test_delete_analysis_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_analysis(contact_center_insights.DeleteAnalysisRequest(), name='name_value')

def test_delete_analysis_rest_error():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.BulkAnalyzeConversationsRequest, dict])
def test_bulk_analyze_conversations_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.bulk_analyze_conversations(request)
    assert response.operation.name == 'operations/spam'

def test_bulk_analyze_conversations_rest_required_fields(request_type=contact_center_insights.BulkAnalyzeConversationsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['filter'] = ''
    request_init['analysis_percentage'] = 0.0
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).bulk_analyze_conversations._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['filter'] = 'filter_value'
    jsonified_request['analysisPercentage'] = 0.20170000000000002
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).bulk_analyze_conversations._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'filter' in jsonified_request
    assert jsonified_request['filter'] == 'filter_value'
    assert 'analysisPercentage' in jsonified_request
    assert jsonified_request['analysisPercentage'] == 0.20170000000000002
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.bulk_analyze_conversations(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_bulk_analyze_conversations_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.bulk_analyze_conversations._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'filter', 'analysisPercentage'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_bulk_analyze_conversations_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_bulk_analyze_conversations') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_bulk_analyze_conversations') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.BulkAnalyzeConversationsRequest.pb(contact_center_insights.BulkAnalyzeConversationsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = contact_center_insights.BulkAnalyzeConversationsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.bulk_analyze_conversations(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_bulk_analyze_conversations_rest_bad_request(transport: str='rest', request_type=contact_center_insights.BulkAnalyzeConversationsRequest):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.bulk_analyze_conversations(request)

def test_bulk_analyze_conversations_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', filter='filter_value', analysis_percentage=0.20170000000000002)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.bulk_analyze_conversations(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/conversations:bulkAnalyze' % client.transport._host, args[1])

def test_bulk_analyze_conversations_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.bulk_analyze_conversations(contact_center_insights.BulkAnalyzeConversationsRequest(), parent='parent_value', filter='filter_value', analysis_percentage=0.20170000000000002)

def test_bulk_analyze_conversations_rest_error():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.BulkDeleteConversationsRequest, dict])
def test_bulk_delete_conversations_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.bulk_delete_conversations(request)
    assert response.operation.name == 'operations/spam'

def test_bulk_delete_conversations_rest_required_fields(request_type=contact_center_insights.BulkDeleteConversationsRequest):
    if False:
        return 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).bulk_delete_conversations._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).bulk_delete_conversations._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.bulk_delete_conversations(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_bulk_delete_conversations_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.bulk_delete_conversations._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_bulk_delete_conversations_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_bulk_delete_conversations') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_bulk_delete_conversations') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.BulkDeleteConversationsRequest.pb(contact_center_insights.BulkDeleteConversationsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = contact_center_insights.BulkDeleteConversationsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.bulk_delete_conversations(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_bulk_delete_conversations_rest_bad_request(transport: str='rest', request_type=contact_center_insights.BulkDeleteConversationsRequest):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.bulk_delete_conversations(request)

def test_bulk_delete_conversations_rest_flattened():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', filter='filter_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.bulk_delete_conversations(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/conversations:bulkDelete' % client.transport._host, args[1])

def test_bulk_delete_conversations_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.bulk_delete_conversations(contact_center_insights.BulkDeleteConversationsRequest(), parent='parent_value', filter='filter_value')

def test_bulk_delete_conversations_rest_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.IngestConversationsRequest, dict])
def test_ingest_conversations_rest(request_type):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.ingest_conversations(request)
    assert response.operation.name == 'operations/spam'

def test_ingest_conversations_rest_required_fields(request_type=contact_center_insights.IngestConversationsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).ingest_conversations._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).ingest_conversations._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.ingest_conversations(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_ingest_conversations_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.ingest_conversations._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_ingest_conversations_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_ingest_conversations') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_ingest_conversations') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.IngestConversationsRequest.pb(contact_center_insights.IngestConversationsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = contact_center_insights.IngestConversationsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.ingest_conversations(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_ingest_conversations_rest_bad_request(transport: str='rest', request_type=contact_center_insights.IngestConversationsRequest):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.ingest_conversations(request)

def test_ingest_conversations_rest_flattened():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.ingest_conversations(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/conversations:ingest' % client.transport._host, args[1])

def test_ingest_conversations_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.ingest_conversations(contact_center_insights.IngestConversationsRequest(), parent='parent_value')

def test_ingest_conversations_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.ExportInsightsDataRequest, dict])
def test_export_insights_data_rest(request_type):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.export_insights_data(request)
    assert response.operation.name == 'operations/spam'

def test_export_insights_data_rest_required_fields(request_type=contact_center_insights.ExportInsightsDataRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).export_insights_data._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).export_insights_data._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.export_insights_data(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_export_insights_data_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.export_insights_data._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_export_insights_data_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_export_insights_data') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_export_insights_data') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.ExportInsightsDataRequest.pb(contact_center_insights.ExportInsightsDataRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = contact_center_insights.ExportInsightsDataRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.export_insights_data(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_export_insights_data_rest_bad_request(transport: str='rest', request_type=contact_center_insights.ExportInsightsDataRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.export_insights_data(request)

def test_export_insights_data_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.export_insights_data(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/insightsdata:export' % client.transport._host, args[1])

def test_export_insights_data_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.export_insights_data(contact_center_insights.ExportInsightsDataRequest(), parent='parent_value')

def test_export_insights_data_rest_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.CreateIssueModelRequest, dict])
def test_create_issue_model_rest(request_type):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['issue_model'] = {'name': 'name_value', 'display_name': 'display_name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'issue_count': 1201, 'state': 1, 'input_data_config': {'medium': 1, 'training_conversations_count': 3025, 'filter': 'filter_value'}, 'training_stats': {'analyzed_conversations_count': 3021, 'unclassified_conversations_count': 3439, 'issue_stats': {}}, 'model_type': 1, 'language_code': 'language_code_value'}
    test_field = contact_center_insights.CreateIssueModelRequest.meta.fields['issue_model']

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
    for (field, value) in request_init['issue_model'].items():
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
                for i in range(0, len(request_init['issue_model'][field])):
                    del request_init['issue_model'][field][i][subfield]
            else:
                del request_init['issue_model'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_issue_model(request)
    assert response.operation.name == 'operations/spam'

def test_create_issue_model_rest_required_fields(request_type=contact_center_insights.CreateIssueModelRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_issue_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_issue_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_issue_model(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_issue_model_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_issue_model._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'issueModel'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_issue_model_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_create_issue_model') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_create_issue_model') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.CreateIssueModelRequest.pb(contact_center_insights.CreateIssueModelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = contact_center_insights.CreateIssueModelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_issue_model(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_issue_model_rest_bad_request(transport: str='rest', request_type=contact_center_insights.CreateIssueModelRequest):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_issue_model(request)

def test_create_issue_model_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', issue_model=resources.IssueModel(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_issue_model(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/issueModels' % client.transport._host, args[1])

def test_create_issue_model_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_issue_model(contact_center_insights.CreateIssueModelRequest(), parent='parent_value', issue_model=resources.IssueModel(name='name_value'))

def test_create_issue_model_rest_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.UpdateIssueModelRequest, dict])
def test_update_issue_model_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'issue_model': {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}}
    request_init['issue_model'] = {'name': 'projects/sample1/locations/sample2/issueModels/sample3', 'display_name': 'display_name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'issue_count': 1201, 'state': 1, 'input_data_config': {'medium': 1, 'training_conversations_count': 3025, 'filter': 'filter_value'}, 'training_stats': {'analyzed_conversations_count': 3021, 'unclassified_conversations_count': 3439, 'issue_stats': {}}, 'model_type': 1, 'language_code': 'language_code_value'}
    test_field = contact_center_insights.UpdateIssueModelRequest.meta.fields['issue_model']

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
    for (field, value) in request_init['issue_model'].items():
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
                for i in range(0, len(request_init['issue_model'][field])):
                    del request_init['issue_model'][field][i][subfield]
            else:
                del request_init['issue_model'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.IssueModel(name='name_value', display_name='display_name_value', issue_count=1201, state=resources.IssueModel.State.UNDEPLOYED, model_type=resources.IssueModel.ModelType.TYPE_V1, language_code='language_code_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.IssueModel.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_issue_model(request)
    assert isinstance(response, resources.IssueModel)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.issue_count == 1201
    assert response.state == resources.IssueModel.State.UNDEPLOYED
    assert response.model_type == resources.IssueModel.ModelType.TYPE_V1
    assert response.language_code == 'language_code_value'

def test_update_issue_model_rest_required_fields(request_type=contact_center_insights.UpdateIssueModelRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_issue_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_issue_model._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.IssueModel()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.IssueModel.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_issue_model(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_issue_model_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_issue_model._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('issueModel',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_issue_model_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_update_issue_model') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_update_issue_model') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.UpdateIssueModelRequest.pb(contact_center_insights.UpdateIssueModelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.IssueModel.to_json(resources.IssueModel())
        request = contact_center_insights.UpdateIssueModelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.IssueModel()
        client.update_issue_model(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_issue_model_rest_bad_request(transport: str='rest', request_type=contact_center_insights.UpdateIssueModelRequest):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'issue_model': {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_issue_model(request)

def test_update_issue_model_rest_flattened():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.IssueModel()
        sample_request = {'issue_model': {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}}
        mock_args = dict(issue_model=resources.IssueModel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.IssueModel.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_issue_model(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{issue_model.name=projects/*/locations/*/issueModels/*}' % client.transport._host, args[1])

def test_update_issue_model_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_issue_model(contact_center_insights.UpdateIssueModelRequest(), issue_model=resources.IssueModel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_issue_model_rest_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.GetIssueModelRequest, dict])
def test_get_issue_model_rest(request_type):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.IssueModel(name='name_value', display_name='display_name_value', issue_count=1201, state=resources.IssueModel.State.UNDEPLOYED, model_type=resources.IssueModel.ModelType.TYPE_V1, language_code='language_code_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.IssueModel.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_issue_model(request)
    assert isinstance(response, resources.IssueModel)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.issue_count == 1201
    assert response.state == resources.IssueModel.State.UNDEPLOYED
    assert response.model_type == resources.IssueModel.ModelType.TYPE_V1
    assert response.language_code == 'language_code_value'

def test_get_issue_model_rest_required_fields(request_type=contact_center_insights.GetIssueModelRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_issue_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_issue_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.IssueModel()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.IssueModel.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_issue_model(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_issue_model_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_issue_model._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_issue_model_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_get_issue_model') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_get_issue_model') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.GetIssueModelRequest.pb(contact_center_insights.GetIssueModelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.IssueModel.to_json(resources.IssueModel())
        request = contact_center_insights.GetIssueModelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.IssueModel()
        client.get_issue_model(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_issue_model_rest_bad_request(transport: str='rest', request_type=contact_center_insights.GetIssueModelRequest):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_issue_model(request)

def test_get_issue_model_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.IssueModel()
        sample_request = {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.IssueModel.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_issue_model(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/issueModels/*}' % client.transport._host, args[1])

def test_get_issue_model_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_issue_model(contact_center_insights.GetIssueModelRequest(), name='name_value')

def test_get_issue_model_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.ListIssueModelsRequest, dict])
def test_list_issue_models_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.ListIssueModelsResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.ListIssueModelsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_issue_models(request)
    assert isinstance(response, contact_center_insights.ListIssueModelsResponse)

def test_list_issue_models_rest_required_fields(request_type=contact_center_insights.ListIssueModelsRequest):
    if False:
        return 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_issue_models._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_issue_models._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = contact_center_insights.ListIssueModelsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = contact_center_insights.ListIssueModelsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_issue_models(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_issue_models_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_issue_models._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_issue_models_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_list_issue_models') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_list_issue_models') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.ListIssueModelsRequest.pb(contact_center_insights.ListIssueModelsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = contact_center_insights.ListIssueModelsResponse.to_json(contact_center_insights.ListIssueModelsResponse())
        request = contact_center_insights.ListIssueModelsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = contact_center_insights.ListIssueModelsResponse()
        client.list_issue_models(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_issue_models_rest_bad_request(transport: str='rest', request_type=contact_center_insights.ListIssueModelsRequest):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_issue_models(request)

def test_list_issue_models_rest_flattened():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.ListIssueModelsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.ListIssueModelsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_issue_models(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/issueModels' % client.transport._host, args[1])

def test_list_issue_models_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_issue_models(contact_center_insights.ListIssueModelsRequest(), parent='parent_value')

def test_list_issue_models_rest_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.DeleteIssueModelRequest, dict])
def test_delete_issue_model_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_issue_model(request)
    assert response.operation.name == 'operations/spam'

def test_delete_issue_model_rest_required_fields(request_type=contact_center_insights.DeleteIssueModelRequest):
    if False:
        return 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_issue_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_issue_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_issue_model(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_issue_model_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_issue_model._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_issue_model_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_delete_issue_model') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_delete_issue_model') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.DeleteIssueModelRequest.pb(contact_center_insights.DeleteIssueModelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = contact_center_insights.DeleteIssueModelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_issue_model(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_issue_model_rest_bad_request(transport: str='rest', request_type=contact_center_insights.DeleteIssueModelRequest):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_issue_model(request)

def test_delete_issue_model_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_issue_model(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/issueModels/*}' % client.transport._host, args[1])

def test_delete_issue_model_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_issue_model(contact_center_insights.DeleteIssueModelRequest(), name='name_value')

def test_delete_issue_model_rest_error():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.DeployIssueModelRequest, dict])
def test_deploy_issue_model_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.deploy_issue_model(request)
    assert response.operation.name == 'operations/spam'

def test_deploy_issue_model_rest_required_fields(request_type=contact_center_insights.DeployIssueModelRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).deploy_issue_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).deploy_issue_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.deploy_issue_model(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_deploy_issue_model_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.deploy_issue_model._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_deploy_issue_model_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_deploy_issue_model') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_deploy_issue_model') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.DeployIssueModelRequest.pb(contact_center_insights.DeployIssueModelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = contact_center_insights.DeployIssueModelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.deploy_issue_model(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_deploy_issue_model_rest_bad_request(transport: str='rest', request_type=contact_center_insights.DeployIssueModelRequest):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.deploy_issue_model(request)

def test_deploy_issue_model_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.deploy_issue_model(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/issueModels/*}:deploy' % client.transport._host, args[1])

def test_deploy_issue_model_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.deploy_issue_model(contact_center_insights.DeployIssueModelRequest(), name='name_value')

def test_deploy_issue_model_rest_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.UndeployIssueModelRequest, dict])
def test_undeploy_issue_model_rest(request_type):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.undeploy_issue_model(request)
    assert response.operation.name == 'operations/spam'

def test_undeploy_issue_model_rest_required_fields(request_type=contact_center_insights.UndeployIssueModelRequest):
    if False:
        return 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).undeploy_issue_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).undeploy_issue_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.undeploy_issue_model(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_undeploy_issue_model_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.undeploy_issue_model._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_undeploy_issue_model_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_undeploy_issue_model') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_undeploy_issue_model') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.UndeployIssueModelRequest.pb(contact_center_insights.UndeployIssueModelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = contact_center_insights.UndeployIssueModelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.undeploy_issue_model(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_undeploy_issue_model_rest_bad_request(transport: str='rest', request_type=contact_center_insights.UndeployIssueModelRequest):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.undeploy_issue_model(request)

def test_undeploy_issue_model_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/issueModels/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.undeploy_issue_model(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/issueModels/*}:undeploy' % client.transport._host, args[1])

def test_undeploy_issue_model_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.undeploy_issue_model(contact_center_insights.UndeployIssueModelRequest(), name='name_value')

def test_undeploy_issue_model_rest_error():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.GetIssueRequest, dict])
def test_get_issue_rest(request_type):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/issueModels/sample3/issues/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Issue(name='name_value', display_name='display_name_value', sample_utterances=['sample_utterances_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Issue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_issue(request)
    assert isinstance(response, resources.Issue)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.sample_utterances == ['sample_utterances_value']

def test_get_issue_rest_required_fields(request_type=contact_center_insights.GetIssueRequest):
    if False:
        return 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_issue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_issue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Issue()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Issue.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_issue(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_issue_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_issue._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_issue_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_get_issue') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_get_issue') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.GetIssueRequest.pb(contact_center_insights.GetIssueRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Issue.to_json(resources.Issue())
        request = contact_center_insights.GetIssueRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Issue()
        client.get_issue(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_issue_rest_bad_request(transport: str='rest', request_type=contact_center_insights.GetIssueRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/issueModels/sample3/issues/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_issue(request)

def test_get_issue_rest_flattened():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Issue()
        sample_request = {'name': 'projects/sample1/locations/sample2/issueModels/sample3/issues/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Issue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_issue(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/issueModels/*/issues/*}' % client.transport._host, args[1])

def test_get_issue_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_issue(contact_center_insights.GetIssueRequest(), name='name_value')

def test_get_issue_rest_error():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.ListIssuesRequest, dict])
def test_list_issues_rest(request_type):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/issueModels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.ListIssuesResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.ListIssuesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_issues(request)
    assert isinstance(response, contact_center_insights.ListIssuesResponse)

def test_list_issues_rest_required_fields(request_type=contact_center_insights.ListIssuesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_issues._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_issues._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = contact_center_insights.ListIssuesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = contact_center_insights.ListIssuesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_issues(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_issues_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_issues._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_issues_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_list_issues') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_list_issues') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.ListIssuesRequest.pb(contact_center_insights.ListIssuesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = contact_center_insights.ListIssuesResponse.to_json(contact_center_insights.ListIssuesResponse())
        request = contact_center_insights.ListIssuesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = contact_center_insights.ListIssuesResponse()
        client.list_issues(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_issues_rest_bad_request(transport: str='rest', request_type=contact_center_insights.ListIssuesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/issueModels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_issues(request)

def test_list_issues_rest_flattened():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.ListIssuesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/issueModels/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.ListIssuesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_issues(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/issueModels/*}/issues' % client.transport._host, args[1])

def test_list_issues_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_issues(contact_center_insights.ListIssuesRequest(), parent='parent_value')

def test_list_issues_rest_error():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.UpdateIssueRequest, dict])
def test_update_issue_rest(request_type):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'issue': {'name': 'projects/sample1/locations/sample2/issueModels/sample3/issues/sample4'}}
    request_init['issue'] = {'name': 'projects/sample1/locations/sample2/issueModels/sample3/issues/sample4', 'display_name': 'display_name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'sample_utterances': ['sample_utterances_value1', 'sample_utterances_value2']}
    test_field = contact_center_insights.UpdateIssueRequest.meta.fields['issue']

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
    for (field, value) in request_init['issue'].items():
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
                for i in range(0, len(request_init['issue'][field])):
                    del request_init['issue'][field][i][subfield]
            else:
                del request_init['issue'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Issue(name='name_value', display_name='display_name_value', sample_utterances=['sample_utterances_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Issue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_issue(request)
    assert isinstance(response, resources.Issue)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.sample_utterances == ['sample_utterances_value']

def test_update_issue_rest_required_fields(request_type=contact_center_insights.UpdateIssueRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_issue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_issue._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Issue()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Issue.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_issue(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_issue_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_issue._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('issue',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_issue_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_update_issue') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_update_issue') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.UpdateIssueRequest.pb(contact_center_insights.UpdateIssueRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Issue.to_json(resources.Issue())
        request = contact_center_insights.UpdateIssueRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Issue()
        client.update_issue(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_issue_rest_bad_request(transport: str='rest', request_type=contact_center_insights.UpdateIssueRequest):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'issue': {'name': 'projects/sample1/locations/sample2/issueModels/sample3/issues/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_issue(request)

def test_update_issue_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Issue()
        sample_request = {'issue': {'name': 'projects/sample1/locations/sample2/issueModels/sample3/issues/sample4'}}
        mock_args = dict(issue=resources.Issue(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Issue.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_issue(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{issue.name=projects/*/locations/*/issueModels/*/issues/*}' % client.transport._host, args[1])

def test_update_issue_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_issue(contact_center_insights.UpdateIssueRequest(), issue=resources.Issue(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_issue_rest_error():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.DeleteIssueRequest, dict])
def test_delete_issue_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/issueModels/sample3/issues/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_issue(request)
    assert response is None

def test_delete_issue_rest_required_fields(request_type=contact_center_insights.DeleteIssueRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_issue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_issue._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_issue(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_issue_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_issue._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_issue_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_delete_issue') as pre:
        pre.assert_not_called()
        pb_message = contact_center_insights.DeleteIssueRequest.pb(contact_center_insights.DeleteIssueRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = contact_center_insights.DeleteIssueRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_issue(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_issue_rest_bad_request(transport: str='rest', request_type=contact_center_insights.DeleteIssueRequest):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/issueModels/sample3/issues/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_issue(request)

def test_delete_issue_rest_flattened():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/issueModels/sample3/issues/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_issue(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/issueModels/*/issues/*}' % client.transport._host, args[1])

def test_delete_issue_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_issue(contact_center_insights.DeleteIssueRequest(), name='name_value')

def test_delete_issue_rest_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.CalculateIssueModelStatsRequest, dict])
def test_calculate_issue_model_stats_rest(request_type):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'issue_model': 'projects/sample1/locations/sample2/issueModels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.CalculateIssueModelStatsResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.CalculateIssueModelStatsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.calculate_issue_model_stats(request)
    assert isinstance(response, contact_center_insights.CalculateIssueModelStatsResponse)

def test_calculate_issue_model_stats_rest_required_fields(request_type=contact_center_insights.CalculateIssueModelStatsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['issue_model'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).calculate_issue_model_stats._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['issueModel'] = 'issue_model_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).calculate_issue_model_stats._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'issueModel' in jsonified_request
    assert jsonified_request['issueModel'] == 'issue_model_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = contact_center_insights.CalculateIssueModelStatsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = contact_center_insights.CalculateIssueModelStatsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.calculate_issue_model_stats(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_calculate_issue_model_stats_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.calculate_issue_model_stats._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('issueModel',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_calculate_issue_model_stats_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_calculate_issue_model_stats') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_calculate_issue_model_stats') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.CalculateIssueModelStatsRequest.pb(contact_center_insights.CalculateIssueModelStatsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = contact_center_insights.CalculateIssueModelStatsResponse.to_json(contact_center_insights.CalculateIssueModelStatsResponse())
        request = contact_center_insights.CalculateIssueModelStatsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = contact_center_insights.CalculateIssueModelStatsResponse()
        client.calculate_issue_model_stats(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_calculate_issue_model_stats_rest_bad_request(transport: str='rest', request_type=contact_center_insights.CalculateIssueModelStatsRequest):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'issue_model': 'projects/sample1/locations/sample2/issueModels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.calculate_issue_model_stats(request)

def test_calculate_issue_model_stats_rest_flattened():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.CalculateIssueModelStatsResponse()
        sample_request = {'issue_model': 'projects/sample1/locations/sample2/issueModels/sample3'}
        mock_args = dict(issue_model='issue_model_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.CalculateIssueModelStatsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.calculate_issue_model_stats(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{issue_model=projects/*/locations/*/issueModels/*}:calculateIssueModelStats' % client.transport._host, args[1])

def test_calculate_issue_model_stats_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.calculate_issue_model_stats(contact_center_insights.CalculateIssueModelStatsRequest(), issue_model='issue_model_value')

def test_calculate_issue_model_stats_rest_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.CreatePhraseMatcherRequest, dict])
def test_create_phrase_matcher_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['phrase_matcher'] = {'name': 'name_value', 'revision_id': 'revision_id_value', 'version_tag': 'version_tag_value', 'revision_create_time': {'seconds': 751, 'nanos': 543}, 'display_name': 'display_name_value', 'type_': 1, 'active': True, 'phrase_match_rule_groups': [{'type_': 1, 'phrase_match_rules': [{'query': 'query_value', 'negated': True, 'config': {'exact_match_config': {'case_sensitive': True}}}]}], 'activation_update_time': {}, 'role_match': 1, 'update_time': {}}
    test_field = contact_center_insights.CreatePhraseMatcherRequest.meta.fields['phrase_matcher']

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
    for (field, value) in request_init['phrase_matcher'].items():
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
                for i in range(0, len(request_init['phrase_matcher'][field])):
                    del request_init['phrase_matcher'][field][i][subfield]
            else:
                del request_init['phrase_matcher'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.PhraseMatcher(name='name_value', revision_id='revision_id_value', version_tag='version_tag_value', display_name='display_name_value', type_=resources.PhraseMatcher.PhraseMatcherType.ALL_OF, active=True, role_match=resources.ConversationParticipant.Role.HUMAN_AGENT)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.PhraseMatcher.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_phrase_matcher(request)
    assert isinstance(response, resources.PhraseMatcher)
    assert response.name == 'name_value'
    assert response.revision_id == 'revision_id_value'
    assert response.version_tag == 'version_tag_value'
    assert response.display_name == 'display_name_value'
    assert response.type_ == resources.PhraseMatcher.PhraseMatcherType.ALL_OF
    assert response.active is True
    assert response.role_match == resources.ConversationParticipant.Role.HUMAN_AGENT

def test_create_phrase_matcher_rest_required_fields(request_type=contact_center_insights.CreatePhraseMatcherRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_phrase_matcher._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_phrase_matcher._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.PhraseMatcher()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.PhraseMatcher.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_phrase_matcher(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_phrase_matcher_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_phrase_matcher._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'phraseMatcher'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_phrase_matcher_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_create_phrase_matcher') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_create_phrase_matcher') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.CreatePhraseMatcherRequest.pb(contact_center_insights.CreatePhraseMatcherRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.PhraseMatcher.to_json(resources.PhraseMatcher())
        request = contact_center_insights.CreatePhraseMatcherRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.PhraseMatcher()
        client.create_phrase_matcher(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_phrase_matcher_rest_bad_request(transport: str='rest', request_type=contact_center_insights.CreatePhraseMatcherRequest):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_phrase_matcher(request)

def test_create_phrase_matcher_rest_flattened():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.PhraseMatcher()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', phrase_matcher=resources.PhraseMatcher(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.PhraseMatcher.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_phrase_matcher(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/phraseMatchers' % client.transport._host, args[1])

def test_create_phrase_matcher_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_phrase_matcher(contact_center_insights.CreatePhraseMatcherRequest(), parent='parent_value', phrase_matcher=resources.PhraseMatcher(name='name_value'))

def test_create_phrase_matcher_rest_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.GetPhraseMatcherRequest, dict])
def test_get_phrase_matcher_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/phraseMatchers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.PhraseMatcher(name='name_value', revision_id='revision_id_value', version_tag='version_tag_value', display_name='display_name_value', type_=resources.PhraseMatcher.PhraseMatcherType.ALL_OF, active=True, role_match=resources.ConversationParticipant.Role.HUMAN_AGENT)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.PhraseMatcher.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_phrase_matcher(request)
    assert isinstance(response, resources.PhraseMatcher)
    assert response.name == 'name_value'
    assert response.revision_id == 'revision_id_value'
    assert response.version_tag == 'version_tag_value'
    assert response.display_name == 'display_name_value'
    assert response.type_ == resources.PhraseMatcher.PhraseMatcherType.ALL_OF
    assert response.active is True
    assert response.role_match == resources.ConversationParticipant.Role.HUMAN_AGENT

def test_get_phrase_matcher_rest_required_fields(request_type=contact_center_insights.GetPhraseMatcherRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_phrase_matcher._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_phrase_matcher._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.PhraseMatcher()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.PhraseMatcher.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_phrase_matcher(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_phrase_matcher_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_phrase_matcher._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_phrase_matcher_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_get_phrase_matcher') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_get_phrase_matcher') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.GetPhraseMatcherRequest.pb(contact_center_insights.GetPhraseMatcherRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.PhraseMatcher.to_json(resources.PhraseMatcher())
        request = contact_center_insights.GetPhraseMatcherRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.PhraseMatcher()
        client.get_phrase_matcher(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_phrase_matcher_rest_bad_request(transport: str='rest', request_type=contact_center_insights.GetPhraseMatcherRequest):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/phraseMatchers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_phrase_matcher(request)

def test_get_phrase_matcher_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.PhraseMatcher()
        sample_request = {'name': 'projects/sample1/locations/sample2/phraseMatchers/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.PhraseMatcher.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_phrase_matcher(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/phraseMatchers/*}' % client.transport._host, args[1])

def test_get_phrase_matcher_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_phrase_matcher(contact_center_insights.GetPhraseMatcherRequest(), name='name_value')

def test_get_phrase_matcher_rest_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.ListPhraseMatchersRequest, dict])
def test_list_phrase_matchers_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.ListPhraseMatchersResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.ListPhraseMatchersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_phrase_matchers(request)
    assert isinstance(response, pagers.ListPhraseMatchersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_phrase_matchers_rest_required_fields(request_type=contact_center_insights.ListPhraseMatchersRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_phrase_matchers._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_phrase_matchers._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = contact_center_insights.ListPhraseMatchersResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = contact_center_insights.ListPhraseMatchersResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_phrase_matchers(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_phrase_matchers_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_phrase_matchers._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_phrase_matchers_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_list_phrase_matchers') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_list_phrase_matchers') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.ListPhraseMatchersRequest.pb(contact_center_insights.ListPhraseMatchersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = contact_center_insights.ListPhraseMatchersResponse.to_json(contact_center_insights.ListPhraseMatchersResponse())
        request = contact_center_insights.ListPhraseMatchersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = contact_center_insights.ListPhraseMatchersResponse()
        client.list_phrase_matchers(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_phrase_matchers_rest_bad_request(transport: str='rest', request_type=contact_center_insights.ListPhraseMatchersRequest):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_phrase_matchers(request)

def test_list_phrase_matchers_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.ListPhraseMatchersResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.ListPhraseMatchersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_phrase_matchers(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/phraseMatchers' % client.transport._host, args[1])

def test_list_phrase_matchers_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_phrase_matchers(contact_center_insights.ListPhraseMatchersRequest(), parent='parent_value')

def test_list_phrase_matchers_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher(), resources.PhraseMatcher(), resources.PhraseMatcher()], next_page_token='abc'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[], next_page_token='def'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher()], next_page_token='ghi'), contact_center_insights.ListPhraseMatchersResponse(phrase_matchers=[resources.PhraseMatcher(), resources.PhraseMatcher()]))
        response = response + response
        response = tuple((contact_center_insights.ListPhraseMatchersResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_phrase_matchers(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.PhraseMatcher) for i in results))
        pages = list(client.list_phrase_matchers(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [contact_center_insights.DeletePhraseMatcherRequest, dict])
def test_delete_phrase_matcher_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/phraseMatchers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_phrase_matcher(request)
    assert response is None

def test_delete_phrase_matcher_rest_required_fields(request_type=contact_center_insights.DeletePhraseMatcherRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_phrase_matcher._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_phrase_matcher._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_phrase_matcher(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_phrase_matcher_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_phrase_matcher._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_phrase_matcher_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_delete_phrase_matcher') as pre:
        pre.assert_not_called()
        pb_message = contact_center_insights.DeletePhraseMatcherRequest.pb(contact_center_insights.DeletePhraseMatcherRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = contact_center_insights.DeletePhraseMatcherRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_phrase_matcher(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_phrase_matcher_rest_bad_request(transport: str='rest', request_type=contact_center_insights.DeletePhraseMatcherRequest):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/phraseMatchers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_phrase_matcher(request)

def test_delete_phrase_matcher_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/phraseMatchers/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_phrase_matcher(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/phraseMatchers/*}' % client.transport._host, args[1])

def test_delete_phrase_matcher_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_phrase_matcher(contact_center_insights.DeletePhraseMatcherRequest(), name='name_value')

def test_delete_phrase_matcher_rest_error():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.UpdatePhraseMatcherRequest, dict])
def test_update_phrase_matcher_rest(request_type):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'phrase_matcher': {'name': 'projects/sample1/locations/sample2/phraseMatchers/sample3'}}
    request_init['phrase_matcher'] = {'name': 'projects/sample1/locations/sample2/phraseMatchers/sample3', 'revision_id': 'revision_id_value', 'version_tag': 'version_tag_value', 'revision_create_time': {'seconds': 751, 'nanos': 543}, 'display_name': 'display_name_value', 'type_': 1, 'active': True, 'phrase_match_rule_groups': [{'type_': 1, 'phrase_match_rules': [{'query': 'query_value', 'negated': True, 'config': {'exact_match_config': {'case_sensitive': True}}}]}], 'activation_update_time': {}, 'role_match': 1, 'update_time': {}}
    test_field = contact_center_insights.UpdatePhraseMatcherRequest.meta.fields['phrase_matcher']

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
    for (field, value) in request_init['phrase_matcher'].items():
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
                for i in range(0, len(request_init['phrase_matcher'][field])):
                    del request_init['phrase_matcher'][field][i][subfield]
            else:
                del request_init['phrase_matcher'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.PhraseMatcher(name='name_value', revision_id='revision_id_value', version_tag='version_tag_value', display_name='display_name_value', type_=resources.PhraseMatcher.PhraseMatcherType.ALL_OF, active=True, role_match=resources.ConversationParticipant.Role.HUMAN_AGENT)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.PhraseMatcher.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_phrase_matcher(request)
    assert isinstance(response, resources.PhraseMatcher)
    assert response.name == 'name_value'
    assert response.revision_id == 'revision_id_value'
    assert response.version_tag == 'version_tag_value'
    assert response.display_name == 'display_name_value'
    assert response.type_ == resources.PhraseMatcher.PhraseMatcherType.ALL_OF
    assert response.active is True
    assert response.role_match == resources.ConversationParticipant.Role.HUMAN_AGENT

def test_update_phrase_matcher_rest_required_fields(request_type=contact_center_insights.UpdatePhraseMatcherRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_phrase_matcher._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_phrase_matcher._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.PhraseMatcher()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.PhraseMatcher.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_phrase_matcher(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_phrase_matcher_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_phrase_matcher._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('phraseMatcher',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_phrase_matcher_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_update_phrase_matcher') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_update_phrase_matcher') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.UpdatePhraseMatcherRequest.pb(contact_center_insights.UpdatePhraseMatcherRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.PhraseMatcher.to_json(resources.PhraseMatcher())
        request = contact_center_insights.UpdatePhraseMatcherRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.PhraseMatcher()
        client.update_phrase_matcher(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_phrase_matcher_rest_bad_request(transport: str='rest', request_type=contact_center_insights.UpdatePhraseMatcherRequest):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'phrase_matcher': {'name': 'projects/sample1/locations/sample2/phraseMatchers/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_phrase_matcher(request)

def test_update_phrase_matcher_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.PhraseMatcher()
        sample_request = {'phrase_matcher': {'name': 'projects/sample1/locations/sample2/phraseMatchers/sample3'}}
        mock_args = dict(phrase_matcher=resources.PhraseMatcher(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.PhraseMatcher.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_phrase_matcher(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{phrase_matcher.name=projects/*/locations/*/phraseMatchers/*}' % client.transport._host, args[1])

def test_update_phrase_matcher_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_phrase_matcher(contact_center_insights.UpdatePhraseMatcherRequest(), phrase_matcher=resources.PhraseMatcher(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_phrase_matcher_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.CalculateStatsRequest, dict])
def test_calculate_stats_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'location': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.CalculateStatsResponse(average_turn_count=1931, conversation_count=1955)
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.CalculateStatsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.calculate_stats(request)
    assert isinstance(response, contact_center_insights.CalculateStatsResponse)
    assert response.average_turn_count == 1931
    assert response.conversation_count == 1955

def test_calculate_stats_rest_required_fields(request_type=contact_center_insights.CalculateStatsRequest):
    if False:
        return 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['location'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).calculate_stats._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['location'] = 'location_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).calculate_stats._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter',))
    jsonified_request.update(unset_fields)
    assert 'location' in jsonified_request
    assert jsonified_request['location'] == 'location_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = contact_center_insights.CalculateStatsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = contact_center_insights.CalculateStatsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.calculate_stats(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_calculate_stats_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.calculate_stats._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter',)) & set(('location',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_calculate_stats_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_calculate_stats') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_calculate_stats') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.CalculateStatsRequest.pb(contact_center_insights.CalculateStatsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = contact_center_insights.CalculateStatsResponse.to_json(contact_center_insights.CalculateStatsResponse())
        request = contact_center_insights.CalculateStatsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = contact_center_insights.CalculateStatsResponse()
        client.calculate_stats(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_calculate_stats_rest_bad_request(transport: str='rest', request_type=contact_center_insights.CalculateStatsRequest):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'location': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.calculate_stats(request)

def test_calculate_stats_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.CalculateStatsResponse()
        sample_request = {'location': 'projects/sample1/locations/sample2'}
        mock_args = dict(location='location_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.CalculateStatsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.calculate_stats(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{location=projects/*/locations/*}/conversations:calculateStats' % client.transport._host, args[1])

def test_calculate_stats_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.calculate_stats(contact_center_insights.CalculateStatsRequest(), location='location_value')

def test_calculate_stats_rest_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.GetSettingsRequest, dict])
def test_get_settings_rest(request_type):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/settings'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Settings(name='name_value', language_code='language_code_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Settings.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_settings(request)
    assert isinstance(response, resources.Settings)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'

def test_get_settings_rest_required_fields(request_type=contact_center_insights.GetSettingsRequest):
    if False:
        return 10
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_settings._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_settings._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Settings()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Settings.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_settings(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_settings_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_settings._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_settings_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_get_settings') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_get_settings') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.GetSettingsRequest.pb(contact_center_insights.GetSettingsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Settings.to_json(resources.Settings())
        request = contact_center_insights.GetSettingsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Settings()
        client.get_settings(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_settings_rest_bad_request(transport: str='rest', request_type=contact_center_insights.GetSettingsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/settings'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_settings(request)

def test_get_settings_rest_flattened():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Settings()
        sample_request = {'name': 'projects/sample1/locations/sample2/settings'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Settings.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_settings(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/settings}' % client.transport._host, args[1])

def test_get_settings_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_settings(contact_center_insights.GetSettingsRequest(), name='name_value')

def test_get_settings_rest_error():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.UpdateSettingsRequest, dict])
def test_update_settings_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'settings': {'name': 'projects/sample1/locations/sample2/settings'}}
    request_init['settings'] = {'name': 'projects/sample1/locations/sample2/settings', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'language_code': 'language_code_value', 'conversation_ttl': {'seconds': 751, 'nanos': 543}, 'pubsub_notification_settings': {}, 'analysis_config': {'runtime_integration_analysis_percentage': 0.4167, 'upload_conversation_analysis_percentage': 0.41590000000000005, 'annotator_selector': {'run_interruption_annotator': True, 'run_silence_annotator': True, 'run_phrase_matcher_annotator': True, 'phrase_matchers': ['phrase_matchers_value1', 'phrase_matchers_value2'], 'run_sentiment_annotator': True, 'run_entity_annotator': True, 'run_intent_annotator': True, 'run_issue_model_annotator': True, 'issue_models': ['issue_models_value1', 'issue_models_value2'], 'run_summarization_annotator': True, 'summarization_config': {'conversation_profile': 'conversation_profile_value', 'summarization_model': 1}}}, 'redaction_config': {'deidentify_template': 'deidentify_template_value', 'inspect_template': 'inspect_template_value'}, 'speech_config': {'speech_recognizer': 'speech_recognizer_value'}}
    test_field = contact_center_insights.UpdateSettingsRequest.meta.fields['settings']

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
    for (field, value) in request_init['settings'].items():
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
                for i in range(0, len(request_init['settings'][field])):
                    del request_init['settings'][field][i][subfield]
            else:
                del request_init['settings'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Settings(name='name_value', language_code='language_code_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Settings.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_settings(request)
    assert isinstance(response, resources.Settings)
    assert response.name == 'name_value'
    assert response.language_code == 'language_code_value'

def test_update_settings_rest_required_fields(request_type=contact_center_insights.UpdateSettingsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_settings._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_settings._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Settings()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Settings.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_settings(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_settings_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_settings._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('settings', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_settings_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_update_settings') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_update_settings') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.UpdateSettingsRequest.pb(contact_center_insights.UpdateSettingsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Settings.to_json(resources.Settings())
        request = contact_center_insights.UpdateSettingsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Settings()
        client.update_settings(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_settings_rest_bad_request(transport: str='rest', request_type=contact_center_insights.UpdateSettingsRequest):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'settings': {'name': 'projects/sample1/locations/sample2/settings'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_settings(request)

def test_update_settings_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Settings()
        sample_request = {'settings': {'name': 'projects/sample1/locations/sample2/settings'}}
        mock_args = dict(settings=resources.Settings(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Settings.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_settings(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{settings.name=projects/*/locations/*/settings}' % client.transport._host, args[1])

def test_update_settings_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_settings(contact_center_insights.UpdateSettingsRequest(), settings=resources.Settings(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_settings_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.CreateViewRequest, dict])
def test_create_view_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['view'] = {'name': 'name_value', 'display_name': 'display_name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'value': 'value_value'}
    test_field = contact_center_insights.CreateViewRequest.meta.fields['view']

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
    for (field, value) in request_init['view'].items():
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
                for i in range(0, len(request_init['view'][field])):
                    del request_init['view'][field][i][subfield]
            else:
                del request_init['view'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.View(name='name_value', display_name='display_name_value', value='value_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.View.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_view(request)
    assert isinstance(response, resources.View)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.value == 'value_value'

def test_create_view_rest_required_fields(request_type=contact_center_insights.CreateViewRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_view._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_view._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.View()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.View.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_view(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_view_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_view._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'view'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_view_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_create_view') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_create_view') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.CreateViewRequest.pb(contact_center_insights.CreateViewRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.View.to_json(resources.View())
        request = contact_center_insights.CreateViewRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.View()
        client.create_view(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_view_rest_bad_request(transport: str='rest', request_type=contact_center_insights.CreateViewRequest):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_view(request)

def test_create_view_rest_flattened():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.View()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', view=resources.View(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.View.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_view(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/views' % client.transport._host, args[1])

def test_create_view_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_view(contact_center_insights.CreateViewRequest(), parent='parent_value', view=resources.View(name='name_value'))

def test_create_view_rest_error():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.GetViewRequest, dict])
def test_get_view_rest(request_type):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/views/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.View(name='name_value', display_name='display_name_value', value='value_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.View.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_view(request)
    assert isinstance(response, resources.View)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.value == 'value_value'

def test_get_view_rest_required_fields(request_type=contact_center_insights.GetViewRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_view._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_view._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.View()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.View.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_view(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_view_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_view._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_view_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_get_view') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_get_view') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.GetViewRequest.pb(contact_center_insights.GetViewRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.View.to_json(resources.View())
        request = contact_center_insights.GetViewRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.View()
        client.get_view(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_view_rest_bad_request(transport: str='rest', request_type=contact_center_insights.GetViewRequest):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/views/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_view(request)

def test_get_view_rest_flattened():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.View()
        sample_request = {'name': 'projects/sample1/locations/sample2/views/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.View.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_view(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/views/*}' % client.transport._host, args[1])

def test_get_view_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_view(contact_center_insights.GetViewRequest(), name='name_value')

def test_get_view_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.ListViewsRequest, dict])
def test_list_views_rest(request_type):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.ListViewsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.ListViewsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_views(request)
    assert isinstance(response, pagers.ListViewsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_views_rest_required_fields(request_type=contact_center_insights.ListViewsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_views._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_views._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = contact_center_insights.ListViewsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = contact_center_insights.ListViewsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_views(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_views_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_views._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_views_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_list_views') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_list_views') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.ListViewsRequest.pb(contact_center_insights.ListViewsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = contact_center_insights.ListViewsResponse.to_json(contact_center_insights.ListViewsResponse())
        request = contact_center_insights.ListViewsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = contact_center_insights.ListViewsResponse()
        client.list_views(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_views_rest_bad_request(transport: str='rest', request_type=contact_center_insights.ListViewsRequest):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_views(request)

def test_list_views_rest_flattened():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = contact_center_insights.ListViewsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = contact_center_insights.ListViewsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_views(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/views' % client.transport._host, args[1])

def test_list_views_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_views(contact_center_insights.ListViewsRequest(), parent='parent_value')

def test_list_views_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (contact_center_insights.ListViewsResponse(views=[resources.View(), resources.View(), resources.View()], next_page_token='abc'), contact_center_insights.ListViewsResponse(views=[], next_page_token='def'), contact_center_insights.ListViewsResponse(views=[resources.View()], next_page_token='ghi'), contact_center_insights.ListViewsResponse(views=[resources.View(), resources.View()]))
        response = response + response
        response = tuple((contact_center_insights.ListViewsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_views(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.View) for i in results))
        pages = list(client.list_views(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [contact_center_insights.UpdateViewRequest, dict])
def test_update_view_rest(request_type):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'view': {'name': 'projects/sample1/locations/sample2/views/sample3'}}
    request_init['view'] = {'name': 'projects/sample1/locations/sample2/views/sample3', 'display_name': 'display_name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'value': 'value_value'}
    test_field = contact_center_insights.UpdateViewRequest.meta.fields['view']

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
    for (field, value) in request_init['view'].items():
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
                for i in range(0, len(request_init['view'][field])):
                    del request_init['view'][field][i][subfield]
            else:
                del request_init['view'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.View(name='name_value', display_name='display_name_value', value='value_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.View.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_view(request)
    assert isinstance(response, resources.View)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.value == 'value_value'

def test_update_view_rest_required_fields(request_type=contact_center_insights.UpdateViewRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_view._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_view._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.View()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.View.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_view(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_view_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_view._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('view',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_view_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'post_update_view') as post, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_update_view') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = contact_center_insights.UpdateViewRequest.pb(contact_center_insights.UpdateViewRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.View.to_json(resources.View())
        request = contact_center_insights.UpdateViewRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.View()
        client.update_view(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_view_rest_bad_request(transport: str='rest', request_type=contact_center_insights.UpdateViewRequest):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'view': {'name': 'projects/sample1/locations/sample2/views/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_view(request)

def test_update_view_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.View()
        sample_request = {'view': {'name': 'projects/sample1/locations/sample2/views/sample3'}}
        mock_args = dict(view=resources.View(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.View.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_view(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{view.name=projects/*/locations/*/views/*}' % client.transport._host, args[1])

def test_update_view_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_view(contact_center_insights.UpdateViewRequest(), view=resources.View(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_view_rest_error():
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [contact_center_insights.DeleteViewRequest, dict])
def test_delete_view_rest(request_type):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/views/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_view(request)
    assert response is None

def test_delete_view_rest_required_fields(request_type=contact_center_insights.DeleteViewRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ContactCenterInsightsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_view._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_view._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_view(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_view_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_view._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_view_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ContactCenterInsightsRestInterceptor())
    client = ContactCenterInsightsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ContactCenterInsightsRestInterceptor, 'pre_delete_view') as pre:
        pre.assert_not_called()
        pb_message = contact_center_insights.DeleteViewRequest.pb(contact_center_insights.DeleteViewRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = contact_center_insights.DeleteViewRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_view(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_view_rest_bad_request(transport: str='rest', request_type=contact_center_insights.DeleteViewRequest):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/views/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_view(request)

def test_delete_view_rest_flattened():
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/views/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_view(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/views/*}' % client.transport._host, args[1])

def test_delete_view_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_view(contact_center_insights.DeleteViewRequest(), name='name_value')

def test_delete_view_rest_error():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ContactCenterInsightsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ContactCenterInsightsClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ContactCenterInsightsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ContactCenterInsightsClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ContactCenterInsightsClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ContactCenterInsightsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ContactCenterInsightsClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ContactCenterInsightsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ContactCenterInsightsClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.ContactCenterInsightsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ContactCenterInsightsGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ContactCenterInsightsGrpcTransport, transports.ContactCenterInsightsGrpcAsyncIOTransport, transports.ContactCenterInsightsRestTransport])
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
    transport = ContactCenterInsightsClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ContactCenterInsightsGrpcTransport)

def test_contact_center_insights_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ContactCenterInsightsTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_contact_center_insights_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.contact_center_insights_v1.services.contact_center_insights.transports.ContactCenterInsightsTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ContactCenterInsightsTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_conversation', 'upload_conversation', 'update_conversation', 'get_conversation', 'list_conversations', 'delete_conversation', 'create_analysis', 'get_analysis', 'list_analyses', 'delete_analysis', 'bulk_analyze_conversations', 'bulk_delete_conversations', 'ingest_conversations', 'export_insights_data', 'create_issue_model', 'update_issue_model', 'get_issue_model', 'list_issue_models', 'delete_issue_model', 'deploy_issue_model', 'undeploy_issue_model', 'get_issue', 'list_issues', 'update_issue', 'delete_issue', 'calculate_issue_model_stats', 'create_phrase_matcher', 'get_phrase_matcher', 'list_phrase_matchers', 'delete_phrase_matcher', 'update_phrase_matcher', 'calculate_stats', 'get_settings', 'update_settings', 'create_view', 'get_view', 'list_views', 'update_view', 'delete_view', 'get_operation', 'cancel_operation', 'list_operations')
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

def test_contact_center_insights_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.contact_center_insights_v1.services.contact_center_insights.transports.ContactCenterInsightsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ContactCenterInsightsTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_contact_center_insights_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.contact_center_insights_v1.services.contact_center_insights.transports.ContactCenterInsightsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ContactCenterInsightsTransport()
        adc.assert_called_once()

def test_contact_center_insights_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ContactCenterInsightsClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ContactCenterInsightsGrpcTransport, transports.ContactCenterInsightsGrpcAsyncIOTransport])
def test_contact_center_insights_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ContactCenterInsightsGrpcTransport, transports.ContactCenterInsightsGrpcAsyncIOTransport, transports.ContactCenterInsightsRestTransport])
def test_contact_center_insights_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ContactCenterInsightsGrpcTransport, grpc_helpers), (transports.ContactCenterInsightsGrpcAsyncIOTransport, grpc_helpers_async)])
def test_contact_center_insights_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('contactcenterinsights.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='contactcenterinsights.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ContactCenterInsightsGrpcTransport, transports.ContactCenterInsightsGrpcAsyncIOTransport])
def test_contact_center_insights_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        return 10
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

def test_contact_center_insights_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.ContactCenterInsightsRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_contact_center_insights_rest_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_contact_center_insights_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='contactcenterinsights.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('contactcenterinsights.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://contactcenterinsights.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_contact_center_insights_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='contactcenterinsights.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('contactcenterinsights.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://contactcenterinsights.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_contact_center_insights_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = ContactCenterInsightsClient(credentials=creds1, transport=transport_name)
    client2 = ContactCenterInsightsClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_conversation._session
    session2 = client2.transport.create_conversation._session
    assert session1 != session2
    session1 = client1.transport.upload_conversation._session
    session2 = client2.transport.upload_conversation._session
    assert session1 != session2
    session1 = client1.transport.update_conversation._session
    session2 = client2.transport.update_conversation._session
    assert session1 != session2
    session1 = client1.transport.get_conversation._session
    session2 = client2.transport.get_conversation._session
    assert session1 != session2
    session1 = client1.transport.list_conversations._session
    session2 = client2.transport.list_conversations._session
    assert session1 != session2
    session1 = client1.transport.delete_conversation._session
    session2 = client2.transport.delete_conversation._session
    assert session1 != session2
    session1 = client1.transport.create_analysis._session
    session2 = client2.transport.create_analysis._session
    assert session1 != session2
    session1 = client1.transport.get_analysis._session
    session2 = client2.transport.get_analysis._session
    assert session1 != session2
    session1 = client1.transport.list_analyses._session
    session2 = client2.transport.list_analyses._session
    assert session1 != session2
    session1 = client1.transport.delete_analysis._session
    session2 = client2.transport.delete_analysis._session
    assert session1 != session2
    session1 = client1.transport.bulk_analyze_conversations._session
    session2 = client2.transport.bulk_analyze_conversations._session
    assert session1 != session2
    session1 = client1.transport.bulk_delete_conversations._session
    session2 = client2.transport.bulk_delete_conversations._session
    assert session1 != session2
    session1 = client1.transport.ingest_conversations._session
    session2 = client2.transport.ingest_conversations._session
    assert session1 != session2
    session1 = client1.transport.export_insights_data._session
    session2 = client2.transport.export_insights_data._session
    assert session1 != session2
    session1 = client1.transport.create_issue_model._session
    session2 = client2.transport.create_issue_model._session
    assert session1 != session2
    session1 = client1.transport.update_issue_model._session
    session2 = client2.transport.update_issue_model._session
    assert session1 != session2
    session1 = client1.transport.get_issue_model._session
    session2 = client2.transport.get_issue_model._session
    assert session1 != session2
    session1 = client1.transport.list_issue_models._session
    session2 = client2.transport.list_issue_models._session
    assert session1 != session2
    session1 = client1.transport.delete_issue_model._session
    session2 = client2.transport.delete_issue_model._session
    assert session1 != session2
    session1 = client1.transport.deploy_issue_model._session
    session2 = client2.transport.deploy_issue_model._session
    assert session1 != session2
    session1 = client1.transport.undeploy_issue_model._session
    session2 = client2.transport.undeploy_issue_model._session
    assert session1 != session2
    session1 = client1.transport.get_issue._session
    session2 = client2.transport.get_issue._session
    assert session1 != session2
    session1 = client1.transport.list_issues._session
    session2 = client2.transport.list_issues._session
    assert session1 != session2
    session1 = client1.transport.update_issue._session
    session2 = client2.transport.update_issue._session
    assert session1 != session2
    session1 = client1.transport.delete_issue._session
    session2 = client2.transport.delete_issue._session
    assert session1 != session2
    session1 = client1.transport.calculate_issue_model_stats._session
    session2 = client2.transport.calculate_issue_model_stats._session
    assert session1 != session2
    session1 = client1.transport.create_phrase_matcher._session
    session2 = client2.transport.create_phrase_matcher._session
    assert session1 != session2
    session1 = client1.transport.get_phrase_matcher._session
    session2 = client2.transport.get_phrase_matcher._session
    assert session1 != session2
    session1 = client1.transport.list_phrase_matchers._session
    session2 = client2.transport.list_phrase_matchers._session
    assert session1 != session2
    session1 = client1.transport.delete_phrase_matcher._session
    session2 = client2.transport.delete_phrase_matcher._session
    assert session1 != session2
    session1 = client1.transport.update_phrase_matcher._session
    session2 = client2.transport.update_phrase_matcher._session
    assert session1 != session2
    session1 = client1.transport.calculate_stats._session
    session2 = client2.transport.calculate_stats._session
    assert session1 != session2
    session1 = client1.transport.get_settings._session
    session2 = client2.transport.get_settings._session
    assert session1 != session2
    session1 = client1.transport.update_settings._session
    session2 = client2.transport.update_settings._session
    assert session1 != session2
    session1 = client1.transport.create_view._session
    session2 = client2.transport.create_view._session
    assert session1 != session2
    session1 = client1.transport.get_view._session
    session2 = client2.transport.get_view._session
    assert session1 != session2
    session1 = client1.transport.list_views._session
    session2 = client2.transport.list_views._session
    assert session1 != session2
    session1 = client1.transport.update_view._session
    session2 = client2.transport.update_view._session
    assert session1 != session2
    session1 = client1.transport.delete_view._session
    session2 = client2.transport.delete_view._session
    assert session1 != session2

def test_contact_center_insights_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ContactCenterInsightsGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_contact_center_insights_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ContactCenterInsightsGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ContactCenterInsightsGrpcTransport, transports.ContactCenterInsightsGrpcAsyncIOTransport])
def test_contact_center_insights_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        while True:
            i = 10
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

@pytest.mark.parametrize('transport_class', [transports.ContactCenterInsightsGrpcTransport, transports.ContactCenterInsightsGrpcAsyncIOTransport])
def test_contact_center_insights_transport_channel_mtls_with_adc(transport_class):
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

def test_contact_center_insights_grpc_lro_client():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_contact_center_insights_grpc_lro_async_client():
    if False:
        print('Hello World!')
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_analysis_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    conversation = 'whelk'
    analysis = 'octopus'
    expected = 'projects/{project}/locations/{location}/conversations/{conversation}/analyses/{analysis}'.format(project=project, location=location, conversation=conversation, analysis=analysis)
    actual = ContactCenterInsightsClient.analysis_path(project, location, conversation, analysis)
    assert expected == actual

def test_parse_analysis_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'oyster', 'location': 'nudibranch', 'conversation': 'cuttlefish', 'analysis': 'mussel'}
    path = ContactCenterInsightsClient.analysis_path(**expected)
    actual = ContactCenterInsightsClient.parse_analysis_path(path)
    assert expected == actual

def test_conversation_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    location = 'nautilus'
    conversation = 'scallop'
    expected = 'projects/{project}/locations/{location}/conversations/{conversation}'.format(project=project, location=location, conversation=conversation)
    actual = ContactCenterInsightsClient.conversation_path(project, location, conversation)
    assert expected == actual

def test_parse_conversation_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'abalone', 'location': 'squid', 'conversation': 'clam'}
    path = ContactCenterInsightsClient.conversation_path(**expected)
    actual = ContactCenterInsightsClient.parse_conversation_path(path)
    assert expected == actual

def test_conversation_profile_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    conversation_profile = 'oyster'
    expected = 'projects/{project}/locations/{location}/conversationProfiles/{conversation_profile}'.format(project=project, location=location, conversation_profile=conversation_profile)
    actual = ContactCenterInsightsClient.conversation_profile_path(project, location, conversation_profile)
    assert expected == actual

def test_parse_conversation_profile_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'conversation_profile': 'mussel'}
    path = ContactCenterInsightsClient.conversation_profile_path(**expected)
    actual = ContactCenterInsightsClient.parse_conversation_profile_path(path)
    assert expected == actual

def test_issue_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    issue_model = 'scallop'
    issue = 'abalone'
    expected = 'projects/{project}/locations/{location}/issueModels/{issue_model}/issues/{issue}'.format(project=project, location=location, issue_model=issue_model, issue=issue)
    actual = ContactCenterInsightsClient.issue_path(project, location, issue_model, issue)
    assert expected == actual

def test_parse_issue_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'squid', 'location': 'clam', 'issue_model': 'whelk', 'issue': 'octopus'}
    path = ContactCenterInsightsClient.issue_path(**expected)
    actual = ContactCenterInsightsClient.parse_issue_path(path)
    assert expected == actual

def test_issue_model_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    location = 'nudibranch'
    issue_model = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/issueModels/{issue_model}'.format(project=project, location=location, issue_model=issue_model)
    actual = ContactCenterInsightsClient.issue_model_path(project, location, issue_model)
    assert expected == actual

def test_parse_issue_model_path():
    if False:
        return 10
    expected = {'project': 'mussel', 'location': 'winkle', 'issue_model': 'nautilus'}
    path = ContactCenterInsightsClient.issue_model_path(**expected)
    actual = ContactCenterInsightsClient.parse_issue_model_path(path)
    assert expected == actual

def test_participant_path():
    if False:
        return 10
    project = 'scallop'
    conversation = 'abalone'
    participant = 'squid'
    expected = 'projects/{project}/conversations/{conversation}/participants/{participant}'.format(project=project, conversation=conversation, participant=participant)
    actual = ContactCenterInsightsClient.participant_path(project, conversation, participant)
    assert expected == actual

def test_parse_participant_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'clam', 'conversation': 'whelk', 'participant': 'octopus'}
    path = ContactCenterInsightsClient.participant_path(**expected)
    actual = ContactCenterInsightsClient.parse_participant_path(path)
    assert expected == actual

def test_phrase_matcher_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    location = 'nudibranch'
    phrase_matcher = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/phraseMatchers/{phrase_matcher}'.format(project=project, location=location, phrase_matcher=phrase_matcher)
    actual = ContactCenterInsightsClient.phrase_matcher_path(project, location, phrase_matcher)
    assert expected == actual

def test_parse_phrase_matcher_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'mussel', 'location': 'winkle', 'phrase_matcher': 'nautilus'}
    path = ContactCenterInsightsClient.phrase_matcher_path(**expected)
    actual = ContactCenterInsightsClient.parse_phrase_matcher_path(path)
    assert expected == actual

def test_recognizer_path():
    if False:
        return 10
    project = 'scallop'
    location = 'abalone'
    recognizer = 'squid'
    expected = 'projects/{project}/locations/{location}/recognizers/{recognizer}'.format(project=project, location=location, recognizer=recognizer)
    actual = ContactCenterInsightsClient.recognizer_path(project, location, recognizer)
    assert expected == actual

def test_parse_recognizer_path():
    if False:
        return 10
    expected = {'project': 'clam', 'location': 'whelk', 'recognizer': 'octopus'}
    path = ContactCenterInsightsClient.recognizer_path(**expected)
    actual = ContactCenterInsightsClient.parse_recognizer_path(path)
    assert expected == actual

def test_settings_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}/settings'.format(project=project, location=location)
    actual = ContactCenterInsightsClient.settings_path(project, location)
    assert expected == actual

def test_parse_settings_path():
    if False:
        print('Hello World!')
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = ContactCenterInsightsClient.settings_path(**expected)
    actual = ContactCenterInsightsClient.parse_settings_path(path)
    assert expected == actual

def test_view_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    view = 'scallop'
    expected = 'projects/{project}/locations/{location}/views/{view}'.format(project=project, location=location, view=view)
    actual = ContactCenterInsightsClient.view_path(project, location, view)
    assert expected == actual

def test_parse_view_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'abalone', 'location': 'squid', 'view': 'clam'}
    path = ContactCenterInsightsClient.view_path(**expected)
    actual = ContactCenterInsightsClient.parse_view_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ContactCenterInsightsClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'octopus'}
    path = ContactCenterInsightsClient.common_billing_account_path(**expected)
    actual = ContactCenterInsightsClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ContactCenterInsightsClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'nudibranch'}
    path = ContactCenterInsightsClient.common_folder_path(**expected)
    actual = ContactCenterInsightsClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ContactCenterInsightsClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'mussel'}
    path = ContactCenterInsightsClient.common_organization_path(**expected)
    actual = ContactCenterInsightsClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = ContactCenterInsightsClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nautilus'}
    path = ContactCenterInsightsClient.common_project_path(**expected)
    actual = ContactCenterInsightsClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ContactCenterInsightsClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'squid', 'location': 'clam'}
    path = ContactCenterInsightsClient.common_location_path(**expected)
    actual = ContactCenterInsightsClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ContactCenterInsightsTransport, '_prep_wrapped_messages') as prep:
        client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ContactCenterInsightsTransport, '_prep_wrapped_messages') as prep:
        transport_class = ContactCenterInsightsClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_cancel_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.CancelOperationRequest):
    if False:
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_list_operations_rest_bad_request(transport: str='rest', request_type=operations_pb2.ListOperationsRequest):
    if False:
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_operations(request)

@pytest.mark.parametrize('request_type', [operations_pb2.ListOperationsRequest, dict])
def test_list_operations_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2'}
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
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = ContactCenterInsightsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = ContactCenterInsightsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ContactCenterInsightsClient, transports.ContactCenterInsightsGrpcTransport), (ContactCenterInsightsAsyncClient, transports.ContactCenterInsightsGrpcAsyncIOTransport)])
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