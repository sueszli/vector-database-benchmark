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
from google.oauth2 import service_account
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
from google.cloud.essential_contacts_v1.services.essential_contacts_service import EssentialContactsServiceAsyncClient, EssentialContactsServiceClient, pagers, transports
from google.cloud.essential_contacts_v1.types import enums, service

def client_cert_source_callback():
    if False:
        return 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        while True:
            i = 10
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
    assert EssentialContactsServiceClient._get_default_mtls_endpoint(None) is None
    assert EssentialContactsServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert EssentialContactsServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert EssentialContactsServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert EssentialContactsServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert EssentialContactsServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(EssentialContactsServiceClient, 'grpc'), (EssentialContactsServiceAsyncClient, 'grpc_asyncio'), (EssentialContactsServiceClient, 'rest')])
def test_essential_contacts_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('essentialcontacts.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://essentialcontacts.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.EssentialContactsServiceGrpcTransport, 'grpc'), (transports.EssentialContactsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.EssentialContactsServiceRestTransport, 'rest')])
def test_essential_contacts_service_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(EssentialContactsServiceClient, 'grpc'), (EssentialContactsServiceAsyncClient, 'grpc_asyncio'), (EssentialContactsServiceClient, 'rest')])
def test_essential_contacts_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('essentialcontacts.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://essentialcontacts.googleapis.com')

def test_essential_contacts_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = EssentialContactsServiceClient.get_transport_class()
    available_transports = [transports.EssentialContactsServiceGrpcTransport, transports.EssentialContactsServiceRestTransport]
    assert transport in available_transports
    transport = EssentialContactsServiceClient.get_transport_class('grpc')
    assert transport == transports.EssentialContactsServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(EssentialContactsServiceClient, transports.EssentialContactsServiceGrpcTransport, 'grpc'), (EssentialContactsServiceAsyncClient, transports.EssentialContactsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (EssentialContactsServiceClient, transports.EssentialContactsServiceRestTransport, 'rest')])
@mock.patch.object(EssentialContactsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EssentialContactsServiceClient))
@mock.patch.object(EssentialContactsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EssentialContactsServiceAsyncClient))
def test_essential_contacts_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(EssentialContactsServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(EssentialContactsServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(EssentialContactsServiceClient, transports.EssentialContactsServiceGrpcTransport, 'grpc', 'true'), (EssentialContactsServiceAsyncClient, transports.EssentialContactsServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (EssentialContactsServiceClient, transports.EssentialContactsServiceGrpcTransport, 'grpc', 'false'), (EssentialContactsServiceAsyncClient, transports.EssentialContactsServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (EssentialContactsServiceClient, transports.EssentialContactsServiceRestTransport, 'rest', 'true'), (EssentialContactsServiceClient, transports.EssentialContactsServiceRestTransport, 'rest', 'false')])
@mock.patch.object(EssentialContactsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EssentialContactsServiceClient))
@mock.patch.object(EssentialContactsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EssentialContactsServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_essential_contacts_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [EssentialContactsServiceClient, EssentialContactsServiceAsyncClient])
@mock.patch.object(EssentialContactsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EssentialContactsServiceClient))
@mock.patch.object(EssentialContactsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EssentialContactsServiceAsyncClient))
def test_essential_contacts_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(EssentialContactsServiceClient, transports.EssentialContactsServiceGrpcTransport, 'grpc'), (EssentialContactsServiceAsyncClient, transports.EssentialContactsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (EssentialContactsServiceClient, transports.EssentialContactsServiceRestTransport, 'rest')])
def test_essential_contacts_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(EssentialContactsServiceClient, transports.EssentialContactsServiceGrpcTransport, 'grpc', grpc_helpers), (EssentialContactsServiceAsyncClient, transports.EssentialContactsServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (EssentialContactsServiceClient, transports.EssentialContactsServiceRestTransport, 'rest', None)])
def test_essential_contacts_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_essential_contacts_service_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.essential_contacts_v1.services.essential_contacts_service.transports.EssentialContactsServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = EssentialContactsServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(EssentialContactsServiceClient, transports.EssentialContactsServiceGrpcTransport, 'grpc', grpc_helpers), (EssentialContactsServiceAsyncClient, transports.EssentialContactsServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_essential_contacts_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
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
        create_channel.assert_called_with('essentialcontacts.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='essentialcontacts.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [service.CreateContactRequest, dict])
def test_create_contact(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_contact), '__call__') as call:
        call.return_value = service.Contact(name='name_value', email='email_value', notification_category_subscriptions=[enums.NotificationCategory.ALL], language_tag='language_tag_value', validation_state=enums.ValidationState.VALID)
        response = client.create_contact(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateContactRequest()
    assert isinstance(response, service.Contact)
    assert response.name == 'name_value'
    assert response.email == 'email_value'
    assert response.notification_category_subscriptions == [enums.NotificationCategory.ALL]
    assert response.language_tag == 'language_tag_value'
    assert response.validation_state == enums.ValidationState.VALID

def test_create_contact_empty_call():
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_contact), '__call__') as call:
        client.create_contact()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateContactRequest()

@pytest.mark.asyncio
async def test_create_contact_async(transport: str='grpc_asyncio', request_type=service.CreateContactRequest):
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_contact), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Contact(name='name_value', email='email_value', notification_category_subscriptions=[enums.NotificationCategory.ALL], language_tag='language_tag_value', validation_state=enums.ValidationState.VALID))
        response = await client.create_contact(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateContactRequest()
    assert isinstance(response, service.Contact)
    assert response.name == 'name_value'
    assert response.email == 'email_value'
    assert response.notification_category_subscriptions == [enums.NotificationCategory.ALL]
    assert response.language_tag == 'language_tag_value'
    assert response.validation_state == enums.ValidationState.VALID

@pytest.mark.asyncio
async def test_create_contact_async_from_dict():
    await test_create_contact_async(request_type=dict)

def test_create_contact_field_headers():
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateContactRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_contact), '__call__') as call:
        call.return_value = service.Contact()
        client.create_contact(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_contact_field_headers_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateContactRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_contact), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Contact())
        await client.create_contact(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_contact_flattened():
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_contact), '__call__') as call:
        call.return_value = service.Contact()
        client.create_contact(parent='parent_value', contact=service.Contact(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].contact
        mock_val = service.Contact(name='name_value')
        assert arg == mock_val

def test_create_contact_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_contact(service.CreateContactRequest(), parent='parent_value', contact=service.Contact(name='name_value'))

@pytest.mark.asyncio
async def test_create_contact_flattened_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_contact), '__call__') as call:
        call.return_value = service.Contact()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Contact())
        response = await client.create_contact(parent='parent_value', contact=service.Contact(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].contact
        mock_val = service.Contact(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_contact_flattened_error_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_contact(service.CreateContactRequest(), parent='parent_value', contact=service.Contact(name='name_value'))

@pytest.mark.parametrize('request_type', [service.UpdateContactRequest, dict])
def test_update_contact(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_contact), '__call__') as call:
        call.return_value = service.Contact(name='name_value', email='email_value', notification_category_subscriptions=[enums.NotificationCategory.ALL], language_tag='language_tag_value', validation_state=enums.ValidationState.VALID)
        response = client.update_contact(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateContactRequest()
    assert isinstance(response, service.Contact)
    assert response.name == 'name_value'
    assert response.email == 'email_value'
    assert response.notification_category_subscriptions == [enums.NotificationCategory.ALL]
    assert response.language_tag == 'language_tag_value'
    assert response.validation_state == enums.ValidationState.VALID

def test_update_contact_empty_call():
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_contact), '__call__') as call:
        client.update_contact()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateContactRequest()

@pytest.mark.asyncio
async def test_update_contact_async(transport: str='grpc_asyncio', request_type=service.UpdateContactRequest):
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_contact), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Contact(name='name_value', email='email_value', notification_category_subscriptions=[enums.NotificationCategory.ALL], language_tag='language_tag_value', validation_state=enums.ValidationState.VALID))
        response = await client.update_contact(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateContactRequest()
    assert isinstance(response, service.Contact)
    assert response.name == 'name_value'
    assert response.email == 'email_value'
    assert response.notification_category_subscriptions == [enums.NotificationCategory.ALL]
    assert response.language_tag == 'language_tag_value'
    assert response.validation_state == enums.ValidationState.VALID

@pytest.mark.asyncio
async def test_update_contact_async_from_dict():
    await test_update_contact_async(request_type=dict)

def test_update_contact_field_headers():
    if False:
        i = 10
        return i + 15
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateContactRequest()
    request.contact.name = 'name_value'
    with mock.patch.object(type(client.transport.update_contact), '__call__') as call:
        call.return_value = service.Contact()
        client.update_contact(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'contact.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_contact_field_headers_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateContactRequest()
    request.contact.name = 'name_value'
    with mock.patch.object(type(client.transport.update_contact), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Contact())
        await client.update_contact(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'contact.name=name_value') in kw['metadata']

def test_update_contact_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_contact), '__call__') as call:
        call.return_value = service.Contact()
        client.update_contact(contact=service.Contact(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].contact
        mock_val = service.Contact(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_contact_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_contact(service.UpdateContactRequest(), contact=service.Contact(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_contact_flattened_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_contact), '__call__') as call:
        call.return_value = service.Contact()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Contact())
        response = await client.update_contact(contact=service.Contact(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].contact
        mock_val = service.Contact(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_contact_flattened_error_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_contact(service.UpdateContactRequest(), contact=service.Contact(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.ListContactsRequest, dict])
def test_list_contacts(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_contacts), '__call__') as call:
        call.return_value = service.ListContactsResponse(next_page_token='next_page_token_value')
        response = client.list_contacts(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListContactsRequest()
    assert isinstance(response, pagers.ListContactsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_contacts_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_contacts), '__call__') as call:
        client.list_contacts()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListContactsRequest()

@pytest.mark.asyncio
async def test_list_contacts_async(transport: str='grpc_asyncio', request_type=service.ListContactsRequest):
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_contacts), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListContactsResponse(next_page_token='next_page_token_value'))
        response = await client.list_contacts(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListContactsRequest()
    assert isinstance(response, pagers.ListContactsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_contacts_async_from_dict():
    await test_list_contacts_async(request_type=dict)

def test_list_contacts_field_headers():
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListContactsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_contacts), '__call__') as call:
        call.return_value = service.ListContactsResponse()
        client.list_contacts(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_contacts_field_headers_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListContactsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_contacts), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListContactsResponse())
        await client.list_contacts(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_contacts_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_contacts), '__call__') as call:
        call.return_value = service.ListContactsResponse()
        client.list_contacts(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_contacts_flattened_error():
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_contacts(service.ListContactsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_contacts_flattened_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_contacts), '__call__') as call:
        call.return_value = service.ListContactsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListContactsResponse())
        response = await client.list_contacts(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_contacts_flattened_error_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_contacts(service.ListContactsRequest(), parent='parent_value')

def test_list_contacts_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_contacts), '__call__') as call:
        call.side_effect = (service.ListContactsResponse(contacts=[service.Contact(), service.Contact(), service.Contact()], next_page_token='abc'), service.ListContactsResponse(contacts=[], next_page_token='def'), service.ListContactsResponse(contacts=[service.Contact()], next_page_token='ghi'), service.ListContactsResponse(contacts=[service.Contact(), service.Contact()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_contacts(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.Contact) for i in results))

def test_list_contacts_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_contacts), '__call__') as call:
        call.side_effect = (service.ListContactsResponse(contacts=[service.Contact(), service.Contact(), service.Contact()], next_page_token='abc'), service.ListContactsResponse(contacts=[], next_page_token='def'), service.ListContactsResponse(contacts=[service.Contact()], next_page_token='ghi'), service.ListContactsResponse(contacts=[service.Contact(), service.Contact()]), RuntimeError)
        pages = list(client.list_contacts(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_contacts_async_pager():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_contacts), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListContactsResponse(contacts=[service.Contact(), service.Contact(), service.Contact()], next_page_token='abc'), service.ListContactsResponse(contacts=[], next_page_token='def'), service.ListContactsResponse(contacts=[service.Contact()], next_page_token='ghi'), service.ListContactsResponse(contacts=[service.Contact(), service.Contact()]), RuntimeError)
        async_pager = await client.list_contacts(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, service.Contact) for i in responses))

@pytest.mark.asyncio
async def test_list_contacts_async_pages():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_contacts), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListContactsResponse(contacts=[service.Contact(), service.Contact(), service.Contact()], next_page_token='abc'), service.ListContactsResponse(contacts=[], next_page_token='def'), service.ListContactsResponse(contacts=[service.Contact()], next_page_token='ghi'), service.ListContactsResponse(contacts=[service.Contact(), service.Contact()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_contacts(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetContactRequest, dict])
def test_get_contact(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_contact), '__call__') as call:
        call.return_value = service.Contact(name='name_value', email='email_value', notification_category_subscriptions=[enums.NotificationCategory.ALL], language_tag='language_tag_value', validation_state=enums.ValidationState.VALID)
        response = client.get_contact(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetContactRequest()
    assert isinstance(response, service.Contact)
    assert response.name == 'name_value'
    assert response.email == 'email_value'
    assert response.notification_category_subscriptions == [enums.NotificationCategory.ALL]
    assert response.language_tag == 'language_tag_value'
    assert response.validation_state == enums.ValidationState.VALID

def test_get_contact_empty_call():
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_contact), '__call__') as call:
        client.get_contact()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetContactRequest()

@pytest.mark.asyncio
async def test_get_contact_async(transport: str='grpc_asyncio', request_type=service.GetContactRequest):
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_contact), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Contact(name='name_value', email='email_value', notification_category_subscriptions=[enums.NotificationCategory.ALL], language_tag='language_tag_value', validation_state=enums.ValidationState.VALID))
        response = await client.get_contact(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetContactRequest()
    assert isinstance(response, service.Contact)
    assert response.name == 'name_value'
    assert response.email == 'email_value'
    assert response.notification_category_subscriptions == [enums.NotificationCategory.ALL]
    assert response.language_tag == 'language_tag_value'
    assert response.validation_state == enums.ValidationState.VALID

@pytest.mark.asyncio
async def test_get_contact_async_from_dict():
    await test_get_contact_async(request_type=dict)

def test_get_contact_field_headers():
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetContactRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_contact), '__call__') as call:
        call.return_value = service.Contact()
        client.get_contact(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_contact_field_headers_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetContactRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_contact), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Contact())
        await client.get_contact(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_contact_flattened():
    if False:
        while True:
            i = 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_contact), '__call__') as call:
        call.return_value = service.Contact()
        client.get_contact(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_contact_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_contact(service.GetContactRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_contact_flattened_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_contact), '__call__') as call:
        call.return_value = service.Contact()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Contact())
        response = await client.get_contact(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_contact_flattened_error_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_contact(service.GetContactRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.DeleteContactRequest, dict])
def test_delete_contact(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_contact), '__call__') as call:
        call.return_value = None
        response = client.delete_contact(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteContactRequest()
    assert response is None

def test_delete_contact_empty_call():
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_contact), '__call__') as call:
        client.delete_contact()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteContactRequest()

@pytest.mark.asyncio
async def test_delete_contact_async(transport: str='grpc_asyncio', request_type=service.DeleteContactRequest):
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_contact), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_contact(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteContactRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_contact_async_from_dict():
    await test_delete_contact_async(request_type=dict)

def test_delete_contact_field_headers():
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteContactRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_contact), '__call__') as call:
        call.return_value = None
        client.delete_contact(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_contact_field_headers_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteContactRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_contact), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_contact(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_contact_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_contact), '__call__') as call:
        call.return_value = None
        client.delete_contact(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_contact_flattened_error():
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_contact(service.DeleteContactRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_contact_flattened_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_contact), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_contact(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_contact_flattened_error_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_contact(service.DeleteContactRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ComputeContactsRequest, dict])
def test_compute_contacts(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.compute_contacts), '__call__') as call:
        call.return_value = service.ComputeContactsResponse(next_page_token='next_page_token_value')
        response = client.compute_contacts(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ComputeContactsRequest()
    assert isinstance(response, pagers.ComputeContactsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_compute_contacts_empty_call():
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.compute_contacts), '__call__') as call:
        client.compute_contacts()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ComputeContactsRequest()

@pytest.mark.asyncio
async def test_compute_contacts_async(transport: str='grpc_asyncio', request_type=service.ComputeContactsRequest):
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.compute_contacts), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ComputeContactsResponse(next_page_token='next_page_token_value'))
        response = await client.compute_contacts(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ComputeContactsRequest()
    assert isinstance(response, pagers.ComputeContactsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_compute_contacts_async_from_dict():
    await test_compute_contacts_async(request_type=dict)

def test_compute_contacts_field_headers():
    if False:
        while True:
            i = 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ComputeContactsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.compute_contacts), '__call__') as call:
        call.return_value = service.ComputeContactsResponse()
        client.compute_contacts(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_compute_contacts_field_headers_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ComputeContactsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.compute_contacts), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ComputeContactsResponse())
        await client.compute_contacts(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_compute_contacts_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.compute_contacts), '__call__') as call:
        call.side_effect = (service.ComputeContactsResponse(contacts=[service.Contact(), service.Contact(), service.Contact()], next_page_token='abc'), service.ComputeContactsResponse(contacts=[], next_page_token='def'), service.ComputeContactsResponse(contacts=[service.Contact()], next_page_token='ghi'), service.ComputeContactsResponse(contacts=[service.Contact(), service.Contact()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.compute_contacts(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.Contact) for i in results))

def test_compute_contacts_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.compute_contacts), '__call__') as call:
        call.side_effect = (service.ComputeContactsResponse(contacts=[service.Contact(), service.Contact(), service.Contact()], next_page_token='abc'), service.ComputeContactsResponse(contacts=[], next_page_token='def'), service.ComputeContactsResponse(contacts=[service.Contact()], next_page_token='ghi'), service.ComputeContactsResponse(contacts=[service.Contact(), service.Contact()]), RuntimeError)
        pages = list(client.compute_contacts(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_compute_contacts_async_pager():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.compute_contacts), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ComputeContactsResponse(contacts=[service.Contact(), service.Contact(), service.Contact()], next_page_token='abc'), service.ComputeContactsResponse(contacts=[], next_page_token='def'), service.ComputeContactsResponse(contacts=[service.Contact()], next_page_token='ghi'), service.ComputeContactsResponse(contacts=[service.Contact(), service.Contact()]), RuntimeError)
        async_pager = await client.compute_contacts(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, service.Contact) for i in responses))

@pytest.mark.asyncio
async def test_compute_contacts_async_pages():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.compute_contacts), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ComputeContactsResponse(contacts=[service.Contact(), service.Contact(), service.Contact()], next_page_token='abc'), service.ComputeContactsResponse(contacts=[], next_page_token='def'), service.ComputeContactsResponse(contacts=[service.Contact()], next_page_token='ghi'), service.ComputeContactsResponse(contacts=[service.Contact(), service.Contact()]), RuntimeError)
        pages = []
        async for page_ in (await client.compute_contacts(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.SendTestMessageRequest, dict])
def test_send_test_message(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.send_test_message), '__call__') as call:
        call.return_value = None
        response = client.send_test_message(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.SendTestMessageRequest()
    assert response is None

def test_send_test_message_empty_call():
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.send_test_message), '__call__') as call:
        client.send_test_message()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.SendTestMessageRequest()

@pytest.mark.asyncio
async def test_send_test_message_async(transport: str='grpc_asyncio', request_type=service.SendTestMessageRequest):
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.send_test_message), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.send_test_message(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.SendTestMessageRequest()
    assert response is None

@pytest.mark.asyncio
async def test_send_test_message_async_from_dict():
    await test_send_test_message_async(request_type=dict)

def test_send_test_message_field_headers():
    if False:
        while True:
            i = 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.SendTestMessageRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.send_test_message), '__call__') as call:
        call.return_value = None
        client.send_test_message(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_send_test_message_field_headers_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.SendTestMessageRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.send_test_message), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.send_test_message(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.CreateContactRequest, dict])
def test_create_contact_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request_init['contact'] = {'name': 'name_value', 'email': 'email_value', 'notification_category_subscriptions': [2], 'language_tag': 'language_tag_value', 'validation_state': 1, 'validate_time': {'seconds': 751, 'nanos': 543}}
    test_field = service.CreateContactRequest.meta.fields['contact']

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
    for (field, value) in request_init['contact'].items():
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
                for i in range(0, len(request_init['contact'][field])):
                    del request_init['contact'][field][i][subfield]
            else:
                del request_init['contact'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.Contact(name='name_value', email='email_value', notification_category_subscriptions=[enums.NotificationCategory.ALL], language_tag='language_tag_value', validation_state=enums.ValidationState.VALID)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.Contact.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_contact(request)
    assert isinstance(response, service.Contact)
    assert response.name == 'name_value'
    assert response.email == 'email_value'
    assert response.notification_category_subscriptions == [enums.NotificationCategory.ALL]
    assert response.language_tag == 'language_tag_value'
    assert response.validation_state == enums.ValidationState.VALID

def test_create_contact_rest_required_fields(request_type=service.CreateContactRequest):
    if False:
        return 10
    transport_class = transports.EssentialContactsServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_contact._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_contact._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.Contact()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.Contact.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_contact(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_contact_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EssentialContactsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_contact._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'contact'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_contact_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EssentialContactsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EssentialContactsServiceRestInterceptor())
    client = EssentialContactsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EssentialContactsServiceRestInterceptor, 'post_create_contact') as post, mock.patch.object(transports.EssentialContactsServiceRestInterceptor, 'pre_create_contact') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateContactRequest.pb(service.CreateContactRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.Contact.to_json(service.Contact())
        request = service.CreateContactRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.Contact()
        client.create_contact(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_contact_rest_bad_request(transport: str='rest', request_type=service.CreateContactRequest):
    if False:
        i = 10
        return i + 15
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_contact(request)

def test_create_contact_rest_flattened():
    if False:
        while True:
            i = 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.Contact()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', contact=service.Contact(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.Contact.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_contact(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*}/contacts' % client.transport._host, args[1])

def test_create_contact_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_contact(service.CreateContactRequest(), parent='parent_value', contact=service.Contact(name='name_value'))

def test_create_contact_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateContactRequest, dict])
def test_update_contact_rest(request_type):
    if False:
        while True:
            i = 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'contact': {'name': 'projects/sample1/contacts/sample2'}}
    request_init['contact'] = {'name': 'projects/sample1/contacts/sample2', 'email': 'email_value', 'notification_category_subscriptions': [2], 'language_tag': 'language_tag_value', 'validation_state': 1, 'validate_time': {'seconds': 751, 'nanos': 543}}
    test_field = service.UpdateContactRequest.meta.fields['contact']

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
    for (field, value) in request_init['contact'].items():
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
                for i in range(0, len(request_init['contact'][field])):
                    del request_init['contact'][field][i][subfield]
            else:
                del request_init['contact'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.Contact(name='name_value', email='email_value', notification_category_subscriptions=[enums.NotificationCategory.ALL], language_tag='language_tag_value', validation_state=enums.ValidationState.VALID)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.Contact.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_contact(request)
    assert isinstance(response, service.Contact)
    assert response.name == 'name_value'
    assert response.email == 'email_value'
    assert response.notification_category_subscriptions == [enums.NotificationCategory.ALL]
    assert response.language_tag == 'language_tag_value'
    assert response.validation_state == enums.ValidationState.VALID

def test_update_contact_rest_required_fields(request_type=service.UpdateContactRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.EssentialContactsServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_contact._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_contact._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.Contact()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.Contact.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_contact(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_contact_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EssentialContactsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_contact._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('contact',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_contact_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.EssentialContactsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EssentialContactsServiceRestInterceptor())
    client = EssentialContactsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EssentialContactsServiceRestInterceptor, 'post_update_contact') as post, mock.patch.object(transports.EssentialContactsServiceRestInterceptor, 'pre_update_contact') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateContactRequest.pb(service.UpdateContactRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.Contact.to_json(service.Contact())
        request = service.UpdateContactRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.Contact()
        client.update_contact(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_contact_rest_bad_request(transport: str='rest', request_type=service.UpdateContactRequest):
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'contact': {'name': 'projects/sample1/contacts/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_contact(request)

def test_update_contact_rest_flattened():
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.Contact()
        sample_request = {'contact': {'name': 'projects/sample1/contacts/sample2'}}
        mock_args = dict(contact=service.Contact(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.Contact.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_contact(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{contact.name=projects/*/contacts/*}' % client.transport._host, args[1])

def test_update_contact_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_contact(service.UpdateContactRequest(), contact=service.Contact(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_contact_rest_error():
    if False:
        while True:
            i = 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListContactsRequest, dict])
def test_list_contacts_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListContactsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListContactsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_contacts(request)
    assert isinstance(response, pagers.ListContactsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_contacts_rest_required_fields(request_type=service.ListContactsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EssentialContactsServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_contacts._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_contacts._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListContactsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListContactsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_contacts(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_contacts_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EssentialContactsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_contacts._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_contacts_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.EssentialContactsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EssentialContactsServiceRestInterceptor())
    client = EssentialContactsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EssentialContactsServiceRestInterceptor, 'post_list_contacts') as post, mock.patch.object(transports.EssentialContactsServiceRestInterceptor, 'pre_list_contacts') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListContactsRequest.pb(service.ListContactsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListContactsResponse.to_json(service.ListContactsResponse())
        request = service.ListContactsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListContactsResponse()
        client.list_contacts(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_contacts_rest_bad_request(transport: str='rest', request_type=service.ListContactsRequest):
    if False:
        i = 10
        return i + 15
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_contacts(request)

def test_list_contacts_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListContactsResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListContactsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_contacts(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*}/contacts' % client.transport._host, args[1])

def test_list_contacts_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_contacts(service.ListContactsRequest(), parent='parent_value')

def test_list_contacts_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListContactsResponse(contacts=[service.Contact(), service.Contact(), service.Contact()], next_page_token='abc'), service.ListContactsResponse(contacts=[], next_page_token='def'), service.ListContactsResponse(contacts=[service.Contact()], next_page_token='ghi'), service.ListContactsResponse(contacts=[service.Contact(), service.Contact()]))
        response = response + response
        response = tuple((service.ListContactsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_contacts(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.Contact) for i in results))
        pages = list(client.list_contacts(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetContactRequest, dict])
def test_get_contact_rest(request_type):
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/contacts/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.Contact(name='name_value', email='email_value', notification_category_subscriptions=[enums.NotificationCategory.ALL], language_tag='language_tag_value', validation_state=enums.ValidationState.VALID)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.Contact.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_contact(request)
    assert isinstance(response, service.Contact)
    assert response.name == 'name_value'
    assert response.email == 'email_value'
    assert response.notification_category_subscriptions == [enums.NotificationCategory.ALL]
    assert response.language_tag == 'language_tag_value'
    assert response.validation_state == enums.ValidationState.VALID

def test_get_contact_rest_required_fields(request_type=service.GetContactRequest):
    if False:
        return 10
    transport_class = transports.EssentialContactsServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_contact._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_contact._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.Contact()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.Contact.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_contact(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_contact_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EssentialContactsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_contact._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_contact_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EssentialContactsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EssentialContactsServiceRestInterceptor())
    client = EssentialContactsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EssentialContactsServiceRestInterceptor, 'post_get_contact') as post, mock.patch.object(transports.EssentialContactsServiceRestInterceptor, 'pre_get_contact') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetContactRequest.pb(service.GetContactRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.Contact.to_json(service.Contact())
        request = service.GetContactRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.Contact()
        client.get_contact(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_contact_rest_bad_request(transport: str='rest', request_type=service.GetContactRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/contacts/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_contact(request)

def test_get_contact_rest_flattened():
    if False:
        while True:
            i = 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.Contact()
        sample_request = {'name': 'projects/sample1/contacts/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.Contact.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_contact(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/contacts/*}' % client.transport._host, args[1])

def test_get_contact_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_contact(service.GetContactRequest(), name='name_value')

def test_get_contact_rest_error():
    if False:
        i = 10
        return i + 15
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteContactRequest, dict])
def test_delete_contact_rest(request_type):
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/contacts/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_contact(request)
    assert response is None

def test_delete_contact_rest_required_fields(request_type=service.DeleteContactRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EssentialContactsServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_contact._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_contact._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_contact(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_contact_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EssentialContactsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_contact._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_contact_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EssentialContactsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EssentialContactsServiceRestInterceptor())
    client = EssentialContactsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EssentialContactsServiceRestInterceptor, 'pre_delete_contact') as pre:
        pre.assert_not_called()
        pb_message = service.DeleteContactRequest.pb(service.DeleteContactRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = service.DeleteContactRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_contact(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_contact_rest_bad_request(transport: str='rest', request_type=service.DeleteContactRequest):
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/contacts/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_contact(request)

def test_delete_contact_rest_flattened():
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/contacts/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_contact(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/contacts/*}' % client.transport._host, args[1])

def test_delete_contact_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_contact(service.DeleteContactRequest(), name='name_value')

def test_delete_contact_rest_error():
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ComputeContactsRequest, dict])
def test_compute_contacts_rest(request_type):
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ComputeContactsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ComputeContactsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.compute_contacts(request)
    assert isinstance(response, pagers.ComputeContactsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_compute_contacts_rest_required_fields(request_type=service.ComputeContactsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EssentialContactsServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).compute_contacts._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).compute_contacts._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('notification_categories', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ComputeContactsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ComputeContactsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.compute_contacts(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_compute_contacts_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EssentialContactsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.compute_contacts._get_unset_required_fields({})
    assert set(unset_fields) == set(('notificationCategories', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_compute_contacts_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EssentialContactsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EssentialContactsServiceRestInterceptor())
    client = EssentialContactsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EssentialContactsServiceRestInterceptor, 'post_compute_contacts') as post, mock.patch.object(transports.EssentialContactsServiceRestInterceptor, 'pre_compute_contacts') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ComputeContactsRequest.pb(service.ComputeContactsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ComputeContactsResponse.to_json(service.ComputeContactsResponse())
        request = service.ComputeContactsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ComputeContactsResponse()
        client.compute_contacts(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_compute_contacts_rest_bad_request(transport: str='rest', request_type=service.ComputeContactsRequest):
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.compute_contacts(request)

def test_compute_contacts_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ComputeContactsResponse(contacts=[service.Contact(), service.Contact(), service.Contact()], next_page_token='abc'), service.ComputeContactsResponse(contacts=[], next_page_token='def'), service.ComputeContactsResponse(contacts=[service.Contact()], next_page_token='ghi'), service.ComputeContactsResponse(contacts=[service.Contact(), service.Contact()]))
        response = response + response
        response = tuple((service.ComputeContactsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.compute_contacts(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.Contact) for i in results))
        pages = list(client.compute_contacts(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.SendTestMessageRequest, dict])
def test_send_test_message_rest(request_type):
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.send_test_message(request)
    assert response is None

def test_send_test_message_rest_required_fields(request_type=service.SendTestMessageRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EssentialContactsServiceRestTransport
    request_init = {}
    request_init['contacts'] = ''
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).send_test_message._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['contacts'] = 'contacts_value'
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).send_test_message._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'contacts' in jsonified_request
    assert jsonified_request['contacts'] == 'contacts_value'
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = None
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = ''
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.send_test_message(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_send_test_message_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EssentialContactsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.send_test_message._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('contacts', 'resource', 'notificationCategory'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_send_test_message_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EssentialContactsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EssentialContactsServiceRestInterceptor())
    client = EssentialContactsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EssentialContactsServiceRestInterceptor, 'pre_send_test_message') as pre:
        pre.assert_not_called()
        pb_message = service.SendTestMessageRequest.pb(service.SendTestMessageRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = service.SendTestMessageRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.send_test_message(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_send_test_message_rest_bad_request(transport: str='rest', request_type=service.SendTestMessageRequest):
    if False:
        i = 10
        return i + 15
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.send_test_message(request)

def test_send_test_message_rest_error():
    if False:
        i = 10
        return i + 15
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.EssentialContactsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.EssentialContactsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EssentialContactsServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.EssentialContactsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = EssentialContactsServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = EssentialContactsServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.EssentialContactsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EssentialContactsServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.EssentialContactsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = EssentialContactsServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.EssentialContactsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.EssentialContactsServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.EssentialContactsServiceGrpcTransport, transports.EssentialContactsServiceGrpcAsyncIOTransport, transports.EssentialContactsServiceRestTransport])
def test_transport_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        i = 10
        return i + 15
    transport = EssentialContactsServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.EssentialContactsServiceGrpcTransport)

def test_essential_contacts_service_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.EssentialContactsServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_essential_contacts_service_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.essential_contacts_v1.services.essential_contacts_service.transports.EssentialContactsServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.EssentialContactsServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_contact', 'update_contact', 'list_contacts', 'get_contact', 'delete_contact', 'compute_contacts', 'send_test_message')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_essential_contacts_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.essential_contacts_v1.services.essential_contacts_service.transports.EssentialContactsServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.EssentialContactsServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_essential_contacts_service_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.essential_contacts_v1.services.essential_contacts_service.transports.EssentialContactsServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.EssentialContactsServiceTransport()
        adc.assert_called_once()

def test_essential_contacts_service_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        EssentialContactsServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.EssentialContactsServiceGrpcTransport, transports.EssentialContactsServiceGrpcAsyncIOTransport])
def test_essential_contacts_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.EssentialContactsServiceGrpcTransport, transports.EssentialContactsServiceGrpcAsyncIOTransport, transports.EssentialContactsServiceRestTransport])
def test_essential_contacts_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.EssentialContactsServiceGrpcTransport, grpc_helpers), (transports.EssentialContactsServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_essential_contacts_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('essentialcontacts.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='essentialcontacts.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.EssentialContactsServiceGrpcTransport, transports.EssentialContactsServiceGrpcAsyncIOTransport])
def test_essential_contacts_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_essential_contacts_service_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.EssentialContactsServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_essential_contacts_service_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='essentialcontacts.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('essentialcontacts.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://essentialcontacts.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_essential_contacts_service_host_with_port(transport_name):
    if False:
        return 10
    client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='essentialcontacts.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('essentialcontacts.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://essentialcontacts.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_essential_contacts_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = EssentialContactsServiceClient(credentials=creds1, transport=transport_name)
    client2 = EssentialContactsServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_contact._session
    session2 = client2.transport.create_contact._session
    assert session1 != session2
    session1 = client1.transport.update_contact._session
    session2 = client2.transport.update_contact._session
    assert session1 != session2
    session1 = client1.transport.list_contacts._session
    session2 = client2.transport.list_contacts._session
    assert session1 != session2
    session1 = client1.transport.get_contact._session
    session2 = client2.transport.get_contact._session
    assert session1 != session2
    session1 = client1.transport.delete_contact._session
    session2 = client2.transport.delete_contact._session
    assert session1 != session2
    session1 = client1.transport.compute_contacts._session
    session2 = client2.transport.compute_contacts._session
    assert session1 != session2
    session1 = client1.transport.send_test_message._session
    session2 = client2.transport.send_test_message._session
    assert session1 != session2

def test_essential_contacts_service_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.EssentialContactsServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_essential_contacts_service_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.EssentialContactsServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.EssentialContactsServiceGrpcTransport, transports.EssentialContactsServiceGrpcAsyncIOTransport])
def test_essential_contacts_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.EssentialContactsServiceGrpcTransport, transports.EssentialContactsServiceGrpcAsyncIOTransport])
def test_essential_contacts_service_transport_channel_mtls_with_adc(transport_class):
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

def test_contact_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    contact = 'clam'
    expected = 'projects/{project}/contacts/{contact}'.format(project=project, contact=contact)
    actual = EssentialContactsServiceClient.contact_path(project, contact)
    assert expected == actual

def test_parse_contact_path():
    if False:
        return 10
    expected = {'project': 'whelk', 'contact': 'octopus'}
    path = EssentialContactsServiceClient.contact_path(**expected)
    actual = EssentialContactsServiceClient.parse_contact_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = EssentialContactsServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'nudibranch'}
    path = EssentialContactsServiceClient.common_billing_account_path(**expected)
    actual = EssentialContactsServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = EssentialContactsServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'mussel'}
    path = EssentialContactsServiceClient.common_folder_path(**expected)
    actual = EssentialContactsServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = EssentialContactsServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'nautilus'}
    path = EssentialContactsServiceClient.common_organization_path(**expected)
    actual = EssentialContactsServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = EssentialContactsServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'abalone'}
    path = EssentialContactsServiceClient.common_project_path(**expected)
    actual = EssentialContactsServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = EssentialContactsServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = EssentialContactsServiceClient.common_location_path(**expected)
    actual = EssentialContactsServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.EssentialContactsServiceTransport, '_prep_wrapped_messages') as prep:
        client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.EssentialContactsServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = EssentialContactsServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = EssentialContactsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = EssentialContactsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(EssentialContactsServiceClient, transports.EssentialContactsServiceGrpcTransport), (EssentialContactsServiceAsyncClient, transports.EssentialContactsServiceGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)