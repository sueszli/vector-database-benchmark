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
from google.protobuf import json_format
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.resourcesettings_v1.services.resource_settings_service import ResourceSettingsServiceAsyncClient, ResourceSettingsServiceClient, pagers, transports
from google.cloud.resourcesettings_v1.types import resource_settings

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
    assert ResourceSettingsServiceClient._get_default_mtls_endpoint(None) is None
    assert ResourceSettingsServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ResourceSettingsServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ResourceSettingsServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ResourceSettingsServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ResourceSettingsServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ResourceSettingsServiceClient, 'grpc'), (ResourceSettingsServiceAsyncClient, 'grpc_asyncio'), (ResourceSettingsServiceClient, 'rest')])
def test_resource_settings_service_client_from_service_account_info(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('resourcesettings.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://resourcesettings.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ResourceSettingsServiceGrpcTransport, 'grpc'), (transports.ResourceSettingsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.ResourceSettingsServiceRestTransport, 'rest')])
def test_resource_settings_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ResourceSettingsServiceClient, 'grpc'), (ResourceSettingsServiceAsyncClient, 'grpc_asyncio'), (ResourceSettingsServiceClient, 'rest')])
def test_resource_settings_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('resourcesettings.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://resourcesettings.googleapis.com')

def test_resource_settings_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = ResourceSettingsServiceClient.get_transport_class()
    available_transports = [transports.ResourceSettingsServiceGrpcTransport, transports.ResourceSettingsServiceRestTransport]
    assert transport in available_transports
    transport = ResourceSettingsServiceClient.get_transport_class('grpc')
    assert transport == transports.ResourceSettingsServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ResourceSettingsServiceClient, transports.ResourceSettingsServiceGrpcTransport, 'grpc'), (ResourceSettingsServiceAsyncClient, transports.ResourceSettingsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (ResourceSettingsServiceClient, transports.ResourceSettingsServiceRestTransport, 'rest')])
@mock.patch.object(ResourceSettingsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ResourceSettingsServiceClient))
@mock.patch.object(ResourceSettingsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ResourceSettingsServiceAsyncClient))
def test_resource_settings_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(ResourceSettingsServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ResourceSettingsServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ResourceSettingsServiceClient, transports.ResourceSettingsServiceGrpcTransport, 'grpc', 'true'), (ResourceSettingsServiceAsyncClient, transports.ResourceSettingsServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ResourceSettingsServiceClient, transports.ResourceSettingsServiceGrpcTransport, 'grpc', 'false'), (ResourceSettingsServiceAsyncClient, transports.ResourceSettingsServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (ResourceSettingsServiceClient, transports.ResourceSettingsServiceRestTransport, 'rest', 'true'), (ResourceSettingsServiceClient, transports.ResourceSettingsServiceRestTransport, 'rest', 'false')])
@mock.patch.object(ResourceSettingsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ResourceSettingsServiceClient))
@mock.patch.object(ResourceSettingsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ResourceSettingsServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_resource_settings_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [ResourceSettingsServiceClient, ResourceSettingsServiceAsyncClient])
@mock.patch.object(ResourceSettingsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ResourceSettingsServiceClient))
@mock.patch.object(ResourceSettingsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ResourceSettingsServiceAsyncClient))
def test_resource_settings_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ResourceSettingsServiceClient, transports.ResourceSettingsServiceGrpcTransport, 'grpc'), (ResourceSettingsServiceAsyncClient, transports.ResourceSettingsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (ResourceSettingsServiceClient, transports.ResourceSettingsServiceRestTransport, 'rest')])
def test_resource_settings_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ResourceSettingsServiceClient, transports.ResourceSettingsServiceGrpcTransport, 'grpc', grpc_helpers), (ResourceSettingsServiceAsyncClient, transports.ResourceSettingsServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (ResourceSettingsServiceClient, transports.ResourceSettingsServiceRestTransport, 'rest', None)])
def test_resource_settings_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_resource_settings_service_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.resourcesettings_v1.services.resource_settings_service.transports.ResourceSettingsServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ResourceSettingsServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ResourceSettingsServiceClient, transports.ResourceSettingsServiceGrpcTransport, 'grpc', grpc_helpers), (ResourceSettingsServiceAsyncClient, transports.ResourceSettingsServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_resource_settings_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('resourcesettings.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='resourcesettings.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [resource_settings.ListSettingsRequest, dict])
def test_list_settings(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_settings), '__call__') as call:
        call.return_value = resource_settings.ListSettingsResponse(next_page_token='next_page_token_value')
        response = client.list_settings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == resource_settings.ListSettingsRequest()
    assert isinstance(response, pagers.ListSettingsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_settings_empty_call():
    if False:
        while True:
            i = 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_settings), '__call__') as call:
        client.list_settings()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == resource_settings.ListSettingsRequest()

@pytest.mark.asyncio
async def test_list_settings_async(transport: str='grpc_asyncio', request_type=resource_settings.ListSettingsRequest):
    client = ResourceSettingsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_settings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resource_settings.ListSettingsResponse(next_page_token='next_page_token_value'))
        response = await client.list_settings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == resource_settings.ListSettingsRequest()
    assert isinstance(response, pagers.ListSettingsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_settings_async_from_dict():
    await test_list_settings_async(request_type=dict)

def test_list_settings_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = resource_settings.ListSettingsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_settings), '__call__') as call:
        call.return_value = resource_settings.ListSettingsResponse()
        client.list_settings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_settings_field_headers_async():
    client = ResourceSettingsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = resource_settings.ListSettingsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_settings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resource_settings.ListSettingsResponse())
        await client.list_settings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_settings_flattened():
    if False:
        while True:
            i = 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_settings), '__call__') as call:
        call.return_value = resource_settings.ListSettingsResponse()
        client.list_settings(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_settings_flattened_error():
    if False:
        return 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_settings(resource_settings.ListSettingsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_settings_flattened_async():
    client = ResourceSettingsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_settings), '__call__') as call:
        call.return_value = resource_settings.ListSettingsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resource_settings.ListSettingsResponse())
        response = await client.list_settings(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_settings_flattened_error_async():
    client = ResourceSettingsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_settings(resource_settings.ListSettingsRequest(), parent='parent_value')

def test_list_settings_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_settings), '__call__') as call:
        call.side_effect = (resource_settings.ListSettingsResponse(settings=[resource_settings.Setting(), resource_settings.Setting(), resource_settings.Setting()], next_page_token='abc'), resource_settings.ListSettingsResponse(settings=[], next_page_token='def'), resource_settings.ListSettingsResponse(settings=[resource_settings.Setting()], next_page_token='ghi'), resource_settings.ListSettingsResponse(settings=[resource_settings.Setting(), resource_settings.Setting()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_settings(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resource_settings.Setting) for i in results))

def test_list_settings_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_settings), '__call__') as call:
        call.side_effect = (resource_settings.ListSettingsResponse(settings=[resource_settings.Setting(), resource_settings.Setting(), resource_settings.Setting()], next_page_token='abc'), resource_settings.ListSettingsResponse(settings=[], next_page_token='def'), resource_settings.ListSettingsResponse(settings=[resource_settings.Setting()], next_page_token='ghi'), resource_settings.ListSettingsResponse(settings=[resource_settings.Setting(), resource_settings.Setting()]), RuntimeError)
        pages = list(client.list_settings(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_settings_async_pager():
    client = ResourceSettingsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_settings), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (resource_settings.ListSettingsResponse(settings=[resource_settings.Setting(), resource_settings.Setting(), resource_settings.Setting()], next_page_token='abc'), resource_settings.ListSettingsResponse(settings=[], next_page_token='def'), resource_settings.ListSettingsResponse(settings=[resource_settings.Setting()], next_page_token='ghi'), resource_settings.ListSettingsResponse(settings=[resource_settings.Setting(), resource_settings.Setting()]), RuntimeError)
        async_pager = await client.list_settings(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resource_settings.Setting) for i in responses))

@pytest.mark.asyncio
async def test_list_settings_async_pages():
    client = ResourceSettingsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_settings), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (resource_settings.ListSettingsResponse(settings=[resource_settings.Setting(), resource_settings.Setting(), resource_settings.Setting()], next_page_token='abc'), resource_settings.ListSettingsResponse(settings=[], next_page_token='def'), resource_settings.ListSettingsResponse(settings=[resource_settings.Setting()], next_page_token='ghi'), resource_settings.ListSettingsResponse(settings=[resource_settings.Setting(), resource_settings.Setting()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_settings(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [resource_settings.GetSettingRequest, dict])
def test_get_setting(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_setting), '__call__') as call:
        call.return_value = resource_settings.Setting(name='name_value', etag='etag_value')
        response = client.get_setting(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == resource_settings.GetSettingRequest()
    assert isinstance(response, resource_settings.Setting)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'

def test_get_setting_empty_call():
    if False:
        i = 10
        return i + 15
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_setting), '__call__') as call:
        client.get_setting()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == resource_settings.GetSettingRequest()

@pytest.mark.asyncio
async def test_get_setting_async(transport: str='grpc_asyncio', request_type=resource_settings.GetSettingRequest):
    client = ResourceSettingsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_setting), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resource_settings.Setting(name='name_value', etag='etag_value'))
        response = await client.get_setting(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == resource_settings.GetSettingRequest()
    assert isinstance(response, resource_settings.Setting)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_setting_async_from_dict():
    await test_get_setting_async(request_type=dict)

def test_get_setting_field_headers():
    if False:
        return 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = resource_settings.GetSettingRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_setting), '__call__') as call:
        call.return_value = resource_settings.Setting()
        client.get_setting(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_setting_field_headers_async():
    client = ResourceSettingsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = resource_settings.GetSettingRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_setting), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resource_settings.Setting())
        await client.get_setting(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_setting_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_setting), '__call__') as call:
        call.return_value = resource_settings.Setting()
        client.get_setting(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_setting_flattened_error():
    if False:
        print('Hello World!')
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_setting(resource_settings.GetSettingRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_setting_flattened_async():
    client = ResourceSettingsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_setting), '__call__') as call:
        call.return_value = resource_settings.Setting()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resource_settings.Setting())
        response = await client.get_setting(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_setting_flattened_error_async():
    client = ResourceSettingsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_setting(resource_settings.GetSettingRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [resource_settings.UpdateSettingRequest, dict])
def test_update_setting(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_setting), '__call__') as call:
        call.return_value = resource_settings.Setting(name='name_value', etag='etag_value')
        response = client.update_setting(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == resource_settings.UpdateSettingRequest()
    assert isinstance(response, resource_settings.Setting)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'

def test_update_setting_empty_call():
    if False:
        return 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_setting), '__call__') as call:
        client.update_setting()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == resource_settings.UpdateSettingRequest()

@pytest.mark.asyncio
async def test_update_setting_async(transport: str='grpc_asyncio', request_type=resource_settings.UpdateSettingRequest):
    client = ResourceSettingsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_setting), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resource_settings.Setting(name='name_value', etag='etag_value'))
        response = await client.update_setting(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == resource_settings.UpdateSettingRequest()
    assert isinstance(response, resource_settings.Setting)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_update_setting_async_from_dict():
    await test_update_setting_async(request_type=dict)

def test_update_setting_field_headers():
    if False:
        return 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = resource_settings.UpdateSettingRequest()
    request.setting.name = 'name_value'
    with mock.patch.object(type(client.transport.update_setting), '__call__') as call:
        call.return_value = resource_settings.Setting()
        client.update_setting(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'setting.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_setting_field_headers_async():
    client = ResourceSettingsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = resource_settings.UpdateSettingRequest()
    request.setting.name = 'name_value'
    with mock.patch.object(type(client.transport.update_setting), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resource_settings.Setting())
        await client.update_setting(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'setting.name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [resource_settings.ListSettingsRequest, dict])
def test_list_settings_rest(request_type):
    if False:
        while True:
            i = 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resource_settings.ListSettingsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resource_settings.ListSettingsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_settings(request)
    assert isinstance(response, pagers.ListSettingsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_settings_rest_required_fields(request_type=resource_settings.ListSettingsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ResourceSettingsServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_settings._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_settings._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token', 'view'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resource_settings.ListSettingsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resource_settings.ListSettingsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_settings(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_settings_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ResourceSettingsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_settings._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken', 'view')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_settings_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ResourceSettingsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ResourceSettingsServiceRestInterceptor())
    client = ResourceSettingsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ResourceSettingsServiceRestInterceptor, 'post_list_settings') as post, mock.patch.object(transports.ResourceSettingsServiceRestInterceptor, 'pre_list_settings') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = resource_settings.ListSettingsRequest.pb(resource_settings.ListSettingsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resource_settings.ListSettingsResponse.to_json(resource_settings.ListSettingsResponse())
        request = resource_settings.ListSettingsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resource_settings.ListSettingsResponse()
        client.list_settings(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_settings_rest_bad_request(transport: str='rest', request_type=resource_settings.ListSettingsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_settings(request)

def test_list_settings_rest_flattened():
    if False:
        while True:
            i = 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resource_settings.ListSettingsResponse()
        sample_request = {'parent': 'organizations/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resource_settings.ListSettingsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_settings(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=organizations/*}/settings' % client.transport._host, args[1])

def test_list_settings_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_settings(resource_settings.ListSettingsRequest(), parent='parent_value')

def test_list_settings_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (resource_settings.ListSettingsResponse(settings=[resource_settings.Setting(), resource_settings.Setting(), resource_settings.Setting()], next_page_token='abc'), resource_settings.ListSettingsResponse(settings=[], next_page_token='def'), resource_settings.ListSettingsResponse(settings=[resource_settings.Setting()], next_page_token='ghi'), resource_settings.ListSettingsResponse(settings=[resource_settings.Setting(), resource_settings.Setting()]))
        response = response + response
        response = tuple((resource_settings.ListSettingsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'organizations/sample1'}
        pager = client.list_settings(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resource_settings.Setting) for i in results))
        pages = list(client.list_settings(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [resource_settings.GetSettingRequest, dict])
def test_get_setting_rest(request_type):
    if False:
        return 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/settings/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resource_settings.Setting(name='name_value', etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resource_settings.Setting.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_setting(request)
    assert isinstance(response, resource_settings.Setting)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'

def test_get_setting_rest_required_fields(request_type=resource_settings.GetSettingRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ResourceSettingsServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_setting._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_setting._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('view',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resource_settings.Setting()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resource_settings.Setting.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_setting(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_setting_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ResourceSettingsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_setting._get_unset_required_fields({})
    assert set(unset_fields) == set(('view',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_setting_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ResourceSettingsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ResourceSettingsServiceRestInterceptor())
    client = ResourceSettingsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ResourceSettingsServiceRestInterceptor, 'post_get_setting') as post, mock.patch.object(transports.ResourceSettingsServiceRestInterceptor, 'pre_get_setting') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = resource_settings.GetSettingRequest.pb(resource_settings.GetSettingRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resource_settings.Setting.to_json(resource_settings.Setting())
        request = resource_settings.GetSettingRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resource_settings.Setting()
        client.get_setting(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_setting_rest_bad_request(transport: str='rest', request_type=resource_settings.GetSettingRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'organizations/sample1/settings/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_setting(request)

def test_get_setting_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resource_settings.Setting()
        sample_request = {'name': 'organizations/sample1/settings/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resource_settings.Setting.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_setting(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=organizations/*/settings/*}' % client.transport._host, args[1])

def test_get_setting_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_setting(resource_settings.GetSettingRequest(), name='name_value')

def test_get_setting_rest_error():
    if False:
        return 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [resource_settings.UpdateSettingRequest, dict])
def test_update_setting_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'setting': {'name': 'organizations/sample1/settings/sample2'}}
    request_init['setting'] = {'name': 'organizations/sample1/settings/sample2', 'metadata': {'display_name': 'display_name_value', 'description': 'description_value', 'read_only': True, 'data_type': 1, 'default_value': {'boolean_value': True, 'string_value': 'string_value_value', 'string_set_value': {'values': ['values_value1', 'values_value2']}, 'enum_value': {'value': 'value_value'}}}, 'local_value': {}, 'effective_value': {}, 'etag': 'etag_value'}
    test_field = resource_settings.UpdateSettingRequest.meta.fields['setting']

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
    for (field, value) in request_init['setting'].items():
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
                for i in range(0, len(request_init['setting'][field])):
                    del request_init['setting'][field][i][subfield]
            else:
                del request_init['setting'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resource_settings.Setting(name='name_value', etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resource_settings.Setting.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_setting(request)
    assert isinstance(response, resource_settings.Setting)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'

def test_update_setting_rest_required_fields(request_type=resource_settings.UpdateSettingRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ResourceSettingsServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_setting._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_setting._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resource_settings.Setting()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resource_settings.Setting.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_setting(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_setting_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ResourceSettingsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_setting._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('setting',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_setting_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ResourceSettingsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ResourceSettingsServiceRestInterceptor())
    client = ResourceSettingsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ResourceSettingsServiceRestInterceptor, 'post_update_setting') as post, mock.patch.object(transports.ResourceSettingsServiceRestInterceptor, 'pre_update_setting') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = resource_settings.UpdateSettingRequest.pb(resource_settings.UpdateSettingRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resource_settings.Setting.to_json(resource_settings.Setting())
        request = resource_settings.UpdateSettingRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resource_settings.Setting()
        client.update_setting(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_setting_rest_bad_request(transport: str='rest', request_type=resource_settings.UpdateSettingRequest):
    if False:
        while True:
            i = 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'setting': {'name': 'organizations/sample1/settings/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_setting(request)

def test_update_setting_rest_error():
    if False:
        while True:
            i = 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.ResourceSettingsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ResourceSettingsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ResourceSettingsServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ResourceSettingsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ResourceSettingsServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ResourceSettingsServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ResourceSettingsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ResourceSettingsServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.ResourceSettingsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ResourceSettingsServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ResourceSettingsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ResourceSettingsServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ResourceSettingsServiceGrpcTransport, transports.ResourceSettingsServiceGrpcAsyncIOTransport, transports.ResourceSettingsServiceRestTransport])
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
    transport = ResourceSettingsServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ResourceSettingsServiceGrpcTransport)

def test_resource_settings_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ResourceSettingsServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_resource_settings_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.resourcesettings_v1.services.resource_settings_service.transports.ResourceSettingsServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ResourceSettingsServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_settings', 'get_setting', 'update_setting')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_resource_settings_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.resourcesettings_v1.services.resource_settings_service.transports.ResourceSettingsServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ResourceSettingsServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_resource_settings_service_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.resourcesettings_v1.services.resource_settings_service.transports.ResourceSettingsServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ResourceSettingsServiceTransport()
        adc.assert_called_once()

def test_resource_settings_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ResourceSettingsServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ResourceSettingsServiceGrpcTransport, transports.ResourceSettingsServiceGrpcAsyncIOTransport])
def test_resource_settings_service_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ResourceSettingsServiceGrpcTransport, transports.ResourceSettingsServiceGrpcAsyncIOTransport, transports.ResourceSettingsServiceRestTransport])
def test_resource_settings_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ResourceSettingsServiceGrpcTransport, grpc_helpers), (transports.ResourceSettingsServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_resource_settings_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('resourcesettings.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='resourcesettings.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ResourceSettingsServiceGrpcTransport, transports.ResourceSettingsServiceGrpcAsyncIOTransport])
def test_resource_settings_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_resource_settings_service_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.ResourceSettingsServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_resource_settings_service_host_no_port(transport_name):
    if False:
        return 10
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='resourcesettings.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('resourcesettings.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://resourcesettings.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_resource_settings_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='resourcesettings.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('resourcesettings.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://resourcesettings.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_resource_settings_service_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = ResourceSettingsServiceClient(credentials=creds1, transport=transport_name)
    client2 = ResourceSettingsServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_settings._session
    session2 = client2.transport.list_settings._session
    assert session1 != session2
    session1 = client1.transport.get_setting._session
    session2 = client2.transport.get_setting._session
    assert session1 != session2
    session1 = client1.transport.update_setting._session
    session2 = client2.transport.update_setting._session
    assert session1 != session2

def test_resource_settings_service_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ResourceSettingsServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_resource_settings_service_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ResourceSettingsServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ResourceSettingsServiceGrpcTransport, transports.ResourceSettingsServiceGrpcAsyncIOTransport])
def test_resource_settings_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.ResourceSettingsServiceGrpcTransport, transports.ResourceSettingsServiceGrpcAsyncIOTransport])
def test_resource_settings_service_transport_channel_mtls_with_adc(transport_class):
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

def test_setting_path():
    if False:
        while True:
            i = 10
    project_number = 'squid'
    setting_name = 'clam'
    expected = 'projects/{project_number}/settings/{setting_name}'.format(project_number=project_number, setting_name=setting_name)
    actual = ResourceSettingsServiceClient.setting_path(project_number, setting_name)
    assert expected == actual

def test_parse_setting_path():
    if False:
        i = 10
        return i + 15
    expected = {'project_number': 'whelk', 'setting_name': 'octopus'}
    path = ResourceSettingsServiceClient.setting_path(**expected)
    actual = ResourceSettingsServiceClient.parse_setting_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ResourceSettingsServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'nudibranch'}
    path = ResourceSettingsServiceClient.common_billing_account_path(**expected)
    actual = ResourceSettingsServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ResourceSettingsServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'mussel'}
    path = ResourceSettingsServiceClient.common_folder_path(**expected)
    actual = ResourceSettingsServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ResourceSettingsServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'nautilus'}
    path = ResourceSettingsServiceClient.common_organization_path(**expected)
    actual = ResourceSettingsServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = ResourceSettingsServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone'}
    path = ResourceSettingsServiceClient.common_project_path(**expected)
    actual = ResourceSettingsServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ResourceSettingsServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = ResourceSettingsServiceClient.common_location_path(**expected)
    actual = ResourceSettingsServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ResourceSettingsServiceTransport, '_prep_wrapped_messages') as prep:
        client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ResourceSettingsServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = ResourceSettingsServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ResourceSettingsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = ResourceSettingsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ResourceSettingsServiceClient, transports.ResourceSettingsServiceGrpcTransport), (ResourceSettingsServiceAsyncClient, transports.ResourceSettingsServiceGrpcAsyncIOTransport)])
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