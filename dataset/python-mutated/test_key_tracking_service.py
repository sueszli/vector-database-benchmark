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
from google.cloud.kms_inventory_v1.services.key_tracking_service import KeyTrackingServiceAsyncClient, KeyTrackingServiceClient, pagers, transports
from google.cloud.kms_inventory_v1.types import key_tracking_service

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
        i = 10
        return i + 15
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert KeyTrackingServiceClient._get_default_mtls_endpoint(None) is None
    assert KeyTrackingServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert KeyTrackingServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert KeyTrackingServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert KeyTrackingServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert KeyTrackingServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(KeyTrackingServiceClient, 'grpc'), (KeyTrackingServiceAsyncClient, 'grpc_asyncio'), (KeyTrackingServiceClient, 'rest')])
def test_key_tracking_service_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('kmsinventory.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://kmsinventory.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.KeyTrackingServiceGrpcTransport, 'grpc'), (transports.KeyTrackingServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.KeyTrackingServiceRestTransport, 'rest')])
def test_key_tracking_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(KeyTrackingServiceClient, 'grpc'), (KeyTrackingServiceAsyncClient, 'grpc_asyncio'), (KeyTrackingServiceClient, 'rest')])
def test_key_tracking_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('kmsinventory.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://kmsinventory.googleapis.com')

def test_key_tracking_service_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = KeyTrackingServiceClient.get_transport_class()
    available_transports = [transports.KeyTrackingServiceGrpcTransport, transports.KeyTrackingServiceRestTransport]
    assert transport in available_transports
    transport = KeyTrackingServiceClient.get_transport_class('grpc')
    assert transport == transports.KeyTrackingServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(KeyTrackingServiceClient, transports.KeyTrackingServiceGrpcTransport, 'grpc'), (KeyTrackingServiceAsyncClient, transports.KeyTrackingServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (KeyTrackingServiceClient, transports.KeyTrackingServiceRestTransport, 'rest')])
@mock.patch.object(KeyTrackingServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(KeyTrackingServiceClient))
@mock.patch.object(KeyTrackingServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(KeyTrackingServiceAsyncClient))
def test_key_tracking_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(KeyTrackingServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(KeyTrackingServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(KeyTrackingServiceClient, transports.KeyTrackingServiceGrpcTransport, 'grpc', 'true'), (KeyTrackingServiceAsyncClient, transports.KeyTrackingServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (KeyTrackingServiceClient, transports.KeyTrackingServiceGrpcTransport, 'grpc', 'false'), (KeyTrackingServiceAsyncClient, transports.KeyTrackingServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (KeyTrackingServiceClient, transports.KeyTrackingServiceRestTransport, 'rest', 'true'), (KeyTrackingServiceClient, transports.KeyTrackingServiceRestTransport, 'rest', 'false')])
@mock.patch.object(KeyTrackingServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(KeyTrackingServiceClient))
@mock.patch.object(KeyTrackingServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(KeyTrackingServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_key_tracking_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [KeyTrackingServiceClient, KeyTrackingServiceAsyncClient])
@mock.patch.object(KeyTrackingServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(KeyTrackingServiceClient))
@mock.patch.object(KeyTrackingServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(KeyTrackingServiceAsyncClient))
def test_key_tracking_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(KeyTrackingServiceClient, transports.KeyTrackingServiceGrpcTransport, 'grpc'), (KeyTrackingServiceAsyncClient, transports.KeyTrackingServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (KeyTrackingServiceClient, transports.KeyTrackingServiceRestTransport, 'rest')])
def test_key_tracking_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(KeyTrackingServiceClient, transports.KeyTrackingServiceGrpcTransport, 'grpc', grpc_helpers), (KeyTrackingServiceAsyncClient, transports.KeyTrackingServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (KeyTrackingServiceClient, transports.KeyTrackingServiceRestTransport, 'rest', None)])
def test_key_tracking_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_key_tracking_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.kms_inventory_v1.services.key_tracking_service.transports.KeyTrackingServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = KeyTrackingServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(KeyTrackingServiceClient, transports.KeyTrackingServiceGrpcTransport, 'grpc', grpc_helpers), (KeyTrackingServiceAsyncClient, transports.KeyTrackingServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_key_tracking_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('kmsinventory.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='kmsinventory.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [key_tracking_service.GetProtectedResourcesSummaryRequest, dict])
def test_get_protected_resources_summary(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_protected_resources_summary), '__call__') as call:
        call.return_value = key_tracking_service.ProtectedResourcesSummary(name='name_value', resource_count=1520, project_count=1407)
        response = client.get_protected_resources_summary(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == key_tracking_service.GetProtectedResourcesSummaryRequest()
    assert isinstance(response, key_tracking_service.ProtectedResourcesSummary)
    assert response.name == 'name_value'
    assert response.resource_count == 1520
    assert response.project_count == 1407

def test_get_protected_resources_summary_empty_call():
    if False:
        while True:
            i = 10
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_protected_resources_summary), '__call__') as call:
        client.get_protected_resources_summary()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == key_tracking_service.GetProtectedResourcesSummaryRequest()

@pytest.mark.asyncio
async def test_get_protected_resources_summary_async(transport: str='grpc_asyncio', request_type=key_tracking_service.GetProtectedResourcesSummaryRequest):
    client = KeyTrackingServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_protected_resources_summary), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(key_tracking_service.ProtectedResourcesSummary(name='name_value', resource_count=1520, project_count=1407))
        response = await client.get_protected_resources_summary(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == key_tracking_service.GetProtectedResourcesSummaryRequest()
    assert isinstance(response, key_tracking_service.ProtectedResourcesSummary)
    assert response.name == 'name_value'
    assert response.resource_count == 1520
    assert response.project_count == 1407

@pytest.mark.asyncio
async def test_get_protected_resources_summary_async_from_dict():
    await test_get_protected_resources_summary_async(request_type=dict)

def test_get_protected_resources_summary_field_headers():
    if False:
        return 10
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = key_tracking_service.GetProtectedResourcesSummaryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_protected_resources_summary), '__call__') as call:
        call.return_value = key_tracking_service.ProtectedResourcesSummary()
        client.get_protected_resources_summary(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_protected_resources_summary_field_headers_async():
    client = KeyTrackingServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = key_tracking_service.GetProtectedResourcesSummaryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_protected_resources_summary), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(key_tracking_service.ProtectedResourcesSummary())
        await client.get_protected_resources_summary(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_protected_resources_summary_flattened():
    if False:
        print('Hello World!')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_protected_resources_summary), '__call__') as call:
        call.return_value = key_tracking_service.ProtectedResourcesSummary()
        client.get_protected_resources_summary(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_protected_resources_summary_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_protected_resources_summary(key_tracking_service.GetProtectedResourcesSummaryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_protected_resources_summary_flattened_async():
    client = KeyTrackingServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_protected_resources_summary), '__call__') as call:
        call.return_value = key_tracking_service.ProtectedResourcesSummary()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(key_tracking_service.ProtectedResourcesSummary())
        response = await client.get_protected_resources_summary(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_protected_resources_summary_flattened_error_async():
    client = KeyTrackingServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_protected_resources_summary(key_tracking_service.GetProtectedResourcesSummaryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [key_tracking_service.SearchProtectedResourcesRequest, dict])
def test_search_protected_resources(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_protected_resources), '__call__') as call:
        call.return_value = key_tracking_service.SearchProtectedResourcesResponse(next_page_token='next_page_token_value')
        response = client.search_protected_resources(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == key_tracking_service.SearchProtectedResourcesRequest()
    assert isinstance(response, pagers.SearchProtectedResourcesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_protected_resources_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_protected_resources), '__call__') as call:
        client.search_protected_resources()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == key_tracking_service.SearchProtectedResourcesRequest()

@pytest.mark.asyncio
async def test_search_protected_resources_async(transport: str='grpc_asyncio', request_type=key_tracking_service.SearchProtectedResourcesRequest):
    client = KeyTrackingServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_protected_resources), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(key_tracking_service.SearchProtectedResourcesResponse(next_page_token='next_page_token_value'))
        response = await client.search_protected_resources(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == key_tracking_service.SearchProtectedResourcesRequest()
    assert isinstance(response, pagers.SearchProtectedResourcesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_search_protected_resources_async_from_dict():
    await test_search_protected_resources_async(request_type=dict)

def test_search_protected_resources_field_headers():
    if False:
        print('Hello World!')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = key_tracking_service.SearchProtectedResourcesRequest()
    request.scope = 'scope_value'
    with mock.patch.object(type(client.transport.search_protected_resources), '__call__') as call:
        call.return_value = key_tracking_service.SearchProtectedResourcesResponse()
        client.search_protected_resources(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'scope=scope_value') in kw['metadata']

@pytest.mark.asyncio
async def test_search_protected_resources_field_headers_async():
    client = KeyTrackingServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = key_tracking_service.SearchProtectedResourcesRequest()
    request.scope = 'scope_value'
    with mock.patch.object(type(client.transport.search_protected_resources), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(key_tracking_service.SearchProtectedResourcesResponse())
        await client.search_protected_resources(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'scope=scope_value') in kw['metadata']

def test_search_protected_resources_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_protected_resources), '__call__') as call:
        call.return_value = key_tracking_service.SearchProtectedResourcesResponse()
        client.search_protected_resources(scope='scope_value', crypto_key='crypto_key_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].scope
        mock_val = 'scope_value'
        assert arg == mock_val
        arg = args[0].crypto_key
        mock_val = 'crypto_key_value'
        assert arg == mock_val

def test_search_protected_resources_flattened_error():
    if False:
        print('Hello World!')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.search_protected_resources(key_tracking_service.SearchProtectedResourcesRequest(), scope='scope_value', crypto_key='crypto_key_value')

@pytest.mark.asyncio
async def test_search_protected_resources_flattened_async():
    client = KeyTrackingServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_protected_resources), '__call__') as call:
        call.return_value = key_tracking_service.SearchProtectedResourcesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(key_tracking_service.SearchProtectedResourcesResponse())
        response = await client.search_protected_resources(scope='scope_value', crypto_key='crypto_key_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].scope
        mock_val = 'scope_value'
        assert arg == mock_val
        arg = args[0].crypto_key
        mock_val = 'crypto_key_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_search_protected_resources_flattened_error_async():
    client = KeyTrackingServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.search_protected_resources(key_tracking_service.SearchProtectedResourcesRequest(), scope='scope_value', crypto_key='crypto_key_value')

def test_search_protected_resources_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_protected_resources), '__call__') as call:
        call.side_effect = (key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource()], next_page_token='abc'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[], next_page_token='def'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource()], next_page_token='ghi'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('scope', ''),)),)
        pager = client.search_protected_resources(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, key_tracking_service.ProtectedResource) for i in results))

def test_search_protected_resources_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_protected_resources), '__call__') as call:
        call.side_effect = (key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource()], next_page_token='abc'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[], next_page_token='def'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource()], next_page_token='ghi'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource()]), RuntimeError)
        pages = list(client.search_protected_resources(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_search_protected_resources_async_pager():
    client = KeyTrackingServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_protected_resources), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource()], next_page_token='abc'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[], next_page_token='def'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource()], next_page_token='ghi'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource()]), RuntimeError)
        async_pager = await client.search_protected_resources(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, key_tracking_service.ProtectedResource) for i in responses))

@pytest.mark.asyncio
async def test_search_protected_resources_async_pages():
    client = KeyTrackingServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_protected_resources), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource()], next_page_token='abc'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[], next_page_token='def'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource()], next_page_token='ghi'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource()]), RuntimeError)
        pages = []
        async for page_ in (await client.search_protected_resources(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [key_tracking_service.GetProtectedResourcesSummaryRequest, dict])
def test_get_protected_resources_summary_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = key_tracking_service.ProtectedResourcesSummary(name='name_value', resource_count=1520, project_count=1407)
        response_value = Response()
        response_value.status_code = 200
        return_value = key_tracking_service.ProtectedResourcesSummary.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_protected_resources_summary(request)
    assert isinstance(response, key_tracking_service.ProtectedResourcesSummary)
    assert response.name == 'name_value'
    assert response.resource_count == 1520
    assert response.project_count == 1407

def test_get_protected_resources_summary_rest_required_fields(request_type=key_tracking_service.GetProtectedResourcesSummaryRequest):
    if False:
        print('Hello World!')
    transport_class = transports.KeyTrackingServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_protected_resources_summary._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_protected_resources_summary._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = key_tracking_service.ProtectedResourcesSummary()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = key_tracking_service.ProtectedResourcesSummary.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_protected_resources_summary(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_protected_resources_summary_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.KeyTrackingServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_protected_resources_summary._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_protected_resources_summary_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.KeyTrackingServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyTrackingServiceRestInterceptor())
    client = KeyTrackingServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyTrackingServiceRestInterceptor, 'post_get_protected_resources_summary') as post, mock.patch.object(transports.KeyTrackingServiceRestInterceptor, 'pre_get_protected_resources_summary') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = key_tracking_service.GetProtectedResourcesSummaryRequest.pb(key_tracking_service.GetProtectedResourcesSummaryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = key_tracking_service.ProtectedResourcesSummary.to_json(key_tracking_service.ProtectedResourcesSummary())
        request = key_tracking_service.GetProtectedResourcesSummaryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = key_tracking_service.ProtectedResourcesSummary()
        client.get_protected_resources_summary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_protected_resources_summary_rest_bad_request(transport: str='rest', request_type=key_tracking_service.GetProtectedResourcesSummaryRequest):
    if False:
        print('Hello World!')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_protected_resources_summary(request)

def test_get_protected_resources_summary_rest_flattened():
    if False:
        return 10
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = key_tracking_service.ProtectedResourcesSummary()
        sample_request = {'name': 'projects/sample1/locations/sample2/keyRings/sample3/cryptoKeys/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = key_tracking_service.ProtectedResourcesSummary.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_protected_resources_summary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/keyRings/*/cryptoKeys/**}/protectedResourcesSummary' % client.transport._host, args[1])

def test_get_protected_resources_summary_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_protected_resources_summary(key_tracking_service.GetProtectedResourcesSummaryRequest(), name='name_value')

def test_get_protected_resources_summary_rest_error():
    if False:
        while True:
            i = 10
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [key_tracking_service.SearchProtectedResourcesRequest, dict])
def test_search_protected_resources_rest(request_type):
    if False:
        while True:
            i = 10
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'scope': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = key_tracking_service.SearchProtectedResourcesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = key_tracking_service.SearchProtectedResourcesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search_protected_resources(request)
    assert isinstance(response, pagers.SearchProtectedResourcesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_protected_resources_rest_required_fields(request_type=key_tracking_service.SearchProtectedResourcesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.KeyTrackingServiceRestTransport
    request_init = {}
    request_init['scope'] = ''
    request_init['crypto_key'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'cryptoKey' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_protected_resources._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'cryptoKey' in jsonified_request
    assert jsonified_request['cryptoKey'] == request_init['crypto_key']
    jsonified_request['scope'] = 'scope_value'
    jsonified_request['cryptoKey'] = 'crypto_key_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_protected_resources._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('crypto_key', 'page_size', 'page_token', 'resource_types'))
    jsonified_request.update(unset_fields)
    assert 'scope' in jsonified_request
    assert jsonified_request['scope'] == 'scope_value'
    assert 'cryptoKey' in jsonified_request
    assert jsonified_request['cryptoKey'] == 'crypto_key_value'
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = key_tracking_service.SearchProtectedResourcesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = key_tracking_service.SearchProtectedResourcesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.search_protected_resources(request)
            expected_params = [('cryptoKey', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_search_protected_resources_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.KeyTrackingServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search_protected_resources._get_unset_required_fields({})
    assert set(unset_fields) == set(('cryptoKey', 'pageSize', 'pageToken', 'resourceTypes')) & set(('scope', 'cryptoKey'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_protected_resources_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.KeyTrackingServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.KeyTrackingServiceRestInterceptor())
    client = KeyTrackingServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.KeyTrackingServiceRestInterceptor, 'post_search_protected_resources') as post, mock.patch.object(transports.KeyTrackingServiceRestInterceptor, 'pre_search_protected_resources') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = key_tracking_service.SearchProtectedResourcesRequest.pb(key_tracking_service.SearchProtectedResourcesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = key_tracking_service.SearchProtectedResourcesResponse.to_json(key_tracking_service.SearchProtectedResourcesResponse())
        request = key_tracking_service.SearchProtectedResourcesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = key_tracking_service.SearchProtectedResourcesResponse()
        client.search_protected_resources(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_protected_resources_rest_bad_request(transport: str='rest', request_type=key_tracking_service.SearchProtectedResourcesRequest):
    if False:
        return 10
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'scope': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search_protected_resources(request)

def test_search_protected_resources_rest_flattened():
    if False:
        print('Hello World!')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = key_tracking_service.SearchProtectedResourcesResponse()
        sample_request = {'scope': 'organizations/sample1'}
        mock_args = dict(scope='scope_value', crypto_key='crypto_key_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = key_tracking_service.SearchProtectedResourcesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.search_protected_resources(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{scope=organizations/*}/protectedResources:search' % client.transport._host, args[1])

def test_search_protected_resources_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.search_protected_resources(key_tracking_service.SearchProtectedResourcesRequest(), scope='scope_value', crypto_key='crypto_key_value')

def test_search_protected_resources_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource()], next_page_token='abc'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[], next_page_token='def'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource()], next_page_token='ghi'), key_tracking_service.SearchProtectedResourcesResponse(protected_resources=[key_tracking_service.ProtectedResource(), key_tracking_service.ProtectedResource()]))
        response = response + response
        response = tuple((key_tracking_service.SearchProtectedResourcesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'scope': 'organizations/sample1'}
        pager = client.search_protected_resources(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, key_tracking_service.ProtectedResource) for i in results))
        pages = list(client.search_protected_resources(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.KeyTrackingServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.KeyTrackingServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = KeyTrackingServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.KeyTrackingServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = KeyTrackingServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = KeyTrackingServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.KeyTrackingServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = KeyTrackingServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.KeyTrackingServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = KeyTrackingServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.KeyTrackingServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.KeyTrackingServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.KeyTrackingServiceGrpcTransport, transports.KeyTrackingServiceGrpcAsyncIOTransport, transports.KeyTrackingServiceRestTransport])
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
    transport = KeyTrackingServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.KeyTrackingServiceGrpcTransport)

def test_key_tracking_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.KeyTrackingServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_key_tracking_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.kms_inventory_v1.services.key_tracking_service.transports.KeyTrackingServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.KeyTrackingServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_protected_resources_summary', 'search_protected_resources')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_key_tracking_service_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.kms_inventory_v1.services.key_tracking_service.transports.KeyTrackingServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.KeyTrackingServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_key_tracking_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.kms_inventory_v1.services.key_tracking_service.transports.KeyTrackingServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.KeyTrackingServiceTransport()
        adc.assert_called_once()

def test_key_tracking_service_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        KeyTrackingServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.KeyTrackingServiceGrpcTransport, transports.KeyTrackingServiceGrpcAsyncIOTransport])
def test_key_tracking_service_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.KeyTrackingServiceGrpcTransport, transports.KeyTrackingServiceGrpcAsyncIOTransport, transports.KeyTrackingServiceRestTransport])
def test_key_tracking_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.KeyTrackingServiceGrpcTransport, grpc_helpers), (transports.KeyTrackingServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_key_tracking_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('kmsinventory.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='kmsinventory.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.KeyTrackingServiceGrpcTransport, transports.KeyTrackingServiceGrpcAsyncIOTransport])
def test_key_tracking_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_key_tracking_service_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.KeyTrackingServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_key_tracking_service_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='kmsinventory.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('kmsinventory.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://kmsinventory.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_key_tracking_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='kmsinventory.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('kmsinventory.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://kmsinventory.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_key_tracking_service_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = KeyTrackingServiceClient(credentials=creds1, transport=transport_name)
    client2 = KeyTrackingServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.get_protected_resources_summary._session
    session2 = client2.transport.get_protected_resources_summary._session
    assert session1 != session2
    session1 = client1.transport.search_protected_resources._session
    session2 = client2.transport.search_protected_resources._session
    assert session1 != session2

def test_key_tracking_service_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.KeyTrackingServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_key_tracking_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.KeyTrackingServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.KeyTrackingServiceGrpcTransport, transports.KeyTrackingServiceGrpcAsyncIOTransport])
def test_key_tracking_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.KeyTrackingServiceGrpcTransport, transports.KeyTrackingServiceGrpcAsyncIOTransport])
def test_key_tracking_service_transport_channel_mtls_with_adc(transport_class):
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

def test_asset_path():
    if False:
        print('Hello World!')
    expected = '*'.format()
    actual = KeyTrackingServiceClient.asset_path()
    assert expected == actual

def test_parse_asset_path():
    if False:
        return 10
    expected = {}
    path = KeyTrackingServiceClient.asset_path(**expected)
    actual = KeyTrackingServiceClient.parse_asset_path(path)
    assert expected == actual

def test_crypto_key_version_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    key_ring = 'whelk'
    crypto_key = 'octopus'
    crypto_key_version = 'oyster'
    expected = 'projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}/cryptoKeyVersions/{crypto_key_version}'.format(project=project, location=location, key_ring=key_ring, crypto_key=crypto_key, crypto_key_version=crypto_key_version)
    actual = KeyTrackingServiceClient.crypto_key_version_path(project, location, key_ring, crypto_key, crypto_key_version)
    assert expected == actual

def test_parse_crypto_key_version_path():
    if False:
        return 10
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'key_ring': 'mussel', 'crypto_key': 'winkle', 'crypto_key_version': 'nautilus'}
    path = KeyTrackingServiceClient.crypto_key_version_path(**expected)
    actual = KeyTrackingServiceClient.parse_crypto_key_version_path(path)
    assert expected == actual

def test_protected_resources_summary_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    location = 'abalone'
    key_ring = 'squid'
    crypto_key = 'clam'
    expected = 'projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}/protectedResourcesSummary'.format(project=project, location=location, key_ring=key_ring, crypto_key=crypto_key)
    actual = KeyTrackingServiceClient.protected_resources_summary_path(project, location, key_ring, crypto_key)
    assert expected == actual

def test_parse_protected_resources_summary_path():
    if False:
        print('Hello World!')
    expected = {'project': 'whelk', 'location': 'octopus', 'key_ring': 'oyster', 'crypto_key': 'nudibranch'}
    path = KeyTrackingServiceClient.protected_resources_summary_path(**expected)
    actual = KeyTrackingServiceClient.parse_protected_resources_summary_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = KeyTrackingServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'mussel'}
    path = KeyTrackingServiceClient.common_billing_account_path(**expected)
    actual = KeyTrackingServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = KeyTrackingServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'nautilus'}
    path = KeyTrackingServiceClient.common_folder_path(**expected)
    actual = KeyTrackingServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = KeyTrackingServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'abalone'}
    path = KeyTrackingServiceClient.common_organization_path(**expected)
    actual = KeyTrackingServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = KeyTrackingServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'clam'}
    path = KeyTrackingServiceClient.common_project_path(**expected)
    actual = KeyTrackingServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = KeyTrackingServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = KeyTrackingServiceClient.common_location_path(**expected)
    actual = KeyTrackingServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.KeyTrackingServiceTransport, '_prep_wrapped_messages') as prep:
        client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.KeyTrackingServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = KeyTrackingServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = KeyTrackingServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
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
        client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = KeyTrackingServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(KeyTrackingServiceClient, transports.KeyTrackingServiceGrpcTransport), (KeyTrackingServiceAsyncClient, transports.KeyTrackingServiceGrpcAsyncIOTransport)])
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