import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
import math
from google.api import monitored_resource_pb2
from google.api_core import gapic_v1, grpc_helpers, grpc_helpers_async, path_template
from google.api_core import client_options
from google.api_core import exceptions as core_exceptions
import google.auth
from google.auth import credentials as ga_credentials
from google.auth.exceptions import MutualTLSChannelError
from google.oauth2 import service_account
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from google.cloud.monitoring_v3.services.uptime_check_service import UptimeCheckServiceAsyncClient, UptimeCheckServiceClient, pagers, transports
from google.cloud.monitoring_v3.types import uptime, uptime_service

def client_cert_source_callback():
    if False:
        return 10
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
    assert UptimeCheckServiceClient._get_default_mtls_endpoint(None) is None
    assert UptimeCheckServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert UptimeCheckServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert UptimeCheckServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert UptimeCheckServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert UptimeCheckServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(UptimeCheckServiceClient, 'grpc'), (UptimeCheckServiceAsyncClient, 'grpc_asyncio')])
def test_uptime_check_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == 'monitoring.googleapis.com:443'

@pytest.mark.parametrize('transport_class,transport_name', [(transports.UptimeCheckServiceGrpcTransport, 'grpc'), (transports.UptimeCheckServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_uptime_check_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(UptimeCheckServiceClient, 'grpc'), (UptimeCheckServiceAsyncClient, 'grpc_asyncio')])
def test_uptime_check_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == 'monitoring.googleapis.com:443'

def test_uptime_check_service_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = UptimeCheckServiceClient.get_transport_class()
    available_transports = [transports.UptimeCheckServiceGrpcTransport]
    assert transport in available_transports
    transport = UptimeCheckServiceClient.get_transport_class('grpc')
    assert transport == transports.UptimeCheckServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(UptimeCheckServiceClient, transports.UptimeCheckServiceGrpcTransport, 'grpc'), (UptimeCheckServiceAsyncClient, transports.UptimeCheckServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
@mock.patch.object(UptimeCheckServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(UptimeCheckServiceClient))
@mock.patch.object(UptimeCheckServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(UptimeCheckServiceAsyncClient))
def test_uptime_check_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(UptimeCheckServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(UptimeCheckServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(UptimeCheckServiceClient, transports.UptimeCheckServiceGrpcTransport, 'grpc', 'true'), (UptimeCheckServiceAsyncClient, transports.UptimeCheckServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (UptimeCheckServiceClient, transports.UptimeCheckServiceGrpcTransport, 'grpc', 'false'), (UptimeCheckServiceAsyncClient, transports.UptimeCheckServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false')])
@mock.patch.object(UptimeCheckServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(UptimeCheckServiceClient))
@mock.patch.object(UptimeCheckServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(UptimeCheckServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_uptime_check_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
    if False:
        while True:
            i = 10
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

@pytest.mark.parametrize('client_class', [UptimeCheckServiceClient, UptimeCheckServiceAsyncClient])
@mock.patch.object(UptimeCheckServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(UptimeCheckServiceClient))
@mock.patch.object(UptimeCheckServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(UptimeCheckServiceAsyncClient))
def test_uptime_check_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(UptimeCheckServiceClient, transports.UptimeCheckServiceGrpcTransport, 'grpc'), (UptimeCheckServiceAsyncClient, transports.UptimeCheckServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_uptime_check_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(UptimeCheckServiceClient, transports.UptimeCheckServiceGrpcTransport, 'grpc', grpc_helpers), (UptimeCheckServiceAsyncClient, transports.UptimeCheckServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_uptime_check_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_uptime_check_service_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.monitoring_v3.services.uptime_check_service.transports.UptimeCheckServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = UptimeCheckServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(UptimeCheckServiceClient, transports.UptimeCheckServiceGrpcTransport, 'grpc', grpc_helpers), (UptimeCheckServiceAsyncClient, transports.UptimeCheckServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_uptime_check_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('monitoring.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), scopes=None, default_host='monitoring.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [uptime_service.ListUptimeCheckConfigsRequest, dict])
def test_list_uptime_check_configs(request_type, transport: str='grpc'):
    if False:
        return 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_uptime_check_configs), '__call__') as call:
        call.return_value = uptime_service.ListUptimeCheckConfigsResponse(next_page_token='next_page_token_value', total_size=1086)
        response = client.list_uptime_check_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.ListUptimeCheckConfigsRequest()
    assert isinstance(response, pagers.ListUptimeCheckConfigsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

def test_list_uptime_check_configs_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_uptime_check_configs), '__call__') as call:
        client.list_uptime_check_configs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.ListUptimeCheckConfigsRequest()

@pytest.mark.asyncio
async def test_list_uptime_check_configs_async(transport: str='grpc_asyncio', request_type=uptime_service.ListUptimeCheckConfigsRequest):
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_uptime_check_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(uptime_service.ListUptimeCheckConfigsResponse(next_page_token='next_page_token_value', total_size=1086))
        response = await client.list_uptime_check_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.ListUptimeCheckConfigsRequest()
    assert isinstance(response, pagers.ListUptimeCheckConfigsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.total_size == 1086

@pytest.mark.asyncio
async def test_list_uptime_check_configs_async_from_dict():
    await test_list_uptime_check_configs_async(request_type=dict)

def test_list_uptime_check_configs_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = uptime_service.ListUptimeCheckConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_uptime_check_configs), '__call__') as call:
        call.return_value = uptime_service.ListUptimeCheckConfigsResponse()
        client.list_uptime_check_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_uptime_check_configs_field_headers_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = uptime_service.ListUptimeCheckConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_uptime_check_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(uptime_service.ListUptimeCheckConfigsResponse())
        await client.list_uptime_check_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_uptime_check_configs_flattened():
    if False:
        while True:
            i = 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_uptime_check_configs), '__call__') as call:
        call.return_value = uptime_service.ListUptimeCheckConfigsResponse()
        client.list_uptime_check_configs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_uptime_check_configs_flattened_error():
    if False:
        return 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_uptime_check_configs(uptime_service.ListUptimeCheckConfigsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_uptime_check_configs_flattened_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_uptime_check_configs), '__call__') as call:
        call.return_value = uptime_service.ListUptimeCheckConfigsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(uptime_service.ListUptimeCheckConfigsResponse())
        response = await client.list_uptime_check_configs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_uptime_check_configs_flattened_error_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_uptime_check_configs(uptime_service.ListUptimeCheckConfigsRequest(), parent='parent_value')

def test_list_uptime_check_configs_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_uptime_check_configs), '__call__') as call:
        call.side_effect = (uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[uptime.UptimeCheckConfig(), uptime.UptimeCheckConfig(), uptime.UptimeCheckConfig()], next_page_token='abc'), uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[], next_page_token='def'), uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[uptime.UptimeCheckConfig()], next_page_token='ghi'), uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[uptime.UptimeCheckConfig(), uptime.UptimeCheckConfig()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_uptime_check_configs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, uptime.UptimeCheckConfig) for i in results))

def test_list_uptime_check_configs_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_uptime_check_configs), '__call__') as call:
        call.side_effect = (uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[uptime.UptimeCheckConfig(), uptime.UptimeCheckConfig(), uptime.UptimeCheckConfig()], next_page_token='abc'), uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[], next_page_token='def'), uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[uptime.UptimeCheckConfig()], next_page_token='ghi'), uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[uptime.UptimeCheckConfig(), uptime.UptimeCheckConfig()]), RuntimeError)
        pages = list(client.list_uptime_check_configs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_uptime_check_configs_async_pager():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_uptime_check_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[uptime.UptimeCheckConfig(), uptime.UptimeCheckConfig(), uptime.UptimeCheckConfig()], next_page_token='abc'), uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[], next_page_token='def'), uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[uptime.UptimeCheckConfig()], next_page_token='ghi'), uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[uptime.UptimeCheckConfig(), uptime.UptimeCheckConfig()]), RuntimeError)
        async_pager = await client.list_uptime_check_configs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, uptime.UptimeCheckConfig) for i in responses))

@pytest.mark.asyncio
async def test_list_uptime_check_configs_async_pages():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_uptime_check_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[uptime.UptimeCheckConfig(), uptime.UptimeCheckConfig(), uptime.UptimeCheckConfig()], next_page_token='abc'), uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[], next_page_token='def'), uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[uptime.UptimeCheckConfig()], next_page_token='ghi'), uptime_service.ListUptimeCheckConfigsResponse(uptime_check_configs=[uptime.UptimeCheckConfig(), uptime.UptimeCheckConfig()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_uptime_check_configs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [uptime_service.GetUptimeCheckConfigRequest, dict])
def test_get_uptime_check_config(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_uptime_check_config), '__call__') as call:
        call.return_value = uptime.UptimeCheckConfig(name='name_value', display_name='display_name_value', checker_type=uptime.UptimeCheckConfig.CheckerType.STATIC_IP_CHECKERS, selected_regions=[uptime.UptimeCheckRegion.USA], is_internal=True)
        response = client.get_uptime_check_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.GetUptimeCheckConfigRequest()
    assert isinstance(response, uptime.UptimeCheckConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.checker_type == uptime.UptimeCheckConfig.CheckerType.STATIC_IP_CHECKERS
    assert response.selected_regions == [uptime.UptimeCheckRegion.USA]
    assert response.is_internal is True

def test_get_uptime_check_config_empty_call():
    if False:
        return 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_uptime_check_config), '__call__') as call:
        client.get_uptime_check_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.GetUptimeCheckConfigRequest()

@pytest.mark.asyncio
async def test_get_uptime_check_config_async(transport: str='grpc_asyncio', request_type=uptime_service.GetUptimeCheckConfigRequest):
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_uptime_check_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(uptime.UptimeCheckConfig(name='name_value', display_name='display_name_value', checker_type=uptime.UptimeCheckConfig.CheckerType.STATIC_IP_CHECKERS, selected_regions=[uptime.UptimeCheckRegion.USA], is_internal=True))
        response = await client.get_uptime_check_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.GetUptimeCheckConfigRequest()
    assert isinstance(response, uptime.UptimeCheckConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.checker_type == uptime.UptimeCheckConfig.CheckerType.STATIC_IP_CHECKERS
    assert response.selected_regions == [uptime.UptimeCheckRegion.USA]
    assert response.is_internal is True

@pytest.mark.asyncio
async def test_get_uptime_check_config_async_from_dict():
    await test_get_uptime_check_config_async(request_type=dict)

def test_get_uptime_check_config_field_headers():
    if False:
        return 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = uptime_service.GetUptimeCheckConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_uptime_check_config), '__call__') as call:
        call.return_value = uptime.UptimeCheckConfig()
        client.get_uptime_check_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_uptime_check_config_field_headers_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = uptime_service.GetUptimeCheckConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_uptime_check_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(uptime.UptimeCheckConfig())
        await client.get_uptime_check_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_uptime_check_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_uptime_check_config), '__call__') as call:
        call.return_value = uptime.UptimeCheckConfig()
        client.get_uptime_check_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_uptime_check_config_flattened_error():
    if False:
        print('Hello World!')
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_uptime_check_config(uptime_service.GetUptimeCheckConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_uptime_check_config_flattened_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_uptime_check_config), '__call__') as call:
        call.return_value = uptime.UptimeCheckConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(uptime.UptimeCheckConfig())
        response = await client.get_uptime_check_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_uptime_check_config_flattened_error_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_uptime_check_config(uptime_service.GetUptimeCheckConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [uptime_service.CreateUptimeCheckConfigRequest, dict])
def test_create_uptime_check_config(request_type, transport: str='grpc'):
    if False:
        return 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_uptime_check_config), '__call__') as call:
        call.return_value = uptime.UptimeCheckConfig(name='name_value', display_name='display_name_value', checker_type=uptime.UptimeCheckConfig.CheckerType.STATIC_IP_CHECKERS, selected_regions=[uptime.UptimeCheckRegion.USA], is_internal=True)
        response = client.create_uptime_check_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.CreateUptimeCheckConfigRequest()
    assert isinstance(response, uptime.UptimeCheckConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.checker_type == uptime.UptimeCheckConfig.CheckerType.STATIC_IP_CHECKERS
    assert response.selected_regions == [uptime.UptimeCheckRegion.USA]
    assert response.is_internal is True

def test_create_uptime_check_config_empty_call():
    if False:
        i = 10
        return i + 15
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_uptime_check_config), '__call__') as call:
        client.create_uptime_check_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.CreateUptimeCheckConfigRequest()

@pytest.mark.asyncio
async def test_create_uptime_check_config_async(transport: str='grpc_asyncio', request_type=uptime_service.CreateUptimeCheckConfigRequest):
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_uptime_check_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(uptime.UptimeCheckConfig(name='name_value', display_name='display_name_value', checker_type=uptime.UptimeCheckConfig.CheckerType.STATIC_IP_CHECKERS, selected_regions=[uptime.UptimeCheckRegion.USA], is_internal=True))
        response = await client.create_uptime_check_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.CreateUptimeCheckConfigRequest()
    assert isinstance(response, uptime.UptimeCheckConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.checker_type == uptime.UptimeCheckConfig.CheckerType.STATIC_IP_CHECKERS
    assert response.selected_regions == [uptime.UptimeCheckRegion.USA]
    assert response.is_internal is True

@pytest.mark.asyncio
async def test_create_uptime_check_config_async_from_dict():
    await test_create_uptime_check_config_async(request_type=dict)

def test_create_uptime_check_config_field_headers():
    if False:
        while True:
            i = 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = uptime_service.CreateUptimeCheckConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_uptime_check_config), '__call__') as call:
        call.return_value = uptime.UptimeCheckConfig()
        client.create_uptime_check_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_uptime_check_config_field_headers_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = uptime_service.CreateUptimeCheckConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_uptime_check_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(uptime.UptimeCheckConfig())
        await client.create_uptime_check_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_uptime_check_config_flattened():
    if False:
        while True:
            i = 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_uptime_check_config), '__call__') as call:
        call.return_value = uptime.UptimeCheckConfig()
        client.create_uptime_check_config(parent='parent_value', uptime_check_config=uptime.UptimeCheckConfig(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].uptime_check_config
        mock_val = uptime.UptimeCheckConfig(name='name_value')
        assert arg == mock_val

def test_create_uptime_check_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_uptime_check_config(uptime_service.CreateUptimeCheckConfigRequest(), parent='parent_value', uptime_check_config=uptime.UptimeCheckConfig(name='name_value'))

@pytest.mark.asyncio
async def test_create_uptime_check_config_flattened_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_uptime_check_config), '__call__') as call:
        call.return_value = uptime.UptimeCheckConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(uptime.UptimeCheckConfig())
        response = await client.create_uptime_check_config(parent='parent_value', uptime_check_config=uptime.UptimeCheckConfig(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].uptime_check_config
        mock_val = uptime.UptimeCheckConfig(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_uptime_check_config_flattened_error_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_uptime_check_config(uptime_service.CreateUptimeCheckConfigRequest(), parent='parent_value', uptime_check_config=uptime.UptimeCheckConfig(name='name_value'))

@pytest.mark.parametrize('request_type', [uptime_service.UpdateUptimeCheckConfigRequest, dict])
def test_update_uptime_check_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_uptime_check_config), '__call__') as call:
        call.return_value = uptime.UptimeCheckConfig(name='name_value', display_name='display_name_value', checker_type=uptime.UptimeCheckConfig.CheckerType.STATIC_IP_CHECKERS, selected_regions=[uptime.UptimeCheckRegion.USA], is_internal=True)
        response = client.update_uptime_check_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.UpdateUptimeCheckConfigRequest()
    assert isinstance(response, uptime.UptimeCheckConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.checker_type == uptime.UptimeCheckConfig.CheckerType.STATIC_IP_CHECKERS
    assert response.selected_regions == [uptime.UptimeCheckRegion.USA]
    assert response.is_internal is True

def test_update_uptime_check_config_empty_call():
    if False:
        print('Hello World!')
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_uptime_check_config), '__call__') as call:
        client.update_uptime_check_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.UpdateUptimeCheckConfigRequest()

@pytest.mark.asyncio
async def test_update_uptime_check_config_async(transport: str='grpc_asyncio', request_type=uptime_service.UpdateUptimeCheckConfigRequest):
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_uptime_check_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(uptime.UptimeCheckConfig(name='name_value', display_name='display_name_value', checker_type=uptime.UptimeCheckConfig.CheckerType.STATIC_IP_CHECKERS, selected_regions=[uptime.UptimeCheckRegion.USA], is_internal=True))
        response = await client.update_uptime_check_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.UpdateUptimeCheckConfigRequest()
    assert isinstance(response, uptime.UptimeCheckConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.checker_type == uptime.UptimeCheckConfig.CheckerType.STATIC_IP_CHECKERS
    assert response.selected_regions == [uptime.UptimeCheckRegion.USA]
    assert response.is_internal is True

@pytest.mark.asyncio
async def test_update_uptime_check_config_async_from_dict():
    await test_update_uptime_check_config_async(request_type=dict)

def test_update_uptime_check_config_field_headers():
    if False:
        while True:
            i = 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = uptime_service.UpdateUptimeCheckConfigRequest()
    request.uptime_check_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_uptime_check_config), '__call__') as call:
        call.return_value = uptime.UptimeCheckConfig()
        client.update_uptime_check_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'uptime_check_config.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_uptime_check_config_field_headers_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = uptime_service.UpdateUptimeCheckConfigRequest()
    request.uptime_check_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_uptime_check_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(uptime.UptimeCheckConfig())
        await client.update_uptime_check_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'uptime_check_config.name=name_value') in kw['metadata']

def test_update_uptime_check_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_uptime_check_config), '__call__') as call:
        call.return_value = uptime.UptimeCheckConfig()
        client.update_uptime_check_config(uptime_check_config=uptime.UptimeCheckConfig(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].uptime_check_config
        mock_val = uptime.UptimeCheckConfig(name='name_value')
        assert arg == mock_val

def test_update_uptime_check_config_flattened_error():
    if False:
        while True:
            i = 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_uptime_check_config(uptime_service.UpdateUptimeCheckConfigRequest(), uptime_check_config=uptime.UptimeCheckConfig(name='name_value'))

@pytest.mark.asyncio
async def test_update_uptime_check_config_flattened_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_uptime_check_config), '__call__') as call:
        call.return_value = uptime.UptimeCheckConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(uptime.UptimeCheckConfig())
        response = await client.update_uptime_check_config(uptime_check_config=uptime.UptimeCheckConfig(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].uptime_check_config
        mock_val = uptime.UptimeCheckConfig(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_uptime_check_config_flattened_error_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_uptime_check_config(uptime_service.UpdateUptimeCheckConfigRequest(), uptime_check_config=uptime.UptimeCheckConfig(name='name_value'))

@pytest.mark.parametrize('request_type', [uptime_service.DeleteUptimeCheckConfigRequest, dict])
def test_delete_uptime_check_config(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_uptime_check_config), '__call__') as call:
        call.return_value = None
        response = client.delete_uptime_check_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.DeleteUptimeCheckConfigRequest()
    assert response is None

def test_delete_uptime_check_config_empty_call():
    if False:
        print('Hello World!')
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_uptime_check_config), '__call__') as call:
        client.delete_uptime_check_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.DeleteUptimeCheckConfigRequest()

@pytest.mark.asyncio
async def test_delete_uptime_check_config_async(transport: str='grpc_asyncio', request_type=uptime_service.DeleteUptimeCheckConfigRequest):
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_uptime_check_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_uptime_check_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.DeleteUptimeCheckConfigRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_uptime_check_config_async_from_dict():
    await test_delete_uptime_check_config_async(request_type=dict)

def test_delete_uptime_check_config_field_headers():
    if False:
        while True:
            i = 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = uptime_service.DeleteUptimeCheckConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_uptime_check_config), '__call__') as call:
        call.return_value = None
        client.delete_uptime_check_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_uptime_check_config_field_headers_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = uptime_service.DeleteUptimeCheckConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_uptime_check_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_uptime_check_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_uptime_check_config_flattened():
    if False:
        while True:
            i = 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_uptime_check_config), '__call__') as call:
        call.return_value = None
        client.delete_uptime_check_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_uptime_check_config_flattened_error():
    if False:
        while True:
            i = 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_uptime_check_config(uptime_service.DeleteUptimeCheckConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_uptime_check_config_flattened_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_uptime_check_config), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_uptime_check_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_uptime_check_config_flattened_error_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_uptime_check_config(uptime_service.DeleteUptimeCheckConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [uptime_service.ListUptimeCheckIpsRequest, dict])
def test_list_uptime_check_ips(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_uptime_check_ips), '__call__') as call:
        call.return_value = uptime_service.ListUptimeCheckIpsResponse(next_page_token='next_page_token_value')
        response = client.list_uptime_check_ips(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.ListUptimeCheckIpsRequest()
    assert isinstance(response, pagers.ListUptimeCheckIpsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_uptime_check_ips_empty_call():
    if False:
        while True:
            i = 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_uptime_check_ips), '__call__') as call:
        client.list_uptime_check_ips()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.ListUptimeCheckIpsRequest()

@pytest.mark.asyncio
async def test_list_uptime_check_ips_async(transport: str='grpc_asyncio', request_type=uptime_service.ListUptimeCheckIpsRequest):
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_uptime_check_ips), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(uptime_service.ListUptimeCheckIpsResponse(next_page_token='next_page_token_value'))
        response = await client.list_uptime_check_ips(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == uptime_service.ListUptimeCheckIpsRequest()
    assert isinstance(response, pagers.ListUptimeCheckIpsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_uptime_check_ips_async_from_dict():
    await test_list_uptime_check_ips_async(request_type=dict)

def test_list_uptime_check_ips_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_uptime_check_ips), '__call__') as call:
        call.side_effect = (uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[uptime.UptimeCheckIp(), uptime.UptimeCheckIp(), uptime.UptimeCheckIp()], next_page_token='abc'), uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[], next_page_token='def'), uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[uptime.UptimeCheckIp()], next_page_token='ghi'), uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[uptime.UptimeCheckIp(), uptime.UptimeCheckIp()]), RuntimeError)
        metadata = ()
        pager = client.list_uptime_check_ips(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, uptime.UptimeCheckIp) for i in results))

def test_list_uptime_check_ips_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_uptime_check_ips), '__call__') as call:
        call.side_effect = (uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[uptime.UptimeCheckIp(), uptime.UptimeCheckIp(), uptime.UptimeCheckIp()], next_page_token='abc'), uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[], next_page_token='def'), uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[uptime.UptimeCheckIp()], next_page_token='ghi'), uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[uptime.UptimeCheckIp(), uptime.UptimeCheckIp()]), RuntimeError)
        pages = list(client.list_uptime_check_ips(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_uptime_check_ips_async_pager():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_uptime_check_ips), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[uptime.UptimeCheckIp(), uptime.UptimeCheckIp(), uptime.UptimeCheckIp()], next_page_token='abc'), uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[], next_page_token='def'), uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[uptime.UptimeCheckIp()], next_page_token='ghi'), uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[uptime.UptimeCheckIp(), uptime.UptimeCheckIp()]), RuntimeError)
        async_pager = await client.list_uptime_check_ips(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, uptime.UptimeCheckIp) for i in responses))

@pytest.mark.asyncio
async def test_list_uptime_check_ips_async_pages():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_uptime_check_ips), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[uptime.UptimeCheckIp(), uptime.UptimeCheckIp(), uptime.UptimeCheckIp()], next_page_token='abc'), uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[], next_page_token='def'), uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[uptime.UptimeCheckIp()], next_page_token='ghi'), uptime_service.ListUptimeCheckIpsResponse(uptime_check_ips=[uptime.UptimeCheckIp(), uptime.UptimeCheckIp()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_uptime_check_ips(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.UptimeCheckServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.UptimeCheckServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = UptimeCheckServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.UptimeCheckServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = UptimeCheckServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = UptimeCheckServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.UptimeCheckServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = UptimeCheckServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.UptimeCheckServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = UptimeCheckServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.UptimeCheckServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.UptimeCheckServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.UptimeCheckServiceGrpcTransport, transports.UptimeCheckServiceGrpcAsyncIOTransport])
def test_transport_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc'])
def test_transport_kind(transport_name):
    if False:
        return 10
    transport = UptimeCheckServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.UptimeCheckServiceGrpcTransport)

def test_uptime_check_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.UptimeCheckServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_uptime_check_service_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.monitoring_v3.services.uptime_check_service.transports.UptimeCheckServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.UptimeCheckServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_uptime_check_configs', 'get_uptime_check_config', 'create_uptime_check_config', 'update_uptime_check_config', 'delete_uptime_check_config', 'list_uptime_check_ips')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_uptime_check_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.monitoring_v3.services.uptime_check_service.transports.UptimeCheckServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.UptimeCheckServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), quota_project_id='octopus')

def test_uptime_check_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.monitoring_v3.services.uptime_check_service.transports.UptimeCheckServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.UptimeCheckServiceTransport()
        adc.assert_called_once()

def test_uptime_check_service_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        UptimeCheckServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.UptimeCheckServiceGrpcTransport, transports.UptimeCheckServiceGrpcAsyncIOTransport])
def test_uptime_check_service_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.UptimeCheckServiceGrpcTransport, transports.UptimeCheckServiceGrpcAsyncIOTransport])
def test_uptime_check_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.UptimeCheckServiceGrpcTransport, grpc_helpers), (transports.UptimeCheckServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_uptime_check_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('monitoring.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), scopes=['1', '2'], default_host='monitoring.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.UptimeCheckServiceGrpcTransport, transports.UptimeCheckServiceGrpcAsyncIOTransport])
def test_uptime_check_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_uptime_check_service_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='monitoring.googleapis.com'), transport=transport_name)
    assert client.transport._host == 'monitoring.googleapis.com:443'

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_uptime_check_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='monitoring.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == 'monitoring.googleapis.com:8000'

def test_uptime_check_service_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.UptimeCheckServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_uptime_check_service_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.UptimeCheckServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.UptimeCheckServiceGrpcTransport, transports.UptimeCheckServiceGrpcAsyncIOTransport])
def test_uptime_check_service_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        i = 10
        return i + 15
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

@pytest.mark.parametrize('transport_class', [transports.UptimeCheckServiceGrpcTransport, transports.UptimeCheckServiceGrpcAsyncIOTransport])
def test_uptime_check_service_transport_channel_mtls_with_adc(transport_class):
    if False:
        print('Hello World!')
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

def test_uptime_check_config_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    uptime_check_config = 'clam'
    expected = 'projects/{project}/uptimeCheckConfigs/{uptime_check_config}'.format(project=project, uptime_check_config=uptime_check_config)
    actual = UptimeCheckServiceClient.uptime_check_config_path(project, uptime_check_config)
    assert expected == actual

def test_parse_uptime_check_config_path():
    if False:
        print('Hello World!')
    expected = {'project': 'whelk', 'uptime_check_config': 'octopus'}
    path = UptimeCheckServiceClient.uptime_check_config_path(**expected)
    actual = UptimeCheckServiceClient.parse_uptime_check_config_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = UptimeCheckServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'nudibranch'}
    path = UptimeCheckServiceClient.common_billing_account_path(**expected)
    actual = UptimeCheckServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = UptimeCheckServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'mussel'}
    path = UptimeCheckServiceClient.common_folder_path(**expected)
    actual = UptimeCheckServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = UptimeCheckServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'nautilus'}
    path = UptimeCheckServiceClient.common_organization_path(**expected)
    actual = UptimeCheckServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = UptimeCheckServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'abalone'}
    path = UptimeCheckServiceClient.common_project_path(**expected)
    actual = UptimeCheckServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = UptimeCheckServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = UptimeCheckServiceClient.common_location_path(**expected)
    actual = UptimeCheckServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.UptimeCheckServiceTransport, '_prep_wrapped_messages') as prep:
        client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.UptimeCheckServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = UptimeCheckServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = UptimeCheckServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        while True:
            i = 10
    transports = ['grpc']
    for transport in transports:
        client = UptimeCheckServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(UptimeCheckServiceClient, transports.UptimeCheckServiceGrpcTransport), (UptimeCheckServiceAsyncClient, transports.UptimeCheckServiceGrpcAsyncIOTransport)])
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