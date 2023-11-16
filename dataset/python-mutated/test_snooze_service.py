import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
import math
from google.api_core import gapic_v1, grpc_helpers, grpc_helpers_async, path_template
from google.api_core import client_options
from google.api_core import exceptions as core_exceptions
import google.auth
from google.auth import credentials as ga_credentials
from google.auth.exceptions import MutualTLSChannelError
from google.oauth2 import service_account
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from google.cloud.monitoring_v3.services.snooze_service import SnoozeServiceAsyncClient, SnoozeServiceClient, pagers, transports
from google.cloud.monitoring_v3.types import common
from google.cloud.monitoring_v3.types import snooze
from google.cloud.monitoring_v3.types import snooze as gm_snooze
from google.cloud.monitoring_v3.types import snooze_service

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        for i in range(10):
            print('nop')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        for i in range(10):
            print('nop')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert SnoozeServiceClient._get_default_mtls_endpoint(None) is None
    assert SnoozeServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert SnoozeServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert SnoozeServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert SnoozeServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert SnoozeServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(SnoozeServiceClient, 'grpc'), (SnoozeServiceAsyncClient, 'grpc_asyncio')])
def test_snooze_service_client_from_service_account_info(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == 'monitoring.googleapis.com:443'

@pytest.mark.parametrize('transport_class,transport_name', [(transports.SnoozeServiceGrpcTransport, 'grpc'), (transports.SnoozeServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_snooze_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(SnoozeServiceClient, 'grpc'), (SnoozeServiceAsyncClient, 'grpc_asyncio')])
def test_snooze_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == 'monitoring.googleapis.com:443'

def test_snooze_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = SnoozeServiceClient.get_transport_class()
    available_transports = [transports.SnoozeServiceGrpcTransport]
    assert transport in available_transports
    transport = SnoozeServiceClient.get_transport_class('grpc')
    assert transport == transports.SnoozeServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(SnoozeServiceClient, transports.SnoozeServiceGrpcTransport, 'grpc'), (SnoozeServiceAsyncClient, transports.SnoozeServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
@mock.patch.object(SnoozeServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SnoozeServiceClient))
@mock.patch.object(SnoozeServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SnoozeServiceAsyncClient))
def test_snooze_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(SnoozeServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(SnoozeServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(SnoozeServiceClient, transports.SnoozeServiceGrpcTransport, 'grpc', 'true'), (SnoozeServiceAsyncClient, transports.SnoozeServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (SnoozeServiceClient, transports.SnoozeServiceGrpcTransport, 'grpc', 'false'), (SnoozeServiceAsyncClient, transports.SnoozeServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false')])
@mock.patch.object(SnoozeServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SnoozeServiceClient))
@mock.patch.object(SnoozeServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SnoozeServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_snooze_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [SnoozeServiceClient, SnoozeServiceAsyncClient])
@mock.patch.object(SnoozeServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SnoozeServiceClient))
@mock.patch.object(SnoozeServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SnoozeServiceAsyncClient))
def test_snooze_service_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        for i in range(10):
            print('nop')
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(SnoozeServiceClient, transports.SnoozeServiceGrpcTransport, 'grpc'), (SnoozeServiceAsyncClient, transports.SnoozeServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_snooze_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(SnoozeServiceClient, transports.SnoozeServiceGrpcTransport, 'grpc', grpc_helpers), (SnoozeServiceAsyncClient, transports.SnoozeServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_snooze_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_snooze_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.monitoring_v3.services.snooze_service.transports.SnoozeServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = SnoozeServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(SnoozeServiceClient, transports.SnoozeServiceGrpcTransport, 'grpc', grpc_helpers), (SnoozeServiceAsyncClient, transports.SnoozeServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_snooze_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('monitoring.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), scopes=None, default_host='monitoring.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [snooze_service.CreateSnoozeRequest, dict])
def test_create_snooze(request_type, transport: str='grpc'):
    if False:
        return 10
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_snooze), '__call__') as call:
        call.return_value = gm_snooze.Snooze(name='name_value', display_name='display_name_value')
        response = client.create_snooze(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snooze_service.CreateSnoozeRequest()
    assert isinstance(response, gm_snooze.Snooze)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_create_snooze_empty_call():
    if False:
        print('Hello World!')
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_snooze), '__call__') as call:
        client.create_snooze()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snooze_service.CreateSnoozeRequest()

@pytest.mark.asyncio
async def test_create_snooze_async(transport: str='grpc_asyncio', request_type=snooze_service.CreateSnoozeRequest):
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_snooze), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gm_snooze.Snooze(name='name_value', display_name='display_name_value'))
        response = await client.create_snooze(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snooze_service.CreateSnoozeRequest()
    assert isinstance(response, gm_snooze.Snooze)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

@pytest.mark.asyncio
async def test_create_snooze_async_from_dict():
    await test_create_snooze_async(request_type=dict)

def test_create_snooze_field_headers():
    if False:
        while True:
            i = 10
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = snooze_service.CreateSnoozeRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_snooze), '__call__') as call:
        call.return_value = gm_snooze.Snooze()
        client.create_snooze(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_snooze_field_headers_async():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = snooze_service.CreateSnoozeRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_snooze), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gm_snooze.Snooze())
        await client.create_snooze(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_snooze_flattened():
    if False:
        return 10
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_snooze), '__call__') as call:
        call.return_value = gm_snooze.Snooze()
        client.create_snooze(parent='parent_value', snooze=gm_snooze.Snooze(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].snooze
        mock_val = gm_snooze.Snooze(name='name_value')
        assert arg == mock_val

def test_create_snooze_flattened_error():
    if False:
        return 10
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_snooze(snooze_service.CreateSnoozeRequest(), parent='parent_value', snooze=gm_snooze.Snooze(name='name_value'))

@pytest.mark.asyncio
async def test_create_snooze_flattened_async():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_snooze), '__call__') as call:
        call.return_value = gm_snooze.Snooze()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gm_snooze.Snooze())
        response = await client.create_snooze(parent='parent_value', snooze=gm_snooze.Snooze(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].snooze
        mock_val = gm_snooze.Snooze(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_snooze_flattened_error_async():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_snooze(snooze_service.CreateSnoozeRequest(), parent='parent_value', snooze=gm_snooze.Snooze(name='name_value'))

@pytest.mark.parametrize('request_type', [snooze_service.ListSnoozesRequest, dict])
def test_list_snoozes(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_snoozes), '__call__') as call:
        call.return_value = snooze_service.ListSnoozesResponse(next_page_token='next_page_token_value')
        response = client.list_snoozes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snooze_service.ListSnoozesRequest()
    assert isinstance(response, pagers.ListSnoozesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_snoozes_empty_call():
    if False:
        return 10
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_snoozes), '__call__') as call:
        client.list_snoozes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snooze_service.ListSnoozesRequest()

@pytest.mark.asyncio
async def test_list_snoozes_async(transport: str='grpc_asyncio', request_type=snooze_service.ListSnoozesRequest):
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_snoozes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(snooze_service.ListSnoozesResponse(next_page_token='next_page_token_value'))
        response = await client.list_snoozes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snooze_service.ListSnoozesRequest()
    assert isinstance(response, pagers.ListSnoozesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_snoozes_async_from_dict():
    await test_list_snoozes_async(request_type=dict)

def test_list_snoozes_field_headers():
    if False:
        return 10
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = snooze_service.ListSnoozesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_snoozes), '__call__') as call:
        call.return_value = snooze_service.ListSnoozesResponse()
        client.list_snoozes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_snoozes_field_headers_async():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = snooze_service.ListSnoozesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_snoozes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(snooze_service.ListSnoozesResponse())
        await client.list_snoozes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_snoozes_flattened():
    if False:
        while True:
            i = 10
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_snoozes), '__call__') as call:
        call.return_value = snooze_service.ListSnoozesResponse()
        client.list_snoozes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_snoozes_flattened_error():
    if False:
        i = 10
        return i + 15
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_snoozes(snooze_service.ListSnoozesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_snoozes_flattened_async():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_snoozes), '__call__') as call:
        call.return_value = snooze_service.ListSnoozesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(snooze_service.ListSnoozesResponse())
        response = await client.list_snoozes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_snoozes_flattened_error_async():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_snoozes(snooze_service.ListSnoozesRequest(), parent='parent_value')

def test_list_snoozes_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_snoozes), '__call__') as call:
        call.side_effect = (snooze_service.ListSnoozesResponse(snoozes=[snooze.Snooze(), snooze.Snooze(), snooze.Snooze()], next_page_token='abc'), snooze_service.ListSnoozesResponse(snoozes=[], next_page_token='def'), snooze_service.ListSnoozesResponse(snoozes=[snooze.Snooze()], next_page_token='ghi'), snooze_service.ListSnoozesResponse(snoozes=[snooze.Snooze(), snooze.Snooze()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_snoozes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, snooze.Snooze) for i in results))

def test_list_snoozes_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_snoozes), '__call__') as call:
        call.side_effect = (snooze_service.ListSnoozesResponse(snoozes=[snooze.Snooze(), snooze.Snooze(), snooze.Snooze()], next_page_token='abc'), snooze_service.ListSnoozesResponse(snoozes=[], next_page_token='def'), snooze_service.ListSnoozesResponse(snoozes=[snooze.Snooze()], next_page_token='ghi'), snooze_service.ListSnoozesResponse(snoozes=[snooze.Snooze(), snooze.Snooze()]), RuntimeError)
        pages = list(client.list_snoozes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_snoozes_async_pager():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_snoozes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (snooze_service.ListSnoozesResponse(snoozes=[snooze.Snooze(), snooze.Snooze(), snooze.Snooze()], next_page_token='abc'), snooze_service.ListSnoozesResponse(snoozes=[], next_page_token='def'), snooze_service.ListSnoozesResponse(snoozes=[snooze.Snooze()], next_page_token='ghi'), snooze_service.ListSnoozesResponse(snoozes=[snooze.Snooze(), snooze.Snooze()]), RuntimeError)
        async_pager = await client.list_snoozes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, snooze.Snooze) for i in responses))

@pytest.mark.asyncio
async def test_list_snoozes_async_pages():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_snoozes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (snooze_service.ListSnoozesResponse(snoozes=[snooze.Snooze(), snooze.Snooze(), snooze.Snooze()], next_page_token='abc'), snooze_service.ListSnoozesResponse(snoozes=[], next_page_token='def'), snooze_service.ListSnoozesResponse(snoozes=[snooze.Snooze()], next_page_token='ghi'), snooze_service.ListSnoozesResponse(snoozes=[snooze.Snooze(), snooze.Snooze()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_snoozes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [snooze_service.GetSnoozeRequest, dict])
def test_get_snooze(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_snooze), '__call__') as call:
        call.return_value = snooze.Snooze(name='name_value', display_name='display_name_value')
        response = client.get_snooze(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snooze_service.GetSnoozeRequest()
    assert isinstance(response, snooze.Snooze)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_get_snooze_empty_call():
    if False:
        return 10
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_snooze), '__call__') as call:
        client.get_snooze()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snooze_service.GetSnoozeRequest()

@pytest.mark.asyncio
async def test_get_snooze_async(transport: str='grpc_asyncio', request_type=snooze_service.GetSnoozeRequest):
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_snooze), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(snooze.Snooze(name='name_value', display_name='display_name_value'))
        response = await client.get_snooze(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snooze_service.GetSnoozeRequest()
    assert isinstance(response, snooze.Snooze)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

@pytest.mark.asyncio
async def test_get_snooze_async_from_dict():
    await test_get_snooze_async(request_type=dict)

def test_get_snooze_field_headers():
    if False:
        while True:
            i = 10
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = snooze_service.GetSnoozeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_snooze), '__call__') as call:
        call.return_value = snooze.Snooze()
        client.get_snooze(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_snooze_field_headers_async():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = snooze_service.GetSnoozeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_snooze), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(snooze.Snooze())
        await client.get_snooze(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_snooze_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_snooze), '__call__') as call:
        call.return_value = snooze.Snooze()
        client.get_snooze(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_snooze_flattened_error():
    if False:
        while True:
            i = 10
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_snooze(snooze_service.GetSnoozeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_snooze_flattened_async():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_snooze), '__call__') as call:
        call.return_value = snooze.Snooze()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(snooze.Snooze())
        response = await client.get_snooze(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_snooze_flattened_error_async():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_snooze(snooze_service.GetSnoozeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [snooze_service.UpdateSnoozeRequest, dict])
def test_update_snooze(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_snooze), '__call__') as call:
        call.return_value = gm_snooze.Snooze(name='name_value', display_name='display_name_value')
        response = client.update_snooze(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snooze_service.UpdateSnoozeRequest()
    assert isinstance(response, gm_snooze.Snooze)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_update_snooze_empty_call():
    if False:
        i = 10
        return i + 15
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_snooze), '__call__') as call:
        client.update_snooze()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snooze_service.UpdateSnoozeRequest()

@pytest.mark.asyncio
async def test_update_snooze_async(transport: str='grpc_asyncio', request_type=snooze_service.UpdateSnoozeRequest):
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_snooze), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gm_snooze.Snooze(name='name_value', display_name='display_name_value'))
        response = await client.update_snooze(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == snooze_service.UpdateSnoozeRequest()
    assert isinstance(response, gm_snooze.Snooze)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

@pytest.mark.asyncio
async def test_update_snooze_async_from_dict():
    await test_update_snooze_async(request_type=dict)

def test_update_snooze_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = snooze_service.UpdateSnoozeRequest()
    request.snooze.name = 'name_value'
    with mock.patch.object(type(client.transport.update_snooze), '__call__') as call:
        call.return_value = gm_snooze.Snooze()
        client.update_snooze(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'snooze.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_snooze_field_headers_async():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = snooze_service.UpdateSnoozeRequest()
    request.snooze.name = 'name_value'
    with mock.patch.object(type(client.transport.update_snooze), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gm_snooze.Snooze())
        await client.update_snooze(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'snooze.name=name_value') in kw['metadata']

def test_update_snooze_flattened():
    if False:
        return 10
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_snooze), '__call__') as call:
        call.return_value = gm_snooze.Snooze()
        client.update_snooze(snooze=gm_snooze.Snooze(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].snooze
        mock_val = gm_snooze.Snooze(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_snooze_flattened_error():
    if False:
        print('Hello World!')
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_snooze(snooze_service.UpdateSnoozeRequest(), snooze=gm_snooze.Snooze(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_snooze_flattened_async():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_snooze), '__call__') as call:
        call.return_value = gm_snooze.Snooze()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gm_snooze.Snooze())
        response = await client.update_snooze(snooze=gm_snooze.Snooze(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].snooze
        mock_val = gm_snooze.Snooze(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_snooze_flattened_error_async():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_snooze(snooze_service.UpdateSnoozeRequest(), snooze=gm_snooze.Snooze(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SnoozeServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.SnoozeServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SnoozeServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.SnoozeServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = SnoozeServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = SnoozeServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.SnoozeServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SnoozeServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.SnoozeServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = SnoozeServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SnoozeServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.SnoozeServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.SnoozeServiceGrpcTransport, transports.SnoozeServiceGrpcAsyncIOTransport])
def test_transport_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc'])
def test_transport_kind(transport_name):
    if False:
        for i in range(10):
            print('nop')
    transport = SnoozeServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.SnoozeServiceGrpcTransport)

def test_snooze_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.SnoozeServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_snooze_service_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.monitoring_v3.services.snooze_service.transports.SnoozeServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.SnoozeServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_snooze', 'list_snoozes', 'get_snooze', 'update_snooze')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_snooze_service_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.monitoring_v3.services.snooze_service.transports.SnoozeServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.SnoozeServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), quota_project_id='octopus')

def test_snooze_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.monitoring_v3.services.snooze_service.transports.SnoozeServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.SnoozeServiceTransport()
        adc.assert_called_once()

def test_snooze_service_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        SnoozeServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.SnoozeServiceGrpcTransport, transports.SnoozeServiceGrpcAsyncIOTransport])
def test_snooze_service_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.SnoozeServiceGrpcTransport, transports.SnoozeServiceGrpcAsyncIOTransport])
def test_snooze_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.SnoozeServiceGrpcTransport, grpc_helpers), (transports.SnoozeServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_snooze_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('monitoring.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), scopes=['1', '2'], default_host='monitoring.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.SnoozeServiceGrpcTransport, transports.SnoozeServiceGrpcAsyncIOTransport])
def test_snooze_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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
def test_snooze_service_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='monitoring.googleapis.com'), transport=transport_name)
    assert client.transport._host == 'monitoring.googleapis.com:443'

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_snooze_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='monitoring.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == 'monitoring.googleapis.com:8000'

def test_snooze_service_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.SnoozeServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_snooze_service_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.SnoozeServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.SnoozeServiceGrpcTransport, transports.SnoozeServiceGrpcAsyncIOTransport])
def test_snooze_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.SnoozeServiceGrpcTransport, transports.SnoozeServiceGrpcAsyncIOTransport])
def test_snooze_service_transport_channel_mtls_with_adc(transport_class):
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

def test_alert_policy_path():
    if False:
        print('Hello World!')
    project = 'squid'
    alert_policy = 'clam'
    expected = 'projects/{project}/alertPolicies/{alert_policy}'.format(project=project, alert_policy=alert_policy)
    actual = SnoozeServiceClient.alert_policy_path(project, alert_policy)
    assert expected == actual

def test_parse_alert_policy_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'whelk', 'alert_policy': 'octopus'}
    path = SnoozeServiceClient.alert_policy_path(**expected)
    actual = SnoozeServiceClient.parse_alert_policy_path(path)
    assert expected == actual

def test_snooze_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    snooze = 'nudibranch'
    expected = 'projects/{project}/snoozes/{snooze}'.format(project=project, snooze=snooze)
    actual = SnoozeServiceClient.snooze_path(project, snooze)
    assert expected == actual

def test_parse_snooze_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'cuttlefish', 'snooze': 'mussel'}
    path = SnoozeServiceClient.snooze_path(**expected)
    actual = SnoozeServiceClient.parse_snooze_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'winkle'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = SnoozeServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'nautilus'}
    path = SnoozeServiceClient.common_billing_account_path(**expected)
    actual = SnoozeServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'scallop'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = SnoozeServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'abalone'}
    path = SnoozeServiceClient.common_folder_path(**expected)
    actual = SnoozeServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'squid'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = SnoozeServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'clam'}
    path = SnoozeServiceClient.common_organization_path(**expected)
    actual = SnoozeServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    expected = 'projects/{project}'.format(project=project)
    actual = SnoozeServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'octopus'}
    path = SnoozeServiceClient.common_project_path(**expected)
    actual = SnoozeServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = SnoozeServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = SnoozeServiceClient.common_location_path(**expected)
    actual = SnoozeServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.SnoozeServiceTransport, '_prep_wrapped_messages') as prep:
        client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.SnoozeServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = SnoozeServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = SnoozeServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        for i in range(10):
            print('nop')
    transports = ['grpc']
    for transport in transports:
        client = SnoozeServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(SnoozeServiceClient, transports.SnoozeServiceGrpcTransport), (SnoozeServiceAsyncClient, transports.SnoozeServiceGrpcAsyncIOTransport)])
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