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
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.type import calendar_period_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from google.cloud.monitoring_v3.services.service_monitoring_service import ServiceMonitoringServiceAsyncClient, ServiceMonitoringServiceClient, pagers, transports
from google.cloud.monitoring_v3.types import service
from google.cloud.monitoring_v3.types import service as gm_service
from google.cloud.monitoring_v3.types import service_service

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
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert ServiceMonitoringServiceClient._get_default_mtls_endpoint(None) is None
    assert ServiceMonitoringServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ServiceMonitoringServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ServiceMonitoringServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ServiceMonitoringServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ServiceMonitoringServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ServiceMonitoringServiceClient, 'grpc'), (ServiceMonitoringServiceAsyncClient, 'grpc_asyncio')])
def test_service_monitoring_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == 'monitoring.googleapis.com:443'

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ServiceMonitoringServiceGrpcTransport, 'grpc'), (transports.ServiceMonitoringServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_service_monitoring_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ServiceMonitoringServiceClient, 'grpc'), (ServiceMonitoringServiceAsyncClient, 'grpc_asyncio')])
def test_service_monitoring_service_client_from_service_account_file(client_class, transport_name):
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

def test_service_monitoring_service_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = ServiceMonitoringServiceClient.get_transport_class()
    available_transports = [transports.ServiceMonitoringServiceGrpcTransport]
    assert transport in available_transports
    transport = ServiceMonitoringServiceClient.get_transport_class('grpc')
    assert transport == transports.ServiceMonitoringServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ServiceMonitoringServiceClient, transports.ServiceMonitoringServiceGrpcTransport, 'grpc'), (ServiceMonitoringServiceAsyncClient, transports.ServiceMonitoringServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
@mock.patch.object(ServiceMonitoringServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceMonitoringServiceClient))
@mock.patch.object(ServiceMonitoringServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceMonitoringServiceAsyncClient))
def test_service_monitoring_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(ServiceMonitoringServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ServiceMonitoringServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ServiceMonitoringServiceClient, transports.ServiceMonitoringServiceGrpcTransport, 'grpc', 'true'), (ServiceMonitoringServiceAsyncClient, transports.ServiceMonitoringServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ServiceMonitoringServiceClient, transports.ServiceMonitoringServiceGrpcTransport, 'grpc', 'false'), (ServiceMonitoringServiceAsyncClient, transports.ServiceMonitoringServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false')])
@mock.patch.object(ServiceMonitoringServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceMonitoringServiceClient))
@mock.patch.object(ServiceMonitoringServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceMonitoringServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_service_monitoring_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [ServiceMonitoringServiceClient, ServiceMonitoringServiceAsyncClient])
@mock.patch.object(ServiceMonitoringServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceMonitoringServiceClient))
@mock.patch.object(ServiceMonitoringServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceMonitoringServiceAsyncClient))
def test_service_monitoring_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ServiceMonitoringServiceClient, transports.ServiceMonitoringServiceGrpcTransport, 'grpc'), (ServiceMonitoringServiceAsyncClient, transports.ServiceMonitoringServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_service_monitoring_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ServiceMonitoringServiceClient, transports.ServiceMonitoringServiceGrpcTransport, 'grpc', grpc_helpers), (ServiceMonitoringServiceAsyncClient, transports.ServiceMonitoringServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_service_monitoring_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_service_monitoring_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.monitoring_v3.services.service_monitoring_service.transports.ServiceMonitoringServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ServiceMonitoringServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ServiceMonitoringServiceClient, transports.ServiceMonitoringServiceGrpcTransport, 'grpc', grpc_helpers), (ServiceMonitoringServiceAsyncClient, transports.ServiceMonitoringServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_service_monitoring_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('monitoring.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), scopes=None, default_host='monitoring.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [service_service.CreateServiceRequest, dict])
def test_create_service(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = gm_service.Service(name='name_value', display_name='display_name_value')
        response = client.create_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.CreateServiceRequest()
    assert isinstance(response, gm_service.Service)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_create_service_empty_call():
    if False:
        while True:
            i = 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        client.create_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.CreateServiceRequest()

@pytest.mark.asyncio
async def test_create_service_async(transport: str='grpc_asyncio', request_type=service_service.CreateServiceRequest):
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gm_service.Service(name='name_value', display_name='display_name_value'))
        response = await client.create_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.CreateServiceRequest()
    assert isinstance(response, gm_service.Service)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

@pytest.mark.asyncio
async def test_create_service_async_from_dict():
    await test_create_service_async(request_type=dict)

def test_create_service_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.CreateServiceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = gm_service.Service()
        client.create_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_service_field_headers_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.CreateServiceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gm_service.Service())
        await client.create_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_service_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = gm_service.Service()
        client.create_service(parent='parent_value', service=gm_service.Service(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].service
        mock_val = gm_service.Service(name='name_value')
        assert arg == mock_val

def test_create_service_flattened_error():
    if False:
        return 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_service(service_service.CreateServiceRequest(), parent='parent_value', service=gm_service.Service(name='name_value'))

@pytest.mark.asyncio
async def test_create_service_flattened_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = gm_service.Service()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gm_service.Service())
        response = await client.create_service(parent='parent_value', service=gm_service.Service(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].service
        mock_val = gm_service.Service(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_service_flattened_error_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_service(service_service.CreateServiceRequest(), parent='parent_value', service=gm_service.Service(name='name_value'))

@pytest.mark.parametrize('request_type', [service_service.GetServiceRequest, dict])
def test_get_service(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = service.Service(name='name_value', display_name='display_name_value')
        response = client.get_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.GetServiceRequest()
    assert isinstance(response, service.Service)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_get_service_empty_call():
    if False:
        i = 10
        return i + 15
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        client.get_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.GetServiceRequest()

@pytest.mark.asyncio
async def test_get_service_async(transport: str='grpc_asyncio', request_type=service_service.GetServiceRequest):
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Service(name='name_value', display_name='display_name_value'))
        response = await client.get_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.GetServiceRequest()
    assert isinstance(response, service.Service)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

@pytest.mark.asyncio
async def test_get_service_async_from_dict():
    await test_get_service_async(request_type=dict)

def test_get_service_field_headers():
    if False:
        while True:
            i = 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.GetServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = service.Service()
        client.get_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_service_field_headers_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.GetServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Service())
        await client.get_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_service_flattened():
    if False:
        i = 10
        return i + 15
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = service.Service()
        client.get_service(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_service_flattened_error():
    if False:
        return 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_service(service_service.GetServiceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_service_flattened_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = service.Service()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Service())
        response = await client.get_service(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_service_flattened_error_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_service(service_service.GetServiceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service_service.ListServicesRequest, dict])
def test_list_services(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = service_service.ListServicesResponse(next_page_token='next_page_token_value')
        response = client.list_services(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.ListServicesRequest()
    assert isinstance(response, pagers.ListServicesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_services_empty_call():
    if False:
        print('Hello World!')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        client.list_services()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.ListServicesRequest()

@pytest.mark.asyncio
async def test_list_services_async(transport: str='grpc_asyncio', request_type=service_service.ListServicesRequest):
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_service.ListServicesResponse(next_page_token='next_page_token_value'))
        response = await client.list_services(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.ListServicesRequest()
    assert isinstance(response, pagers.ListServicesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_services_async_from_dict():
    await test_list_services_async(request_type=dict)

def test_list_services_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.ListServicesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = service_service.ListServicesResponse()
        client.list_services(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_services_field_headers_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.ListServicesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_service.ListServicesResponse())
        await client.list_services(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_services_flattened():
    if False:
        i = 10
        return i + 15
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = service_service.ListServicesResponse()
        client.list_services(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_services_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_services(service_service.ListServicesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_services_flattened_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = service_service.ListServicesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_service.ListServicesResponse())
        response = await client.list_services(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_services_flattened_error_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_services(service_service.ListServicesRequest(), parent='parent_value')

def test_list_services_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.side_effect = (service_service.ListServicesResponse(services=[service.Service(), service.Service(), service.Service()], next_page_token='abc'), service_service.ListServicesResponse(services=[], next_page_token='def'), service_service.ListServicesResponse(services=[service.Service()], next_page_token='ghi'), service_service.ListServicesResponse(services=[service.Service(), service.Service()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_services(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.Service) for i in results))

def test_list_services_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.side_effect = (service_service.ListServicesResponse(services=[service.Service(), service.Service(), service.Service()], next_page_token='abc'), service_service.ListServicesResponse(services=[], next_page_token='def'), service_service.ListServicesResponse(services=[service.Service()], next_page_token='ghi'), service_service.ListServicesResponse(services=[service.Service(), service.Service()]), RuntimeError)
        pages = list(client.list_services(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_services_async_pager():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_services), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service_service.ListServicesResponse(services=[service.Service(), service.Service(), service.Service()], next_page_token='abc'), service_service.ListServicesResponse(services=[], next_page_token='def'), service_service.ListServicesResponse(services=[service.Service()], next_page_token='ghi'), service_service.ListServicesResponse(services=[service.Service(), service.Service()]), RuntimeError)
        async_pager = await client.list_services(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, service.Service) for i in responses))

@pytest.mark.asyncio
async def test_list_services_async_pages():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_services), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service_service.ListServicesResponse(services=[service.Service(), service.Service(), service.Service()], next_page_token='abc'), service_service.ListServicesResponse(services=[], next_page_token='def'), service_service.ListServicesResponse(services=[service.Service()], next_page_token='ghi'), service_service.ListServicesResponse(services=[service.Service(), service.Service()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_services(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service_service.UpdateServiceRequest, dict])
def test_update_service(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_service), '__call__') as call:
        call.return_value = gm_service.Service(name='name_value', display_name='display_name_value')
        response = client.update_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.UpdateServiceRequest()
    assert isinstance(response, gm_service.Service)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_update_service_empty_call():
    if False:
        return 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_service), '__call__') as call:
        client.update_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.UpdateServiceRequest()

@pytest.mark.asyncio
async def test_update_service_async(transport: str='grpc_asyncio', request_type=service_service.UpdateServiceRequest):
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gm_service.Service(name='name_value', display_name='display_name_value'))
        response = await client.update_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.UpdateServiceRequest()
    assert isinstance(response, gm_service.Service)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

@pytest.mark.asyncio
async def test_update_service_async_from_dict():
    await test_update_service_async(request_type=dict)

def test_update_service_field_headers():
    if False:
        while True:
            i = 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.UpdateServiceRequest()
    request.service.name = 'name_value'
    with mock.patch.object(type(client.transport.update_service), '__call__') as call:
        call.return_value = gm_service.Service()
        client.update_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_service_field_headers_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.UpdateServiceRequest()
    request.service.name = 'name_value'
    with mock.patch.object(type(client.transport.update_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gm_service.Service())
        await client.update_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service.name=name_value') in kw['metadata']

def test_update_service_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_service), '__call__') as call:
        call.return_value = gm_service.Service()
        client.update_service(service=gm_service.Service(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service
        mock_val = gm_service.Service(name='name_value')
        assert arg == mock_val

def test_update_service_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_service(service_service.UpdateServiceRequest(), service=gm_service.Service(name='name_value'))

@pytest.mark.asyncio
async def test_update_service_flattened_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_service), '__call__') as call:
        call.return_value = gm_service.Service()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gm_service.Service())
        response = await client.update_service(service=gm_service.Service(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service
        mock_val = gm_service.Service(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_service_flattened_error_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_service(service_service.UpdateServiceRequest(), service=gm_service.Service(name='name_value'))

@pytest.mark.parametrize('request_type', [service_service.DeleteServiceRequest, dict])
def test_delete_service(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = None
        response = client.delete_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.DeleteServiceRequest()
    assert response is None

def test_delete_service_empty_call():
    if False:
        i = 10
        return i + 15
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        client.delete_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.DeleteServiceRequest()

@pytest.mark.asyncio
async def test_delete_service_async(transport: str='grpc_asyncio', request_type=service_service.DeleteServiceRequest):
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.DeleteServiceRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_service_async_from_dict():
    await test_delete_service_async(request_type=dict)

def test_delete_service_field_headers():
    if False:
        i = 10
        return i + 15
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.DeleteServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = None
        client.delete_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_service_field_headers_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.DeleteServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_service_flattened():
    if False:
        print('Hello World!')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = None
        client.delete_service(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_service_flattened_error():
    if False:
        return 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_service(service_service.DeleteServiceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_service_flattened_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_service(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_service_flattened_error_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_service(service_service.DeleteServiceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service_service.CreateServiceLevelObjectiveRequest, dict])
def test_create_service_level_objective(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_service_level_objective), '__call__') as call:
        call.return_value = service.ServiceLevelObjective(name='name_value', display_name='display_name_value', goal=0.419, calendar_period=calendar_period_pb2.CalendarPeriod.DAY)
        response = client.create_service_level_objective(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.CreateServiceLevelObjectiveRequest()
    assert isinstance(response, service.ServiceLevelObjective)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert math.isclose(response.goal, 0.419, rel_tol=1e-06)

def test_create_service_level_objective_empty_call():
    if False:
        i = 10
        return i + 15
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_service_level_objective), '__call__') as call:
        client.create_service_level_objective()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.CreateServiceLevelObjectiveRequest()

@pytest.mark.asyncio
async def test_create_service_level_objective_async(transport: str='grpc_asyncio', request_type=service_service.CreateServiceLevelObjectiveRequest):
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_service_level_objective), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ServiceLevelObjective(name='name_value', display_name='display_name_value', goal=0.419))
        response = await client.create_service_level_objective(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.CreateServiceLevelObjectiveRequest()
    assert isinstance(response, service.ServiceLevelObjective)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert math.isclose(response.goal, 0.419, rel_tol=1e-06)

@pytest.mark.asyncio
async def test_create_service_level_objective_async_from_dict():
    await test_create_service_level_objective_async(request_type=dict)

def test_create_service_level_objective_field_headers():
    if False:
        return 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.CreateServiceLevelObjectiveRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_service_level_objective), '__call__') as call:
        call.return_value = service.ServiceLevelObjective()
        client.create_service_level_objective(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_service_level_objective_field_headers_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.CreateServiceLevelObjectiveRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_service_level_objective), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ServiceLevelObjective())
        await client.create_service_level_objective(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_service_level_objective_flattened():
    if False:
        while True:
            i = 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_service_level_objective), '__call__') as call:
        call.return_value = service.ServiceLevelObjective()
        client.create_service_level_objective(parent='parent_value', service_level_objective=service.ServiceLevelObjective(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].service_level_objective
        mock_val = service.ServiceLevelObjective(name='name_value')
        assert arg == mock_val

def test_create_service_level_objective_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_service_level_objective(service_service.CreateServiceLevelObjectiveRequest(), parent='parent_value', service_level_objective=service.ServiceLevelObjective(name='name_value'))

@pytest.mark.asyncio
async def test_create_service_level_objective_flattened_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_service_level_objective), '__call__') as call:
        call.return_value = service.ServiceLevelObjective()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ServiceLevelObjective())
        response = await client.create_service_level_objective(parent='parent_value', service_level_objective=service.ServiceLevelObjective(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].service_level_objective
        mock_val = service.ServiceLevelObjective(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_service_level_objective_flattened_error_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_service_level_objective(service_service.CreateServiceLevelObjectiveRequest(), parent='parent_value', service_level_objective=service.ServiceLevelObjective(name='name_value'))

@pytest.mark.parametrize('request_type', [service_service.GetServiceLevelObjectiveRequest, dict])
def test_get_service_level_objective(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service_level_objective), '__call__') as call:
        call.return_value = service.ServiceLevelObjective(name='name_value', display_name='display_name_value', goal=0.419, calendar_period=calendar_period_pb2.CalendarPeriod.DAY)
        response = client.get_service_level_objective(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.GetServiceLevelObjectiveRequest()
    assert isinstance(response, service.ServiceLevelObjective)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert math.isclose(response.goal, 0.419, rel_tol=1e-06)

def test_get_service_level_objective_empty_call():
    if False:
        print('Hello World!')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_service_level_objective), '__call__') as call:
        client.get_service_level_objective()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.GetServiceLevelObjectiveRequest()

@pytest.mark.asyncio
async def test_get_service_level_objective_async(transport: str='grpc_asyncio', request_type=service_service.GetServiceLevelObjectiveRequest):
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service_level_objective), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ServiceLevelObjective(name='name_value', display_name='display_name_value', goal=0.419))
        response = await client.get_service_level_objective(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.GetServiceLevelObjectiveRequest()
    assert isinstance(response, service.ServiceLevelObjective)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert math.isclose(response.goal, 0.419, rel_tol=1e-06)

@pytest.mark.asyncio
async def test_get_service_level_objective_async_from_dict():
    await test_get_service_level_objective_async(request_type=dict)

def test_get_service_level_objective_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.GetServiceLevelObjectiveRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_service_level_objective), '__call__') as call:
        call.return_value = service.ServiceLevelObjective()
        client.get_service_level_objective(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_service_level_objective_field_headers_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.GetServiceLevelObjectiveRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_service_level_objective), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ServiceLevelObjective())
        await client.get_service_level_objective(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_service_level_objective_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_service_level_objective), '__call__') as call:
        call.return_value = service.ServiceLevelObjective()
        client.get_service_level_objective(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_service_level_objective_flattened_error():
    if False:
        while True:
            i = 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_service_level_objective(service_service.GetServiceLevelObjectiveRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_service_level_objective_flattened_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_service_level_objective), '__call__') as call:
        call.return_value = service.ServiceLevelObjective()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ServiceLevelObjective())
        response = await client.get_service_level_objective(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_service_level_objective_flattened_error_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_service_level_objective(service_service.GetServiceLevelObjectiveRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service_service.ListServiceLevelObjectivesRequest, dict])
def test_list_service_level_objectives(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_service_level_objectives), '__call__') as call:
        call.return_value = service_service.ListServiceLevelObjectivesResponse(next_page_token='next_page_token_value')
        response = client.list_service_level_objectives(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.ListServiceLevelObjectivesRequest()
    assert isinstance(response, pagers.ListServiceLevelObjectivesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_service_level_objectives_empty_call():
    if False:
        i = 10
        return i + 15
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_service_level_objectives), '__call__') as call:
        client.list_service_level_objectives()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.ListServiceLevelObjectivesRequest()

@pytest.mark.asyncio
async def test_list_service_level_objectives_async(transport: str='grpc_asyncio', request_type=service_service.ListServiceLevelObjectivesRequest):
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_service_level_objectives), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_service.ListServiceLevelObjectivesResponse(next_page_token='next_page_token_value'))
        response = await client.list_service_level_objectives(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.ListServiceLevelObjectivesRequest()
    assert isinstance(response, pagers.ListServiceLevelObjectivesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_service_level_objectives_async_from_dict():
    await test_list_service_level_objectives_async(request_type=dict)

def test_list_service_level_objectives_field_headers():
    if False:
        while True:
            i = 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.ListServiceLevelObjectivesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_service_level_objectives), '__call__') as call:
        call.return_value = service_service.ListServiceLevelObjectivesResponse()
        client.list_service_level_objectives(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_service_level_objectives_field_headers_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.ListServiceLevelObjectivesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_service_level_objectives), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_service.ListServiceLevelObjectivesResponse())
        await client.list_service_level_objectives(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_service_level_objectives_flattened():
    if False:
        while True:
            i = 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_service_level_objectives), '__call__') as call:
        call.return_value = service_service.ListServiceLevelObjectivesResponse()
        client.list_service_level_objectives(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_service_level_objectives_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_service_level_objectives(service_service.ListServiceLevelObjectivesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_service_level_objectives_flattened_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_service_level_objectives), '__call__') as call:
        call.return_value = service_service.ListServiceLevelObjectivesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_service.ListServiceLevelObjectivesResponse())
        response = await client.list_service_level_objectives(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_service_level_objectives_flattened_error_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_service_level_objectives(service_service.ListServiceLevelObjectivesRequest(), parent='parent_value')

def test_list_service_level_objectives_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_service_level_objectives), '__call__') as call:
        call.side_effect = (service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[service.ServiceLevelObjective(), service.ServiceLevelObjective(), service.ServiceLevelObjective()], next_page_token='abc'), service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[], next_page_token='def'), service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[service.ServiceLevelObjective()], next_page_token='ghi'), service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[service.ServiceLevelObjective(), service.ServiceLevelObjective()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_service_level_objectives(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.ServiceLevelObjective) for i in results))

def test_list_service_level_objectives_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_service_level_objectives), '__call__') as call:
        call.side_effect = (service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[service.ServiceLevelObjective(), service.ServiceLevelObjective(), service.ServiceLevelObjective()], next_page_token='abc'), service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[], next_page_token='def'), service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[service.ServiceLevelObjective()], next_page_token='ghi'), service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[service.ServiceLevelObjective(), service.ServiceLevelObjective()]), RuntimeError)
        pages = list(client.list_service_level_objectives(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_service_level_objectives_async_pager():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_service_level_objectives), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[service.ServiceLevelObjective(), service.ServiceLevelObjective(), service.ServiceLevelObjective()], next_page_token='abc'), service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[], next_page_token='def'), service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[service.ServiceLevelObjective()], next_page_token='ghi'), service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[service.ServiceLevelObjective(), service.ServiceLevelObjective()]), RuntimeError)
        async_pager = await client.list_service_level_objectives(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, service.ServiceLevelObjective) for i in responses))

@pytest.mark.asyncio
async def test_list_service_level_objectives_async_pages():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_service_level_objectives), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[service.ServiceLevelObjective(), service.ServiceLevelObjective(), service.ServiceLevelObjective()], next_page_token='abc'), service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[], next_page_token='def'), service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[service.ServiceLevelObjective()], next_page_token='ghi'), service_service.ListServiceLevelObjectivesResponse(service_level_objectives=[service.ServiceLevelObjective(), service.ServiceLevelObjective()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_service_level_objectives(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service_service.UpdateServiceLevelObjectiveRequest, dict])
def test_update_service_level_objective(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_service_level_objective), '__call__') as call:
        call.return_value = service.ServiceLevelObjective(name='name_value', display_name='display_name_value', goal=0.419, calendar_period=calendar_period_pb2.CalendarPeriod.DAY)
        response = client.update_service_level_objective(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.UpdateServiceLevelObjectiveRequest()
    assert isinstance(response, service.ServiceLevelObjective)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert math.isclose(response.goal, 0.419, rel_tol=1e-06)

def test_update_service_level_objective_empty_call():
    if False:
        print('Hello World!')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_service_level_objective), '__call__') as call:
        client.update_service_level_objective()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.UpdateServiceLevelObjectiveRequest()

@pytest.mark.asyncio
async def test_update_service_level_objective_async(transport: str='grpc_asyncio', request_type=service_service.UpdateServiceLevelObjectiveRequest):
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_service_level_objective), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ServiceLevelObjective(name='name_value', display_name='display_name_value', goal=0.419))
        response = await client.update_service_level_objective(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.UpdateServiceLevelObjectiveRequest()
    assert isinstance(response, service.ServiceLevelObjective)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert math.isclose(response.goal, 0.419, rel_tol=1e-06)

@pytest.mark.asyncio
async def test_update_service_level_objective_async_from_dict():
    await test_update_service_level_objective_async(request_type=dict)

def test_update_service_level_objective_field_headers():
    if False:
        print('Hello World!')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.UpdateServiceLevelObjectiveRequest()
    request.service_level_objective.name = 'name_value'
    with mock.patch.object(type(client.transport.update_service_level_objective), '__call__') as call:
        call.return_value = service.ServiceLevelObjective()
        client.update_service_level_objective(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_level_objective.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_service_level_objective_field_headers_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.UpdateServiceLevelObjectiveRequest()
    request.service_level_objective.name = 'name_value'
    with mock.patch.object(type(client.transport.update_service_level_objective), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ServiceLevelObjective())
        await client.update_service_level_objective(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_level_objective.name=name_value') in kw['metadata']

def test_update_service_level_objective_flattened():
    if False:
        print('Hello World!')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_service_level_objective), '__call__') as call:
        call.return_value = service.ServiceLevelObjective()
        client.update_service_level_objective(service_level_objective=service.ServiceLevelObjective(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_level_objective
        mock_val = service.ServiceLevelObjective(name='name_value')
        assert arg == mock_val

def test_update_service_level_objective_flattened_error():
    if False:
        while True:
            i = 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_service_level_objective(service_service.UpdateServiceLevelObjectiveRequest(), service_level_objective=service.ServiceLevelObjective(name='name_value'))

@pytest.mark.asyncio
async def test_update_service_level_objective_flattened_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_service_level_objective), '__call__') as call:
        call.return_value = service.ServiceLevelObjective()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ServiceLevelObjective())
        response = await client.update_service_level_objective(service_level_objective=service.ServiceLevelObjective(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_level_objective
        mock_val = service.ServiceLevelObjective(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_service_level_objective_flattened_error_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_service_level_objective(service_service.UpdateServiceLevelObjectiveRequest(), service_level_objective=service.ServiceLevelObjective(name='name_value'))

@pytest.mark.parametrize('request_type', [service_service.DeleteServiceLevelObjectiveRequest, dict])
def test_delete_service_level_objective(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_service_level_objective), '__call__') as call:
        call.return_value = None
        response = client.delete_service_level_objective(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.DeleteServiceLevelObjectiveRequest()
    assert response is None

def test_delete_service_level_objective_empty_call():
    if False:
        print('Hello World!')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_service_level_objective), '__call__') as call:
        client.delete_service_level_objective()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.DeleteServiceLevelObjectiveRequest()

@pytest.mark.asyncio
async def test_delete_service_level_objective_async(transport: str='grpc_asyncio', request_type=service_service.DeleteServiceLevelObjectiveRequest):
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_service_level_objective), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_service_level_objective(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_service.DeleteServiceLevelObjectiveRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_service_level_objective_async_from_dict():
    await test_delete_service_level_objective_async(request_type=dict)

def test_delete_service_level_objective_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.DeleteServiceLevelObjectiveRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_service_level_objective), '__call__') as call:
        call.return_value = None
        client.delete_service_level_objective(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_service_level_objective_field_headers_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_service.DeleteServiceLevelObjectiveRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_service_level_objective), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_service_level_objective(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_service_level_objective_flattened():
    if False:
        return 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_service_level_objective), '__call__') as call:
        call.return_value = None
        client.delete_service_level_objective(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_service_level_objective_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_service_level_objective(service_service.DeleteServiceLevelObjectiveRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_service_level_objective_flattened_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_service_level_objective), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_service_level_objective(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_service_level_objective_flattened_error_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_service_level_objective(service_service.DeleteServiceLevelObjectiveRequest(), name='name_value')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.ServiceMonitoringServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ServiceMonitoringServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ServiceMonitoringServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ServiceMonitoringServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ServiceMonitoringServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ServiceMonitoringServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ServiceMonitoringServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ServiceMonitoringServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.ServiceMonitoringServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ServiceMonitoringServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ServiceMonitoringServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ServiceMonitoringServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ServiceMonitoringServiceGrpcTransport, transports.ServiceMonitoringServiceGrpcAsyncIOTransport])
def test_transport_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc'])
def test_transport_kind(transport_name):
    if False:
        while True:
            i = 10
    transport = ServiceMonitoringServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ServiceMonitoringServiceGrpcTransport)

def test_service_monitoring_service_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ServiceMonitoringServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_service_monitoring_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.monitoring_v3.services.service_monitoring_service.transports.ServiceMonitoringServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ServiceMonitoringServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_service', 'get_service', 'list_services', 'update_service', 'delete_service', 'create_service_level_objective', 'get_service_level_objective', 'list_service_level_objectives', 'update_service_level_objective', 'delete_service_level_objective')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_service_monitoring_service_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.monitoring_v3.services.service_monitoring_service.transports.ServiceMonitoringServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ServiceMonitoringServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), quota_project_id='octopus')

def test_service_monitoring_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.monitoring_v3.services.service_monitoring_service.transports.ServiceMonitoringServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ServiceMonitoringServiceTransport()
        adc.assert_called_once()

def test_service_monitoring_service_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ServiceMonitoringServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ServiceMonitoringServiceGrpcTransport, transports.ServiceMonitoringServiceGrpcAsyncIOTransport])
def test_service_monitoring_service_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ServiceMonitoringServiceGrpcTransport, transports.ServiceMonitoringServiceGrpcAsyncIOTransport])
def test_service_monitoring_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ServiceMonitoringServiceGrpcTransport, grpc_helpers), (transports.ServiceMonitoringServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_service_monitoring_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('monitoring.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/monitoring', 'https://www.googleapis.com/auth/monitoring.read'), scopes=['1', '2'], default_host='monitoring.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ServiceMonitoringServiceGrpcTransport, transports.ServiceMonitoringServiceGrpcAsyncIOTransport])
def test_service_monitoring_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_service_monitoring_service_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='monitoring.googleapis.com'), transport=transport_name)
    assert client.transport._host == 'monitoring.googleapis.com:443'

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_service_monitoring_service_host_with_port(transport_name):
    if False:
        return 10
    client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='monitoring.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == 'monitoring.googleapis.com:8000'

def test_service_monitoring_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ServiceMonitoringServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_service_monitoring_service_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ServiceMonitoringServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ServiceMonitoringServiceGrpcTransport, transports.ServiceMonitoringServiceGrpcAsyncIOTransport])
def test_service_monitoring_service_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        print('Hello World!')
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

@pytest.mark.parametrize('transport_class', [transports.ServiceMonitoringServiceGrpcTransport, transports.ServiceMonitoringServiceGrpcAsyncIOTransport])
def test_service_monitoring_service_transport_channel_mtls_with_adc(transport_class):
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

def test_service_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    service = 'clam'
    expected = 'projects/{project}/services/{service}'.format(project=project, service=service)
    actual = ServiceMonitoringServiceClient.service_path(project, service)
    assert expected == actual

def test_parse_service_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'whelk', 'service': 'octopus'}
    path = ServiceMonitoringServiceClient.service_path(**expected)
    actual = ServiceMonitoringServiceClient.parse_service_path(path)
    assert expected == actual

def test_service_level_objective_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'oyster'
    service = 'nudibranch'
    service_level_objective = 'cuttlefish'
    expected = 'projects/{project}/services/{service}/serviceLevelObjectives/{service_level_objective}'.format(project=project, service=service, service_level_objective=service_level_objective)
    actual = ServiceMonitoringServiceClient.service_level_objective_path(project, service, service_level_objective)
    assert expected == actual

def test_parse_service_level_objective_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'mussel', 'service': 'winkle', 'service_level_objective': 'nautilus'}
    path = ServiceMonitoringServiceClient.service_level_objective_path(**expected)
    actual = ServiceMonitoringServiceClient.parse_service_level_objective_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ServiceMonitoringServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'abalone'}
    path = ServiceMonitoringServiceClient.common_billing_account_path(**expected)
    actual = ServiceMonitoringServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ServiceMonitoringServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'clam'}
    path = ServiceMonitoringServiceClient.common_folder_path(**expected)
    actual = ServiceMonitoringServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ServiceMonitoringServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'octopus'}
    path = ServiceMonitoringServiceClient.common_organization_path(**expected)
    actual = ServiceMonitoringServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = ServiceMonitoringServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch'}
    path = ServiceMonitoringServiceClient.common_project_path(**expected)
    actual = ServiceMonitoringServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ServiceMonitoringServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = ServiceMonitoringServiceClient.common_location_path(**expected)
    actual = ServiceMonitoringServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ServiceMonitoringServiceTransport, '_prep_wrapped_messages') as prep:
        client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ServiceMonitoringServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = ServiceMonitoringServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ServiceMonitoringServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['grpc']
    for transport in transports:
        client = ServiceMonitoringServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ServiceMonitoringServiceClient, transports.ServiceMonitoringServiceGrpcTransport), (ServiceMonitoringServiceAsyncClient, transports.ServiceMonitoringServiceGrpcAsyncIOTransport)])
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