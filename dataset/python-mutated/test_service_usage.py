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
from google.protobuf import json_format
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.service_usage_v1.services.service_usage import ServiceUsageAsyncClient, ServiceUsageClient, pagers, transports
from google.cloud.service_usage_v1.types import resources, serviceusage

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
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert ServiceUsageClient._get_default_mtls_endpoint(None) is None
    assert ServiceUsageClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ServiceUsageClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ServiceUsageClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ServiceUsageClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ServiceUsageClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ServiceUsageClient, 'grpc'), (ServiceUsageAsyncClient, 'grpc_asyncio'), (ServiceUsageClient, 'rest')])
def test_service_usage_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('serviceusage.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://serviceusage.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ServiceUsageGrpcTransport, 'grpc'), (transports.ServiceUsageGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.ServiceUsageRestTransport, 'rest')])
def test_service_usage_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ServiceUsageClient, 'grpc'), (ServiceUsageAsyncClient, 'grpc_asyncio'), (ServiceUsageClient, 'rest')])
def test_service_usage_client_from_service_account_file(client_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_file') as factory:
        factory.return_value = creds
        client = client_class.from_service_account_file('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        client = client_class.from_service_account_json('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('serviceusage.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://serviceusage.googleapis.com')

def test_service_usage_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = ServiceUsageClient.get_transport_class()
    available_transports = [transports.ServiceUsageGrpcTransport, transports.ServiceUsageRestTransport]
    assert transport in available_transports
    transport = ServiceUsageClient.get_transport_class('grpc')
    assert transport == transports.ServiceUsageGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ServiceUsageClient, transports.ServiceUsageGrpcTransport, 'grpc'), (ServiceUsageAsyncClient, transports.ServiceUsageGrpcAsyncIOTransport, 'grpc_asyncio'), (ServiceUsageClient, transports.ServiceUsageRestTransport, 'rest')])
@mock.patch.object(ServiceUsageClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceUsageClient))
@mock.patch.object(ServiceUsageAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceUsageAsyncClient))
def test_service_usage_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(ServiceUsageClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ServiceUsageClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ServiceUsageClient, transports.ServiceUsageGrpcTransport, 'grpc', 'true'), (ServiceUsageAsyncClient, transports.ServiceUsageGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ServiceUsageClient, transports.ServiceUsageGrpcTransport, 'grpc', 'false'), (ServiceUsageAsyncClient, transports.ServiceUsageGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (ServiceUsageClient, transports.ServiceUsageRestTransport, 'rest', 'true'), (ServiceUsageClient, transports.ServiceUsageRestTransport, 'rest', 'false')])
@mock.patch.object(ServiceUsageClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceUsageClient))
@mock.patch.object(ServiceUsageAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceUsageAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_service_usage_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
    if False:
        return 10
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

@pytest.mark.parametrize('client_class', [ServiceUsageClient, ServiceUsageAsyncClient])
@mock.patch.object(ServiceUsageClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceUsageClient))
@mock.patch.object(ServiceUsageAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceUsageAsyncClient))
def test_service_usage_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ServiceUsageClient, transports.ServiceUsageGrpcTransport, 'grpc'), (ServiceUsageAsyncClient, transports.ServiceUsageGrpcAsyncIOTransport, 'grpc_asyncio'), (ServiceUsageClient, transports.ServiceUsageRestTransport, 'rest')])
def test_service_usage_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ServiceUsageClient, transports.ServiceUsageGrpcTransport, 'grpc', grpc_helpers), (ServiceUsageAsyncClient, transports.ServiceUsageGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (ServiceUsageClient, transports.ServiceUsageRestTransport, 'rest', None)])
def test_service_usage_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_service_usage_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.service_usage_v1.services.service_usage.transports.ServiceUsageGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ServiceUsageClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ServiceUsageClient, transports.ServiceUsageGrpcTransport, 'grpc', grpc_helpers), (ServiceUsageAsyncClient, transports.ServiceUsageGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_service_usage_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('serviceusage.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/service.management'), scopes=None, default_host='serviceusage.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [serviceusage.EnableServiceRequest, dict])
def test_enable_service(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.enable_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.enable_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.EnableServiceRequest()
    assert isinstance(response, future.Future)

def test_enable_service_empty_call():
    if False:
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.enable_service), '__call__') as call:
        client.enable_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.EnableServiceRequest()

@pytest.mark.asyncio
async def test_enable_service_async(transport: str='grpc_asyncio', request_type=serviceusage.EnableServiceRequest):
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.enable_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.enable_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.EnableServiceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_enable_service_async_from_dict():
    await test_enable_service_async(request_type=dict)

def test_enable_service_field_headers():
    if False:
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials())
    request = serviceusage.EnableServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.enable_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.enable_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_enable_service_field_headers_async():
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = serviceusage.EnableServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.enable_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.enable_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [serviceusage.DisableServiceRequest, dict])
def test_disable_service(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.disable_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.disable_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.DisableServiceRequest()
    assert isinstance(response, future.Future)

def test_disable_service_empty_call():
    if False:
        print('Hello World!')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.disable_service), '__call__') as call:
        client.disable_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.DisableServiceRequest()

@pytest.mark.asyncio
async def test_disable_service_async(transport: str='grpc_asyncio', request_type=serviceusage.DisableServiceRequest):
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.disable_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.disable_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.DisableServiceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_disable_service_async_from_dict():
    await test_disable_service_async(request_type=dict)

def test_disable_service_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials())
    request = serviceusage.DisableServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.disable_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.disable_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_disable_service_field_headers_async():
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = serviceusage.DisableServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.disable_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.disable_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [serviceusage.GetServiceRequest, dict])
def test_get_service(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = resources.Service(name='name_value', parent='parent_value', state=resources.State.DISABLED)
        response = client.get_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.GetServiceRequest()
    assert isinstance(response, resources.Service)
    assert response.name == 'name_value'
    assert response.parent == 'parent_value'
    assert response.state == resources.State.DISABLED

def test_get_service_empty_call():
    if False:
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        client.get_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.GetServiceRequest()

@pytest.mark.asyncio
async def test_get_service_async(transport: str='grpc_asyncio', request_type=serviceusage.GetServiceRequest):
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Service(name='name_value', parent='parent_value', state=resources.State.DISABLED))
        response = await client.get_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.GetServiceRequest()
    assert isinstance(response, resources.Service)
    assert response.name == 'name_value'
    assert response.parent == 'parent_value'
    assert response.state == resources.State.DISABLED

@pytest.mark.asyncio
async def test_get_service_async_from_dict():
    await test_get_service_async(request_type=dict)

def test_get_service_field_headers():
    if False:
        print('Hello World!')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials())
    request = serviceusage.GetServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = resources.Service()
        client.get_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_service_field_headers_async():
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = serviceusage.GetServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Service())
        await client.get_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [serviceusage.ListServicesRequest, dict])
def test_list_services(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = serviceusage.ListServicesResponse(next_page_token='next_page_token_value')
        response = client.list_services(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.ListServicesRequest()
    assert isinstance(response, pagers.ListServicesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_services_empty_call():
    if False:
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        client.list_services()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.ListServicesRequest()

@pytest.mark.asyncio
async def test_list_services_async(transport: str='grpc_asyncio', request_type=serviceusage.ListServicesRequest):
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(serviceusage.ListServicesResponse(next_page_token='next_page_token_value'))
        response = await client.list_services(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.ListServicesRequest()
    assert isinstance(response, pagers.ListServicesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_services_async_from_dict():
    await test_list_services_async(request_type=dict)

def test_list_services_field_headers():
    if False:
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials())
    request = serviceusage.ListServicesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = serviceusage.ListServicesResponse()
        client.list_services(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_services_field_headers_async():
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = serviceusage.ListServicesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(serviceusage.ListServicesResponse())
        await client.list_services(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_services_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.side_effect = (serviceusage.ListServicesResponse(services=[resources.Service(), resources.Service(), resources.Service()], next_page_token='abc'), serviceusage.ListServicesResponse(services=[], next_page_token='def'), serviceusage.ListServicesResponse(services=[resources.Service()], next_page_token='ghi'), serviceusage.ListServicesResponse(services=[resources.Service(), resources.Service()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_services(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Service) for i in results))

def test_list_services_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.side_effect = (serviceusage.ListServicesResponse(services=[resources.Service(), resources.Service(), resources.Service()], next_page_token='abc'), serviceusage.ListServicesResponse(services=[], next_page_token='def'), serviceusage.ListServicesResponse(services=[resources.Service()], next_page_token='ghi'), serviceusage.ListServicesResponse(services=[resources.Service(), resources.Service()]), RuntimeError)
        pages = list(client.list_services(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_services_async_pager():
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_services), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (serviceusage.ListServicesResponse(services=[resources.Service(), resources.Service(), resources.Service()], next_page_token='abc'), serviceusage.ListServicesResponse(services=[], next_page_token='def'), serviceusage.ListServicesResponse(services=[resources.Service()], next_page_token='ghi'), serviceusage.ListServicesResponse(services=[resources.Service(), resources.Service()]), RuntimeError)
        async_pager = await client.list_services(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Service) for i in responses))

@pytest.mark.asyncio
async def test_list_services_async_pages():
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_services), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (serviceusage.ListServicesResponse(services=[resources.Service(), resources.Service(), resources.Service()], next_page_token='abc'), serviceusage.ListServicesResponse(services=[], next_page_token='def'), serviceusage.ListServicesResponse(services=[resources.Service()], next_page_token='ghi'), serviceusage.ListServicesResponse(services=[resources.Service(), resources.Service()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_services(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [serviceusage.BatchEnableServicesRequest, dict])
def test_batch_enable_services(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_enable_services), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_enable_services(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.BatchEnableServicesRequest()
    assert isinstance(response, future.Future)

def test_batch_enable_services_empty_call():
    if False:
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_enable_services), '__call__') as call:
        client.batch_enable_services()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.BatchEnableServicesRequest()

@pytest.mark.asyncio
async def test_batch_enable_services_async(transport: str='grpc_asyncio', request_type=serviceusage.BatchEnableServicesRequest):
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_enable_services), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_enable_services(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.BatchEnableServicesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_enable_services_async_from_dict():
    await test_batch_enable_services_async(request_type=dict)

def test_batch_enable_services_field_headers():
    if False:
        return 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials())
    request = serviceusage.BatchEnableServicesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_enable_services), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_enable_services(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_enable_services_field_headers_async():
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = serviceusage.BatchEnableServicesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_enable_services), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_enable_services(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [serviceusage.BatchGetServicesRequest, dict])
def test_batch_get_services(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_get_services), '__call__') as call:
        call.return_value = serviceusage.BatchGetServicesResponse()
        response = client.batch_get_services(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.BatchGetServicesRequest()
    assert isinstance(response, serviceusage.BatchGetServicesResponse)

def test_batch_get_services_empty_call():
    if False:
        return 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_get_services), '__call__') as call:
        client.batch_get_services()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.BatchGetServicesRequest()

@pytest.mark.asyncio
async def test_batch_get_services_async(transport: str='grpc_asyncio', request_type=serviceusage.BatchGetServicesRequest):
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_get_services), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(serviceusage.BatchGetServicesResponse())
        response = await client.batch_get_services(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == serviceusage.BatchGetServicesRequest()
    assert isinstance(response, serviceusage.BatchGetServicesResponse)

@pytest.mark.asyncio
async def test_batch_get_services_async_from_dict():
    await test_batch_get_services_async(request_type=dict)

def test_batch_get_services_field_headers():
    if False:
        while True:
            i = 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials())
    request = serviceusage.BatchGetServicesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_get_services), '__call__') as call:
        call.return_value = serviceusage.BatchGetServicesResponse()
        client.batch_get_services(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_get_services_field_headers_async():
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = serviceusage.BatchGetServicesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_get_services), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(serviceusage.BatchGetServicesResponse())
        await client.batch_get_services(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [serviceusage.EnableServiceRequest, dict])
def test_enable_service_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'sample1/sample2/services/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.enable_service(request)
    assert response.operation.name == 'operations/spam'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_enable_service_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ServiceUsageRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceUsageRestInterceptor())
    client = ServiceUsageClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ServiceUsageRestInterceptor, 'post_enable_service') as post, mock.patch.object(transports.ServiceUsageRestInterceptor, 'pre_enable_service') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = serviceusage.EnableServiceRequest.pb(serviceusage.EnableServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = serviceusage.EnableServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.enable_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_enable_service_rest_bad_request(transport: str='rest', request_type=serviceusage.EnableServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'sample1/sample2/services/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.enable_service(request)

def test_enable_service_rest_error():
    if False:
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [serviceusage.DisableServiceRequest, dict])
def test_disable_service_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'sample1/sample2/services/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.disable_service(request)
    assert response.operation.name == 'operations/spam'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_disable_service_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ServiceUsageRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceUsageRestInterceptor())
    client = ServiceUsageClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ServiceUsageRestInterceptor, 'post_disable_service') as post, mock.patch.object(transports.ServiceUsageRestInterceptor, 'pre_disable_service') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = serviceusage.DisableServiceRequest.pb(serviceusage.DisableServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = serviceusage.DisableServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.disable_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_disable_service_rest_bad_request(transport: str='rest', request_type=serviceusage.DisableServiceRequest):
    if False:
        print('Hello World!')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'sample1/sample2/services/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.disable_service(request)

def test_disable_service_rest_error():
    if False:
        print('Hello World!')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [serviceusage.GetServiceRequest, dict])
def test_get_service_rest(request_type):
    if False:
        while True:
            i = 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'sample1/sample2/services/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Service(name='name_value', parent='parent_value', state=resources.State.DISABLED)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Service.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_service(request)
    assert isinstance(response, resources.Service)
    assert response.name == 'name_value'
    assert response.parent == 'parent_value'
    assert response.state == resources.State.DISABLED

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_service_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ServiceUsageRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceUsageRestInterceptor())
    client = ServiceUsageClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ServiceUsageRestInterceptor, 'post_get_service') as post, mock.patch.object(transports.ServiceUsageRestInterceptor, 'pre_get_service') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = serviceusage.GetServiceRequest.pb(serviceusage.GetServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Service.to_json(resources.Service())
        request = serviceusage.GetServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Service()
        client.get_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_service_rest_bad_request(transport: str='rest', request_type=serviceusage.GetServiceRequest):
    if False:
        while True:
            i = 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'sample1/sample2/services/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_service(request)

def test_get_service_rest_error():
    if False:
        print('Hello World!')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [serviceusage.ListServicesRequest, dict])
def test_list_services_rest(request_type):
    if False:
        while True:
            i = 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = serviceusage.ListServicesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = serviceusage.ListServicesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_services(request)
    assert isinstance(response, pagers.ListServicesPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_services_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ServiceUsageRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceUsageRestInterceptor())
    client = ServiceUsageClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ServiceUsageRestInterceptor, 'post_list_services') as post, mock.patch.object(transports.ServiceUsageRestInterceptor, 'pre_list_services') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = serviceusage.ListServicesRequest.pb(serviceusage.ListServicesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = serviceusage.ListServicesResponse.to_json(serviceusage.ListServicesResponse())
        request = serviceusage.ListServicesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = serviceusage.ListServicesResponse()
        client.list_services(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_services_rest_bad_request(transport: str='rest', request_type=serviceusage.ListServicesRequest):
    if False:
        while True:
            i = 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_services(request)

def test_list_services_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (serviceusage.ListServicesResponse(services=[resources.Service(), resources.Service(), resources.Service()], next_page_token='abc'), serviceusage.ListServicesResponse(services=[], next_page_token='def'), serviceusage.ListServicesResponse(services=[resources.Service()], next_page_token='ghi'), serviceusage.ListServicesResponse(services=[resources.Service(), resources.Service()]))
        response = response + response
        response = tuple((serviceusage.ListServicesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'sample1/sample2'}
        pager = client.list_services(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Service) for i in results))
        pages = list(client.list_services(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [serviceusage.BatchEnableServicesRequest, dict])
def test_batch_enable_services_rest(request_type):
    if False:
        return 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_enable_services(request)
    assert response.operation.name == 'operations/spam'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_enable_services_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ServiceUsageRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceUsageRestInterceptor())
    client = ServiceUsageClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ServiceUsageRestInterceptor, 'post_batch_enable_services') as post, mock.patch.object(transports.ServiceUsageRestInterceptor, 'pre_batch_enable_services') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = serviceusage.BatchEnableServicesRequest.pb(serviceusage.BatchEnableServicesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = serviceusage.BatchEnableServicesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_enable_services(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_enable_services_rest_bad_request(transport: str='rest', request_type=serviceusage.BatchEnableServicesRequest):
    if False:
        return 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_enable_services(request)

def test_batch_enable_services_rest_error():
    if False:
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [serviceusage.BatchGetServicesRequest, dict])
def test_batch_get_services_rest(request_type):
    if False:
        print('Hello World!')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = serviceusage.BatchGetServicesResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = serviceusage.BatchGetServicesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_get_services(request)
    assert isinstance(response, serviceusage.BatchGetServicesResponse)

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_get_services_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ServiceUsageRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceUsageRestInterceptor())
    client = ServiceUsageClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ServiceUsageRestInterceptor, 'post_batch_get_services') as post, mock.patch.object(transports.ServiceUsageRestInterceptor, 'pre_batch_get_services') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = serviceusage.BatchGetServicesRequest.pb(serviceusage.BatchGetServicesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = serviceusage.BatchGetServicesResponse.to_json(serviceusage.BatchGetServicesResponse())
        request = serviceusage.BatchGetServicesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = serviceusage.BatchGetServicesResponse()
        client.batch_get_services(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_get_services_rest_bad_request(transport: str='rest', request_type=serviceusage.BatchGetServicesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'sample1/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_get_services(request)

def test_batch_get_services_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.ServiceUsageGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ServiceUsageGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ServiceUsageClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ServiceUsageGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ServiceUsageClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ServiceUsageClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ServiceUsageGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ServiceUsageClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.ServiceUsageGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ServiceUsageClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ServiceUsageGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ServiceUsageGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ServiceUsageGrpcTransport, transports.ServiceUsageGrpcAsyncIOTransport, transports.ServiceUsageRestTransport])
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
        print('Hello World!')
    transport = ServiceUsageClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ServiceUsageGrpcTransport)

def test_service_usage_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ServiceUsageTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_service_usage_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.service_usage_v1.services.service_usage.transports.ServiceUsageTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ServiceUsageTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('enable_service', 'disable_service', 'get_service', 'list_services', 'batch_enable_services', 'batch_get_services', 'get_operation', 'list_operations')
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

def test_service_usage_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.service_usage_v1.services.service_usage.transports.ServiceUsageTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ServiceUsageTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/service.management'), quota_project_id='octopus')

def test_service_usage_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.service_usage_v1.services.service_usage.transports.ServiceUsageTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ServiceUsageTransport()
        adc.assert_called_once()

def test_service_usage_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ServiceUsageClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/service.management'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ServiceUsageGrpcTransport, transports.ServiceUsageGrpcAsyncIOTransport])
def test_service_usage_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/service.management'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ServiceUsageGrpcTransport, transports.ServiceUsageGrpcAsyncIOTransport, transports.ServiceUsageRestTransport])
def test_service_usage_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ServiceUsageGrpcTransport, grpc_helpers), (transports.ServiceUsageGrpcAsyncIOTransport, grpc_helpers_async)])
def test_service_usage_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('serviceusage.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/service.management'), scopes=['1', '2'], default_host='serviceusage.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ServiceUsageGrpcTransport, transports.ServiceUsageGrpcAsyncIOTransport])
def test_service_usage_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        i = 10
        return i + 15
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

def test_service_usage_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.ServiceUsageRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_service_usage_rest_lro_client():
    if False:
        while True:
            i = 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_service_usage_host_no_port(transport_name):
    if False:
        return 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='serviceusage.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('serviceusage.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://serviceusage.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_service_usage_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='serviceusage.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('serviceusage.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://serviceusage.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_service_usage_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = ServiceUsageClient(credentials=creds1, transport=transport_name)
    client2 = ServiceUsageClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.enable_service._session
    session2 = client2.transport.enable_service._session
    assert session1 != session2
    session1 = client1.transport.disable_service._session
    session2 = client2.transport.disable_service._session
    assert session1 != session2
    session1 = client1.transport.get_service._session
    session2 = client2.transport.get_service._session
    assert session1 != session2
    session1 = client1.transport.list_services._session
    session2 = client2.transport.list_services._session
    assert session1 != session2
    session1 = client1.transport.batch_enable_services._session
    session2 = client2.transport.batch_enable_services._session
    assert session1 != session2
    session1 = client1.transport.batch_get_services._session
    session2 = client2.transport.batch_get_services._session
    assert session1 != session2

def test_service_usage_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ServiceUsageGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_service_usage_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ServiceUsageGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ServiceUsageGrpcTransport, transports.ServiceUsageGrpcAsyncIOTransport])
def test_service_usage_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.ServiceUsageGrpcTransport, transports.ServiceUsageGrpcAsyncIOTransport])
def test_service_usage_transport_channel_mtls_with_adc(transport_class):
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

def test_service_usage_grpc_lro_client():
    if False:
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_service_usage_grpc_lro_async_client():
    if False:
        while True:
            i = 10
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_service_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    service = 'clam'
    expected = 'projects/{project}/services/{service}'.format(project=project, service=service)
    actual = ServiceUsageClient.service_path(project, service)
    assert expected == actual

def test_parse_service_path():
    if False:
        return 10
    expected = {'project': 'whelk', 'service': 'octopus'}
    path = ServiceUsageClient.service_path(**expected)
    actual = ServiceUsageClient.parse_service_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ServiceUsageClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'nudibranch'}
    path = ServiceUsageClient.common_billing_account_path(**expected)
    actual = ServiceUsageClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ServiceUsageClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'mussel'}
    path = ServiceUsageClient.common_folder_path(**expected)
    actual = ServiceUsageClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ServiceUsageClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'nautilus'}
    path = ServiceUsageClient.common_organization_path(**expected)
    actual = ServiceUsageClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = ServiceUsageClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'abalone'}
    path = ServiceUsageClient.common_project_path(**expected)
    actual = ServiceUsageClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ServiceUsageClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = ServiceUsageClient.common_location_path(**expected)
    actual = ServiceUsageClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ServiceUsageTransport, '_prep_wrapped_messages') as prep:
        client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ServiceUsageTransport, '_prep_wrapped_messages') as prep:
        transport_class = ServiceUsageClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        print('Hello World!')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'operations/sample1'}, request)
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
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'operations/sample1'}
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
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_operations(request)

@pytest.mark.parametrize('request_type', [operations_pb2.ListOperationsRequest, dict])
def test_list_operations_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
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

def test_get_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = ServiceUsageAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = ServiceUsageClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ServiceUsageClient, transports.ServiceUsageGrpcTransport), (ServiceUsageAsyncClient, transports.ServiceUsageGrpcAsyncIOTransport)])
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