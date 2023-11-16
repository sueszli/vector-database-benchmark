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
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import json_format
from google.protobuf import struct_pb2
from google.protobuf import timestamp_pb2
from google.rpc import status_pb2
from google.rpc.context import attribute_context_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.servicecontrol_v2.services.service_controller import ServiceControllerAsyncClient, ServiceControllerClient, transports
from google.cloud.servicecontrol_v2.types import service_controller

def client_cert_source_callback():
    if False:
        for i in range(10):
            print('nop')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        print('Hello World!')
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
    assert ServiceControllerClient._get_default_mtls_endpoint(None) is None
    assert ServiceControllerClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ServiceControllerClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ServiceControllerClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ServiceControllerClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ServiceControllerClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ServiceControllerClient, 'grpc'), (ServiceControllerAsyncClient, 'grpc_asyncio'), (ServiceControllerClient, 'rest')])
def test_service_controller_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('servicecontrol.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://servicecontrol.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ServiceControllerGrpcTransport, 'grpc'), (transports.ServiceControllerGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.ServiceControllerRestTransport, 'rest')])
def test_service_controller_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ServiceControllerClient, 'grpc'), (ServiceControllerAsyncClient, 'grpc_asyncio'), (ServiceControllerClient, 'rest')])
def test_service_controller_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('servicecontrol.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://servicecontrol.googleapis.com')

def test_service_controller_client_get_transport_class():
    if False:
        return 10
    transport = ServiceControllerClient.get_transport_class()
    available_transports = [transports.ServiceControllerGrpcTransport, transports.ServiceControllerRestTransport]
    assert transport in available_transports
    transport = ServiceControllerClient.get_transport_class('grpc')
    assert transport == transports.ServiceControllerGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ServiceControllerClient, transports.ServiceControllerGrpcTransport, 'grpc'), (ServiceControllerAsyncClient, transports.ServiceControllerGrpcAsyncIOTransport, 'grpc_asyncio'), (ServiceControllerClient, transports.ServiceControllerRestTransport, 'rest')])
@mock.patch.object(ServiceControllerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceControllerClient))
@mock.patch.object(ServiceControllerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceControllerAsyncClient))
def test_service_controller_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(ServiceControllerClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ServiceControllerClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ServiceControllerClient, transports.ServiceControllerGrpcTransport, 'grpc', 'true'), (ServiceControllerAsyncClient, transports.ServiceControllerGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ServiceControllerClient, transports.ServiceControllerGrpcTransport, 'grpc', 'false'), (ServiceControllerAsyncClient, transports.ServiceControllerGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (ServiceControllerClient, transports.ServiceControllerRestTransport, 'rest', 'true'), (ServiceControllerClient, transports.ServiceControllerRestTransport, 'rest', 'false')])
@mock.patch.object(ServiceControllerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceControllerClient))
@mock.patch.object(ServiceControllerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceControllerAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_service_controller_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [ServiceControllerClient, ServiceControllerAsyncClient])
@mock.patch.object(ServiceControllerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceControllerClient))
@mock.patch.object(ServiceControllerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceControllerAsyncClient))
def test_service_controller_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ServiceControllerClient, transports.ServiceControllerGrpcTransport, 'grpc'), (ServiceControllerAsyncClient, transports.ServiceControllerGrpcAsyncIOTransport, 'grpc_asyncio'), (ServiceControllerClient, transports.ServiceControllerRestTransport, 'rest')])
def test_service_controller_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ServiceControllerClient, transports.ServiceControllerGrpcTransport, 'grpc', grpc_helpers), (ServiceControllerAsyncClient, transports.ServiceControllerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (ServiceControllerClient, transports.ServiceControllerRestTransport, 'rest', None)])
def test_service_controller_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_service_controller_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.servicecontrol_v2.services.service_controller.transports.ServiceControllerGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ServiceControllerClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ServiceControllerClient, transports.ServiceControllerGrpcTransport, 'grpc', grpc_helpers), (ServiceControllerAsyncClient, transports.ServiceControllerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_service_controller_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('servicecontrol.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/servicecontrol'), scopes=None, default_host='servicecontrol.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [service_controller.CheckRequest, dict])
def test_check(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.check), '__call__') as call:
        call.return_value = service_controller.CheckResponse()
        response = client.check(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_controller.CheckRequest()
    assert isinstance(response, service_controller.CheckResponse)

def test_check_empty_call():
    if False:
        while True:
            i = 10
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.check), '__call__') as call:
        client.check()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_controller.CheckRequest()

@pytest.mark.asyncio
async def test_check_async(transport: str='grpc_asyncio', request_type=service_controller.CheckRequest):
    client = ServiceControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.check), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_controller.CheckResponse())
        response = await client.check(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_controller.CheckRequest()
    assert isinstance(response, service_controller.CheckResponse)

@pytest.mark.asyncio
async def test_check_async_from_dict():
    await test_check_async(request_type=dict)

def test_check_field_headers():
    if False:
        i = 10
        return i + 15
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_controller.CheckRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.check), '__call__') as call:
        call.return_value = service_controller.CheckResponse()
        client.check(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_check_field_headers_async():
    client = ServiceControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_controller.CheckRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.check), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_controller.CheckResponse())
        await client.check(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service_controller.ReportRequest, dict])
def test_report(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.report), '__call__') as call:
        call.return_value = service_controller.ReportResponse()
        response = client.report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_controller.ReportRequest()
    assert isinstance(response, service_controller.ReportResponse)

def test_report_empty_call():
    if False:
        print('Hello World!')
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.report), '__call__') as call:
        client.report()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_controller.ReportRequest()

@pytest.mark.asyncio
async def test_report_async(transport: str='grpc_asyncio', request_type=service_controller.ReportRequest):
    client = ServiceControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_controller.ReportResponse())
        response = await client.report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service_controller.ReportRequest()
    assert isinstance(response, service_controller.ReportResponse)

@pytest.mark.asyncio
async def test_report_async_from_dict():
    await test_report_async(request_type=dict)

def test_report_field_headers():
    if False:
        print('Hello World!')
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_controller.ReportRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.report), '__call__') as call:
        call.return_value = service_controller.ReportResponse()
        client.report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_report_field_headers_async():
    client = ServiceControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service_controller.ReportRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_controller.ReportResponse())
        await client.report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service_controller.CheckRequest, dict])
def test_check_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service_controller.CheckResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = service_controller.CheckResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.check(request)
    assert isinstance(response, service_controller.CheckResponse)

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_check_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ServiceControllerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceControllerRestInterceptor())
    client = ServiceControllerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ServiceControllerRestInterceptor, 'post_check') as post, mock.patch.object(transports.ServiceControllerRestInterceptor, 'pre_check') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service_controller.CheckRequest.pb(service_controller.CheckRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service_controller.CheckResponse.to_json(service_controller.CheckResponse())
        request = service_controller.CheckRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service_controller.CheckResponse()
        client.check(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_check_rest_bad_request(transport: str='rest', request_type=service_controller.CheckRequest):
    if False:
        print('Hello World!')
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.check(request)

def test_check_rest_error():
    if False:
        while True:
            i = 10
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service_controller.ReportRequest, dict])
def test_report_rest(request_type):
    if False:
        while True:
            i = 10
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service_controller.ReportResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = service_controller.ReportResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.report(request)
    assert isinstance(response, service_controller.ReportResponse)

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_report_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ServiceControllerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceControllerRestInterceptor())
    client = ServiceControllerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ServiceControllerRestInterceptor, 'post_report') as post, mock.patch.object(transports.ServiceControllerRestInterceptor, 'pre_report') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service_controller.ReportRequest.pb(service_controller.ReportRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service_controller.ReportResponse.to_json(service_controller.ReportResponse())
        request = service_controller.ReportRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service_controller.ReportResponse()
        client.report(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_report_rest_bad_request(transport: str='rest', request_type=service_controller.ReportRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.report(request)

def test_report_rest_error():
    if False:
        print('Hello World!')
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.ServiceControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ServiceControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ServiceControllerClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ServiceControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ServiceControllerClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ServiceControllerClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ServiceControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ServiceControllerClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.ServiceControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ServiceControllerClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.ServiceControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ServiceControllerGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ServiceControllerGrpcTransport, transports.ServiceControllerGrpcAsyncIOTransport, transports.ServiceControllerRestTransport])
def test_transport_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        return 10
    transport = ServiceControllerClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ServiceControllerGrpcTransport)

def test_service_controller_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ServiceControllerTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_service_controller_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.servicecontrol_v2.services.service_controller.transports.ServiceControllerTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ServiceControllerTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('check', 'report')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_service_controller_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.servicecontrol_v2.services.service_controller.transports.ServiceControllerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ServiceControllerTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/servicecontrol'), quota_project_id='octopus')

def test_service_controller_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.servicecontrol_v2.services.service_controller.transports.ServiceControllerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ServiceControllerTransport()
        adc.assert_called_once()

def test_service_controller_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ServiceControllerClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/servicecontrol'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ServiceControllerGrpcTransport, transports.ServiceControllerGrpcAsyncIOTransport])
def test_service_controller_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/servicecontrol'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ServiceControllerGrpcTransport, transports.ServiceControllerGrpcAsyncIOTransport, transports.ServiceControllerRestTransport])
def test_service_controller_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ServiceControllerGrpcTransport, grpc_helpers), (transports.ServiceControllerGrpcAsyncIOTransport, grpc_helpers_async)])
def test_service_controller_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('servicecontrol.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/servicecontrol'), scopes=['1', '2'], default_host='servicecontrol.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ServiceControllerGrpcTransport, transports.ServiceControllerGrpcAsyncIOTransport])
def test_service_controller_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_service_controller_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.ServiceControllerRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_service_controller_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='servicecontrol.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('servicecontrol.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://servicecontrol.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_service_controller_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='servicecontrol.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('servicecontrol.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://servicecontrol.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_service_controller_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = ServiceControllerClient(credentials=creds1, transport=transport_name)
    client2 = ServiceControllerClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.check._session
    session2 = client2.transport.check._session
    assert session1 != session2
    session1 = client1.transport.report._session
    session2 = client2.transport.report._session
    assert session1 != session2

def test_service_controller_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ServiceControllerGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_service_controller_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ServiceControllerGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ServiceControllerGrpcTransport, transports.ServiceControllerGrpcAsyncIOTransport])
def test_service_controller_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.ServiceControllerGrpcTransport, transports.ServiceControllerGrpcAsyncIOTransport])
def test_service_controller_transport_channel_mtls_with_adc(transport_class):
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

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ServiceControllerClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'clam'}
    path = ServiceControllerClient.common_billing_account_path(**expected)
    actual = ServiceControllerClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ServiceControllerClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'octopus'}
    path = ServiceControllerClient.common_folder_path(**expected)
    actual = ServiceControllerClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ServiceControllerClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'nudibranch'}
    path = ServiceControllerClient.common_organization_path(**expected)
    actual = ServiceControllerClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = ServiceControllerClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'mussel'}
    path = ServiceControllerClient.common_project_path(**expected)
    actual = ServiceControllerClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ServiceControllerClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = ServiceControllerClient.common_location_path(**expected)
    actual = ServiceControllerClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ServiceControllerTransport, '_prep_wrapped_messages') as prep:
        client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ServiceControllerTransport, '_prep_wrapped_messages') as prep:
        transport_class = ServiceControllerClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ServiceControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        while True:
            i = 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = ServiceControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ServiceControllerClient, transports.ServiceControllerGrpcTransport), (ServiceControllerAsyncClient, transports.ServiceControllerGrpcAsyncIOTransport)])
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