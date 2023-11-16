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
from google.cloud.certificate_manager_v1.services.certificate_manager import CertificateManagerAsyncClient, CertificateManagerClient, pagers, transports
from google.cloud.certificate_manager_v1.types import certificate_issuance_config
from google.cloud.certificate_manager_v1.types import certificate_issuance_config as gcc_certificate_issuance_config
from google.cloud.certificate_manager_v1.types import certificate_manager

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        print('Hello World!')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert CertificateManagerClient._get_default_mtls_endpoint(None) is None
    assert CertificateManagerClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CertificateManagerClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CertificateManagerClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CertificateManagerClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CertificateManagerClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CertificateManagerClient, 'grpc'), (CertificateManagerAsyncClient, 'grpc_asyncio'), (CertificateManagerClient, 'rest')])
def test_certificate_manager_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('certificatemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://certificatemanager.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CertificateManagerGrpcTransport, 'grpc'), (transports.CertificateManagerGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.CertificateManagerRestTransport, 'rest')])
def test_certificate_manager_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(CertificateManagerClient, 'grpc'), (CertificateManagerAsyncClient, 'grpc_asyncio'), (CertificateManagerClient, 'rest')])
def test_certificate_manager_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('certificatemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://certificatemanager.googleapis.com')

def test_certificate_manager_client_get_transport_class():
    if False:
        return 10
    transport = CertificateManagerClient.get_transport_class()
    available_transports = [transports.CertificateManagerGrpcTransport, transports.CertificateManagerRestTransport]
    assert transport in available_transports
    transport = CertificateManagerClient.get_transport_class('grpc')
    assert transport == transports.CertificateManagerGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CertificateManagerClient, transports.CertificateManagerGrpcTransport, 'grpc'), (CertificateManagerAsyncClient, transports.CertificateManagerGrpcAsyncIOTransport, 'grpc_asyncio'), (CertificateManagerClient, transports.CertificateManagerRestTransport, 'rest')])
@mock.patch.object(CertificateManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CertificateManagerClient))
@mock.patch.object(CertificateManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CertificateManagerAsyncClient))
def test_certificate_manager_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(CertificateManagerClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CertificateManagerClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CertificateManagerClient, transports.CertificateManagerGrpcTransport, 'grpc', 'true'), (CertificateManagerAsyncClient, transports.CertificateManagerGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CertificateManagerClient, transports.CertificateManagerGrpcTransport, 'grpc', 'false'), (CertificateManagerAsyncClient, transports.CertificateManagerGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (CertificateManagerClient, transports.CertificateManagerRestTransport, 'rest', 'true'), (CertificateManagerClient, transports.CertificateManagerRestTransport, 'rest', 'false')])
@mock.patch.object(CertificateManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CertificateManagerClient))
@mock.patch.object(CertificateManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CertificateManagerAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_certificate_manager_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [CertificateManagerClient, CertificateManagerAsyncClient])
@mock.patch.object(CertificateManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CertificateManagerClient))
@mock.patch.object(CertificateManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CertificateManagerAsyncClient))
def test_certificate_manager_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CertificateManagerClient, transports.CertificateManagerGrpcTransport, 'grpc'), (CertificateManagerAsyncClient, transports.CertificateManagerGrpcAsyncIOTransport, 'grpc_asyncio'), (CertificateManagerClient, transports.CertificateManagerRestTransport, 'rest')])
def test_certificate_manager_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CertificateManagerClient, transports.CertificateManagerGrpcTransport, 'grpc', grpc_helpers), (CertificateManagerAsyncClient, transports.CertificateManagerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (CertificateManagerClient, transports.CertificateManagerRestTransport, 'rest', None)])
def test_certificate_manager_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_certificate_manager_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.certificate_manager_v1.services.certificate_manager.transports.CertificateManagerGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CertificateManagerClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CertificateManagerClient, transports.CertificateManagerGrpcTransport, 'grpc', grpc_helpers), (CertificateManagerAsyncClient, transports.CertificateManagerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_certificate_manager_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('certificatemanager.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='certificatemanager.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [certificate_manager.ListCertificatesRequest, dict])
def test_list_certificates(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.return_value = certificate_manager.ListCertificatesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_certificates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.ListCertificatesRequest()
    assert isinstance(response, pagers.ListCertificatesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_certificates_empty_call():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        client.list_certificates()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.ListCertificatesRequest()

@pytest.mark.asyncio
async def test_list_certificates_async(transport: str='grpc_asyncio', request_type=certificate_manager.ListCertificatesRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.ListCertificatesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_certificates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.ListCertificatesRequest()
    assert isinstance(response, pagers.ListCertificatesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_certificates_async_from_dict():
    await test_list_certificates_async(request_type=dict)

def test_list_certificates_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.ListCertificatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.return_value = certificate_manager.ListCertificatesResponse()
        client.list_certificates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_certificates_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.ListCertificatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.ListCertificatesResponse())
        await client.list_certificates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_certificates_flattened():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.return_value = certificate_manager.ListCertificatesResponse()
        client.list_certificates(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_certificates_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_certificates(certificate_manager.ListCertificatesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_certificates_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.return_value = certificate_manager.ListCertificatesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.ListCertificatesResponse())
        response = await client.list_certificates(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_certificates_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_certificates(certificate_manager.ListCertificatesRequest(), parent='parent_value')

def test_list_certificates_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.side_effect = (certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate(), certificate_manager.Certificate(), certificate_manager.Certificate()], next_page_token='abc'), certificate_manager.ListCertificatesResponse(certificates=[], next_page_token='def'), certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate()], next_page_token='ghi'), certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate(), certificate_manager.Certificate()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_certificates(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, certificate_manager.Certificate) for i in results))

def test_list_certificates_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.side_effect = (certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate(), certificate_manager.Certificate(), certificate_manager.Certificate()], next_page_token='abc'), certificate_manager.ListCertificatesResponse(certificates=[], next_page_token='def'), certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate()], next_page_token='ghi'), certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate(), certificate_manager.Certificate()]), RuntimeError)
        pages = list(client.list_certificates(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_certificates_async_pager():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_certificates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate(), certificate_manager.Certificate(), certificate_manager.Certificate()], next_page_token='abc'), certificate_manager.ListCertificatesResponse(certificates=[], next_page_token='def'), certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate()], next_page_token='ghi'), certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate(), certificate_manager.Certificate()]), RuntimeError)
        async_pager = await client.list_certificates(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, certificate_manager.Certificate) for i in responses))

@pytest.mark.asyncio
async def test_list_certificates_async_pages():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_certificates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate(), certificate_manager.Certificate(), certificate_manager.Certificate()], next_page_token='abc'), certificate_manager.ListCertificatesResponse(certificates=[], next_page_token='def'), certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate()], next_page_token='ghi'), certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate(), certificate_manager.Certificate()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_certificates(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [certificate_manager.GetCertificateRequest, dict])
def test_get_certificate(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_certificate), '__call__') as call:
        call.return_value = certificate_manager.Certificate(name='name_value', description='description_value', san_dnsnames=['san_dnsnames_value'], pem_certificate='pem_certificate_value', scope=certificate_manager.Certificate.Scope.EDGE_CACHE)
        response = client.get_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.GetCertificateRequest()
    assert isinstance(response, certificate_manager.Certificate)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.san_dnsnames == ['san_dnsnames_value']
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.scope == certificate_manager.Certificate.Scope.EDGE_CACHE

def test_get_certificate_empty_call():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_certificate), '__call__') as call:
        client.get_certificate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.GetCertificateRequest()

@pytest.mark.asyncio
async def test_get_certificate_async(transport: str='grpc_asyncio', request_type=certificate_manager.GetCertificateRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.Certificate(name='name_value', description='description_value', san_dnsnames=['san_dnsnames_value'], pem_certificate='pem_certificate_value', scope=certificate_manager.Certificate.Scope.EDGE_CACHE))
        response = await client.get_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.GetCertificateRequest()
    assert isinstance(response, certificate_manager.Certificate)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.san_dnsnames == ['san_dnsnames_value']
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.scope == certificate_manager.Certificate.Scope.EDGE_CACHE

@pytest.mark.asyncio
async def test_get_certificate_async_from_dict():
    await test_get_certificate_async(request_type=dict)

def test_get_certificate_field_headers():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.GetCertificateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_certificate), '__call__') as call:
        call.return_value = certificate_manager.Certificate()
        client.get_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_certificate_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.GetCertificateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.Certificate())
        await client.get_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_certificate_flattened():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_certificate), '__call__') as call:
        call.return_value = certificate_manager.Certificate()
        client.get_certificate(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_certificate_flattened_error():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_certificate(certificate_manager.GetCertificateRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_certificate_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_certificate), '__call__') as call:
        call.return_value = certificate_manager.Certificate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.Certificate())
        response = await client.get_certificate(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_certificate_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_certificate(certificate_manager.GetCertificateRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [certificate_manager.CreateCertificateRequest, dict])
def test_create_certificate(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_certificate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.CreateCertificateRequest()
    assert isinstance(response, future.Future)

def test_create_certificate_empty_call():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_certificate), '__call__') as call:
        client.create_certificate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.CreateCertificateRequest()

@pytest.mark.asyncio
async def test_create_certificate_async(transport: str='grpc_asyncio', request_type=certificate_manager.CreateCertificateRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.CreateCertificateRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_certificate_async_from_dict():
    await test_create_certificate_async(request_type=dict)

def test_create_certificate_field_headers():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.CreateCertificateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_certificate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_certificate_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.CreateCertificateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_certificate_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_certificate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_certificate(parent='parent_value', certificate=certificate_manager.Certificate(name='name_value'), certificate_id='certificate_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].certificate
        mock_val = certificate_manager.Certificate(name='name_value')
        assert arg == mock_val
        arg = args[0].certificate_id
        mock_val = 'certificate_id_value'
        assert arg == mock_val

def test_create_certificate_flattened_error():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_certificate(certificate_manager.CreateCertificateRequest(), parent='parent_value', certificate=certificate_manager.Certificate(name='name_value'), certificate_id='certificate_id_value')

@pytest.mark.asyncio
async def test_create_certificate_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_certificate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_certificate(parent='parent_value', certificate=certificate_manager.Certificate(name='name_value'), certificate_id='certificate_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].certificate
        mock_val = certificate_manager.Certificate(name='name_value')
        assert arg == mock_val
        arg = args[0].certificate_id
        mock_val = 'certificate_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_certificate_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_certificate(certificate_manager.CreateCertificateRequest(), parent='parent_value', certificate=certificate_manager.Certificate(name='name_value'), certificate_id='certificate_id_value')

@pytest.mark.parametrize('request_type', [certificate_manager.UpdateCertificateRequest, dict])
def test_update_certificate(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_certificate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.UpdateCertificateRequest()
    assert isinstance(response, future.Future)

def test_update_certificate_empty_call():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_certificate), '__call__') as call:
        client.update_certificate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.UpdateCertificateRequest()

@pytest.mark.asyncio
async def test_update_certificate_async(transport: str='grpc_asyncio', request_type=certificate_manager.UpdateCertificateRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.UpdateCertificateRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_certificate_async_from_dict():
    await test_update_certificate_async(request_type=dict)

def test_update_certificate_field_headers():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.UpdateCertificateRequest()
    request.certificate.name = 'name_value'
    with mock.patch.object(type(client.transport.update_certificate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'certificate.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_certificate_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.UpdateCertificateRequest()
    request.certificate.name = 'name_value'
    with mock.patch.object(type(client.transport.update_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'certificate.name=name_value') in kw['metadata']

def test_update_certificate_flattened():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_certificate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_certificate(certificate=certificate_manager.Certificate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].certificate
        mock_val = certificate_manager.Certificate(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_certificate_flattened_error():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_certificate(certificate_manager.UpdateCertificateRequest(), certificate=certificate_manager.Certificate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_certificate_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_certificate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_certificate(certificate=certificate_manager.Certificate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].certificate
        mock_val = certificate_manager.Certificate(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_certificate_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_certificate(certificate_manager.UpdateCertificateRequest(), certificate=certificate_manager.Certificate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [certificate_manager.DeleteCertificateRequest, dict])
def test_delete_certificate(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_certificate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.DeleteCertificateRequest()
    assert isinstance(response, future.Future)

def test_delete_certificate_empty_call():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_certificate), '__call__') as call:
        client.delete_certificate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.DeleteCertificateRequest()

@pytest.mark.asyncio
async def test_delete_certificate_async(transport: str='grpc_asyncio', request_type=certificate_manager.DeleteCertificateRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.DeleteCertificateRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_certificate_async_from_dict():
    await test_delete_certificate_async(request_type=dict)

def test_delete_certificate_field_headers():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.DeleteCertificateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_certificate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_certificate_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.DeleteCertificateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_certificate_flattened():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_certificate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_certificate(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_certificate_flattened_error():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_certificate(certificate_manager.DeleteCertificateRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_certificate_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_certificate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_certificate(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_certificate_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_certificate(certificate_manager.DeleteCertificateRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [certificate_manager.ListCertificateMapsRequest, dict])
def test_list_certificate_maps(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_certificate_maps), '__call__') as call:
        call.return_value = certificate_manager.ListCertificateMapsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_certificate_maps(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.ListCertificateMapsRequest()
    assert isinstance(response, pagers.ListCertificateMapsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_certificate_maps_empty_call():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_certificate_maps), '__call__') as call:
        client.list_certificate_maps()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.ListCertificateMapsRequest()

@pytest.mark.asyncio
async def test_list_certificate_maps_async(transport: str='grpc_asyncio', request_type=certificate_manager.ListCertificateMapsRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_certificate_maps), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.ListCertificateMapsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_certificate_maps(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.ListCertificateMapsRequest()
    assert isinstance(response, pagers.ListCertificateMapsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_certificate_maps_async_from_dict():
    await test_list_certificate_maps_async(request_type=dict)

def test_list_certificate_maps_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.ListCertificateMapsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_certificate_maps), '__call__') as call:
        call.return_value = certificate_manager.ListCertificateMapsResponse()
        client.list_certificate_maps(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_certificate_maps_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.ListCertificateMapsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_certificate_maps), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.ListCertificateMapsResponse())
        await client.list_certificate_maps(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_certificate_maps_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_certificate_maps), '__call__') as call:
        call.return_value = certificate_manager.ListCertificateMapsResponse()
        client.list_certificate_maps(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_certificate_maps_flattened_error():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_certificate_maps(certificate_manager.ListCertificateMapsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_certificate_maps_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_certificate_maps), '__call__') as call:
        call.return_value = certificate_manager.ListCertificateMapsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.ListCertificateMapsResponse())
        response = await client.list_certificate_maps(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_certificate_maps_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_certificate_maps(certificate_manager.ListCertificateMapsRequest(), parent='parent_value')

def test_list_certificate_maps_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_certificate_maps), '__call__') as call:
        call.side_effect = (certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap(), certificate_manager.CertificateMap(), certificate_manager.CertificateMap()], next_page_token='abc'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[], next_page_token='def'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap()], next_page_token='ghi'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap(), certificate_manager.CertificateMap()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_certificate_maps(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, certificate_manager.CertificateMap) for i in results))

def test_list_certificate_maps_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_certificate_maps), '__call__') as call:
        call.side_effect = (certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap(), certificate_manager.CertificateMap(), certificate_manager.CertificateMap()], next_page_token='abc'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[], next_page_token='def'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap()], next_page_token='ghi'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap(), certificate_manager.CertificateMap()]), RuntimeError)
        pages = list(client.list_certificate_maps(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_certificate_maps_async_pager():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_certificate_maps), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap(), certificate_manager.CertificateMap(), certificate_manager.CertificateMap()], next_page_token='abc'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[], next_page_token='def'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap()], next_page_token='ghi'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap(), certificate_manager.CertificateMap()]), RuntimeError)
        async_pager = await client.list_certificate_maps(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, certificate_manager.CertificateMap) for i in responses))

@pytest.mark.asyncio
async def test_list_certificate_maps_async_pages():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_certificate_maps), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap(), certificate_manager.CertificateMap(), certificate_manager.CertificateMap()], next_page_token='abc'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[], next_page_token='def'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap()], next_page_token='ghi'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap(), certificate_manager.CertificateMap()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_certificate_maps(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [certificate_manager.GetCertificateMapRequest, dict])
def test_get_certificate_map(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_certificate_map), '__call__') as call:
        call.return_value = certificate_manager.CertificateMap(name='name_value', description='description_value')
        response = client.get_certificate_map(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.GetCertificateMapRequest()
    assert isinstance(response, certificate_manager.CertificateMap)
    assert response.name == 'name_value'
    assert response.description == 'description_value'

def test_get_certificate_map_empty_call():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_certificate_map), '__call__') as call:
        client.get_certificate_map()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.GetCertificateMapRequest()

@pytest.mark.asyncio
async def test_get_certificate_map_async(transport: str='grpc_asyncio', request_type=certificate_manager.GetCertificateMapRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_certificate_map), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.CertificateMap(name='name_value', description='description_value'))
        response = await client.get_certificate_map(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.GetCertificateMapRequest()
    assert isinstance(response, certificate_manager.CertificateMap)
    assert response.name == 'name_value'
    assert response.description == 'description_value'

@pytest.mark.asyncio
async def test_get_certificate_map_async_from_dict():
    await test_get_certificate_map_async(request_type=dict)

def test_get_certificate_map_field_headers():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.GetCertificateMapRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_certificate_map), '__call__') as call:
        call.return_value = certificate_manager.CertificateMap()
        client.get_certificate_map(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_certificate_map_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.GetCertificateMapRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_certificate_map), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.CertificateMap())
        await client.get_certificate_map(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_certificate_map_flattened():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_certificate_map), '__call__') as call:
        call.return_value = certificate_manager.CertificateMap()
        client.get_certificate_map(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_certificate_map_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_certificate_map(certificate_manager.GetCertificateMapRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_certificate_map_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_certificate_map), '__call__') as call:
        call.return_value = certificate_manager.CertificateMap()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.CertificateMap())
        response = await client.get_certificate_map(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_certificate_map_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_certificate_map(certificate_manager.GetCertificateMapRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [certificate_manager.CreateCertificateMapRequest, dict])
def test_create_certificate_map(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_certificate_map), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_certificate_map(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.CreateCertificateMapRequest()
    assert isinstance(response, future.Future)

def test_create_certificate_map_empty_call():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_certificate_map), '__call__') as call:
        client.create_certificate_map()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.CreateCertificateMapRequest()

@pytest.mark.asyncio
async def test_create_certificate_map_async(transport: str='grpc_asyncio', request_type=certificate_manager.CreateCertificateMapRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_certificate_map), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_certificate_map(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.CreateCertificateMapRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_certificate_map_async_from_dict():
    await test_create_certificate_map_async(request_type=dict)

def test_create_certificate_map_field_headers():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.CreateCertificateMapRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_certificate_map), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_certificate_map(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_certificate_map_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.CreateCertificateMapRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_certificate_map), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_certificate_map(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_certificate_map_flattened():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_certificate_map), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_certificate_map(parent='parent_value', certificate_map=certificate_manager.CertificateMap(name='name_value'), certificate_map_id='certificate_map_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].certificate_map
        mock_val = certificate_manager.CertificateMap(name='name_value')
        assert arg == mock_val
        arg = args[0].certificate_map_id
        mock_val = 'certificate_map_id_value'
        assert arg == mock_val

def test_create_certificate_map_flattened_error():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_certificate_map(certificate_manager.CreateCertificateMapRequest(), parent='parent_value', certificate_map=certificate_manager.CertificateMap(name='name_value'), certificate_map_id='certificate_map_id_value')

@pytest.mark.asyncio
async def test_create_certificate_map_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_certificate_map), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_certificate_map(parent='parent_value', certificate_map=certificate_manager.CertificateMap(name='name_value'), certificate_map_id='certificate_map_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].certificate_map
        mock_val = certificate_manager.CertificateMap(name='name_value')
        assert arg == mock_val
        arg = args[0].certificate_map_id
        mock_val = 'certificate_map_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_certificate_map_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_certificate_map(certificate_manager.CreateCertificateMapRequest(), parent='parent_value', certificate_map=certificate_manager.CertificateMap(name='name_value'), certificate_map_id='certificate_map_id_value')

@pytest.mark.parametrize('request_type', [certificate_manager.UpdateCertificateMapRequest, dict])
def test_update_certificate_map(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_certificate_map), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_certificate_map(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.UpdateCertificateMapRequest()
    assert isinstance(response, future.Future)

def test_update_certificate_map_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_certificate_map), '__call__') as call:
        client.update_certificate_map()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.UpdateCertificateMapRequest()

@pytest.mark.asyncio
async def test_update_certificate_map_async(transport: str='grpc_asyncio', request_type=certificate_manager.UpdateCertificateMapRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_certificate_map), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_certificate_map(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.UpdateCertificateMapRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_certificate_map_async_from_dict():
    await test_update_certificate_map_async(request_type=dict)

def test_update_certificate_map_field_headers():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.UpdateCertificateMapRequest()
    request.certificate_map.name = 'name_value'
    with mock.patch.object(type(client.transport.update_certificate_map), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_certificate_map(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'certificate_map.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_certificate_map_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.UpdateCertificateMapRequest()
    request.certificate_map.name = 'name_value'
    with mock.patch.object(type(client.transport.update_certificate_map), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_certificate_map(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'certificate_map.name=name_value') in kw['metadata']

def test_update_certificate_map_flattened():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_certificate_map), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_certificate_map(certificate_map=certificate_manager.CertificateMap(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].certificate_map
        mock_val = certificate_manager.CertificateMap(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_certificate_map_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_certificate_map(certificate_manager.UpdateCertificateMapRequest(), certificate_map=certificate_manager.CertificateMap(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_certificate_map_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_certificate_map), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_certificate_map(certificate_map=certificate_manager.CertificateMap(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].certificate_map
        mock_val = certificate_manager.CertificateMap(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_certificate_map_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_certificate_map(certificate_manager.UpdateCertificateMapRequest(), certificate_map=certificate_manager.CertificateMap(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [certificate_manager.DeleteCertificateMapRequest, dict])
def test_delete_certificate_map(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_certificate_map), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_certificate_map(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.DeleteCertificateMapRequest()
    assert isinstance(response, future.Future)

def test_delete_certificate_map_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_certificate_map), '__call__') as call:
        client.delete_certificate_map()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.DeleteCertificateMapRequest()

@pytest.mark.asyncio
async def test_delete_certificate_map_async(transport: str='grpc_asyncio', request_type=certificate_manager.DeleteCertificateMapRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_certificate_map), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_certificate_map(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.DeleteCertificateMapRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_certificate_map_async_from_dict():
    await test_delete_certificate_map_async(request_type=dict)

def test_delete_certificate_map_field_headers():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.DeleteCertificateMapRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_certificate_map), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_certificate_map(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_certificate_map_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.DeleteCertificateMapRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_certificate_map), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_certificate_map(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_certificate_map_flattened():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_certificate_map), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_certificate_map(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_certificate_map_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_certificate_map(certificate_manager.DeleteCertificateMapRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_certificate_map_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_certificate_map), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_certificate_map(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_certificate_map_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_certificate_map(certificate_manager.DeleteCertificateMapRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [certificate_manager.ListCertificateMapEntriesRequest, dict])
def test_list_certificate_map_entries(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_certificate_map_entries), '__call__') as call:
        call.return_value = certificate_manager.ListCertificateMapEntriesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_certificate_map_entries(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.ListCertificateMapEntriesRequest()
    assert isinstance(response, pagers.ListCertificateMapEntriesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_certificate_map_entries_empty_call():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_certificate_map_entries), '__call__') as call:
        client.list_certificate_map_entries()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.ListCertificateMapEntriesRequest()

@pytest.mark.asyncio
async def test_list_certificate_map_entries_async(transport: str='grpc_asyncio', request_type=certificate_manager.ListCertificateMapEntriesRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_certificate_map_entries), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.ListCertificateMapEntriesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_certificate_map_entries(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.ListCertificateMapEntriesRequest()
    assert isinstance(response, pagers.ListCertificateMapEntriesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_certificate_map_entries_async_from_dict():
    await test_list_certificate_map_entries_async(request_type=dict)

def test_list_certificate_map_entries_field_headers():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.ListCertificateMapEntriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_certificate_map_entries), '__call__') as call:
        call.return_value = certificate_manager.ListCertificateMapEntriesResponse()
        client.list_certificate_map_entries(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_certificate_map_entries_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.ListCertificateMapEntriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_certificate_map_entries), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.ListCertificateMapEntriesResponse())
        await client.list_certificate_map_entries(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_certificate_map_entries_flattened():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_certificate_map_entries), '__call__') as call:
        call.return_value = certificate_manager.ListCertificateMapEntriesResponse()
        client.list_certificate_map_entries(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_certificate_map_entries_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_certificate_map_entries(certificate_manager.ListCertificateMapEntriesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_certificate_map_entries_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_certificate_map_entries), '__call__') as call:
        call.return_value = certificate_manager.ListCertificateMapEntriesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.ListCertificateMapEntriesResponse())
        response = await client.list_certificate_map_entries(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_certificate_map_entries_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_certificate_map_entries(certificate_manager.ListCertificateMapEntriesRequest(), parent='parent_value')

def test_list_certificate_map_entries_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_certificate_map_entries), '__call__') as call:
        call.side_effect = (certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry()], next_page_token='abc'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[], next_page_token='def'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry()], next_page_token='ghi'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_certificate_map_entries(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, certificate_manager.CertificateMapEntry) for i in results))

def test_list_certificate_map_entries_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_certificate_map_entries), '__call__') as call:
        call.side_effect = (certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry()], next_page_token='abc'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[], next_page_token='def'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry()], next_page_token='ghi'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry()]), RuntimeError)
        pages = list(client.list_certificate_map_entries(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_certificate_map_entries_async_pager():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_certificate_map_entries), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry()], next_page_token='abc'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[], next_page_token='def'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry()], next_page_token='ghi'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry()]), RuntimeError)
        async_pager = await client.list_certificate_map_entries(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, certificate_manager.CertificateMapEntry) for i in responses))

@pytest.mark.asyncio
async def test_list_certificate_map_entries_async_pages():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_certificate_map_entries), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry()], next_page_token='abc'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[], next_page_token='def'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry()], next_page_token='ghi'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_certificate_map_entries(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [certificate_manager.GetCertificateMapEntryRequest, dict])
def test_get_certificate_map_entry(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_certificate_map_entry), '__call__') as call:
        call.return_value = certificate_manager.CertificateMapEntry(name='name_value', description='description_value', certificates=['certificates_value'], state=certificate_manager.ServingState.ACTIVE, hostname='hostname_value')
        response = client.get_certificate_map_entry(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.GetCertificateMapEntryRequest()
    assert isinstance(response, certificate_manager.CertificateMapEntry)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.certificates == ['certificates_value']
    assert response.state == certificate_manager.ServingState.ACTIVE

def test_get_certificate_map_entry_empty_call():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_certificate_map_entry), '__call__') as call:
        client.get_certificate_map_entry()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.GetCertificateMapEntryRequest()

@pytest.mark.asyncio
async def test_get_certificate_map_entry_async(transport: str='grpc_asyncio', request_type=certificate_manager.GetCertificateMapEntryRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_certificate_map_entry), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.CertificateMapEntry(name='name_value', description='description_value', certificates=['certificates_value'], state=certificate_manager.ServingState.ACTIVE))
        response = await client.get_certificate_map_entry(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.GetCertificateMapEntryRequest()
    assert isinstance(response, certificate_manager.CertificateMapEntry)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.certificates == ['certificates_value']
    assert response.state == certificate_manager.ServingState.ACTIVE

@pytest.mark.asyncio
async def test_get_certificate_map_entry_async_from_dict():
    await test_get_certificate_map_entry_async(request_type=dict)

def test_get_certificate_map_entry_field_headers():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.GetCertificateMapEntryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_certificate_map_entry), '__call__') as call:
        call.return_value = certificate_manager.CertificateMapEntry()
        client.get_certificate_map_entry(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_certificate_map_entry_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.GetCertificateMapEntryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_certificate_map_entry), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.CertificateMapEntry())
        await client.get_certificate_map_entry(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_certificate_map_entry_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_certificate_map_entry), '__call__') as call:
        call.return_value = certificate_manager.CertificateMapEntry()
        client.get_certificate_map_entry(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_certificate_map_entry_flattened_error():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_certificate_map_entry(certificate_manager.GetCertificateMapEntryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_certificate_map_entry_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_certificate_map_entry), '__call__') as call:
        call.return_value = certificate_manager.CertificateMapEntry()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.CertificateMapEntry())
        response = await client.get_certificate_map_entry(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_certificate_map_entry_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_certificate_map_entry(certificate_manager.GetCertificateMapEntryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [certificate_manager.CreateCertificateMapEntryRequest, dict])
def test_create_certificate_map_entry(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_certificate_map_entry), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_certificate_map_entry(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.CreateCertificateMapEntryRequest()
    assert isinstance(response, future.Future)

def test_create_certificate_map_entry_empty_call():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_certificate_map_entry), '__call__') as call:
        client.create_certificate_map_entry()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.CreateCertificateMapEntryRequest()

@pytest.mark.asyncio
async def test_create_certificate_map_entry_async(transport: str='grpc_asyncio', request_type=certificate_manager.CreateCertificateMapEntryRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_certificate_map_entry), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_certificate_map_entry(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.CreateCertificateMapEntryRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_certificate_map_entry_async_from_dict():
    await test_create_certificate_map_entry_async(request_type=dict)

def test_create_certificate_map_entry_field_headers():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.CreateCertificateMapEntryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_certificate_map_entry), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_certificate_map_entry(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_certificate_map_entry_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.CreateCertificateMapEntryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_certificate_map_entry), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_certificate_map_entry(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_certificate_map_entry_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_certificate_map_entry), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_certificate_map_entry(parent='parent_value', certificate_map_entry=certificate_manager.CertificateMapEntry(name='name_value'), certificate_map_entry_id='certificate_map_entry_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].certificate_map_entry
        mock_val = certificate_manager.CertificateMapEntry(name='name_value')
        assert arg == mock_val
        arg = args[0].certificate_map_entry_id
        mock_val = 'certificate_map_entry_id_value'
        assert arg == mock_val

def test_create_certificate_map_entry_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_certificate_map_entry(certificate_manager.CreateCertificateMapEntryRequest(), parent='parent_value', certificate_map_entry=certificate_manager.CertificateMapEntry(name='name_value'), certificate_map_entry_id='certificate_map_entry_id_value')

@pytest.mark.asyncio
async def test_create_certificate_map_entry_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_certificate_map_entry), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_certificate_map_entry(parent='parent_value', certificate_map_entry=certificate_manager.CertificateMapEntry(name='name_value'), certificate_map_entry_id='certificate_map_entry_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].certificate_map_entry
        mock_val = certificate_manager.CertificateMapEntry(name='name_value')
        assert arg == mock_val
        arg = args[0].certificate_map_entry_id
        mock_val = 'certificate_map_entry_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_certificate_map_entry_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_certificate_map_entry(certificate_manager.CreateCertificateMapEntryRequest(), parent='parent_value', certificate_map_entry=certificate_manager.CertificateMapEntry(name='name_value'), certificate_map_entry_id='certificate_map_entry_id_value')

@pytest.mark.parametrize('request_type', [certificate_manager.UpdateCertificateMapEntryRequest, dict])
def test_update_certificate_map_entry(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_certificate_map_entry), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_certificate_map_entry(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.UpdateCertificateMapEntryRequest()
    assert isinstance(response, future.Future)

def test_update_certificate_map_entry_empty_call():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_certificate_map_entry), '__call__') as call:
        client.update_certificate_map_entry()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.UpdateCertificateMapEntryRequest()

@pytest.mark.asyncio
async def test_update_certificate_map_entry_async(transport: str='grpc_asyncio', request_type=certificate_manager.UpdateCertificateMapEntryRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_certificate_map_entry), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_certificate_map_entry(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.UpdateCertificateMapEntryRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_certificate_map_entry_async_from_dict():
    await test_update_certificate_map_entry_async(request_type=dict)

def test_update_certificate_map_entry_field_headers():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.UpdateCertificateMapEntryRequest()
    request.certificate_map_entry.name = 'name_value'
    with mock.patch.object(type(client.transport.update_certificate_map_entry), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_certificate_map_entry(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'certificate_map_entry.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_certificate_map_entry_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.UpdateCertificateMapEntryRequest()
    request.certificate_map_entry.name = 'name_value'
    with mock.patch.object(type(client.transport.update_certificate_map_entry), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_certificate_map_entry(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'certificate_map_entry.name=name_value') in kw['metadata']

def test_update_certificate_map_entry_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_certificate_map_entry), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_certificate_map_entry(certificate_map_entry=certificate_manager.CertificateMapEntry(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].certificate_map_entry
        mock_val = certificate_manager.CertificateMapEntry(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_certificate_map_entry_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_certificate_map_entry(certificate_manager.UpdateCertificateMapEntryRequest(), certificate_map_entry=certificate_manager.CertificateMapEntry(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_certificate_map_entry_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_certificate_map_entry), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_certificate_map_entry(certificate_map_entry=certificate_manager.CertificateMapEntry(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].certificate_map_entry
        mock_val = certificate_manager.CertificateMapEntry(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_certificate_map_entry_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_certificate_map_entry(certificate_manager.UpdateCertificateMapEntryRequest(), certificate_map_entry=certificate_manager.CertificateMapEntry(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [certificate_manager.DeleteCertificateMapEntryRequest, dict])
def test_delete_certificate_map_entry(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_certificate_map_entry), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_certificate_map_entry(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.DeleteCertificateMapEntryRequest()
    assert isinstance(response, future.Future)

def test_delete_certificate_map_entry_empty_call():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_certificate_map_entry), '__call__') as call:
        client.delete_certificate_map_entry()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.DeleteCertificateMapEntryRequest()

@pytest.mark.asyncio
async def test_delete_certificate_map_entry_async(transport: str='grpc_asyncio', request_type=certificate_manager.DeleteCertificateMapEntryRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_certificate_map_entry), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_certificate_map_entry(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.DeleteCertificateMapEntryRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_certificate_map_entry_async_from_dict():
    await test_delete_certificate_map_entry_async(request_type=dict)

def test_delete_certificate_map_entry_field_headers():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.DeleteCertificateMapEntryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_certificate_map_entry), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_certificate_map_entry(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_certificate_map_entry_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.DeleteCertificateMapEntryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_certificate_map_entry), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_certificate_map_entry(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_certificate_map_entry_flattened():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_certificate_map_entry), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_certificate_map_entry(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_certificate_map_entry_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_certificate_map_entry(certificate_manager.DeleteCertificateMapEntryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_certificate_map_entry_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_certificate_map_entry), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_certificate_map_entry(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_certificate_map_entry_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_certificate_map_entry(certificate_manager.DeleteCertificateMapEntryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [certificate_manager.ListDnsAuthorizationsRequest, dict])
def test_list_dns_authorizations(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_dns_authorizations), '__call__') as call:
        call.return_value = certificate_manager.ListDnsAuthorizationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_dns_authorizations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.ListDnsAuthorizationsRequest()
    assert isinstance(response, pagers.ListDnsAuthorizationsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_dns_authorizations_empty_call():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_dns_authorizations), '__call__') as call:
        client.list_dns_authorizations()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.ListDnsAuthorizationsRequest()

@pytest.mark.asyncio
async def test_list_dns_authorizations_async(transport: str='grpc_asyncio', request_type=certificate_manager.ListDnsAuthorizationsRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_dns_authorizations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.ListDnsAuthorizationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_dns_authorizations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.ListDnsAuthorizationsRequest()
    assert isinstance(response, pagers.ListDnsAuthorizationsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_dns_authorizations_async_from_dict():
    await test_list_dns_authorizations_async(request_type=dict)

def test_list_dns_authorizations_field_headers():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.ListDnsAuthorizationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_dns_authorizations), '__call__') as call:
        call.return_value = certificate_manager.ListDnsAuthorizationsResponse()
        client.list_dns_authorizations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_dns_authorizations_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.ListDnsAuthorizationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_dns_authorizations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.ListDnsAuthorizationsResponse())
        await client.list_dns_authorizations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_dns_authorizations_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_dns_authorizations), '__call__') as call:
        call.return_value = certificate_manager.ListDnsAuthorizationsResponse()
        client.list_dns_authorizations(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_dns_authorizations_flattened_error():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_dns_authorizations(certificate_manager.ListDnsAuthorizationsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_dns_authorizations_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_dns_authorizations), '__call__') as call:
        call.return_value = certificate_manager.ListDnsAuthorizationsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.ListDnsAuthorizationsResponse())
        response = await client.list_dns_authorizations(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_dns_authorizations_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_dns_authorizations(certificate_manager.ListDnsAuthorizationsRequest(), parent='parent_value')

def test_list_dns_authorizations_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_dns_authorizations), '__call__') as call:
        call.side_effect = (certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization()], next_page_token='abc'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[], next_page_token='def'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization()], next_page_token='ghi'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_dns_authorizations(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, certificate_manager.DnsAuthorization) for i in results))

def test_list_dns_authorizations_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_dns_authorizations), '__call__') as call:
        call.side_effect = (certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization()], next_page_token='abc'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[], next_page_token='def'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization()], next_page_token='ghi'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization()]), RuntimeError)
        pages = list(client.list_dns_authorizations(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_dns_authorizations_async_pager():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_dns_authorizations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization()], next_page_token='abc'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[], next_page_token='def'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization()], next_page_token='ghi'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization()]), RuntimeError)
        async_pager = await client.list_dns_authorizations(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, certificate_manager.DnsAuthorization) for i in responses))

@pytest.mark.asyncio
async def test_list_dns_authorizations_async_pages():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_dns_authorizations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization()], next_page_token='abc'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[], next_page_token='def'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization()], next_page_token='ghi'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_dns_authorizations(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [certificate_manager.GetDnsAuthorizationRequest, dict])
def test_get_dns_authorization(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_dns_authorization), '__call__') as call:
        call.return_value = certificate_manager.DnsAuthorization(name='name_value', description='description_value', domain='domain_value')
        response = client.get_dns_authorization(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.GetDnsAuthorizationRequest()
    assert isinstance(response, certificate_manager.DnsAuthorization)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.domain == 'domain_value'

def test_get_dns_authorization_empty_call():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_dns_authorization), '__call__') as call:
        client.get_dns_authorization()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.GetDnsAuthorizationRequest()

@pytest.mark.asyncio
async def test_get_dns_authorization_async(transport: str='grpc_asyncio', request_type=certificate_manager.GetDnsAuthorizationRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_dns_authorization), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.DnsAuthorization(name='name_value', description='description_value', domain='domain_value'))
        response = await client.get_dns_authorization(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.GetDnsAuthorizationRequest()
    assert isinstance(response, certificate_manager.DnsAuthorization)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.domain == 'domain_value'

@pytest.mark.asyncio
async def test_get_dns_authorization_async_from_dict():
    await test_get_dns_authorization_async(request_type=dict)

def test_get_dns_authorization_field_headers():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.GetDnsAuthorizationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_dns_authorization), '__call__') as call:
        call.return_value = certificate_manager.DnsAuthorization()
        client.get_dns_authorization(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_dns_authorization_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.GetDnsAuthorizationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_dns_authorization), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.DnsAuthorization())
        await client.get_dns_authorization(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_dns_authorization_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_dns_authorization), '__call__') as call:
        call.return_value = certificate_manager.DnsAuthorization()
        client.get_dns_authorization(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_dns_authorization_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_dns_authorization(certificate_manager.GetDnsAuthorizationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_dns_authorization_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_dns_authorization), '__call__') as call:
        call.return_value = certificate_manager.DnsAuthorization()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_manager.DnsAuthorization())
        response = await client.get_dns_authorization(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_dns_authorization_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_dns_authorization(certificate_manager.GetDnsAuthorizationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [certificate_manager.CreateDnsAuthorizationRequest, dict])
def test_create_dns_authorization(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_dns_authorization), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_dns_authorization(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.CreateDnsAuthorizationRequest()
    assert isinstance(response, future.Future)

def test_create_dns_authorization_empty_call():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_dns_authorization), '__call__') as call:
        client.create_dns_authorization()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.CreateDnsAuthorizationRequest()

@pytest.mark.asyncio
async def test_create_dns_authorization_async(transport: str='grpc_asyncio', request_type=certificate_manager.CreateDnsAuthorizationRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_dns_authorization), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_dns_authorization(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.CreateDnsAuthorizationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_dns_authorization_async_from_dict():
    await test_create_dns_authorization_async(request_type=dict)

def test_create_dns_authorization_field_headers():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.CreateDnsAuthorizationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_dns_authorization), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_dns_authorization(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_dns_authorization_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.CreateDnsAuthorizationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_dns_authorization), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_dns_authorization(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_dns_authorization_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_dns_authorization), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_dns_authorization(parent='parent_value', dns_authorization=certificate_manager.DnsAuthorization(name='name_value'), dns_authorization_id='dns_authorization_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].dns_authorization
        mock_val = certificate_manager.DnsAuthorization(name='name_value')
        assert arg == mock_val
        arg = args[0].dns_authorization_id
        mock_val = 'dns_authorization_id_value'
        assert arg == mock_val

def test_create_dns_authorization_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_dns_authorization(certificate_manager.CreateDnsAuthorizationRequest(), parent='parent_value', dns_authorization=certificate_manager.DnsAuthorization(name='name_value'), dns_authorization_id='dns_authorization_id_value')

@pytest.mark.asyncio
async def test_create_dns_authorization_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_dns_authorization), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_dns_authorization(parent='parent_value', dns_authorization=certificate_manager.DnsAuthorization(name='name_value'), dns_authorization_id='dns_authorization_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].dns_authorization
        mock_val = certificate_manager.DnsAuthorization(name='name_value')
        assert arg == mock_val
        arg = args[0].dns_authorization_id
        mock_val = 'dns_authorization_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_dns_authorization_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_dns_authorization(certificate_manager.CreateDnsAuthorizationRequest(), parent='parent_value', dns_authorization=certificate_manager.DnsAuthorization(name='name_value'), dns_authorization_id='dns_authorization_id_value')

@pytest.mark.parametrize('request_type', [certificate_manager.UpdateDnsAuthorizationRequest, dict])
def test_update_dns_authorization(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_dns_authorization), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_dns_authorization(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.UpdateDnsAuthorizationRequest()
    assert isinstance(response, future.Future)

def test_update_dns_authorization_empty_call():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_dns_authorization), '__call__') as call:
        client.update_dns_authorization()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.UpdateDnsAuthorizationRequest()

@pytest.mark.asyncio
async def test_update_dns_authorization_async(transport: str='grpc_asyncio', request_type=certificate_manager.UpdateDnsAuthorizationRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_dns_authorization), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_dns_authorization(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.UpdateDnsAuthorizationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_dns_authorization_async_from_dict():
    await test_update_dns_authorization_async(request_type=dict)

def test_update_dns_authorization_field_headers():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.UpdateDnsAuthorizationRequest()
    request.dns_authorization.name = 'name_value'
    with mock.patch.object(type(client.transport.update_dns_authorization), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_dns_authorization(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dns_authorization.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_dns_authorization_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.UpdateDnsAuthorizationRequest()
    request.dns_authorization.name = 'name_value'
    with mock.patch.object(type(client.transport.update_dns_authorization), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_dns_authorization(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'dns_authorization.name=name_value') in kw['metadata']

def test_update_dns_authorization_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_dns_authorization), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_dns_authorization(dns_authorization=certificate_manager.DnsAuthorization(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dns_authorization
        mock_val = certificate_manager.DnsAuthorization(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_dns_authorization_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_dns_authorization(certificate_manager.UpdateDnsAuthorizationRequest(), dns_authorization=certificate_manager.DnsAuthorization(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_dns_authorization_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_dns_authorization), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_dns_authorization(dns_authorization=certificate_manager.DnsAuthorization(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].dns_authorization
        mock_val = certificate_manager.DnsAuthorization(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_dns_authorization_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_dns_authorization(certificate_manager.UpdateDnsAuthorizationRequest(), dns_authorization=certificate_manager.DnsAuthorization(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [certificate_manager.DeleteDnsAuthorizationRequest, dict])
def test_delete_dns_authorization(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_dns_authorization), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_dns_authorization(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.DeleteDnsAuthorizationRequest()
    assert isinstance(response, future.Future)

def test_delete_dns_authorization_empty_call():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_dns_authorization), '__call__') as call:
        client.delete_dns_authorization()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.DeleteDnsAuthorizationRequest()

@pytest.mark.asyncio
async def test_delete_dns_authorization_async(transport: str='grpc_asyncio', request_type=certificate_manager.DeleteDnsAuthorizationRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_dns_authorization), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_dns_authorization(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_manager.DeleteDnsAuthorizationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_dns_authorization_async_from_dict():
    await test_delete_dns_authorization_async(request_type=dict)

def test_delete_dns_authorization_field_headers():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.DeleteDnsAuthorizationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_dns_authorization), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_dns_authorization(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_dns_authorization_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_manager.DeleteDnsAuthorizationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_dns_authorization), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_dns_authorization(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_dns_authorization_flattened():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_dns_authorization), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_dns_authorization(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_dns_authorization_flattened_error():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_dns_authorization(certificate_manager.DeleteDnsAuthorizationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_dns_authorization_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_dns_authorization), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_dns_authorization(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_dns_authorization_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_dns_authorization(certificate_manager.DeleteDnsAuthorizationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [certificate_issuance_config.ListCertificateIssuanceConfigsRequest, dict])
def test_list_certificate_issuance_configs(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_certificate_issuance_configs), '__call__') as call:
        call.return_value = certificate_issuance_config.ListCertificateIssuanceConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_certificate_issuance_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_issuance_config.ListCertificateIssuanceConfigsRequest()
    assert isinstance(response, pagers.ListCertificateIssuanceConfigsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_certificate_issuance_configs_empty_call():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_certificate_issuance_configs), '__call__') as call:
        client.list_certificate_issuance_configs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_issuance_config.ListCertificateIssuanceConfigsRequest()

@pytest.mark.asyncio
async def test_list_certificate_issuance_configs_async(transport: str='grpc_asyncio', request_type=certificate_issuance_config.ListCertificateIssuanceConfigsRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_certificate_issuance_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_issuance_config.ListCertificateIssuanceConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_certificate_issuance_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_issuance_config.ListCertificateIssuanceConfigsRequest()
    assert isinstance(response, pagers.ListCertificateIssuanceConfigsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_certificate_issuance_configs_async_from_dict():
    await test_list_certificate_issuance_configs_async(request_type=dict)

def test_list_certificate_issuance_configs_field_headers():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_issuance_config.ListCertificateIssuanceConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_certificate_issuance_configs), '__call__') as call:
        call.return_value = certificate_issuance_config.ListCertificateIssuanceConfigsResponse()
        client.list_certificate_issuance_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_certificate_issuance_configs_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_issuance_config.ListCertificateIssuanceConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_certificate_issuance_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_issuance_config.ListCertificateIssuanceConfigsResponse())
        await client.list_certificate_issuance_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_certificate_issuance_configs_flattened():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_certificate_issuance_configs), '__call__') as call:
        call.return_value = certificate_issuance_config.ListCertificateIssuanceConfigsResponse()
        client.list_certificate_issuance_configs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_certificate_issuance_configs_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_certificate_issuance_configs(certificate_issuance_config.ListCertificateIssuanceConfigsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_certificate_issuance_configs_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_certificate_issuance_configs), '__call__') as call:
        call.return_value = certificate_issuance_config.ListCertificateIssuanceConfigsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_issuance_config.ListCertificateIssuanceConfigsResponse())
        response = await client.list_certificate_issuance_configs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_certificate_issuance_configs_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_certificate_issuance_configs(certificate_issuance_config.ListCertificateIssuanceConfigsRequest(), parent='parent_value')

def test_list_certificate_issuance_configs_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_certificate_issuance_configs), '__call__') as call:
        call.side_effect = (certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig()], next_page_token='abc'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[], next_page_token='def'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig()], next_page_token='ghi'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_certificate_issuance_configs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, certificate_issuance_config.CertificateIssuanceConfig) for i in results))

def test_list_certificate_issuance_configs_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_certificate_issuance_configs), '__call__') as call:
        call.side_effect = (certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig()], next_page_token='abc'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[], next_page_token='def'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig()], next_page_token='ghi'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig()]), RuntimeError)
        pages = list(client.list_certificate_issuance_configs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_certificate_issuance_configs_async_pager():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_certificate_issuance_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig()], next_page_token='abc'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[], next_page_token='def'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig()], next_page_token='ghi'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig()]), RuntimeError)
        async_pager = await client.list_certificate_issuance_configs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, certificate_issuance_config.CertificateIssuanceConfig) for i in responses))

@pytest.mark.asyncio
async def test_list_certificate_issuance_configs_async_pages():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_certificate_issuance_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig()], next_page_token='abc'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[], next_page_token='def'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig()], next_page_token='ghi'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_certificate_issuance_configs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [certificate_issuance_config.GetCertificateIssuanceConfigRequest, dict])
def test_get_certificate_issuance_config(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_certificate_issuance_config), '__call__') as call:
        call.return_value = certificate_issuance_config.CertificateIssuanceConfig(name='name_value', description='description_value', rotation_window_percentage=2788, key_algorithm=certificate_issuance_config.CertificateIssuanceConfig.KeyAlgorithm.RSA_2048)
        response = client.get_certificate_issuance_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_issuance_config.GetCertificateIssuanceConfigRequest()
    assert isinstance(response, certificate_issuance_config.CertificateIssuanceConfig)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.rotation_window_percentage == 2788
    assert response.key_algorithm == certificate_issuance_config.CertificateIssuanceConfig.KeyAlgorithm.RSA_2048

def test_get_certificate_issuance_config_empty_call():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_certificate_issuance_config), '__call__') as call:
        client.get_certificate_issuance_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_issuance_config.GetCertificateIssuanceConfigRequest()

@pytest.mark.asyncio
async def test_get_certificate_issuance_config_async(transport: str='grpc_asyncio', request_type=certificate_issuance_config.GetCertificateIssuanceConfigRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_certificate_issuance_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_issuance_config.CertificateIssuanceConfig(name='name_value', description='description_value', rotation_window_percentage=2788, key_algorithm=certificate_issuance_config.CertificateIssuanceConfig.KeyAlgorithm.RSA_2048))
        response = await client.get_certificate_issuance_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_issuance_config.GetCertificateIssuanceConfigRequest()
    assert isinstance(response, certificate_issuance_config.CertificateIssuanceConfig)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.rotation_window_percentage == 2788
    assert response.key_algorithm == certificate_issuance_config.CertificateIssuanceConfig.KeyAlgorithm.RSA_2048

@pytest.mark.asyncio
async def test_get_certificate_issuance_config_async_from_dict():
    await test_get_certificate_issuance_config_async(request_type=dict)

def test_get_certificate_issuance_config_field_headers():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_issuance_config.GetCertificateIssuanceConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_certificate_issuance_config), '__call__') as call:
        call.return_value = certificate_issuance_config.CertificateIssuanceConfig()
        client.get_certificate_issuance_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_certificate_issuance_config_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_issuance_config.GetCertificateIssuanceConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_certificate_issuance_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_issuance_config.CertificateIssuanceConfig())
        await client.get_certificate_issuance_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_certificate_issuance_config_flattened():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_certificate_issuance_config), '__call__') as call:
        call.return_value = certificate_issuance_config.CertificateIssuanceConfig()
        client.get_certificate_issuance_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_certificate_issuance_config_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_certificate_issuance_config(certificate_issuance_config.GetCertificateIssuanceConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_certificate_issuance_config_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_certificate_issuance_config), '__call__') as call:
        call.return_value = certificate_issuance_config.CertificateIssuanceConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate_issuance_config.CertificateIssuanceConfig())
        response = await client.get_certificate_issuance_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_certificate_issuance_config_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_certificate_issuance_config(certificate_issuance_config.GetCertificateIssuanceConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest, dict])
def test_create_certificate_issuance_config(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_certificate_issuance_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_certificate_issuance_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest()
    assert isinstance(response, future.Future)

def test_create_certificate_issuance_config_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_certificate_issuance_config), '__call__') as call:
        client.create_certificate_issuance_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest()

@pytest.mark.asyncio
async def test_create_certificate_issuance_config_async(transport: str='grpc_asyncio', request_type=gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_certificate_issuance_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_certificate_issuance_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_certificate_issuance_config_async_from_dict():
    await test_create_certificate_issuance_config_async(request_type=dict)

def test_create_certificate_issuance_config_field_headers():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_certificate_issuance_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_certificate_issuance_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_certificate_issuance_config_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_certificate_issuance_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_certificate_issuance_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_certificate_issuance_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_certificate_issuance_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_certificate_issuance_config(parent='parent_value', certificate_issuance_config=gcc_certificate_issuance_config.CertificateIssuanceConfig(name='name_value'), certificate_issuance_config_id='certificate_issuance_config_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].certificate_issuance_config
        mock_val = gcc_certificate_issuance_config.CertificateIssuanceConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].certificate_issuance_config_id
        mock_val = 'certificate_issuance_config_id_value'
        assert arg == mock_val

def test_create_certificate_issuance_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_certificate_issuance_config(gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest(), parent='parent_value', certificate_issuance_config=gcc_certificate_issuance_config.CertificateIssuanceConfig(name='name_value'), certificate_issuance_config_id='certificate_issuance_config_id_value')

@pytest.mark.asyncio
async def test_create_certificate_issuance_config_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_certificate_issuance_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_certificate_issuance_config(parent='parent_value', certificate_issuance_config=gcc_certificate_issuance_config.CertificateIssuanceConfig(name='name_value'), certificate_issuance_config_id='certificate_issuance_config_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].certificate_issuance_config
        mock_val = gcc_certificate_issuance_config.CertificateIssuanceConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].certificate_issuance_config_id
        mock_val = 'certificate_issuance_config_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_certificate_issuance_config_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_certificate_issuance_config(gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest(), parent='parent_value', certificate_issuance_config=gcc_certificate_issuance_config.CertificateIssuanceConfig(name='name_value'), certificate_issuance_config_id='certificate_issuance_config_id_value')

@pytest.mark.parametrize('request_type', [certificate_issuance_config.DeleteCertificateIssuanceConfigRequest, dict])
def test_delete_certificate_issuance_config(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_certificate_issuance_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_certificate_issuance_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_issuance_config.DeleteCertificateIssuanceConfigRequest()
    assert isinstance(response, future.Future)

def test_delete_certificate_issuance_config_empty_call():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_certificate_issuance_config), '__call__') as call:
        client.delete_certificate_issuance_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_issuance_config.DeleteCertificateIssuanceConfigRequest()

@pytest.mark.asyncio
async def test_delete_certificate_issuance_config_async(transport: str='grpc_asyncio', request_type=certificate_issuance_config.DeleteCertificateIssuanceConfigRequest):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_certificate_issuance_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_certificate_issuance_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == certificate_issuance_config.DeleteCertificateIssuanceConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_certificate_issuance_config_async_from_dict():
    await test_delete_certificate_issuance_config_async(request_type=dict)

def test_delete_certificate_issuance_config_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_issuance_config.DeleteCertificateIssuanceConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_certificate_issuance_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_certificate_issuance_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_certificate_issuance_config_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = certificate_issuance_config.DeleteCertificateIssuanceConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_certificate_issuance_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_certificate_issuance_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_certificate_issuance_config_flattened():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_certificate_issuance_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_certificate_issuance_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_certificate_issuance_config_flattened_error():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_certificate_issuance_config(certificate_issuance_config.DeleteCertificateIssuanceConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_certificate_issuance_config_flattened_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_certificate_issuance_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_certificate_issuance_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_certificate_issuance_config_flattened_error_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_certificate_issuance_config(certificate_issuance_config.DeleteCertificateIssuanceConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [certificate_manager.ListCertificatesRequest, dict])
def test_list_certificates_rest(request_type):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.ListCertificatesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.ListCertificatesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_certificates(request)
    assert isinstance(response, pagers.ListCertificatesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_certificates_rest_required_fields(request_type=certificate_manager.ListCertificatesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_certificates._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_certificates._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = certificate_manager.ListCertificatesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = certificate_manager.ListCertificatesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_certificates(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_certificates_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_certificates._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_certificates_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_list_certificates') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_list_certificates') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.ListCertificatesRequest.pb(certificate_manager.ListCertificatesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = certificate_manager.ListCertificatesResponse.to_json(certificate_manager.ListCertificatesResponse())
        request = certificate_manager.ListCertificatesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = certificate_manager.ListCertificatesResponse()
        client.list_certificates(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_certificates_rest_bad_request(transport: str='rest', request_type=certificate_manager.ListCertificatesRequest):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_certificates(request)

def test_list_certificates_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.ListCertificatesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.ListCertificatesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_certificates(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/certificates' % client.transport._host, args[1])

def test_list_certificates_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_certificates(certificate_manager.ListCertificatesRequest(), parent='parent_value')

def test_list_certificates_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate(), certificate_manager.Certificate(), certificate_manager.Certificate()], next_page_token='abc'), certificate_manager.ListCertificatesResponse(certificates=[], next_page_token='def'), certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate()], next_page_token='ghi'), certificate_manager.ListCertificatesResponse(certificates=[certificate_manager.Certificate(), certificate_manager.Certificate()]))
        response = response + response
        response = tuple((certificate_manager.ListCertificatesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_certificates(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, certificate_manager.Certificate) for i in results))
        pages = list(client.list_certificates(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [certificate_manager.GetCertificateRequest, dict])
def test_get_certificate_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificates/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.Certificate(name='name_value', description='description_value', san_dnsnames=['san_dnsnames_value'], pem_certificate='pem_certificate_value', scope=certificate_manager.Certificate.Scope.EDGE_CACHE)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.Certificate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_certificate(request)
    assert isinstance(response, certificate_manager.Certificate)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.san_dnsnames == ['san_dnsnames_value']
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.scope == certificate_manager.Certificate.Scope.EDGE_CACHE

def test_get_certificate_rest_required_fields(request_type=certificate_manager.GetCertificateRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_certificate._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_certificate._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = certificate_manager.Certificate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = certificate_manager.Certificate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_certificate(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_certificate_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_certificate._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_certificate_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_get_certificate') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_get_certificate') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.GetCertificateRequest.pb(certificate_manager.GetCertificateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = certificate_manager.Certificate.to_json(certificate_manager.Certificate())
        request = certificate_manager.GetCertificateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = certificate_manager.Certificate()
        client.get_certificate(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_certificate_rest_bad_request(transport: str='rest', request_type=certificate_manager.GetCertificateRequest):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificates/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_certificate(request)

def test_get_certificate_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.Certificate()
        sample_request = {'name': 'projects/sample1/locations/sample2/certificates/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.Certificate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_certificate(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/certificates/*}' % client.transport._host, args[1])

def test_get_certificate_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_certificate(certificate_manager.GetCertificateRequest(), name='name_value')

def test_get_certificate_rest_error():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.CreateCertificateRequest, dict])
def test_create_certificate_rest(request_type):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['certificate'] = {'name': 'name_value', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'self_managed': {'pem_certificate': 'pem_certificate_value', 'pem_private_key': 'pem_private_key_value'}, 'managed': {'domains': ['domains_value1', 'domains_value2'], 'dns_authorizations': ['dns_authorizations_value1', 'dns_authorizations_value2'], 'issuance_config': 'issuance_config_value', 'state': 1, 'provisioning_issue': {'reason': 1, 'details': 'details_value'}, 'authorization_attempt_info': [{'domain': 'domain_value', 'state': 1, 'failure_reason': 1, 'details': 'details_value'}]}, 'san_dnsnames': ['san_dnsnames_value1', 'san_dnsnames_value2'], 'pem_certificate': 'pem_certificate_value', 'expire_time': {}, 'scope': 1}
    test_field = certificate_manager.CreateCertificateRequest.meta.fields['certificate']

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
    for (field, value) in request_init['certificate'].items():
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
                for i in range(0, len(request_init['certificate'][field])):
                    del request_init['certificate'][field][i][subfield]
            else:
                del request_init['certificate'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_certificate(request)
    assert response.operation.name == 'operations/spam'

def test_create_certificate_rest_required_fields(request_type=certificate_manager.CreateCertificateRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['certificate_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'certificateId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_certificate._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'certificateId' in jsonified_request
    assert jsonified_request['certificateId'] == request_init['certificate_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['certificateId'] = 'certificate_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_certificate._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('certificate_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'certificateId' in jsonified_request
    assert jsonified_request['certificateId'] == 'certificate_id_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_certificate(request)
            expected_params = [('certificateId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_certificate_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_certificate._get_unset_required_fields({})
    assert set(unset_fields) == set(('certificateId',)) & set(('parent', 'certificateId', 'certificate'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_certificate_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_create_certificate') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_create_certificate') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.CreateCertificateRequest.pb(certificate_manager.CreateCertificateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = certificate_manager.CreateCertificateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_certificate(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_certificate_rest_bad_request(transport: str='rest', request_type=certificate_manager.CreateCertificateRequest):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_certificate(request)

def test_create_certificate_rest_flattened():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', certificate=certificate_manager.Certificate(name='name_value'), certificate_id='certificate_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_certificate(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/certificates' % client.transport._host, args[1])

def test_create_certificate_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_certificate(certificate_manager.CreateCertificateRequest(), parent='parent_value', certificate=certificate_manager.Certificate(name='name_value'), certificate_id='certificate_id_value')

def test_create_certificate_rest_error():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.UpdateCertificateRequest, dict])
def test_update_certificate_rest(request_type):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'certificate': {'name': 'projects/sample1/locations/sample2/certificates/sample3'}}
    request_init['certificate'] = {'name': 'projects/sample1/locations/sample2/certificates/sample3', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'self_managed': {'pem_certificate': 'pem_certificate_value', 'pem_private_key': 'pem_private_key_value'}, 'managed': {'domains': ['domains_value1', 'domains_value2'], 'dns_authorizations': ['dns_authorizations_value1', 'dns_authorizations_value2'], 'issuance_config': 'issuance_config_value', 'state': 1, 'provisioning_issue': {'reason': 1, 'details': 'details_value'}, 'authorization_attempt_info': [{'domain': 'domain_value', 'state': 1, 'failure_reason': 1, 'details': 'details_value'}]}, 'san_dnsnames': ['san_dnsnames_value1', 'san_dnsnames_value2'], 'pem_certificate': 'pem_certificate_value', 'expire_time': {}, 'scope': 1}
    test_field = certificate_manager.UpdateCertificateRequest.meta.fields['certificate']

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
    for (field, value) in request_init['certificate'].items():
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
                for i in range(0, len(request_init['certificate'][field])):
                    del request_init['certificate'][field][i][subfield]
            else:
                del request_init['certificate'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_certificate(request)
    assert response.operation.name == 'operations/spam'

def test_update_certificate_rest_required_fields(request_type=certificate_manager.UpdateCertificateRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_certificate._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_certificate._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_certificate(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_certificate_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_certificate._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('certificate', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_certificate_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_update_certificate') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_update_certificate') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.UpdateCertificateRequest.pb(certificate_manager.UpdateCertificateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = certificate_manager.UpdateCertificateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_certificate(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_certificate_rest_bad_request(transport: str='rest', request_type=certificate_manager.UpdateCertificateRequest):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'certificate': {'name': 'projects/sample1/locations/sample2/certificates/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_certificate(request)

def test_update_certificate_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'certificate': {'name': 'projects/sample1/locations/sample2/certificates/sample3'}}
        mock_args = dict(certificate=certificate_manager.Certificate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_certificate(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{certificate.name=projects/*/locations/*/certificates/*}' % client.transport._host, args[1])

def test_update_certificate_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_certificate(certificate_manager.UpdateCertificateRequest(), certificate=certificate_manager.Certificate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_certificate_rest_error():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.DeleteCertificateRequest, dict])
def test_delete_certificate_rest(request_type):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificates/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_certificate(request)
    assert response.operation.name == 'operations/spam'

def test_delete_certificate_rest_required_fields(request_type=certificate_manager.DeleteCertificateRequest):
    if False:
        return 10
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_certificate._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_certificate._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_certificate(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_certificate_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_certificate._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_certificate_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_delete_certificate') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_delete_certificate') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.DeleteCertificateRequest.pb(certificate_manager.DeleteCertificateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = certificate_manager.DeleteCertificateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_certificate(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_certificate_rest_bad_request(transport: str='rest', request_type=certificate_manager.DeleteCertificateRequest):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificates/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_certificate(request)

def test_delete_certificate_rest_flattened():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/certificates/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_certificate(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/certificates/*}' % client.transport._host, args[1])

def test_delete_certificate_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_certificate(certificate_manager.DeleteCertificateRequest(), name='name_value')

def test_delete_certificate_rest_error():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.ListCertificateMapsRequest, dict])
def test_list_certificate_maps_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.ListCertificateMapsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.ListCertificateMapsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_certificate_maps(request)
    assert isinstance(response, pagers.ListCertificateMapsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_certificate_maps_rest_required_fields(request_type=certificate_manager.ListCertificateMapsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_certificate_maps._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_certificate_maps._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = certificate_manager.ListCertificateMapsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = certificate_manager.ListCertificateMapsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_certificate_maps(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_certificate_maps_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_certificate_maps._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_certificate_maps_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_list_certificate_maps') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_list_certificate_maps') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.ListCertificateMapsRequest.pb(certificate_manager.ListCertificateMapsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = certificate_manager.ListCertificateMapsResponse.to_json(certificate_manager.ListCertificateMapsResponse())
        request = certificate_manager.ListCertificateMapsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = certificate_manager.ListCertificateMapsResponse()
        client.list_certificate_maps(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_certificate_maps_rest_bad_request(transport: str='rest', request_type=certificate_manager.ListCertificateMapsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_certificate_maps(request)

def test_list_certificate_maps_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.ListCertificateMapsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.ListCertificateMapsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_certificate_maps(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/certificateMaps' % client.transport._host, args[1])

def test_list_certificate_maps_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_certificate_maps(certificate_manager.ListCertificateMapsRequest(), parent='parent_value')

def test_list_certificate_maps_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap(), certificate_manager.CertificateMap(), certificate_manager.CertificateMap()], next_page_token='abc'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[], next_page_token='def'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap()], next_page_token='ghi'), certificate_manager.ListCertificateMapsResponse(certificate_maps=[certificate_manager.CertificateMap(), certificate_manager.CertificateMap()]))
        response = response + response
        response = tuple((certificate_manager.ListCertificateMapsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_certificate_maps(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, certificate_manager.CertificateMap) for i in results))
        pages = list(client.list_certificate_maps(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [certificate_manager.GetCertificateMapRequest, dict])
def test_get_certificate_map_rest(request_type):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.CertificateMap(name='name_value', description='description_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.CertificateMap.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_certificate_map(request)
    assert isinstance(response, certificate_manager.CertificateMap)
    assert response.name == 'name_value'
    assert response.description == 'description_value'

def test_get_certificate_map_rest_required_fields(request_type=certificate_manager.GetCertificateMapRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_certificate_map._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_certificate_map._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = certificate_manager.CertificateMap()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = certificate_manager.CertificateMap.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_certificate_map(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_certificate_map_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_certificate_map._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_certificate_map_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_get_certificate_map') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_get_certificate_map') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.GetCertificateMapRequest.pb(certificate_manager.GetCertificateMapRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = certificate_manager.CertificateMap.to_json(certificate_manager.CertificateMap())
        request = certificate_manager.GetCertificateMapRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = certificate_manager.CertificateMap()
        client.get_certificate_map(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_certificate_map_rest_bad_request(transport: str='rest', request_type=certificate_manager.GetCertificateMapRequest):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_certificate_map(request)

def test_get_certificate_map_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.CertificateMap()
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.CertificateMap.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_certificate_map(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/certificateMaps/*}' % client.transport._host, args[1])

def test_get_certificate_map_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_certificate_map(certificate_manager.GetCertificateMapRequest(), name='name_value')

def test_get_certificate_map_rest_error():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.CreateCertificateMapRequest, dict])
def test_create_certificate_map_rest(request_type):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['certificate_map'] = {'name': 'name_value', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'gclb_targets': [{'target_https_proxy': 'target_https_proxy_value', 'target_ssl_proxy': 'target_ssl_proxy_value', 'ip_configs': [{'ip_address': 'ip_address_value', 'ports': [569, 570]}]}]}
    test_field = certificate_manager.CreateCertificateMapRequest.meta.fields['certificate_map']

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
    for (field, value) in request_init['certificate_map'].items():
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
                for i in range(0, len(request_init['certificate_map'][field])):
                    del request_init['certificate_map'][field][i][subfield]
            else:
                del request_init['certificate_map'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_certificate_map(request)
    assert response.operation.name == 'operations/spam'

def test_create_certificate_map_rest_required_fields(request_type=certificate_manager.CreateCertificateMapRequest):
    if False:
        return 10
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['certificate_map_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'certificateMapId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_certificate_map._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'certificateMapId' in jsonified_request
    assert jsonified_request['certificateMapId'] == request_init['certificate_map_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['certificateMapId'] = 'certificate_map_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_certificate_map._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('certificate_map_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'certificateMapId' in jsonified_request
    assert jsonified_request['certificateMapId'] == 'certificate_map_id_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_certificate_map(request)
            expected_params = [('certificateMapId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_certificate_map_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_certificate_map._get_unset_required_fields({})
    assert set(unset_fields) == set(('certificateMapId',)) & set(('parent', 'certificateMapId', 'certificateMap'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_certificate_map_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_create_certificate_map') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_create_certificate_map') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.CreateCertificateMapRequest.pb(certificate_manager.CreateCertificateMapRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = certificate_manager.CreateCertificateMapRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_certificate_map(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_certificate_map_rest_bad_request(transport: str='rest', request_type=certificate_manager.CreateCertificateMapRequest):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_certificate_map(request)

def test_create_certificate_map_rest_flattened():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', certificate_map=certificate_manager.CertificateMap(name='name_value'), certificate_map_id='certificate_map_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_certificate_map(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/certificateMaps' % client.transport._host, args[1])

def test_create_certificate_map_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_certificate_map(certificate_manager.CreateCertificateMapRequest(), parent='parent_value', certificate_map=certificate_manager.CertificateMap(name='name_value'), certificate_map_id='certificate_map_id_value')

def test_create_certificate_map_rest_error():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.UpdateCertificateMapRequest, dict])
def test_update_certificate_map_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'certificate_map': {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3'}}
    request_init['certificate_map'] = {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'gclb_targets': [{'target_https_proxy': 'target_https_proxy_value', 'target_ssl_proxy': 'target_ssl_proxy_value', 'ip_configs': [{'ip_address': 'ip_address_value', 'ports': [569, 570]}]}]}
    test_field = certificate_manager.UpdateCertificateMapRequest.meta.fields['certificate_map']

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
    for (field, value) in request_init['certificate_map'].items():
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
                for i in range(0, len(request_init['certificate_map'][field])):
                    del request_init['certificate_map'][field][i][subfield]
            else:
                del request_init['certificate_map'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_certificate_map(request)
    assert response.operation.name == 'operations/spam'

def test_update_certificate_map_rest_required_fields(request_type=certificate_manager.UpdateCertificateMapRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_certificate_map._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_certificate_map._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_certificate_map(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_certificate_map_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_certificate_map._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('certificateMap', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_certificate_map_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_update_certificate_map') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_update_certificate_map') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.UpdateCertificateMapRequest.pb(certificate_manager.UpdateCertificateMapRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = certificate_manager.UpdateCertificateMapRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_certificate_map(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_certificate_map_rest_bad_request(transport: str='rest', request_type=certificate_manager.UpdateCertificateMapRequest):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'certificate_map': {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_certificate_map(request)

def test_update_certificate_map_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'certificate_map': {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3'}}
        mock_args = dict(certificate_map=certificate_manager.CertificateMap(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_certificate_map(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{certificate_map.name=projects/*/locations/*/certificateMaps/*}' % client.transport._host, args[1])

def test_update_certificate_map_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_certificate_map(certificate_manager.UpdateCertificateMapRequest(), certificate_map=certificate_manager.CertificateMap(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_certificate_map_rest_error():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.DeleteCertificateMapRequest, dict])
def test_delete_certificate_map_rest(request_type):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_certificate_map(request)
    assert response.operation.name == 'operations/spam'

def test_delete_certificate_map_rest_required_fields(request_type=certificate_manager.DeleteCertificateMapRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_certificate_map._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_certificate_map._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_certificate_map(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_certificate_map_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_certificate_map._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_certificate_map_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_delete_certificate_map') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_delete_certificate_map') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.DeleteCertificateMapRequest.pb(certificate_manager.DeleteCertificateMapRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = certificate_manager.DeleteCertificateMapRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_certificate_map(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_certificate_map_rest_bad_request(transport: str='rest', request_type=certificate_manager.DeleteCertificateMapRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_certificate_map(request)

def test_delete_certificate_map_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_certificate_map(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/certificateMaps/*}' % client.transport._host, args[1])

def test_delete_certificate_map_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_certificate_map(certificate_manager.DeleteCertificateMapRequest(), name='name_value')

def test_delete_certificate_map_rest_error():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.ListCertificateMapEntriesRequest, dict])
def test_list_certificate_map_entries_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/certificateMaps/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.ListCertificateMapEntriesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.ListCertificateMapEntriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_certificate_map_entries(request)
    assert isinstance(response, pagers.ListCertificateMapEntriesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_certificate_map_entries_rest_required_fields(request_type=certificate_manager.ListCertificateMapEntriesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_certificate_map_entries._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_certificate_map_entries._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = certificate_manager.ListCertificateMapEntriesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = certificate_manager.ListCertificateMapEntriesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_certificate_map_entries(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_certificate_map_entries_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_certificate_map_entries._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_certificate_map_entries_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_list_certificate_map_entries') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_list_certificate_map_entries') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.ListCertificateMapEntriesRequest.pb(certificate_manager.ListCertificateMapEntriesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = certificate_manager.ListCertificateMapEntriesResponse.to_json(certificate_manager.ListCertificateMapEntriesResponse())
        request = certificate_manager.ListCertificateMapEntriesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = certificate_manager.ListCertificateMapEntriesResponse()
        client.list_certificate_map_entries(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_certificate_map_entries_rest_bad_request(transport: str='rest', request_type=certificate_manager.ListCertificateMapEntriesRequest):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/certificateMaps/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_certificate_map_entries(request)

def test_list_certificate_map_entries_rest_flattened():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.ListCertificateMapEntriesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/certificateMaps/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.ListCertificateMapEntriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_certificate_map_entries(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/certificateMaps/*}/certificateMapEntries' % client.transport._host, args[1])

def test_list_certificate_map_entries_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_certificate_map_entries(certificate_manager.ListCertificateMapEntriesRequest(), parent='parent_value')

def test_list_certificate_map_entries_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry()], next_page_token='abc'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[], next_page_token='def'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry()], next_page_token='ghi'), certificate_manager.ListCertificateMapEntriesResponse(certificate_map_entries=[certificate_manager.CertificateMapEntry(), certificate_manager.CertificateMapEntry()]))
        response = response + response
        response = tuple((certificate_manager.ListCertificateMapEntriesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/certificateMaps/sample3'}
        pager = client.list_certificate_map_entries(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, certificate_manager.CertificateMapEntry) for i in results))
        pages = list(client.list_certificate_map_entries(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [certificate_manager.GetCertificateMapEntryRequest, dict])
def test_get_certificate_map_entry_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3/certificateMapEntries/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.CertificateMapEntry(name='name_value', description='description_value', certificates=['certificates_value'], state=certificate_manager.ServingState.ACTIVE, hostname='hostname_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.CertificateMapEntry.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_certificate_map_entry(request)
    assert isinstance(response, certificate_manager.CertificateMapEntry)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.certificates == ['certificates_value']
    assert response.state == certificate_manager.ServingState.ACTIVE

def test_get_certificate_map_entry_rest_required_fields(request_type=certificate_manager.GetCertificateMapEntryRequest):
    if False:
        return 10
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_certificate_map_entry._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_certificate_map_entry._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = certificate_manager.CertificateMapEntry()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = certificate_manager.CertificateMapEntry.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_certificate_map_entry(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_certificate_map_entry_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_certificate_map_entry._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_certificate_map_entry_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_get_certificate_map_entry') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_get_certificate_map_entry') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.GetCertificateMapEntryRequest.pb(certificate_manager.GetCertificateMapEntryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = certificate_manager.CertificateMapEntry.to_json(certificate_manager.CertificateMapEntry())
        request = certificate_manager.GetCertificateMapEntryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = certificate_manager.CertificateMapEntry()
        client.get_certificate_map_entry(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_certificate_map_entry_rest_bad_request(transport: str='rest', request_type=certificate_manager.GetCertificateMapEntryRequest):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3/certificateMapEntries/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_certificate_map_entry(request)

def test_get_certificate_map_entry_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.CertificateMapEntry()
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3/certificateMapEntries/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.CertificateMapEntry.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_certificate_map_entry(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/certificateMaps/*/certificateMapEntries/*}' % client.transport._host, args[1])

def test_get_certificate_map_entry_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_certificate_map_entry(certificate_manager.GetCertificateMapEntryRequest(), name='name_value')

def test_get_certificate_map_entry_rest_error():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.CreateCertificateMapEntryRequest, dict])
def test_create_certificate_map_entry_rest(request_type):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/certificateMaps/sample3'}
    request_init['certificate_map_entry'] = {'name': 'name_value', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'hostname': 'hostname_value', 'matcher': 1, 'certificates': ['certificates_value1', 'certificates_value2'], 'state': 1}
    test_field = certificate_manager.CreateCertificateMapEntryRequest.meta.fields['certificate_map_entry']

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
    for (field, value) in request_init['certificate_map_entry'].items():
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
                for i in range(0, len(request_init['certificate_map_entry'][field])):
                    del request_init['certificate_map_entry'][field][i][subfield]
            else:
                del request_init['certificate_map_entry'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_certificate_map_entry(request)
    assert response.operation.name == 'operations/spam'

def test_create_certificate_map_entry_rest_required_fields(request_type=certificate_manager.CreateCertificateMapEntryRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['certificate_map_entry_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'certificateMapEntryId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_certificate_map_entry._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'certificateMapEntryId' in jsonified_request
    assert jsonified_request['certificateMapEntryId'] == request_init['certificate_map_entry_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['certificateMapEntryId'] = 'certificate_map_entry_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_certificate_map_entry._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('certificate_map_entry_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'certificateMapEntryId' in jsonified_request
    assert jsonified_request['certificateMapEntryId'] == 'certificate_map_entry_id_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_certificate_map_entry(request)
            expected_params = [('certificateMapEntryId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_certificate_map_entry_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_certificate_map_entry._get_unset_required_fields({})
    assert set(unset_fields) == set(('certificateMapEntryId',)) & set(('parent', 'certificateMapEntryId', 'certificateMapEntry'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_certificate_map_entry_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_create_certificate_map_entry') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_create_certificate_map_entry') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.CreateCertificateMapEntryRequest.pb(certificate_manager.CreateCertificateMapEntryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = certificate_manager.CreateCertificateMapEntryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_certificate_map_entry(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_certificate_map_entry_rest_bad_request(transport: str='rest', request_type=certificate_manager.CreateCertificateMapEntryRequest):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/certificateMaps/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_certificate_map_entry(request)

def test_create_certificate_map_entry_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/certificateMaps/sample3'}
        mock_args = dict(parent='parent_value', certificate_map_entry=certificate_manager.CertificateMapEntry(name='name_value'), certificate_map_entry_id='certificate_map_entry_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_certificate_map_entry(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/certificateMaps/*}/certificateMapEntries' % client.transport._host, args[1])

def test_create_certificate_map_entry_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_certificate_map_entry(certificate_manager.CreateCertificateMapEntryRequest(), parent='parent_value', certificate_map_entry=certificate_manager.CertificateMapEntry(name='name_value'), certificate_map_entry_id='certificate_map_entry_id_value')

def test_create_certificate_map_entry_rest_error():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.UpdateCertificateMapEntryRequest, dict])
def test_update_certificate_map_entry_rest(request_type):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'certificate_map_entry': {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3/certificateMapEntries/sample4'}}
    request_init['certificate_map_entry'] = {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3/certificateMapEntries/sample4', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'hostname': 'hostname_value', 'matcher': 1, 'certificates': ['certificates_value1', 'certificates_value2'], 'state': 1}
    test_field = certificate_manager.UpdateCertificateMapEntryRequest.meta.fields['certificate_map_entry']

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
    for (field, value) in request_init['certificate_map_entry'].items():
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
                for i in range(0, len(request_init['certificate_map_entry'][field])):
                    del request_init['certificate_map_entry'][field][i][subfield]
            else:
                del request_init['certificate_map_entry'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_certificate_map_entry(request)
    assert response.operation.name == 'operations/spam'

def test_update_certificate_map_entry_rest_required_fields(request_type=certificate_manager.UpdateCertificateMapEntryRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_certificate_map_entry._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_certificate_map_entry._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_certificate_map_entry(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_certificate_map_entry_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_certificate_map_entry._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('certificateMapEntry', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_certificate_map_entry_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_update_certificate_map_entry') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_update_certificate_map_entry') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.UpdateCertificateMapEntryRequest.pb(certificate_manager.UpdateCertificateMapEntryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = certificate_manager.UpdateCertificateMapEntryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_certificate_map_entry(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_certificate_map_entry_rest_bad_request(transport: str='rest', request_type=certificate_manager.UpdateCertificateMapEntryRequest):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'certificate_map_entry': {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3/certificateMapEntries/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_certificate_map_entry(request)

def test_update_certificate_map_entry_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'certificate_map_entry': {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3/certificateMapEntries/sample4'}}
        mock_args = dict(certificate_map_entry=certificate_manager.CertificateMapEntry(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_certificate_map_entry(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{certificate_map_entry.name=projects/*/locations/*/certificateMaps/*/certificateMapEntries/*}' % client.transport._host, args[1])

def test_update_certificate_map_entry_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_certificate_map_entry(certificate_manager.UpdateCertificateMapEntryRequest(), certificate_map_entry=certificate_manager.CertificateMapEntry(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_certificate_map_entry_rest_error():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.DeleteCertificateMapEntryRequest, dict])
def test_delete_certificate_map_entry_rest(request_type):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3/certificateMapEntries/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_certificate_map_entry(request)
    assert response.operation.name == 'operations/spam'

def test_delete_certificate_map_entry_rest_required_fields(request_type=certificate_manager.DeleteCertificateMapEntryRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_certificate_map_entry._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_certificate_map_entry._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_certificate_map_entry(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_certificate_map_entry_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_certificate_map_entry._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_certificate_map_entry_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_delete_certificate_map_entry') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_delete_certificate_map_entry') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.DeleteCertificateMapEntryRequest.pb(certificate_manager.DeleteCertificateMapEntryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = certificate_manager.DeleteCertificateMapEntryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_certificate_map_entry(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_certificate_map_entry_rest_bad_request(transport: str='rest', request_type=certificate_manager.DeleteCertificateMapEntryRequest):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3/certificateMapEntries/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_certificate_map_entry(request)

def test_delete_certificate_map_entry_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateMaps/sample3/certificateMapEntries/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_certificate_map_entry(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/certificateMaps/*/certificateMapEntries/*}' % client.transport._host, args[1])

def test_delete_certificate_map_entry_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_certificate_map_entry(certificate_manager.DeleteCertificateMapEntryRequest(), name='name_value')

def test_delete_certificate_map_entry_rest_error():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.ListDnsAuthorizationsRequest, dict])
def test_list_dns_authorizations_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.ListDnsAuthorizationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.ListDnsAuthorizationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_dns_authorizations(request)
    assert isinstance(response, pagers.ListDnsAuthorizationsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_dns_authorizations_rest_required_fields(request_type=certificate_manager.ListDnsAuthorizationsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_dns_authorizations._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_dns_authorizations._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = certificate_manager.ListDnsAuthorizationsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = certificate_manager.ListDnsAuthorizationsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_dns_authorizations(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_dns_authorizations_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_dns_authorizations._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_dns_authorizations_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_list_dns_authorizations') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_list_dns_authorizations') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.ListDnsAuthorizationsRequest.pb(certificate_manager.ListDnsAuthorizationsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = certificate_manager.ListDnsAuthorizationsResponse.to_json(certificate_manager.ListDnsAuthorizationsResponse())
        request = certificate_manager.ListDnsAuthorizationsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = certificate_manager.ListDnsAuthorizationsResponse()
        client.list_dns_authorizations(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_dns_authorizations_rest_bad_request(transport: str='rest', request_type=certificate_manager.ListDnsAuthorizationsRequest):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_dns_authorizations(request)

def test_list_dns_authorizations_rest_flattened():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.ListDnsAuthorizationsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.ListDnsAuthorizationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_dns_authorizations(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/dnsAuthorizations' % client.transport._host, args[1])

def test_list_dns_authorizations_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_dns_authorizations(certificate_manager.ListDnsAuthorizationsRequest(), parent='parent_value')

def test_list_dns_authorizations_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization()], next_page_token='abc'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[], next_page_token='def'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization()], next_page_token='ghi'), certificate_manager.ListDnsAuthorizationsResponse(dns_authorizations=[certificate_manager.DnsAuthorization(), certificate_manager.DnsAuthorization()]))
        response = response + response
        response = tuple((certificate_manager.ListDnsAuthorizationsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_dns_authorizations(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, certificate_manager.DnsAuthorization) for i in results))
        pages = list(client.list_dns_authorizations(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [certificate_manager.GetDnsAuthorizationRequest, dict])
def test_get_dns_authorization_rest(request_type):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/dnsAuthorizations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.DnsAuthorization(name='name_value', description='description_value', domain='domain_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.DnsAuthorization.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_dns_authorization(request)
    assert isinstance(response, certificate_manager.DnsAuthorization)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.domain == 'domain_value'

def test_get_dns_authorization_rest_required_fields(request_type=certificate_manager.GetDnsAuthorizationRequest):
    if False:
        return 10
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_dns_authorization._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_dns_authorization._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = certificate_manager.DnsAuthorization()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = certificate_manager.DnsAuthorization.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_dns_authorization(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_dns_authorization_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_dns_authorization._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_dns_authorization_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_get_dns_authorization') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_get_dns_authorization') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.GetDnsAuthorizationRequest.pb(certificate_manager.GetDnsAuthorizationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = certificate_manager.DnsAuthorization.to_json(certificate_manager.DnsAuthorization())
        request = certificate_manager.GetDnsAuthorizationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = certificate_manager.DnsAuthorization()
        client.get_dns_authorization(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_dns_authorization_rest_bad_request(transport: str='rest', request_type=certificate_manager.GetDnsAuthorizationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/dnsAuthorizations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_dns_authorization(request)

def test_get_dns_authorization_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_manager.DnsAuthorization()
        sample_request = {'name': 'projects/sample1/locations/sample2/dnsAuthorizations/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_manager.DnsAuthorization.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_dns_authorization(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/dnsAuthorizations/*}' % client.transport._host, args[1])

def test_get_dns_authorization_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_dns_authorization(certificate_manager.GetDnsAuthorizationRequest(), name='name_value')

def test_get_dns_authorization_rest_error():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.CreateDnsAuthorizationRequest, dict])
def test_create_dns_authorization_rest(request_type):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['dns_authorization'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'domain': 'domain_value', 'dns_resource_record': {'name': 'name_value', 'type_': 'type__value', 'data': 'data_value'}}
    test_field = certificate_manager.CreateDnsAuthorizationRequest.meta.fields['dns_authorization']

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
    for (field, value) in request_init['dns_authorization'].items():
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
                for i in range(0, len(request_init['dns_authorization'][field])):
                    del request_init['dns_authorization'][field][i][subfield]
            else:
                del request_init['dns_authorization'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_dns_authorization(request)
    assert response.operation.name == 'operations/spam'

def test_create_dns_authorization_rest_required_fields(request_type=certificate_manager.CreateDnsAuthorizationRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['dns_authorization_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'dnsAuthorizationId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_dns_authorization._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'dnsAuthorizationId' in jsonified_request
    assert jsonified_request['dnsAuthorizationId'] == request_init['dns_authorization_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['dnsAuthorizationId'] = 'dns_authorization_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_dns_authorization._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('dns_authorization_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'dnsAuthorizationId' in jsonified_request
    assert jsonified_request['dnsAuthorizationId'] == 'dns_authorization_id_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_dns_authorization(request)
            expected_params = [('dnsAuthorizationId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_dns_authorization_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_dns_authorization._get_unset_required_fields({})
    assert set(unset_fields) == set(('dnsAuthorizationId',)) & set(('parent', 'dnsAuthorizationId', 'dnsAuthorization'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_dns_authorization_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_create_dns_authorization') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_create_dns_authorization') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.CreateDnsAuthorizationRequest.pb(certificate_manager.CreateDnsAuthorizationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = certificate_manager.CreateDnsAuthorizationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_dns_authorization(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_dns_authorization_rest_bad_request(transport: str='rest', request_type=certificate_manager.CreateDnsAuthorizationRequest):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_dns_authorization(request)

def test_create_dns_authorization_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', dns_authorization=certificate_manager.DnsAuthorization(name='name_value'), dns_authorization_id='dns_authorization_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_dns_authorization(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/dnsAuthorizations' % client.transport._host, args[1])

def test_create_dns_authorization_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_dns_authorization(certificate_manager.CreateDnsAuthorizationRequest(), parent='parent_value', dns_authorization=certificate_manager.DnsAuthorization(name='name_value'), dns_authorization_id='dns_authorization_id_value')

def test_create_dns_authorization_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.UpdateDnsAuthorizationRequest, dict])
def test_update_dns_authorization_rest(request_type):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'dns_authorization': {'name': 'projects/sample1/locations/sample2/dnsAuthorizations/sample3'}}
    request_init['dns_authorization'] = {'name': 'projects/sample1/locations/sample2/dnsAuthorizations/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'domain': 'domain_value', 'dns_resource_record': {'name': 'name_value', 'type_': 'type__value', 'data': 'data_value'}}
    test_field = certificate_manager.UpdateDnsAuthorizationRequest.meta.fields['dns_authorization']

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
    for (field, value) in request_init['dns_authorization'].items():
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
                for i in range(0, len(request_init['dns_authorization'][field])):
                    del request_init['dns_authorization'][field][i][subfield]
            else:
                del request_init['dns_authorization'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_dns_authorization(request)
    assert response.operation.name == 'operations/spam'

def test_update_dns_authorization_rest_required_fields(request_type=certificate_manager.UpdateDnsAuthorizationRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_dns_authorization._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_dns_authorization._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_dns_authorization(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_dns_authorization_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_dns_authorization._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('dnsAuthorization', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_dns_authorization_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_update_dns_authorization') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_update_dns_authorization') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.UpdateDnsAuthorizationRequest.pb(certificate_manager.UpdateDnsAuthorizationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = certificate_manager.UpdateDnsAuthorizationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_dns_authorization(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_dns_authorization_rest_bad_request(transport: str='rest', request_type=certificate_manager.UpdateDnsAuthorizationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'dns_authorization': {'name': 'projects/sample1/locations/sample2/dnsAuthorizations/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_dns_authorization(request)

def test_update_dns_authorization_rest_flattened():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'dns_authorization': {'name': 'projects/sample1/locations/sample2/dnsAuthorizations/sample3'}}
        mock_args = dict(dns_authorization=certificate_manager.DnsAuthorization(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_dns_authorization(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{dns_authorization.name=projects/*/locations/*/dnsAuthorizations/*}' % client.transport._host, args[1])

def test_update_dns_authorization_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_dns_authorization(certificate_manager.UpdateDnsAuthorizationRequest(), dns_authorization=certificate_manager.DnsAuthorization(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_dns_authorization_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_manager.DeleteDnsAuthorizationRequest, dict])
def test_delete_dns_authorization_rest(request_type):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/dnsAuthorizations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_dns_authorization(request)
    assert response.operation.name == 'operations/spam'

def test_delete_dns_authorization_rest_required_fields(request_type=certificate_manager.DeleteDnsAuthorizationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_dns_authorization._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_dns_authorization._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_dns_authorization(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_dns_authorization_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_dns_authorization._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_dns_authorization_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_delete_dns_authorization') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_delete_dns_authorization') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_manager.DeleteDnsAuthorizationRequest.pb(certificate_manager.DeleteDnsAuthorizationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = certificate_manager.DeleteDnsAuthorizationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_dns_authorization(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_dns_authorization_rest_bad_request(transport: str='rest', request_type=certificate_manager.DeleteDnsAuthorizationRequest):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/dnsAuthorizations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_dns_authorization(request)

def test_delete_dns_authorization_rest_flattened():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/dnsAuthorizations/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_dns_authorization(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/dnsAuthorizations/*}' % client.transport._host, args[1])

def test_delete_dns_authorization_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_dns_authorization(certificate_manager.DeleteDnsAuthorizationRequest(), name='name_value')

def test_delete_dns_authorization_rest_error():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_issuance_config.ListCertificateIssuanceConfigsRequest, dict])
def test_list_certificate_issuance_configs_rest(request_type):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_issuance_config.ListCertificateIssuanceConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_issuance_config.ListCertificateIssuanceConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_certificate_issuance_configs(request)
    assert isinstance(response, pagers.ListCertificateIssuanceConfigsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_certificate_issuance_configs_rest_required_fields(request_type=certificate_issuance_config.ListCertificateIssuanceConfigsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_certificate_issuance_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_certificate_issuance_configs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = certificate_issuance_config.ListCertificateIssuanceConfigsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = certificate_issuance_config.ListCertificateIssuanceConfigsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_certificate_issuance_configs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_certificate_issuance_configs_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_certificate_issuance_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_certificate_issuance_configs_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_list_certificate_issuance_configs') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_list_certificate_issuance_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_issuance_config.ListCertificateIssuanceConfigsRequest.pb(certificate_issuance_config.ListCertificateIssuanceConfigsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = certificate_issuance_config.ListCertificateIssuanceConfigsResponse.to_json(certificate_issuance_config.ListCertificateIssuanceConfigsResponse())
        request = certificate_issuance_config.ListCertificateIssuanceConfigsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = certificate_issuance_config.ListCertificateIssuanceConfigsResponse()
        client.list_certificate_issuance_configs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_certificate_issuance_configs_rest_bad_request(transport: str='rest', request_type=certificate_issuance_config.ListCertificateIssuanceConfigsRequest):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_certificate_issuance_configs(request)

def test_list_certificate_issuance_configs_rest_flattened():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_issuance_config.ListCertificateIssuanceConfigsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_issuance_config.ListCertificateIssuanceConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_certificate_issuance_configs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/certificateIssuanceConfigs' % client.transport._host, args[1])

def test_list_certificate_issuance_configs_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_certificate_issuance_configs(certificate_issuance_config.ListCertificateIssuanceConfigsRequest(), parent='parent_value')

def test_list_certificate_issuance_configs_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig()], next_page_token='abc'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[], next_page_token='def'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig()], next_page_token='ghi'), certificate_issuance_config.ListCertificateIssuanceConfigsResponse(certificate_issuance_configs=[certificate_issuance_config.CertificateIssuanceConfig(), certificate_issuance_config.CertificateIssuanceConfig()]))
        response = response + response
        response = tuple((certificate_issuance_config.ListCertificateIssuanceConfigsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_certificate_issuance_configs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, certificate_issuance_config.CertificateIssuanceConfig) for i in results))
        pages = list(client.list_certificate_issuance_configs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [certificate_issuance_config.GetCertificateIssuanceConfigRequest, dict])
def test_get_certificate_issuance_config_rest(request_type):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateIssuanceConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_issuance_config.CertificateIssuanceConfig(name='name_value', description='description_value', rotation_window_percentage=2788, key_algorithm=certificate_issuance_config.CertificateIssuanceConfig.KeyAlgorithm.RSA_2048)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_issuance_config.CertificateIssuanceConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_certificate_issuance_config(request)
    assert isinstance(response, certificate_issuance_config.CertificateIssuanceConfig)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.rotation_window_percentage == 2788
    assert response.key_algorithm == certificate_issuance_config.CertificateIssuanceConfig.KeyAlgorithm.RSA_2048

def test_get_certificate_issuance_config_rest_required_fields(request_type=certificate_issuance_config.GetCertificateIssuanceConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_certificate_issuance_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_certificate_issuance_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = certificate_issuance_config.CertificateIssuanceConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = certificate_issuance_config.CertificateIssuanceConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_certificate_issuance_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_certificate_issuance_config_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_certificate_issuance_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_certificate_issuance_config_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_get_certificate_issuance_config') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_get_certificate_issuance_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_issuance_config.GetCertificateIssuanceConfigRequest.pb(certificate_issuance_config.GetCertificateIssuanceConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = certificate_issuance_config.CertificateIssuanceConfig.to_json(certificate_issuance_config.CertificateIssuanceConfig())
        request = certificate_issuance_config.GetCertificateIssuanceConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = certificate_issuance_config.CertificateIssuanceConfig()
        client.get_certificate_issuance_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_certificate_issuance_config_rest_bad_request(transport: str='rest', request_type=certificate_issuance_config.GetCertificateIssuanceConfigRequest):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateIssuanceConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_certificate_issuance_config(request)

def test_get_certificate_issuance_config_rest_flattened():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate_issuance_config.CertificateIssuanceConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateIssuanceConfigs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate_issuance_config.CertificateIssuanceConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_certificate_issuance_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/certificateIssuanceConfigs/*}' % client.transport._host, args[1])

def test_get_certificate_issuance_config_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_certificate_issuance_config(certificate_issuance_config.GetCertificateIssuanceConfigRequest(), name='name_value')

def test_get_certificate_issuance_config_rest_error():
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest, dict])
def test_create_certificate_issuance_config_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['certificate_issuance_config'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'description': 'description_value', 'certificate_authority_config': {'certificate_authority_service_config': {'ca_pool': 'ca_pool_value'}}, 'lifetime': {'seconds': 751, 'nanos': 543}, 'rotation_window_percentage': 2788, 'key_algorithm': 1}
    test_field = gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest.meta.fields['certificate_issuance_config']

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
    for (field, value) in request_init['certificate_issuance_config'].items():
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
                for i in range(0, len(request_init['certificate_issuance_config'][field])):
                    del request_init['certificate_issuance_config'][field][i][subfield]
            else:
                del request_init['certificate_issuance_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_certificate_issuance_config(request)
    assert response.operation.name == 'operations/spam'

def test_create_certificate_issuance_config_rest_required_fields(request_type=gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['certificate_issuance_config_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'certificateIssuanceConfigId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_certificate_issuance_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'certificateIssuanceConfigId' in jsonified_request
    assert jsonified_request['certificateIssuanceConfigId'] == request_init['certificate_issuance_config_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['certificateIssuanceConfigId'] = 'certificate_issuance_config_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_certificate_issuance_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('certificate_issuance_config_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'certificateIssuanceConfigId' in jsonified_request
    assert jsonified_request['certificateIssuanceConfigId'] == 'certificate_issuance_config_id_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_certificate_issuance_config(request)
            expected_params = [('certificateIssuanceConfigId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_certificate_issuance_config_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_certificate_issuance_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('certificateIssuanceConfigId',)) & set(('parent', 'certificateIssuanceConfigId', 'certificateIssuanceConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_certificate_issuance_config_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_create_certificate_issuance_config') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_create_certificate_issuance_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest.pb(gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_certificate_issuance_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_certificate_issuance_config_rest_bad_request(transport: str='rest', request_type=gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_certificate_issuance_config(request)

def test_create_certificate_issuance_config_rest_flattened():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', certificate_issuance_config=gcc_certificate_issuance_config.CertificateIssuanceConfig(name='name_value'), certificate_issuance_config_id='certificate_issuance_config_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_certificate_issuance_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/certificateIssuanceConfigs' % client.transport._host, args[1])

def test_create_certificate_issuance_config_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_certificate_issuance_config(gcc_certificate_issuance_config.CreateCertificateIssuanceConfigRequest(), parent='parent_value', certificate_issuance_config=gcc_certificate_issuance_config.CertificateIssuanceConfig(name='name_value'), certificate_issuance_config_id='certificate_issuance_config_id_value')

def test_create_certificate_issuance_config_rest_error():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [certificate_issuance_config.DeleteCertificateIssuanceConfigRequest, dict])
def test_delete_certificate_issuance_config_rest(request_type):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateIssuanceConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_certificate_issuance_config(request)
    assert response.operation.name == 'operations/spam'

def test_delete_certificate_issuance_config_rest_required_fields(request_type=certificate_issuance_config.DeleteCertificateIssuanceConfigRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CertificateManagerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_certificate_issuance_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_certificate_issuance_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_certificate_issuance_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_certificate_issuance_config_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_certificate_issuance_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_certificate_issuance_config_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateManagerRestInterceptor())
    client = CertificateManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateManagerRestInterceptor, 'post_delete_certificate_issuance_config') as post, mock.patch.object(transports.CertificateManagerRestInterceptor, 'pre_delete_certificate_issuance_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = certificate_issuance_config.DeleteCertificateIssuanceConfigRequest.pb(certificate_issuance_config.DeleteCertificateIssuanceConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = certificate_issuance_config.DeleteCertificateIssuanceConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_certificate_issuance_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_certificate_issuance_config_rest_bad_request(transport: str='rest', request_type=certificate_issuance_config.DeleteCertificateIssuanceConfigRequest):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateIssuanceConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_certificate_issuance_config(request)

def test_delete_certificate_issuance_config_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateIssuanceConfigs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_certificate_issuance_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/certificateIssuanceConfigs/*}' % client.transport._host, args[1])

def test_delete_certificate_issuance_config_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_certificate_issuance_config(certificate_issuance_config.DeleteCertificateIssuanceConfigRequest(), name='name_value')

def test_delete_certificate_issuance_config_rest_error():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CertificateManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CertificateManagerClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CertificateManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CertificateManagerClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CertificateManagerClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CertificateManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CertificateManagerClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.CertificateManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CertificateManagerClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.CertificateManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CertificateManagerGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CertificateManagerGrpcTransport, transports.CertificateManagerGrpcAsyncIOTransport, transports.CertificateManagerRestTransport])
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
        for i in range(10):
            print('nop')
    transport = CertificateManagerClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CertificateManagerGrpcTransport)

def test_certificate_manager_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CertificateManagerTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_certificate_manager_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.certificate_manager_v1.services.certificate_manager.transports.CertificateManagerTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CertificateManagerTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_certificates', 'get_certificate', 'create_certificate', 'update_certificate', 'delete_certificate', 'list_certificate_maps', 'get_certificate_map', 'create_certificate_map', 'update_certificate_map', 'delete_certificate_map', 'list_certificate_map_entries', 'get_certificate_map_entry', 'create_certificate_map_entry', 'update_certificate_map_entry', 'delete_certificate_map_entry', 'list_dns_authorizations', 'get_dns_authorization', 'create_dns_authorization', 'update_dns_authorization', 'delete_dns_authorization', 'list_certificate_issuance_configs', 'get_certificate_issuance_config', 'create_certificate_issuance_config', 'delete_certificate_issuance_config', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_certificate_manager_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.certificate_manager_v1.services.certificate_manager.transports.CertificateManagerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CertificateManagerTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_certificate_manager_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.certificate_manager_v1.services.certificate_manager.transports.CertificateManagerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CertificateManagerTransport()
        adc.assert_called_once()

def test_certificate_manager_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CertificateManagerClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CertificateManagerGrpcTransport, transports.CertificateManagerGrpcAsyncIOTransport])
def test_certificate_manager_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CertificateManagerGrpcTransport, transports.CertificateManagerGrpcAsyncIOTransport, transports.CertificateManagerRestTransport])
def test_certificate_manager_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CertificateManagerGrpcTransport, grpc_helpers), (transports.CertificateManagerGrpcAsyncIOTransport, grpc_helpers_async)])
def test_certificate_manager_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('certificatemanager.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='certificatemanager.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CertificateManagerGrpcTransport, transports.CertificateManagerGrpcAsyncIOTransport])
def test_certificate_manager_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_certificate_manager_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.CertificateManagerRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_certificate_manager_rest_lro_client():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_certificate_manager_host_no_port(transport_name):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='certificatemanager.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('certificatemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://certificatemanager.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_certificate_manager_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='certificatemanager.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('certificatemanager.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://certificatemanager.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_certificate_manager_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = CertificateManagerClient(credentials=creds1, transport=transport_name)
    client2 = CertificateManagerClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_certificates._session
    session2 = client2.transport.list_certificates._session
    assert session1 != session2
    session1 = client1.transport.get_certificate._session
    session2 = client2.transport.get_certificate._session
    assert session1 != session2
    session1 = client1.transport.create_certificate._session
    session2 = client2.transport.create_certificate._session
    assert session1 != session2
    session1 = client1.transport.update_certificate._session
    session2 = client2.transport.update_certificate._session
    assert session1 != session2
    session1 = client1.transport.delete_certificate._session
    session2 = client2.transport.delete_certificate._session
    assert session1 != session2
    session1 = client1.transport.list_certificate_maps._session
    session2 = client2.transport.list_certificate_maps._session
    assert session1 != session2
    session1 = client1.transport.get_certificate_map._session
    session2 = client2.transport.get_certificate_map._session
    assert session1 != session2
    session1 = client1.transport.create_certificate_map._session
    session2 = client2.transport.create_certificate_map._session
    assert session1 != session2
    session1 = client1.transport.update_certificate_map._session
    session2 = client2.transport.update_certificate_map._session
    assert session1 != session2
    session1 = client1.transport.delete_certificate_map._session
    session2 = client2.transport.delete_certificate_map._session
    assert session1 != session2
    session1 = client1.transport.list_certificate_map_entries._session
    session2 = client2.transport.list_certificate_map_entries._session
    assert session1 != session2
    session1 = client1.transport.get_certificate_map_entry._session
    session2 = client2.transport.get_certificate_map_entry._session
    assert session1 != session2
    session1 = client1.transport.create_certificate_map_entry._session
    session2 = client2.transport.create_certificate_map_entry._session
    assert session1 != session2
    session1 = client1.transport.update_certificate_map_entry._session
    session2 = client2.transport.update_certificate_map_entry._session
    assert session1 != session2
    session1 = client1.transport.delete_certificate_map_entry._session
    session2 = client2.transport.delete_certificate_map_entry._session
    assert session1 != session2
    session1 = client1.transport.list_dns_authorizations._session
    session2 = client2.transport.list_dns_authorizations._session
    assert session1 != session2
    session1 = client1.transport.get_dns_authorization._session
    session2 = client2.transport.get_dns_authorization._session
    assert session1 != session2
    session1 = client1.transport.create_dns_authorization._session
    session2 = client2.transport.create_dns_authorization._session
    assert session1 != session2
    session1 = client1.transport.update_dns_authorization._session
    session2 = client2.transport.update_dns_authorization._session
    assert session1 != session2
    session1 = client1.transport.delete_dns_authorization._session
    session2 = client2.transport.delete_dns_authorization._session
    assert session1 != session2
    session1 = client1.transport.list_certificate_issuance_configs._session
    session2 = client2.transport.list_certificate_issuance_configs._session
    assert session1 != session2
    session1 = client1.transport.get_certificate_issuance_config._session
    session2 = client2.transport.get_certificate_issuance_config._session
    assert session1 != session2
    session1 = client1.transport.create_certificate_issuance_config._session
    session2 = client2.transport.create_certificate_issuance_config._session
    assert session1 != session2
    session1 = client1.transport.delete_certificate_issuance_config._session
    session2 = client2.transport.delete_certificate_issuance_config._session
    assert session1 != session2

def test_certificate_manager_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CertificateManagerGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_certificate_manager_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CertificateManagerGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CertificateManagerGrpcTransport, transports.CertificateManagerGrpcAsyncIOTransport])
def test_certificate_manager_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.CertificateManagerGrpcTransport, transports.CertificateManagerGrpcAsyncIOTransport])
def test_certificate_manager_transport_channel_mtls_with_adc(transport_class):
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

def test_certificate_manager_grpc_lro_client():
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_certificate_manager_grpc_lro_async_client():
    if False:
        print('Hello World!')
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_ca_pool_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    ca_pool = 'whelk'
    expected = 'projects/{project}/locations/{location}/caPools/{ca_pool}'.format(project=project, location=location, ca_pool=ca_pool)
    actual = CertificateManagerClient.ca_pool_path(project, location, ca_pool)
    assert expected == actual

def test_parse_ca_pool_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'ca_pool': 'nudibranch'}
    path = CertificateManagerClient.ca_pool_path(**expected)
    actual = CertificateManagerClient.parse_ca_pool_path(path)
    assert expected == actual

def test_certificate_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    location = 'mussel'
    certificate = 'winkle'
    expected = 'projects/{project}/locations/{location}/certificates/{certificate}'.format(project=project, location=location, certificate=certificate)
    actual = CertificateManagerClient.certificate_path(project, location, certificate)
    assert expected == actual

def test_parse_certificate_path():
    if False:
        return 10
    expected = {'project': 'nautilus', 'location': 'scallop', 'certificate': 'abalone'}
    path = CertificateManagerClient.certificate_path(**expected)
    actual = CertificateManagerClient.parse_certificate_path(path)
    assert expected == actual

def test_certificate_issuance_config_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    certificate_issuance_config = 'whelk'
    expected = 'projects/{project}/locations/{location}/certificateIssuanceConfigs/{certificate_issuance_config}'.format(project=project, location=location, certificate_issuance_config=certificate_issuance_config)
    actual = CertificateManagerClient.certificate_issuance_config_path(project, location, certificate_issuance_config)
    assert expected == actual

def test_parse_certificate_issuance_config_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'location': 'oyster', 'certificate_issuance_config': 'nudibranch'}
    path = CertificateManagerClient.certificate_issuance_config_path(**expected)
    actual = CertificateManagerClient.parse_certificate_issuance_config_path(path)
    assert expected == actual

def test_certificate_map_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    certificate_map = 'winkle'
    expected = 'projects/{project}/locations/{location}/certificateMaps/{certificate_map}'.format(project=project, location=location, certificate_map=certificate_map)
    actual = CertificateManagerClient.certificate_map_path(project, location, certificate_map)
    assert expected == actual

def test_parse_certificate_map_path():
    if False:
        return 10
    expected = {'project': 'nautilus', 'location': 'scallop', 'certificate_map': 'abalone'}
    path = CertificateManagerClient.certificate_map_path(**expected)
    actual = CertificateManagerClient.parse_certificate_map_path(path)
    assert expected == actual

def test_certificate_map_entry_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    certificate_map = 'whelk'
    certificate_map_entry = 'octopus'
    expected = 'projects/{project}/locations/{location}/certificateMaps/{certificate_map}/certificateMapEntries/{certificate_map_entry}'.format(project=project, location=location, certificate_map=certificate_map, certificate_map_entry=certificate_map_entry)
    actual = CertificateManagerClient.certificate_map_entry_path(project, location, certificate_map, certificate_map_entry)
    assert expected == actual

def test_parse_certificate_map_entry_path():
    if False:
        print('Hello World!')
    expected = {'project': 'oyster', 'location': 'nudibranch', 'certificate_map': 'cuttlefish', 'certificate_map_entry': 'mussel'}
    path = CertificateManagerClient.certificate_map_entry_path(**expected)
    actual = CertificateManagerClient.parse_certificate_map_entry_path(path)
    assert expected == actual

def test_dns_authorization_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    dns_authorization = 'scallop'
    expected = 'projects/{project}/locations/{location}/dnsAuthorizations/{dns_authorization}'.format(project=project, location=location, dns_authorization=dns_authorization)
    actual = CertificateManagerClient.dns_authorization_path(project, location, dns_authorization)
    assert expected == actual

def test_parse_dns_authorization_path():
    if False:
        print('Hello World!')
    expected = {'project': 'abalone', 'location': 'squid', 'dns_authorization': 'clam'}
    path = CertificateManagerClient.dns_authorization_path(**expected)
    actual = CertificateManagerClient.parse_dns_authorization_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CertificateManagerClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'octopus'}
    path = CertificateManagerClient.common_billing_account_path(**expected)
    actual = CertificateManagerClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CertificateManagerClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'nudibranch'}
    path = CertificateManagerClient.common_folder_path(**expected)
    actual = CertificateManagerClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CertificateManagerClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'mussel'}
    path = CertificateManagerClient.common_organization_path(**expected)
    actual = CertificateManagerClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = CertificateManagerClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nautilus'}
    path = CertificateManagerClient.common_project_path(**expected)
    actual = CertificateManagerClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CertificateManagerClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'squid', 'location': 'clam'}
    path = CertificateManagerClient.common_location_path(**expected)
    actual = CertificateManagerClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CertificateManagerTransport, '_prep_wrapped_messages') as prep:
        client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CertificateManagerTransport, '_prep_wrapped_messages') as prep:
        transport_class = CertificateManagerClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_delete_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.DeleteOperationRequest):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/operations/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.DeleteOperationRequest, dict])
def test_delete_operation_rest(request_type):
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/operations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = '{}'
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_operation(request)
    assert response is None

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_delete_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = operations_pb2.DeleteOperationRequest()
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert response is None

@pytest.mark.asyncio
async def test_delete_operation_async(transport: str='grpc'):
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = operations_pb2.DeleteOperationRequest()
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert response is None

def test_delete_operation_field_headers():
    if False:
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = operations_pb2.DeleteOperationRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_operation_field_headers_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = operations_pb2.DeleteOperationRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

def test_delete_operation_from_dict():
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = CertificateManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = CertificateManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CertificateManagerClient, transports.CertificateManagerGrpcTransport), (CertificateManagerAsyncClient, transports.CertificateManagerGrpcAsyncIOTransport)])
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