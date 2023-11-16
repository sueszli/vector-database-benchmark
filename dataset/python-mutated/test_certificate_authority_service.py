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
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.protobuf import wrappers_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.security.privateca_v1beta1.services.certificate_authority_service import CertificateAuthorityServiceAsyncClient, CertificateAuthorityServiceClient, pagers, transports
from google.cloud.security.privateca_v1beta1.types import resources, service

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        i = 10
        return i + 15
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert CertificateAuthorityServiceClient._get_default_mtls_endpoint(None) is None
    assert CertificateAuthorityServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CertificateAuthorityServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CertificateAuthorityServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CertificateAuthorityServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CertificateAuthorityServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CertificateAuthorityServiceClient, 'grpc'), (CertificateAuthorityServiceAsyncClient, 'grpc_asyncio'), (CertificateAuthorityServiceClient, 'rest')])
def test_certificate_authority_service_client_from_service_account_info(client_class, transport_name):
    if False:
        while True:
            i = 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('privateca.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://privateca.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CertificateAuthorityServiceGrpcTransport, 'grpc'), (transports.CertificateAuthorityServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.CertificateAuthorityServiceRestTransport, 'rest')])
def test_certificate_authority_service_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(CertificateAuthorityServiceClient, 'grpc'), (CertificateAuthorityServiceAsyncClient, 'grpc_asyncio'), (CertificateAuthorityServiceClient, 'rest')])
def test_certificate_authority_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('privateca.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://privateca.googleapis.com')

def test_certificate_authority_service_client_get_transport_class():
    if False:
        return 10
    transport = CertificateAuthorityServiceClient.get_transport_class()
    available_transports = [transports.CertificateAuthorityServiceGrpcTransport, transports.CertificateAuthorityServiceRestTransport]
    assert transport in available_transports
    transport = CertificateAuthorityServiceClient.get_transport_class('grpc')
    assert transport == transports.CertificateAuthorityServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CertificateAuthorityServiceClient, transports.CertificateAuthorityServiceGrpcTransport, 'grpc'), (CertificateAuthorityServiceAsyncClient, transports.CertificateAuthorityServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (CertificateAuthorityServiceClient, transports.CertificateAuthorityServiceRestTransport, 'rest')])
@mock.patch.object(CertificateAuthorityServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CertificateAuthorityServiceClient))
@mock.patch.object(CertificateAuthorityServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CertificateAuthorityServiceAsyncClient))
def test_certificate_authority_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(CertificateAuthorityServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CertificateAuthorityServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CertificateAuthorityServiceClient, transports.CertificateAuthorityServiceGrpcTransport, 'grpc', 'true'), (CertificateAuthorityServiceAsyncClient, transports.CertificateAuthorityServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CertificateAuthorityServiceClient, transports.CertificateAuthorityServiceGrpcTransport, 'grpc', 'false'), (CertificateAuthorityServiceAsyncClient, transports.CertificateAuthorityServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (CertificateAuthorityServiceClient, transports.CertificateAuthorityServiceRestTransport, 'rest', 'true'), (CertificateAuthorityServiceClient, transports.CertificateAuthorityServiceRestTransport, 'rest', 'false')])
@mock.patch.object(CertificateAuthorityServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CertificateAuthorityServiceClient))
@mock.patch.object(CertificateAuthorityServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CertificateAuthorityServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_certificate_authority_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [CertificateAuthorityServiceClient, CertificateAuthorityServiceAsyncClient])
@mock.patch.object(CertificateAuthorityServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CertificateAuthorityServiceClient))
@mock.patch.object(CertificateAuthorityServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CertificateAuthorityServiceAsyncClient))
def test_certificate_authority_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CertificateAuthorityServiceClient, transports.CertificateAuthorityServiceGrpcTransport, 'grpc'), (CertificateAuthorityServiceAsyncClient, transports.CertificateAuthorityServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (CertificateAuthorityServiceClient, transports.CertificateAuthorityServiceRestTransport, 'rest')])
def test_certificate_authority_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CertificateAuthorityServiceClient, transports.CertificateAuthorityServiceGrpcTransport, 'grpc', grpc_helpers), (CertificateAuthorityServiceAsyncClient, transports.CertificateAuthorityServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (CertificateAuthorityServiceClient, transports.CertificateAuthorityServiceRestTransport, 'rest', None)])
def test_certificate_authority_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_certificate_authority_service_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.security.privateca_v1beta1.services.certificate_authority_service.transports.CertificateAuthorityServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CertificateAuthorityServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CertificateAuthorityServiceClient, transports.CertificateAuthorityServiceGrpcTransport, 'grpc', grpc_helpers), (CertificateAuthorityServiceAsyncClient, transports.CertificateAuthorityServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_certificate_authority_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('privateca.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='privateca.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [service.CreateCertificateRequest, dict])
def test_create_certificate(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_certificate), '__call__') as call:
        call.return_value = resources.Certificate(name='name_value', pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value'], pem_csr='pem_csr_value')
        response = client.create_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCertificateRequest()
    assert isinstance(response, resources.Certificate)
    assert response.name == 'name_value'
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']

def test_create_certificate_empty_call():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_certificate), '__call__') as call:
        client.create_certificate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCertificateRequest()

@pytest.mark.asyncio
async def test_create_certificate_async(transport: str='grpc_asyncio', request_type=service.CreateCertificateRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Certificate(name='name_value', pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value']))
        response = await client.create_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCertificateRequest()
    assert isinstance(response, resources.Certificate)
    assert response.name == 'name_value'
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']

@pytest.mark.asyncio
async def test_create_certificate_async_from_dict():
    await test_create_certificate_async(request_type=dict)

def test_create_certificate_field_headers():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateCertificateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_certificate), '__call__') as call:
        call.return_value = resources.Certificate()
        client.create_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_certificate_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateCertificateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Certificate())
        await client.create_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_certificate_flattened():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_certificate), '__call__') as call:
        call.return_value = resources.Certificate()
        client.create_certificate(parent='parent_value', certificate=resources.Certificate(name='name_value'), certificate_id='certificate_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].certificate
        mock_val = resources.Certificate(name='name_value')
        assert arg == mock_val
        arg = args[0].certificate_id
        mock_val = 'certificate_id_value'
        assert arg == mock_val

def test_create_certificate_flattened_error():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_certificate(service.CreateCertificateRequest(), parent='parent_value', certificate=resources.Certificate(name='name_value'), certificate_id='certificate_id_value')

@pytest.mark.asyncio
async def test_create_certificate_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_certificate), '__call__') as call:
        call.return_value = resources.Certificate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Certificate())
        response = await client.create_certificate(parent='parent_value', certificate=resources.Certificate(name='name_value'), certificate_id='certificate_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].certificate
        mock_val = resources.Certificate(name='name_value')
        assert arg == mock_val
        arg = args[0].certificate_id
        mock_val = 'certificate_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_certificate_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_certificate(service.CreateCertificateRequest(), parent='parent_value', certificate=resources.Certificate(name='name_value'), certificate_id='certificate_id_value')

@pytest.mark.parametrize('request_type', [service.GetCertificateRequest, dict])
def test_get_certificate(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_certificate), '__call__') as call:
        call.return_value = resources.Certificate(name='name_value', pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value'], pem_csr='pem_csr_value')
        response = client.get_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCertificateRequest()
    assert isinstance(response, resources.Certificate)
    assert response.name == 'name_value'
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']

def test_get_certificate_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_certificate), '__call__') as call:
        client.get_certificate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCertificateRequest()

@pytest.mark.asyncio
async def test_get_certificate_async(transport: str='grpc_asyncio', request_type=service.GetCertificateRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Certificate(name='name_value', pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value']))
        response = await client.get_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCertificateRequest()
    assert isinstance(response, resources.Certificate)
    assert response.name == 'name_value'
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']

@pytest.mark.asyncio
async def test_get_certificate_async_from_dict():
    await test_get_certificate_async(request_type=dict)

def test_get_certificate_field_headers():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetCertificateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_certificate), '__call__') as call:
        call.return_value = resources.Certificate()
        client.get_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_certificate_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetCertificateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Certificate())
        await client.get_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_certificate_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_certificate), '__call__') as call:
        call.return_value = resources.Certificate()
        client.get_certificate(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_certificate_flattened_error():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_certificate(service.GetCertificateRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_certificate_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_certificate), '__call__') as call:
        call.return_value = resources.Certificate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Certificate())
        response = await client.get_certificate(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_certificate_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_certificate(service.GetCertificateRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListCertificatesRequest, dict])
def test_list_certificates(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.return_value = service.ListCertificatesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_certificates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCertificatesRequest()
    assert isinstance(response, pagers.ListCertificatesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_certificates_empty_call():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        client.list_certificates()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCertificatesRequest()

@pytest.mark.asyncio
async def test_list_certificates_async(transport: str='grpc_asyncio', request_type=service.ListCertificatesRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCertificatesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_certificates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCertificatesRequest()
    assert isinstance(response, pagers.ListCertificatesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_certificates_async_from_dict():
    await test_list_certificates_async(request_type=dict)

def test_list_certificates_field_headers():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListCertificatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.return_value = service.ListCertificatesResponse()
        client.list_certificates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_certificates_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListCertificatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCertificatesResponse())
        await client.list_certificates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_certificates_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.return_value = service.ListCertificatesResponse()
        client.list_certificates(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_certificates_flattened_error():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_certificates(service.ListCertificatesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_certificates_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.return_value = service.ListCertificatesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCertificatesResponse())
        response = await client.list_certificates(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_certificates_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_certificates(service.ListCertificatesRequest(), parent='parent_value')

def test_list_certificates_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.side_effect = (service.ListCertificatesResponse(certificates=[resources.Certificate(), resources.Certificate(), resources.Certificate()], next_page_token='abc'), service.ListCertificatesResponse(certificates=[], next_page_token='def'), service.ListCertificatesResponse(certificates=[resources.Certificate()], next_page_token='ghi'), service.ListCertificatesResponse(certificates=[resources.Certificate(), resources.Certificate()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_certificates(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Certificate) for i in results))

def test_list_certificates_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_certificates), '__call__') as call:
        call.side_effect = (service.ListCertificatesResponse(certificates=[resources.Certificate(), resources.Certificate(), resources.Certificate()], next_page_token='abc'), service.ListCertificatesResponse(certificates=[], next_page_token='def'), service.ListCertificatesResponse(certificates=[resources.Certificate()], next_page_token='ghi'), service.ListCertificatesResponse(certificates=[resources.Certificate(), resources.Certificate()]), RuntimeError)
        pages = list(client.list_certificates(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_certificates_async_pager():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_certificates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListCertificatesResponse(certificates=[resources.Certificate(), resources.Certificate(), resources.Certificate()], next_page_token='abc'), service.ListCertificatesResponse(certificates=[], next_page_token='def'), service.ListCertificatesResponse(certificates=[resources.Certificate()], next_page_token='ghi'), service.ListCertificatesResponse(certificates=[resources.Certificate(), resources.Certificate()]), RuntimeError)
        async_pager = await client.list_certificates(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Certificate) for i in responses))

@pytest.mark.asyncio
async def test_list_certificates_async_pages():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_certificates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListCertificatesResponse(certificates=[resources.Certificate(), resources.Certificate(), resources.Certificate()], next_page_token='abc'), service.ListCertificatesResponse(certificates=[], next_page_token='def'), service.ListCertificatesResponse(certificates=[resources.Certificate()], next_page_token='ghi'), service.ListCertificatesResponse(certificates=[resources.Certificate(), resources.Certificate()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_certificates(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.RevokeCertificateRequest, dict])
def test_revoke_certificate(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.revoke_certificate), '__call__') as call:
        call.return_value = resources.Certificate(name='name_value', pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value'], pem_csr='pem_csr_value')
        response = client.revoke_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RevokeCertificateRequest()
    assert isinstance(response, resources.Certificate)
    assert response.name == 'name_value'
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']

def test_revoke_certificate_empty_call():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.revoke_certificate), '__call__') as call:
        client.revoke_certificate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RevokeCertificateRequest()

@pytest.mark.asyncio
async def test_revoke_certificate_async(transport: str='grpc_asyncio', request_type=service.RevokeCertificateRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.revoke_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Certificate(name='name_value', pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value']))
        response = await client.revoke_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RevokeCertificateRequest()
    assert isinstance(response, resources.Certificate)
    assert response.name == 'name_value'
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']

@pytest.mark.asyncio
async def test_revoke_certificate_async_from_dict():
    await test_revoke_certificate_async(request_type=dict)

def test_revoke_certificate_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RevokeCertificateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.revoke_certificate), '__call__') as call:
        call.return_value = resources.Certificate()
        client.revoke_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_revoke_certificate_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RevokeCertificateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.revoke_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Certificate())
        await client.revoke_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_revoke_certificate_flattened():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.revoke_certificate), '__call__') as call:
        call.return_value = resources.Certificate()
        client.revoke_certificate(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_revoke_certificate_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.revoke_certificate(service.RevokeCertificateRequest(), name='name_value')

@pytest.mark.asyncio
async def test_revoke_certificate_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.revoke_certificate), '__call__') as call:
        call.return_value = resources.Certificate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Certificate())
        response = await client.revoke_certificate(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_revoke_certificate_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.revoke_certificate(service.RevokeCertificateRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.UpdateCertificateRequest, dict])
def test_update_certificate(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_certificate), '__call__') as call:
        call.return_value = resources.Certificate(name='name_value', pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value'], pem_csr='pem_csr_value')
        response = client.update_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCertificateRequest()
    assert isinstance(response, resources.Certificate)
    assert response.name == 'name_value'
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']

def test_update_certificate_empty_call():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_certificate), '__call__') as call:
        client.update_certificate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCertificateRequest()

@pytest.mark.asyncio
async def test_update_certificate_async(transport: str='grpc_asyncio', request_type=service.UpdateCertificateRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Certificate(name='name_value', pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value']))
        response = await client.update_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCertificateRequest()
    assert isinstance(response, resources.Certificate)
    assert response.name == 'name_value'
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']

@pytest.mark.asyncio
async def test_update_certificate_async_from_dict():
    await test_update_certificate_async(request_type=dict)

def test_update_certificate_field_headers():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCertificateRequest()
    request.certificate.name = 'name_value'
    with mock.patch.object(type(client.transport.update_certificate), '__call__') as call:
        call.return_value = resources.Certificate()
        client.update_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'certificate.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_certificate_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCertificateRequest()
    request.certificate.name = 'name_value'
    with mock.patch.object(type(client.transport.update_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Certificate())
        await client.update_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'certificate.name=name_value') in kw['metadata']

def test_update_certificate_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_certificate), '__call__') as call:
        call.return_value = resources.Certificate()
        client.update_certificate(certificate=resources.Certificate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].certificate
        mock_val = resources.Certificate(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_certificate_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_certificate(service.UpdateCertificateRequest(), certificate=resources.Certificate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_certificate_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_certificate), '__call__') as call:
        call.return_value = resources.Certificate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Certificate())
        response = await client.update_certificate(certificate=resources.Certificate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].certificate
        mock_val = resources.Certificate(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_certificate_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_certificate(service.UpdateCertificateRequest(), certificate=resources.Certificate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.ActivateCertificateAuthorityRequest, dict])
def test_activate_certificate_authority(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.activate_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.activate_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ActivateCertificateAuthorityRequest()
    assert isinstance(response, future.Future)

def test_activate_certificate_authority_empty_call():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.activate_certificate_authority), '__call__') as call:
        client.activate_certificate_authority()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ActivateCertificateAuthorityRequest()

@pytest.mark.asyncio
async def test_activate_certificate_authority_async(transport: str='grpc_asyncio', request_type=service.ActivateCertificateAuthorityRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.activate_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.activate_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ActivateCertificateAuthorityRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_activate_certificate_authority_async_from_dict():
    await test_activate_certificate_authority_async(request_type=dict)

def test_activate_certificate_authority_field_headers():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ActivateCertificateAuthorityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.activate_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.activate_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_activate_certificate_authority_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ActivateCertificateAuthorityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.activate_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.activate_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_activate_certificate_authority_flattened():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.activate_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.activate_certificate_authority(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_activate_certificate_authority_flattened_error():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.activate_certificate_authority(service.ActivateCertificateAuthorityRequest(), name='name_value')

@pytest.mark.asyncio
async def test_activate_certificate_authority_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.activate_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.activate_certificate_authority(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_activate_certificate_authority_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.activate_certificate_authority(service.ActivateCertificateAuthorityRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.CreateCertificateAuthorityRequest, dict])
def test_create_certificate_authority(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCertificateAuthorityRequest()
    assert isinstance(response, future.Future)

def test_create_certificate_authority_empty_call():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_certificate_authority), '__call__') as call:
        client.create_certificate_authority()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCertificateAuthorityRequest()

@pytest.mark.asyncio
async def test_create_certificate_authority_async(transport: str='grpc_asyncio', request_type=service.CreateCertificateAuthorityRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateCertificateAuthorityRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_certificate_authority_async_from_dict():
    await test_create_certificate_authority_async(request_type=dict)

def test_create_certificate_authority_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateCertificateAuthorityRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_certificate_authority_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateCertificateAuthorityRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_certificate_authority_flattened():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_certificate_authority(parent='parent_value', certificate_authority=resources.CertificateAuthority(name='name_value'), certificate_authority_id='certificate_authority_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].certificate_authority
        mock_val = resources.CertificateAuthority(name='name_value')
        assert arg == mock_val
        arg = args[0].certificate_authority_id
        mock_val = 'certificate_authority_id_value'
        assert arg == mock_val

def test_create_certificate_authority_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_certificate_authority(service.CreateCertificateAuthorityRequest(), parent='parent_value', certificate_authority=resources.CertificateAuthority(name='name_value'), certificate_authority_id='certificate_authority_id_value')

@pytest.mark.asyncio
async def test_create_certificate_authority_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_certificate_authority(parent='parent_value', certificate_authority=resources.CertificateAuthority(name='name_value'), certificate_authority_id='certificate_authority_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].certificate_authority
        mock_val = resources.CertificateAuthority(name='name_value')
        assert arg == mock_val
        arg = args[0].certificate_authority_id
        mock_val = 'certificate_authority_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_certificate_authority_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_certificate_authority(service.CreateCertificateAuthorityRequest(), parent='parent_value', certificate_authority=resources.CertificateAuthority(name='name_value'), certificate_authority_id='certificate_authority_id_value')

@pytest.mark.parametrize('request_type', [service.DisableCertificateAuthorityRequest, dict])
def test_disable_certificate_authority(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.disable_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.disable_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DisableCertificateAuthorityRequest()
    assert isinstance(response, future.Future)

def test_disable_certificate_authority_empty_call():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.disable_certificate_authority), '__call__') as call:
        client.disable_certificate_authority()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DisableCertificateAuthorityRequest()

@pytest.mark.asyncio
async def test_disable_certificate_authority_async(transport: str='grpc_asyncio', request_type=service.DisableCertificateAuthorityRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.disable_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.disable_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DisableCertificateAuthorityRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_disable_certificate_authority_async_from_dict():
    await test_disable_certificate_authority_async(request_type=dict)

def test_disable_certificate_authority_field_headers():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DisableCertificateAuthorityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.disable_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.disable_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_disable_certificate_authority_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DisableCertificateAuthorityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.disable_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.disable_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_disable_certificate_authority_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.disable_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.disable_certificate_authority(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_disable_certificate_authority_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.disable_certificate_authority(service.DisableCertificateAuthorityRequest(), name='name_value')

@pytest.mark.asyncio
async def test_disable_certificate_authority_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.disable_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.disable_certificate_authority(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_disable_certificate_authority_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.disable_certificate_authority(service.DisableCertificateAuthorityRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.EnableCertificateAuthorityRequest, dict])
def test_enable_certificate_authority(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.enable_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.enable_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.EnableCertificateAuthorityRequest()
    assert isinstance(response, future.Future)

def test_enable_certificate_authority_empty_call():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.enable_certificate_authority), '__call__') as call:
        client.enable_certificate_authority()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.EnableCertificateAuthorityRequest()

@pytest.mark.asyncio
async def test_enable_certificate_authority_async(transport: str='grpc_asyncio', request_type=service.EnableCertificateAuthorityRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.enable_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.enable_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.EnableCertificateAuthorityRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_enable_certificate_authority_async_from_dict():
    await test_enable_certificate_authority_async(request_type=dict)

def test_enable_certificate_authority_field_headers():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.EnableCertificateAuthorityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.enable_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.enable_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_enable_certificate_authority_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.EnableCertificateAuthorityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.enable_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.enable_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_enable_certificate_authority_flattened():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.enable_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.enable_certificate_authority(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_enable_certificate_authority_flattened_error():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.enable_certificate_authority(service.EnableCertificateAuthorityRequest(), name='name_value')

@pytest.mark.asyncio
async def test_enable_certificate_authority_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.enable_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.enable_certificate_authority(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_enable_certificate_authority_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.enable_certificate_authority(service.EnableCertificateAuthorityRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.FetchCertificateAuthorityCsrRequest, dict])
def test_fetch_certificate_authority_csr(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.fetch_certificate_authority_csr), '__call__') as call:
        call.return_value = service.FetchCertificateAuthorityCsrResponse(pem_csr='pem_csr_value')
        response = client.fetch_certificate_authority_csr(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.FetchCertificateAuthorityCsrRequest()
    assert isinstance(response, service.FetchCertificateAuthorityCsrResponse)
    assert response.pem_csr == 'pem_csr_value'

def test_fetch_certificate_authority_csr_empty_call():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.fetch_certificate_authority_csr), '__call__') as call:
        client.fetch_certificate_authority_csr()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.FetchCertificateAuthorityCsrRequest()

@pytest.mark.asyncio
async def test_fetch_certificate_authority_csr_async(transport: str='grpc_asyncio', request_type=service.FetchCertificateAuthorityCsrRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.fetch_certificate_authority_csr), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.FetchCertificateAuthorityCsrResponse(pem_csr='pem_csr_value'))
        response = await client.fetch_certificate_authority_csr(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.FetchCertificateAuthorityCsrRequest()
    assert isinstance(response, service.FetchCertificateAuthorityCsrResponse)
    assert response.pem_csr == 'pem_csr_value'

@pytest.mark.asyncio
async def test_fetch_certificate_authority_csr_async_from_dict():
    await test_fetch_certificate_authority_csr_async(request_type=dict)

def test_fetch_certificate_authority_csr_field_headers():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.FetchCertificateAuthorityCsrRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.fetch_certificate_authority_csr), '__call__') as call:
        call.return_value = service.FetchCertificateAuthorityCsrResponse()
        client.fetch_certificate_authority_csr(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_fetch_certificate_authority_csr_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.FetchCertificateAuthorityCsrRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.fetch_certificate_authority_csr), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.FetchCertificateAuthorityCsrResponse())
        await client.fetch_certificate_authority_csr(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_fetch_certificate_authority_csr_flattened():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.fetch_certificate_authority_csr), '__call__') as call:
        call.return_value = service.FetchCertificateAuthorityCsrResponse()
        client.fetch_certificate_authority_csr(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_fetch_certificate_authority_csr_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.fetch_certificate_authority_csr(service.FetchCertificateAuthorityCsrRequest(), name='name_value')

@pytest.mark.asyncio
async def test_fetch_certificate_authority_csr_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.fetch_certificate_authority_csr), '__call__') as call:
        call.return_value = service.FetchCertificateAuthorityCsrResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.FetchCertificateAuthorityCsrResponse())
        response = await client.fetch_certificate_authority_csr(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_fetch_certificate_authority_csr_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.fetch_certificate_authority_csr(service.FetchCertificateAuthorityCsrRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.GetCertificateAuthorityRequest, dict])
def test_get_certificate_authority(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_certificate_authority), '__call__') as call:
        call.return_value = resources.CertificateAuthority(name='name_value', type_=resources.CertificateAuthority.Type.SELF_SIGNED, tier=resources.CertificateAuthority.Tier.ENTERPRISE, state=resources.CertificateAuthority.State.ENABLED, pem_ca_certificates=['pem_ca_certificates_value'], gcs_bucket='gcs_bucket_value')
        response = client.get_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCertificateAuthorityRequest()
    assert isinstance(response, resources.CertificateAuthority)
    assert response.name == 'name_value'
    assert response.type_ == resources.CertificateAuthority.Type.SELF_SIGNED
    assert response.tier == resources.CertificateAuthority.Tier.ENTERPRISE
    assert response.state == resources.CertificateAuthority.State.ENABLED
    assert response.pem_ca_certificates == ['pem_ca_certificates_value']
    assert response.gcs_bucket == 'gcs_bucket_value'

def test_get_certificate_authority_empty_call():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_certificate_authority), '__call__') as call:
        client.get_certificate_authority()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCertificateAuthorityRequest()

@pytest.mark.asyncio
async def test_get_certificate_authority_async(transport: str='grpc_asyncio', request_type=service.GetCertificateAuthorityRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CertificateAuthority(name='name_value', type_=resources.CertificateAuthority.Type.SELF_SIGNED, tier=resources.CertificateAuthority.Tier.ENTERPRISE, state=resources.CertificateAuthority.State.ENABLED, pem_ca_certificates=['pem_ca_certificates_value'], gcs_bucket='gcs_bucket_value'))
        response = await client.get_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCertificateAuthorityRequest()
    assert isinstance(response, resources.CertificateAuthority)
    assert response.name == 'name_value'
    assert response.type_ == resources.CertificateAuthority.Type.SELF_SIGNED
    assert response.tier == resources.CertificateAuthority.Tier.ENTERPRISE
    assert response.state == resources.CertificateAuthority.State.ENABLED
    assert response.pem_ca_certificates == ['pem_ca_certificates_value']
    assert response.gcs_bucket == 'gcs_bucket_value'

@pytest.mark.asyncio
async def test_get_certificate_authority_async_from_dict():
    await test_get_certificate_authority_async(request_type=dict)

def test_get_certificate_authority_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetCertificateAuthorityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_certificate_authority), '__call__') as call:
        call.return_value = resources.CertificateAuthority()
        client.get_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_certificate_authority_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetCertificateAuthorityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CertificateAuthority())
        await client.get_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_certificate_authority_flattened():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_certificate_authority), '__call__') as call:
        call.return_value = resources.CertificateAuthority()
        client.get_certificate_authority(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_certificate_authority_flattened_error():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_certificate_authority(service.GetCertificateAuthorityRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_certificate_authority_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_certificate_authority), '__call__') as call:
        call.return_value = resources.CertificateAuthority()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CertificateAuthority())
        response = await client.get_certificate_authority(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_certificate_authority_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_certificate_authority(service.GetCertificateAuthorityRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListCertificateAuthoritiesRequest, dict])
def test_list_certificate_authorities(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_certificate_authorities), '__call__') as call:
        call.return_value = service.ListCertificateAuthoritiesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_certificate_authorities(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCertificateAuthoritiesRequest()
    assert isinstance(response, pagers.ListCertificateAuthoritiesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_certificate_authorities_empty_call():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_certificate_authorities), '__call__') as call:
        client.list_certificate_authorities()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCertificateAuthoritiesRequest()

@pytest.mark.asyncio
async def test_list_certificate_authorities_async(transport: str='grpc_asyncio', request_type=service.ListCertificateAuthoritiesRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_certificate_authorities), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCertificateAuthoritiesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_certificate_authorities(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCertificateAuthoritiesRequest()
    assert isinstance(response, pagers.ListCertificateAuthoritiesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_certificate_authorities_async_from_dict():
    await test_list_certificate_authorities_async(request_type=dict)

def test_list_certificate_authorities_field_headers():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListCertificateAuthoritiesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_certificate_authorities), '__call__') as call:
        call.return_value = service.ListCertificateAuthoritiesResponse()
        client.list_certificate_authorities(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_certificate_authorities_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListCertificateAuthoritiesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_certificate_authorities), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCertificateAuthoritiesResponse())
        await client.list_certificate_authorities(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_certificate_authorities_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_certificate_authorities), '__call__') as call:
        call.return_value = service.ListCertificateAuthoritiesResponse()
        client.list_certificate_authorities(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_certificate_authorities_flattened_error():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_certificate_authorities(service.ListCertificateAuthoritiesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_certificate_authorities_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_certificate_authorities), '__call__') as call:
        call.return_value = service.ListCertificateAuthoritiesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCertificateAuthoritiesResponse())
        response = await client.list_certificate_authorities(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_certificate_authorities_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_certificate_authorities(service.ListCertificateAuthoritiesRequest(), parent='parent_value')

def test_list_certificate_authorities_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_certificate_authorities), '__call__') as call:
        call.side_effect = (service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority(), resources.CertificateAuthority(), resources.CertificateAuthority()], next_page_token='abc'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[], next_page_token='def'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority()], next_page_token='ghi'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority(), resources.CertificateAuthority()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_certificate_authorities(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.CertificateAuthority) for i in results))

def test_list_certificate_authorities_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_certificate_authorities), '__call__') as call:
        call.side_effect = (service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority(), resources.CertificateAuthority(), resources.CertificateAuthority()], next_page_token='abc'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[], next_page_token='def'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority()], next_page_token='ghi'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority(), resources.CertificateAuthority()]), RuntimeError)
        pages = list(client.list_certificate_authorities(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_certificate_authorities_async_pager():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_certificate_authorities), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority(), resources.CertificateAuthority(), resources.CertificateAuthority()], next_page_token='abc'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[], next_page_token='def'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority()], next_page_token='ghi'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority(), resources.CertificateAuthority()]), RuntimeError)
        async_pager = await client.list_certificate_authorities(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.CertificateAuthority) for i in responses))

@pytest.mark.asyncio
async def test_list_certificate_authorities_async_pages():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_certificate_authorities), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority(), resources.CertificateAuthority(), resources.CertificateAuthority()], next_page_token='abc'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[], next_page_token='def'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority()], next_page_token='ghi'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority(), resources.CertificateAuthority()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_certificate_authorities(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.RestoreCertificateAuthorityRequest, dict])
def test_restore_certificate_authority(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.restore_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.restore_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RestoreCertificateAuthorityRequest()
    assert isinstance(response, future.Future)

def test_restore_certificate_authority_empty_call():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.restore_certificate_authority), '__call__') as call:
        client.restore_certificate_authority()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RestoreCertificateAuthorityRequest()

@pytest.mark.asyncio
async def test_restore_certificate_authority_async(transport: str='grpc_asyncio', request_type=service.RestoreCertificateAuthorityRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.restore_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.restore_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RestoreCertificateAuthorityRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_restore_certificate_authority_async_from_dict():
    await test_restore_certificate_authority_async(request_type=dict)

def test_restore_certificate_authority_field_headers():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RestoreCertificateAuthorityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.restore_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.restore_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_restore_certificate_authority_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RestoreCertificateAuthorityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.restore_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.restore_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_restore_certificate_authority_flattened():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.restore_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.restore_certificate_authority(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_restore_certificate_authority_flattened_error():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.restore_certificate_authority(service.RestoreCertificateAuthorityRequest(), name='name_value')

@pytest.mark.asyncio
async def test_restore_certificate_authority_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.restore_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.restore_certificate_authority(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_restore_certificate_authority_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.restore_certificate_authority(service.RestoreCertificateAuthorityRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ScheduleDeleteCertificateAuthorityRequest, dict])
def test_schedule_delete_certificate_authority(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.schedule_delete_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.schedule_delete_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ScheduleDeleteCertificateAuthorityRequest()
    assert isinstance(response, future.Future)

def test_schedule_delete_certificate_authority_empty_call():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.schedule_delete_certificate_authority), '__call__') as call:
        client.schedule_delete_certificate_authority()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ScheduleDeleteCertificateAuthorityRequest()

@pytest.mark.asyncio
async def test_schedule_delete_certificate_authority_async(transport: str='grpc_asyncio', request_type=service.ScheduleDeleteCertificateAuthorityRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.schedule_delete_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.schedule_delete_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ScheduleDeleteCertificateAuthorityRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_schedule_delete_certificate_authority_async_from_dict():
    await test_schedule_delete_certificate_authority_async(request_type=dict)

def test_schedule_delete_certificate_authority_field_headers():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ScheduleDeleteCertificateAuthorityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.schedule_delete_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.schedule_delete_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_schedule_delete_certificate_authority_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ScheduleDeleteCertificateAuthorityRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.schedule_delete_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.schedule_delete_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_schedule_delete_certificate_authority_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.schedule_delete_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.schedule_delete_certificate_authority(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_schedule_delete_certificate_authority_flattened_error():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.schedule_delete_certificate_authority(service.ScheduleDeleteCertificateAuthorityRequest(), name='name_value')

@pytest.mark.asyncio
async def test_schedule_delete_certificate_authority_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.schedule_delete_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.schedule_delete_certificate_authority(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_schedule_delete_certificate_authority_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.schedule_delete_certificate_authority(service.ScheduleDeleteCertificateAuthorityRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.UpdateCertificateAuthorityRequest, dict])
def test_update_certificate_authority(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCertificateAuthorityRequest()
    assert isinstance(response, future.Future)

def test_update_certificate_authority_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_certificate_authority), '__call__') as call:
        client.update_certificate_authority()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCertificateAuthorityRequest()

@pytest.mark.asyncio
async def test_update_certificate_authority_async(transport: str='grpc_asyncio', request_type=service.UpdateCertificateAuthorityRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCertificateAuthorityRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_certificate_authority_async_from_dict():
    await test_update_certificate_authority_async(request_type=dict)

def test_update_certificate_authority_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCertificateAuthorityRequest()
    request.certificate_authority.name = 'name_value'
    with mock.patch.object(type(client.transport.update_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_certificate_authority(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'certificate_authority.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_certificate_authority_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCertificateAuthorityRequest()
    request.certificate_authority.name = 'name_value'
    with mock.patch.object(type(client.transport.update_certificate_authority), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_certificate_authority(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'certificate_authority.name=name_value') in kw['metadata']

def test_update_certificate_authority_flattened():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_certificate_authority(certificate_authority=resources.CertificateAuthority(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].certificate_authority
        mock_val = resources.CertificateAuthority(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_certificate_authority_flattened_error():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_certificate_authority(service.UpdateCertificateAuthorityRequest(), certificate_authority=resources.CertificateAuthority(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_certificate_authority_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_certificate_authority), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_certificate_authority(certificate_authority=resources.CertificateAuthority(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].certificate_authority
        mock_val = resources.CertificateAuthority(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_certificate_authority_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_certificate_authority(service.UpdateCertificateAuthorityRequest(), certificate_authority=resources.CertificateAuthority(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.GetCertificateRevocationListRequest, dict])
def test_get_certificate_revocation_list(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_certificate_revocation_list), '__call__') as call:
        call.return_value = resources.CertificateRevocationList(name='name_value', sequence_number=1601, pem_crl='pem_crl_value', access_url='access_url_value', state=resources.CertificateRevocationList.State.ACTIVE)
        response = client.get_certificate_revocation_list(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCertificateRevocationListRequest()
    assert isinstance(response, resources.CertificateRevocationList)
    assert response.name == 'name_value'
    assert response.sequence_number == 1601
    assert response.pem_crl == 'pem_crl_value'
    assert response.access_url == 'access_url_value'
    assert response.state == resources.CertificateRevocationList.State.ACTIVE

def test_get_certificate_revocation_list_empty_call():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_certificate_revocation_list), '__call__') as call:
        client.get_certificate_revocation_list()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCertificateRevocationListRequest()

@pytest.mark.asyncio
async def test_get_certificate_revocation_list_async(transport: str='grpc_asyncio', request_type=service.GetCertificateRevocationListRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_certificate_revocation_list), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CertificateRevocationList(name='name_value', sequence_number=1601, pem_crl='pem_crl_value', access_url='access_url_value', state=resources.CertificateRevocationList.State.ACTIVE))
        response = await client.get_certificate_revocation_list(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetCertificateRevocationListRequest()
    assert isinstance(response, resources.CertificateRevocationList)
    assert response.name == 'name_value'
    assert response.sequence_number == 1601
    assert response.pem_crl == 'pem_crl_value'
    assert response.access_url == 'access_url_value'
    assert response.state == resources.CertificateRevocationList.State.ACTIVE

@pytest.mark.asyncio
async def test_get_certificate_revocation_list_async_from_dict():
    await test_get_certificate_revocation_list_async(request_type=dict)

def test_get_certificate_revocation_list_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetCertificateRevocationListRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_certificate_revocation_list), '__call__') as call:
        call.return_value = resources.CertificateRevocationList()
        client.get_certificate_revocation_list(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_certificate_revocation_list_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetCertificateRevocationListRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_certificate_revocation_list), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CertificateRevocationList())
        await client.get_certificate_revocation_list(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_certificate_revocation_list_flattened():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_certificate_revocation_list), '__call__') as call:
        call.return_value = resources.CertificateRevocationList()
        client.get_certificate_revocation_list(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_certificate_revocation_list_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_certificate_revocation_list(service.GetCertificateRevocationListRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_certificate_revocation_list_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_certificate_revocation_list), '__call__') as call:
        call.return_value = resources.CertificateRevocationList()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.CertificateRevocationList())
        response = await client.get_certificate_revocation_list(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_certificate_revocation_list_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_certificate_revocation_list(service.GetCertificateRevocationListRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListCertificateRevocationListsRequest, dict])
def test_list_certificate_revocation_lists(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_certificate_revocation_lists), '__call__') as call:
        call.return_value = service.ListCertificateRevocationListsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_certificate_revocation_lists(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCertificateRevocationListsRequest()
    assert isinstance(response, pagers.ListCertificateRevocationListsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_certificate_revocation_lists_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_certificate_revocation_lists), '__call__') as call:
        client.list_certificate_revocation_lists()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCertificateRevocationListsRequest()

@pytest.mark.asyncio
async def test_list_certificate_revocation_lists_async(transport: str='grpc_asyncio', request_type=service.ListCertificateRevocationListsRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_certificate_revocation_lists), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCertificateRevocationListsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_certificate_revocation_lists(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListCertificateRevocationListsRequest()
    assert isinstance(response, pagers.ListCertificateRevocationListsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_certificate_revocation_lists_async_from_dict():
    await test_list_certificate_revocation_lists_async(request_type=dict)

def test_list_certificate_revocation_lists_field_headers():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListCertificateRevocationListsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_certificate_revocation_lists), '__call__') as call:
        call.return_value = service.ListCertificateRevocationListsResponse()
        client.list_certificate_revocation_lists(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_certificate_revocation_lists_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListCertificateRevocationListsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_certificate_revocation_lists), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCertificateRevocationListsResponse())
        await client.list_certificate_revocation_lists(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_certificate_revocation_lists_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_certificate_revocation_lists), '__call__') as call:
        call.return_value = service.ListCertificateRevocationListsResponse()
        client.list_certificate_revocation_lists(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_certificate_revocation_lists_flattened_error():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_certificate_revocation_lists(service.ListCertificateRevocationListsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_certificate_revocation_lists_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_certificate_revocation_lists), '__call__') as call:
        call.return_value = service.ListCertificateRevocationListsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListCertificateRevocationListsResponse())
        response = await client.list_certificate_revocation_lists(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_certificate_revocation_lists_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_certificate_revocation_lists(service.ListCertificateRevocationListsRequest(), parent='parent_value')

def test_list_certificate_revocation_lists_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_certificate_revocation_lists), '__call__') as call:
        call.side_effect = (service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList(), resources.CertificateRevocationList(), resources.CertificateRevocationList()], next_page_token='abc'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[], next_page_token='def'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList()], next_page_token='ghi'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList(), resources.CertificateRevocationList()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_certificate_revocation_lists(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.CertificateRevocationList) for i in results))

def test_list_certificate_revocation_lists_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_certificate_revocation_lists), '__call__') as call:
        call.side_effect = (service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList(), resources.CertificateRevocationList(), resources.CertificateRevocationList()], next_page_token='abc'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[], next_page_token='def'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList()], next_page_token='ghi'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList(), resources.CertificateRevocationList()]), RuntimeError)
        pages = list(client.list_certificate_revocation_lists(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_certificate_revocation_lists_async_pager():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_certificate_revocation_lists), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList(), resources.CertificateRevocationList(), resources.CertificateRevocationList()], next_page_token='abc'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[], next_page_token='def'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList()], next_page_token='ghi'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList(), resources.CertificateRevocationList()]), RuntimeError)
        async_pager = await client.list_certificate_revocation_lists(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.CertificateRevocationList) for i in responses))

@pytest.mark.asyncio
async def test_list_certificate_revocation_lists_async_pages():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_certificate_revocation_lists), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList(), resources.CertificateRevocationList(), resources.CertificateRevocationList()], next_page_token='abc'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[], next_page_token='def'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList()], next_page_token='ghi'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList(), resources.CertificateRevocationList()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_certificate_revocation_lists(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.UpdateCertificateRevocationListRequest, dict])
def test_update_certificate_revocation_list(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_certificate_revocation_list), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_certificate_revocation_list(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCertificateRevocationListRequest()
    assert isinstance(response, future.Future)

def test_update_certificate_revocation_list_empty_call():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_certificate_revocation_list), '__call__') as call:
        client.update_certificate_revocation_list()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCertificateRevocationListRequest()

@pytest.mark.asyncio
async def test_update_certificate_revocation_list_async(transport: str='grpc_asyncio', request_type=service.UpdateCertificateRevocationListRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_certificate_revocation_list), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_certificate_revocation_list(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateCertificateRevocationListRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_certificate_revocation_list_async_from_dict():
    await test_update_certificate_revocation_list_async(request_type=dict)

def test_update_certificate_revocation_list_field_headers():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCertificateRevocationListRequest()
    request.certificate_revocation_list.name = 'name_value'
    with mock.patch.object(type(client.transport.update_certificate_revocation_list), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_certificate_revocation_list(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'certificate_revocation_list.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_certificate_revocation_list_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateCertificateRevocationListRequest()
    request.certificate_revocation_list.name = 'name_value'
    with mock.patch.object(type(client.transport.update_certificate_revocation_list), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_certificate_revocation_list(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'certificate_revocation_list.name=name_value') in kw['metadata']

def test_update_certificate_revocation_list_flattened():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_certificate_revocation_list), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_certificate_revocation_list(certificate_revocation_list=resources.CertificateRevocationList(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].certificate_revocation_list
        mock_val = resources.CertificateRevocationList(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_certificate_revocation_list_flattened_error():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_certificate_revocation_list(service.UpdateCertificateRevocationListRequest(), certificate_revocation_list=resources.CertificateRevocationList(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_certificate_revocation_list_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_certificate_revocation_list), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_certificate_revocation_list(certificate_revocation_list=resources.CertificateRevocationList(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].certificate_revocation_list
        mock_val = resources.CertificateRevocationList(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_certificate_revocation_list_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_certificate_revocation_list(service.UpdateCertificateRevocationListRequest(), certificate_revocation_list=resources.CertificateRevocationList(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.GetReusableConfigRequest, dict])
def test_get_reusable_config(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_reusable_config), '__call__') as call:
        call.return_value = resources.ReusableConfig(name='name_value', description='description_value')
        response = client.get_reusable_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetReusableConfigRequest()
    assert isinstance(response, resources.ReusableConfig)
    assert response.name == 'name_value'
    assert response.description == 'description_value'

def test_get_reusable_config_empty_call():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_reusable_config), '__call__') as call:
        client.get_reusable_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetReusableConfigRequest()

@pytest.mark.asyncio
async def test_get_reusable_config_async(transport: str='grpc_asyncio', request_type=service.GetReusableConfigRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_reusable_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ReusableConfig(name='name_value', description='description_value'))
        response = await client.get_reusable_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetReusableConfigRequest()
    assert isinstance(response, resources.ReusableConfig)
    assert response.name == 'name_value'
    assert response.description == 'description_value'

@pytest.mark.asyncio
async def test_get_reusable_config_async_from_dict():
    await test_get_reusable_config_async(request_type=dict)

def test_get_reusable_config_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetReusableConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_reusable_config), '__call__') as call:
        call.return_value = resources.ReusableConfig()
        client.get_reusable_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_reusable_config_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetReusableConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_reusable_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ReusableConfig())
        await client.get_reusable_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_reusable_config_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_reusable_config), '__call__') as call:
        call.return_value = resources.ReusableConfig()
        client.get_reusable_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_reusable_config_flattened_error():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_reusable_config(service.GetReusableConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_reusable_config_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_reusable_config), '__call__') as call:
        call.return_value = resources.ReusableConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ReusableConfig())
        response = await client.get_reusable_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_reusable_config_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_reusable_config(service.GetReusableConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListReusableConfigsRequest, dict])
def test_list_reusable_configs(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_reusable_configs), '__call__') as call:
        call.return_value = service.ListReusableConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_reusable_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListReusableConfigsRequest()
    assert isinstance(response, pagers.ListReusableConfigsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_reusable_configs_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_reusable_configs), '__call__') as call:
        client.list_reusable_configs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListReusableConfigsRequest()

@pytest.mark.asyncio
async def test_list_reusable_configs_async(transport: str='grpc_asyncio', request_type=service.ListReusableConfigsRequest):
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_reusable_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListReusableConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_reusable_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListReusableConfigsRequest()
    assert isinstance(response, pagers.ListReusableConfigsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_reusable_configs_async_from_dict():
    await test_list_reusable_configs_async(request_type=dict)

def test_list_reusable_configs_field_headers():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListReusableConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_reusable_configs), '__call__') as call:
        call.return_value = service.ListReusableConfigsResponse()
        client.list_reusable_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_reusable_configs_field_headers_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListReusableConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_reusable_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListReusableConfigsResponse())
        await client.list_reusable_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_reusable_configs_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_reusable_configs), '__call__') as call:
        call.return_value = service.ListReusableConfigsResponse()
        client.list_reusable_configs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_reusable_configs_flattened_error():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_reusable_configs(service.ListReusableConfigsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_reusable_configs_flattened_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_reusable_configs), '__call__') as call:
        call.return_value = service.ListReusableConfigsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListReusableConfigsResponse())
        response = await client.list_reusable_configs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_reusable_configs_flattened_error_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_reusable_configs(service.ListReusableConfigsRequest(), parent='parent_value')

def test_list_reusable_configs_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_reusable_configs), '__call__') as call:
        call.side_effect = (service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig(), resources.ReusableConfig(), resources.ReusableConfig()], next_page_token='abc'), service.ListReusableConfigsResponse(reusable_configs=[], next_page_token='def'), service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig()], next_page_token='ghi'), service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig(), resources.ReusableConfig()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_reusable_configs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.ReusableConfig) for i in results))

def test_list_reusable_configs_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_reusable_configs), '__call__') as call:
        call.side_effect = (service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig(), resources.ReusableConfig(), resources.ReusableConfig()], next_page_token='abc'), service.ListReusableConfigsResponse(reusable_configs=[], next_page_token='def'), service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig()], next_page_token='ghi'), service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig(), resources.ReusableConfig()]), RuntimeError)
        pages = list(client.list_reusable_configs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_reusable_configs_async_pager():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_reusable_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig(), resources.ReusableConfig(), resources.ReusableConfig()], next_page_token='abc'), service.ListReusableConfigsResponse(reusable_configs=[], next_page_token='def'), service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig()], next_page_token='ghi'), service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig(), resources.ReusableConfig()]), RuntimeError)
        async_pager = await client.list_reusable_configs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.ReusableConfig) for i in responses))

@pytest.mark.asyncio
async def test_list_reusable_configs_async_pages():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_reusable_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig(), resources.ReusableConfig(), resources.ReusableConfig()], next_page_token='abc'), service.ListReusableConfigsResponse(reusable_configs=[], next_page_token='def'), service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig()], next_page_token='ghi'), service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig(), resources.ReusableConfig()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_reusable_configs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.CreateCertificateRequest, dict])
def test_create_certificate_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request_init['certificate'] = {'name': 'name_value', 'pem_csr': 'pem_csr_value', 'config': {'subject_config': {'subject': {'country_code': 'country_code_value', 'organization': 'organization_value', 'organizational_unit': 'organizational_unit_value', 'locality': 'locality_value', 'province': 'province_value', 'street_address': 'street_address_value', 'postal_code': 'postal_code_value'}, 'common_name': 'common_name_value', 'subject_alt_name': {'dns_names': ['dns_names_value1', 'dns_names_value2'], 'uris': ['uris_value1', 'uris_value2'], 'email_addresses': ['email_addresses_value1', 'email_addresses_value2'], 'ip_addresses': ['ip_addresses_value1', 'ip_addresses_value2'], 'custom_sans': [{'object_id': {'object_id_path': [1456, 1457]}, 'critical': True, 'value': b'value_blob'}]}}, 'reusable_config': {'reusable_config': 'reusable_config_value', 'reusable_config_values': {'key_usage': {'base_key_usage': {'digital_signature': True, 'content_commitment': True, 'key_encipherment': True, 'data_encipherment': True, 'key_agreement': True, 'cert_sign': True, 'crl_sign': True, 'encipher_only': True, 'decipher_only': True}, 'extended_key_usage': {'server_auth': True, 'client_auth': True, 'code_signing': True, 'email_protection': True, 'time_stamping': True, 'ocsp_signing': True}, 'unknown_extended_key_usages': {}}, 'ca_options': {'is_ca': {'value': True}, 'max_issuer_path_length': {'value': 541}}, 'policy_ids': {}, 'aia_ocsp_servers': ['aia_ocsp_servers_value1', 'aia_ocsp_servers_value2'], 'additional_extensions': {}}}, 'public_key': {'type_': 1, 'key': b'key_blob'}}, 'lifetime': {'seconds': 751, 'nanos': 543}, 'revocation_details': {'revocation_state': 1, 'revocation_time': {'seconds': 751, 'nanos': 543}}, 'pem_certificate': 'pem_certificate_value', 'certificate_description': {'subject_description': {'subject': {}, 'common_name': 'common_name_value', 'subject_alt_name': {}, 'hex_serial_number': 'hex_serial_number_value', 'lifetime': {}, 'not_before_time': {}, 'not_after_time': {}}, 'config_values': {}, 'public_key': {}, 'subject_key_id': {'key_id': 'key_id_value'}, 'authority_key_id': {}, 'crl_distribution_points': ['crl_distribution_points_value1', 'crl_distribution_points_value2'], 'aia_issuing_certificate_urls': ['aia_issuing_certificate_urls_value1', 'aia_issuing_certificate_urls_value2'], 'cert_fingerprint': {'sha256_hash': 'sha256_hash_value'}}, 'pem_certificate_chain': ['pem_certificate_chain_value1', 'pem_certificate_chain_value2'], 'create_time': {}, 'update_time': {}, 'labels': {}}
    test_field = service.CreateCertificateRequest.meta.fields['certificate']

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
        return_value = resources.Certificate(name='name_value', pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value'], pem_csr='pem_csr_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Certificate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_certificate(request)
    assert isinstance(response, resources.Certificate)
    assert response.name == 'name_value'
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']

def test_create_certificate_rest_required_fields(request_type=service.CreateCertificateRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_certificate._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_certificate._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('certificate_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Certificate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Certificate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_certificate(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_certificate_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_certificate._get_unset_required_fields({})
    assert set(unset_fields) == set(('certificateId', 'requestId')) & set(('parent', 'certificate'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_certificate_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_create_certificate') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_create_certificate') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateCertificateRequest.pb(service.CreateCertificateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Certificate.to_json(resources.Certificate())
        request = service.CreateCertificateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Certificate()
        client.create_certificate(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_certificate_rest_bad_request(transport: str='rest', request_type=service.CreateCertificateRequest):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_certificate(request)

def test_create_certificate_rest_flattened():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Certificate()
        sample_request = {'parent': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
        mock_args = dict(parent='parent_value', certificate=resources.Certificate(name='name_value'), certificate_id='certificate_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Certificate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_certificate(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*/certificateAuthorities/*}/certificates' % client.transport._host, args[1])

def test_create_certificate_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_certificate(service.CreateCertificateRequest(), parent='parent_value', certificate=resources.Certificate(name='name_value'), certificate_id='certificate_id_value')

def test_create_certificate_rest_error():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetCertificateRequest, dict])
def test_get_certificate_rest(request_type):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificates/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Certificate(name='name_value', pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value'], pem_csr='pem_csr_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Certificate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_certificate(request)
    assert isinstance(response, resources.Certificate)
    assert response.name == 'name_value'
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']

def test_get_certificate_rest_required_fields(request_type=service.GetCertificateRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CertificateAuthorityServiceRestTransport
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
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Certificate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Certificate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_certificate(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_certificate_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_certificate._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_certificate_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_get_certificate') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_get_certificate') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetCertificateRequest.pb(service.GetCertificateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Certificate.to_json(resources.Certificate())
        request = service.GetCertificateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Certificate()
        client.get_certificate(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_certificate_rest_bad_request(transport: str='rest', request_type=service.GetCertificateRequest):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificates/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_certificate(request)

def test_get_certificate_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Certificate()
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificates/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Certificate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_certificate(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/certificateAuthorities/*/certificates/*}' % client.transport._host, args[1])

def test_get_certificate_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_certificate(service.GetCertificateRequest(), name='name_value')

def test_get_certificate_rest_error():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListCertificatesRequest, dict])
def test_list_certificates_rest(request_type):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListCertificatesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListCertificatesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_certificates(request)
    assert isinstance(response, pagers.ListCertificatesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_certificates_rest_required_fields(request_type=service.ListCertificatesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CertificateAuthorityServiceRestTransport
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
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListCertificatesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListCertificatesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_certificates(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_certificates_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_certificates._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_certificates_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_list_certificates') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_list_certificates') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListCertificatesRequest.pb(service.ListCertificatesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListCertificatesResponse.to_json(service.ListCertificatesResponse())
        request = service.ListCertificatesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListCertificatesResponse()
        client.list_certificates(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_certificates_rest_bad_request(transport: str='rest', request_type=service.ListCertificatesRequest):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_certificates(request)

def test_list_certificates_rest_flattened():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListCertificatesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListCertificatesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_certificates(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*/certificateAuthorities/*}/certificates' % client.transport._host, args[1])

def test_list_certificates_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_certificates(service.ListCertificatesRequest(), parent='parent_value')

def test_list_certificates_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListCertificatesResponse(certificates=[resources.Certificate(), resources.Certificate(), resources.Certificate()], next_page_token='abc'), service.ListCertificatesResponse(certificates=[], next_page_token='def'), service.ListCertificatesResponse(certificates=[resources.Certificate()], next_page_token='ghi'), service.ListCertificatesResponse(certificates=[resources.Certificate(), resources.Certificate()]))
        response = response + response
        response = tuple((service.ListCertificatesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
        pager = client.list_certificates(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Certificate) for i in results))
        pages = list(client.list_certificates(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.RevokeCertificateRequest, dict])
def test_revoke_certificate_rest(request_type):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificates/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Certificate(name='name_value', pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value'], pem_csr='pem_csr_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Certificate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.revoke_certificate(request)
    assert isinstance(response, resources.Certificate)
    assert response.name == 'name_value'
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']

def test_revoke_certificate_rest_required_fields(request_type=service.RevokeCertificateRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).revoke_certificate._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).revoke_certificate._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Certificate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Certificate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.revoke_certificate(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_revoke_certificate_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.revoke_certificate._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'reason'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_revoke_certificate_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_revoke_certificate') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_revoke_certificate') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.RevokeCertificateRequest.pb(service.RevokeCertificateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Certificate.to_json(resources.Certificate())
        request = service.RevokeCertificateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Certificate()
        client.revoke_certificate(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_revoke_certificate_rest_bad_request(transport: str='rest', request_type=service.RevokeCertificateRequest):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificates/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.revoke_certificate(request)

def test_revoke_certificate_rest_flattened():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Certificate()
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificates/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Certificate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.revoke_certificate(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/certificateAuthorities/*/certificates/*}:revoke' % client.transport._host, args[1])

def test_revoke_certificate_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.revoke_certificate(service.RevokeCertificateRequest(), name='name_value')

def test_revoke_certificate_rest_error():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateCertificateRequest, dict])
def test_update_certificate_rest(request_type):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'certificate': {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificates/sample4'}}
    request_init['certificate'] = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificates/sample4', 'pem_csr': 'pem_csr_value', 'config': {'subject_config': {'subject': {'country_code': 'country_code_value', 'organization': 'organization_value', 'organizational_unit': 'organizational_unit_value', 'locality': 'locality_value', 'province': 'province_value', 'street_address': 'street_address_value', 'postal_code': 'postal_code_value'}, 'common_name': 'common_name_value', 'subject_alt_name': {'dns_names': ['dns_names_value1', 'dns_names_value2'], 'uris': ['uris_value1', 'uris_value2'], 'email_addresses': ['email_addresses_value1', 'email_addresses_value2'], 'ip_addresses': ['ip_addresses_value1', 'ip_addresses_value2'], 'custom_sans': [{'object_id': {'object_id_path': [1456, 1457]}, 'critical': True, 'value': b'value_blob'}]}}, 'reusable_config': {'reusable_config': 'reusable_config_value', 'reusable_config_values': {'key_usage': {'base_key_usage': {'digital_signature': True, 'content_commitment': True, 'key_encipherment': True, 'data_encipherment': True, 'key_agreement': True, 'cert_sign': True, 'crl_sign': True, 'encipher_only': True, 'decipher_only': True}, 'extended_key_usage': {'server_auth': True, 'client_auth': True, 'code_signing': True, 'email_protection': True, 'time_stamping': True, 'ocsp_signing': True}, 'unknown_extended_key_usages': {}}, 'ca_options': {'is_ca': {'value': True}, 'max_issuer_path_length': {'value': 541}}, 'policy_ids': {}, 'aia_ocsp_servers': ['aia_ocsp_servers_value1', 'aia_ocsp_servers_value2'], 'additional_extensions': {}}}, 'public_key': {'type_': 1, 'key': b'key_blob'}}, 'lifetime': {'seconds': 751, 'nanos': 543}, 'revocation_details': {'revocation_state': 1, 'revocation_time': {'seconds': 751, 'nanos': 543}}, 'pem_certificate': 'pem_certificate_value', 'certificate_description': {'subject_description': {'subject': {}, 'common_name': 'common_name_value', 'subject_alt_name': {}, 'hex_serial_number': 'hex_serial_number_value', 'lifetime': {}, 'not_before_time': {}, 'not_after_time': {}}, 'config_values': {}, 'public_key': {}, 'subject_key_id': {'key_id': 'key_id_value'}, 'authority_key_id': {}, 'crl_distribution_points': ['crl_distribution_points_value1', 'crl_distribution_points_value2'], 'aia_issuing_certificate_urls': ['aia_issuing_certificate_urls_value1', 'aia_issuing_certificate_urls_value2'], 'cert_fingerprint': {'sha256_hash': 'sha256_hash_value'}}, 'pem_certificate_chain': ['pem_certificate_chain_value1', 'pem_certificate_chain_value2'], 'create_time': {}, 'update_time': {}, 'labels': {}}
    test_field = service.UpdateCertificateRequest.meta.fields['certificate']

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
        return_value = resources.Certificate(name='name_value', pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value'], pem_csr='pem_csr_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Certificate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_certificate(request)
    assert isinstance(response, resources.Certificate)
    assert response.name == 'name_value'
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']

def test_update_certificate_rest_required_fields(request_type=service.UpdateCertificateRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_certificate._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_certificate._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Certificate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Certificate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_certificate(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_certificate_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_certificate._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('certificate', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_certificate_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_update_certificate') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_update_certificate') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateCertificateRequest.pb(service.UpdateCertificateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Certificate.to_json(resources.Certificate())
        request = service.UpdateCertificateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Certificate()
        client.update_certificate(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_certificate_rest_bad_request(transport: str='rest', request_type=service.UpdateCertificateRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'certificate': {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificates/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_certificate(request)

def test_update_certificate_rest_flattened():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Certificate()
        sample_request = {'certificate': {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificates/sample4'}}
        mock_args = dict(certificate=resources.Certificate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Certificate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_certificate(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{certificate.name=projects/*/locations/*/certificateAuthorities/*/certificates/*}' % client.transport._host, args[1])

def test_update_certificate_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_certificate(service.UpdateCertificateRequest(), certificate=resources.Certificate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_certificate_rest_error():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ActivateCertificateAuthorityRequest, dict])
def test_activate_certificate_authority_rest(request_type):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.activate_certificate_authority(request)
    assert response.operation.name == 'operations/spam'

def test_activate_certificate_authority_rest_required_fields(request_type=service.ActivateCertificateAuthorityRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['pem_ca_certificate'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).activate_certificate_authority._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['pemCaCertificate'] = 'pem_ca_certificate_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).activate_certificate_authority._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'pemCaCertificate' in jsonified_request
    assert jsonified_request['pemCaCertificate'] == 'pem_ca_certificate_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.activate_certificate_authority(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_activate_certificate_authority_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.activate_certificate_authority._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'pemCaCertificate', 'subordinateConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_activate_certificate_authority_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_activate_certificate_authority') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_activate_certificate_authority') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ActivateCertificateAuthorityRequest.pb(service.ActivateCertificateAuthorityRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.ActivateCertificateAuthorityRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.activate_certificate_authority(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_activate_certificate_authority_rest_bad_request(transport: str='rest', request_type=service.ActivateCertificateAuthorityRequest):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.activate_certificate_authority(request)

def test_activate_certificate_authority_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.activate_certificate_authority(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/certificateAuthorities/*}:activate' % client.transport._host, args[1])

def test_activate_certificate_authority_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.activate_certificate_authority(service.ActivateCertificateAuthorityRequest(), name='name_value')

def test_activate_certificate_authority_rest_error():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateCertificateAuthorityRequest, dict])
def test_create_certificate_authority_rest(request_type):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['certificate_authority'] = {'name': 'name_value', 'type_': 1, 'tier': 1, 'config': {'subject_config': {'subject': {'country_code': 'country_code_value', 'organization': 'organization_value', 'organizational_unit': 'organizational_unit_value', 'locality': 'locality_value', 'province': 'province_value', 'street_address': 'street_address_value', 'postal_code': 'postal_code_value'}, 'common_name': 'common_name_value', 'subject_alt_name': {'dns_names': ['dns_names_value1', 'dns_names_value2'], 'uris': ['uris_value1', 'uris_value2'], 'email_addresses': ['email_addresses_value1', 'email_addresses_value2'], 'ip_addresses': ['ip_addresses_value1', 'ip_addresses_value2'], 'custom_sans': [{'object_id': {'object_id_path': [1456, 1457]}, 'critical': True, 'value': b'value_blob'}]}}, 'reusable_config': {'reusable_config': 'reusable_config_value', 'reusable_config_values': {'key_usage': {'base_key_usage': {'digital_signature': True, 'content_commitment': True, 'key_encipherment': True, 'data_encipherment': True, 'key_agreement': True, 'cert_sign': True, 'crl_sign': True, 'encipher_only': True, 'decipher_only': True}, 'extended_key_usage': {'server_auth': True, 'client_auth': True, 'code_signing': True, 'email_protection': True, 'time_stamping': True, 'ocsp_signing': True}, 'unknown_extended_key_usages': {}}, 'ca_options': {'is_ca': {'value': True}, 'max_issuer_path_length': {'value': 541}}, 'policy_ids': {}, 'aia_ocsp_servers': ['aia_ocsp_servers_value1', 'aia_ocsp_servers_value2'], 'additional_extensions': {}}}, 'public_key': {'type_': 1, 'key': b'key_blob'}}, 'lifetime': {'seconds': 751, 'nanos': 543}, 'key_spec': {'cloud_kms_key_version': 'cloud_kms_key_version_value', 'algorithm': 1}, 'certificate_policy': {'allowed_config_list': {'allowed_config_values': {}}, 'overwrite_config_values': {}, 'allowed_locations_and_organizations': {}, 'allowed_common_names': ['allowed_common_names_value1', 'allowed_common_names_value2'], 'allowed_sans': {'allowed_dns_names': ['allowed_dns_names_value1', 'allowed_dns_names_value2'], 'allowed_uris': ['allowed_uris_value1', 'allowed_uris_value2'], 'allowed_email_addresses': ['allowed_email_addresses_value1', 'allowed_email_addresses_value2'], 'allowed_ips': ['allowed_ips_value1', 'allowed_ips_value2'], 'allow_globbing_dns_wildcards': True, 'allow_custom_sans': True}, 'maximum_lifetime': {}, 'allowed_issuance_modes': {'allow_csr_based_issuance': True, 'allow_config_based_issuance': True}}, 'issuing_options': {'include_ca_cert_url': True, 'include_crl_access_url': True}, 'subordinate_config': {'certificate_authority': 'certificate_authority_value', 'pem_issuer_chain': {'pem_certificates': ['pem_certificates_value1', 'pem_certificates_value2']}}, 'state': 1, 'pem_ca_certificates': ['pem_ca_certificates_value1', 'pem_ca_certificates_value2'], 'ca_certificate_descriptions': [{'subject_description': {'subject': {}, 'common_name': 'common_name_value', 'subject_alt_name': {}, 'hex_serial_number': 'hex_serial_number_value', 'lifetime': {}, 'not_before_time': {'seconds': 751, 'nanos': 543}, 'not_after_time': {}}, 'config_values': {}, 'public_key': {}, 'subject_key_id': {'key_id': 'key_id_value'}, 'authority_key_id': {}, 'crl_distribution_points': ['crl_distribution_points_value1', 'crl_distribution_points_value2'], 'aia_issuing_certificate_urls': ['aia_issuing_certificate_urls_value1', 'aia_issuing_certificate_urls_value2'], 'cert_fingerprint': {'sha256_hash': 'sha256_hash_value'}}], 'gcs_bucket': 'gcs_bucket_value', 'access_urls': {'ca_certificate_access_url': 'ca_certificate_access_url_value', 'crl_access_url': 'crl_access_url_value'}, 'create_time': {}, 'update_time': {}, 'delete_time': {}, 'labels': {}}
    test_field = service.CreateCertificateAuthorityRequest.meta.fields['certificate_authority']

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
    for (field, value) in request_init['certificate_authority'].items():
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
                for i in range(0, len(request_init['certificate_authority'][field])):
                    del request_init['certificate_authority'][field][i][subfield]
            else:
                del request_init['certificate_authority'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_certificate_authority(request)
    assert response.operation.name == 'operations/spam'

def test_create_certificate_authority_rest_required_fields(request_type=service.CreateCertificateAuthorityRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['certificate_authority_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'certificateAuthorityId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_certificate_authority._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'certificateAuthorityId' in jsonified_request
    assert jsonified_request['certificateAuthorityId'] == request_init['certificate_authority_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['certificateAuthorityId'] = 'certificate_authority_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_certificate_authority._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('certificate_authority_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'certificateAuthorityId' in jsonified_request
    assert jsonified_request['certificateAuthorityId'] == 'certificate_authority_id_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_certificate_authority(request)
            expected_params = [('certificateAuthorityId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_certificate_authority_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_certificate_authority._get_unset_required_fields({})
    assert set(unset_fields) == set(('certificateAuthorityId', 'requestId')) & set(('parent', 'certificateAuthorityId', 'certificateAuthority'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_certificate_authority_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_create_certificate_authority') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_create_certificate_authority') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateCertificateAuthorityRequest.pb(service.CreateCertificateAuthorityRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateCertificateAuthorityRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_certificate_authority(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_certificate_authority_rest_bad_request(transport: str='rest', request_type=service.CreateCertificateAuthorityRequest):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_certificate_authority(request)

def test_create_certificate_authority_rest_flattened():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', certificate_authority=resources.CertificateAuthority(name='name_value'), certificate_authority_id='certificate_authority_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_certificate_authority(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*}/certificateAuthorities' % client.transport._host, args[1])

def test_create_certificate_authority_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_certificate_authority(service.CreateCertificateAuthorityRequest(), parent='parent_value', certificate_authority=resources.CertificateAuthority(name='name_value'), certificate_authority_id='certificate_authority_id_value')

def test_create_certificate_authority_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DisableCertificateAuthorityRequest, dict])
def test_disable_certificate_authority_rest(request_type):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.disable_certificate_authority(request)
    assert response.operation.name == 'operations/spam'

def test_disable_certificate_authority_rest_required_fields(request_type=service.DisableCertificateAuthorityRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).disable_certificate_authority._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).disable_certificate_authority._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.disable_certificate_authority(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_disable_certificate_authority_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.disable_certificate_authority._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_disable_certificate_authority_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_disable_certificate_authority') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_disable_certificate_authority') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DisableCertificateAuthorityRequest.pb(service.DisableCertificateAuthorityRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.DisableCertificateAuthorityRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.disable_certificate_authority(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_disable_certificate_authority_rest_bad_request(transport: str='rest', request_type=service.DisableCertificateAuthorityRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.disable_certificate_authority(request)

def test_disable_certificate_authority_rest_flattened():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.disable_certificate_authority(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/certificateAuthorities/*}:disable' % client.transport._host, args[1])

def test_disable_certificate_authority_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.disable_certificate_authority(service.DisableCertificateAuthorityRequest(), name='name_value')

def test_disable_certificate_authority_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.EnableCertificateAuthorityRequest, dict])
def test_enable_certificate_authority_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.enable_certificate_authority(request)
    assert response.operation.name == 'operations/spam'

def test_enable_certificate_authority_rest_required_fields(request_type=service.EnableCertificateAuthorityRequest):
    if False:
        return 10
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).enable_certificate_authority._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).enable_certificate_authority._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.enable_certificate_authority(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_enable_certificate_authority_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.enable_certificate_authority._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_enable_certificate_authority_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_enable_certificate_authority') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_enable_certificate_authority') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.EnableCertificateAuthorityRequest.pb(service.EnableCertificateAuthorityRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.EnableCertificateAuthorityRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.enable_certificate_authority(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_enable_certificate_authority_rest_bad_request(transport: str='rest', request_type=service.EnableCertificateAuthorityRequest):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.enable_certificate_authority(request)

def test_enable_certificate_authority_rest_flattened():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.enable_certificate_authority(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/certificateAuthorities/*}:enable' % client.transport._host, args[1])

def test_enable_certificate_authority_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.enable_certificate_authority(service.EnableCertificateAuthorityRequest(), name='name_value')

def test_enable_certificate_authority_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.FetchCertificateAuthorityCsrRequest, dict])
def test_fetch_certificate_authority_csr_rest(request_type):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.FetchCertificateAuthorityCsrResponse(pem_csr='pem_csr_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = service.FetchCertificateAuthorityCsrResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.fetch_certificate_authority_csr(request)
    assert isinstance(response, service.FetchCertificateAuthorityCsrResponse)
    assert response.pem_csr == 'pem_csr_value'

def test_fetch_certificate_authority_csr_rest_required_fields(request_type=service.FetchCertificateAuthorityCsrRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).fetch_certificate_authority_csr._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).fetch_certificate_authority_csr._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.FetchCertificateAuthorityCsrResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.FetchCertificateAuthorityCsrResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.fetch_certificate_authority_csr(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_fetch_certificate_authority_csr_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.fetch_certificate_authority_csr._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_fetch_certificate_authority_csr_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_fetch_certificate_authority_csr') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_fetch_certificate_authority_csr') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.FetchCertificateAuthorityCsrRequest.pb(service.FetchCertificateAuthorityCsrRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.FetchCertificateAuthorityCsrResponse.to_json(service.FetchCertificateAuthorityCsrResponse())
        request = service.FetchCertificateAuthorityCsrRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.FetchCertificateAuthorityCsrResponse()
        client.fetch_certificate_authority_csr(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_fetch_certificate_authority_csr_rest_bad_request(transport: str='rest', request_type=service.FetchCertificateAuthorityCsrRequest):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.fetch_certificate_authority_csr(request)

def test_fetch_certificate_authority_csr_rest_flattened():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.FetchCertificateAuthorityCsrResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.FetchCertificateAuthorityCsrResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.fetch_certificate_authority_csr(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/certificateAuthorities/*}:fetch' % client.transport._host, args[1])

def test_fetch_certificate_authority_csr_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.fetch_certificate_authority_csr(service.FetchCertificateAuthorityCsrRequest(), name='name_value')

def test_fetch_certificate_authority_csr_rest_error():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetCertificateAuthorityRequest, dict])
def test_get_certificate_authority_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CertificateAuthority(name='name_value', type_=resources.CertificateAuthority.Type.SELF_SIGNED, tier=resources.CertificateAuthority.Tier.ENTERPRISE, state=resources.CertificateAuthority.State.ENABLED, pem_ca_certificates=['pem_ca_certificates_value'], gcs_bucket='gcs_bucket_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CertificateAuthority.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_certificate_authority(request)
    assert isinstance(response, resources.CertificateAuthority)
    assert response.name == 'name_value'
    assert response.type_ == resources.CertificateAuthority.Type.SELF_SIGNED
    assert response.tier == resources.CertificateAuthority.Tier.ENTERPRISE
    assert response.state == resources.CertificateAuthority.State.ENABLED
    assert response.pem_ca_certificates == ['pem_ca_certificates_value']
    assert response.gcs_bucket == 'gcs_bucket_value'

def test_get_certificate_authority_rest_required_fields(request_type=service.GetCertificateAuthorityRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_certificate_authority._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_certificate_authority._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.CertificateAuthority()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.CertificateAuthority.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_certificate_authority(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_certificate_authority_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_certificate_authority._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_certificate_authority_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_get_certificate_authority') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_get_certificate_authority') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetCertificateAuthorityRequest.pb(service.GetCertificateAuthorityRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.CertificateAuthority.to_json(resources.CertificateAuthority())
        request = service.GetCertificateAuthorityRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.CertificateAuthority()
        client.get_certificate_authority(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_certificate_authority_rest_bad_request(transport: str='rest', request_type=service.GetCertificateAuthorityRequest):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_certificate_authority(request)

def test_get_certificate_authority_rest_flattened():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CertificateAuthority()
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CertificateAuthority.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_certificate_authority(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/certificateAuthorities/*}' % client.transport._host, args[1])

def test_get_certificate_authority_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_certificate_authority(service.GetCertificateAuthorityRequest(), name='name_value')

def test_get_certificate_authority_rest_error():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListCertificateAuthoritiesRequest, dict])
def test_list_certificate_authorities_rest(request_type):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListCertificateAuthoritiesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListCertificateAuthoritiesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_certificate_authorities(request)
    assert isinstance(response, pagers.ListCertificateAuthoritiesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_certificate_authorities_rest_required_fields(request_type=service.ListCertificateAuthoritiesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_certificate_authorities._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_certificate_authorities._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListCertificateAuthoritiesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListCertificateAuthoritiesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_certificate_authorities(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_certificate_authorities_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_certificate_authorities._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_certificate_authorities_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_list_certificate_authorities') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_list_certificate_authorities') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListCertificateAuthoritiesRequest.pb(service.ListCertificateAuthoritiesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListCertificateAuthoritiesResponse.to_json(service.ListCertificateAuthoritiesResponse())
        request = service.ListCertificateAuthoritiesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListCertificateAuthoritiesResponse()
        client.list_certificate_authorities(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_certificate_authorities_rest_bad_request(transport: str='rest', request_type=service.ListCertificateAuthoritiesRequest):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_certificate_authorities(request)

def test_list_certificate_authorities_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListCertificateAuthoritiesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListCertificateAuthoritiesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_certificate_authorities(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*}/certificateAuthorities' % client.transport._host, args[1])

def test_list_certificate_authorities_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_certificate_authorities(service.ListCertificateAuthoritiesRequest(), parent='parent_value')

def test_list_certificate_authorities_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority(), resources.CertificateAuthority(), resources.CertificateAuthority()], next_page_token='abc'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[], next_page_token='def'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority()], next_page_token='ghi'), service.ListCertificateAuthoritiesResponse(certificate_authorities=[resources.CertificateAuthority(), resources.CertificateAuthority()]))
        response = response + response
        response = tuple((service.ListCertificateAuthoritiesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_certificate_authorities(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.CertificateAuthority) for i in results))
        pages = list(client.list_certificate_authorities(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.RestoreCertificateAuthorityRequest, dict])
def test_restore_certificate_authority_rest(request_type):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.restore_certificate_authority(request)
    assert response.operation.name == 'operations/spam'

def test_restore_certificate_authority_rest_required_fields(request_type=service.RestoreCertificateAuthorityRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).restore_certificate_authority._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).restore_certificate_authority._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.restore_certificate_authority(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_restore_certificate_authority_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.restore_certificate_authority._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_restore_certificate_authority_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_restore_certificate_authority') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_restore_certificate_authority') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.RestoreCertificateAuthorityRequest.pb(service.RestoreCertificateAuthorityRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.RestoreCertificateAuthorityRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.restore_certificate_authority(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_restore_certificate_authority_rest_bad_request(transport: str='rest', request_type=service.RestoreCertificateAuthorityRequest):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.restore_certificate_authority(request)

def test_restore_certificate_authority_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.restore_certificate_authority(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/certificateAuthorities/*}:restore' % client.transport._host, args[1])

def test_restore_certificate_authority_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.restore_certificate_authority(service.RestoreCertificateAuthorityRequest(), name='name_value')

def test_restore_certificate_authority_rest_error():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ScheduleDeleteCertificateAuthorityRequest, dict])
def test_schedule_delete_certificate_authority_rest(request_type):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.schedule_delete_certificate_authority(request)
    assert response.operation.name == 'operations/spam'

def test_schedule_delete_certificate_authority_rest_required_fields(request_type=service.ScheduleDeleteCertificateAuthorityRequest):
    if False:
        return 10
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).schedule_delete_certificate_authority._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).schedule_delete_certificate_authority._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.schedule_delete_certificate_authority(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_schedule_delete_certificate_authority_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.schedule_delete_certificate_authority._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_schedule_delete_certificate_authority_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_schedule_delete_certificate_authority') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_schedule_delete_certificate_authority') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ScheduleDeleteCertificateAuthorityRequest.pb(service.ScheduleDeleteCertificateAuthorityRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.ScheduleDeleteCertificateAuthorityRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.schedule_delete_certificate_authority(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_schedule_delete_certificate_authority_rest_bad_request(transport: str='rest', request_type=service.ScheduleDeleteCertificateAuthorityRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.schedule_delete_certificate_authority(request)

def test_schedule_delete_certificate_authority_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.schedule_delete_certificate_authority(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/certificateAuthorities/*}:scheduleDelete' % client.transport._host, args[1])

def test_schedule_delete_certificate_authority_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.schedule_delete_certificate_authority(service.ScheduleDeleteCertificateAuthorityRequest(), name='name_value')

def test_schedule_delete_certificate_authority_rest_error():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateCertificateAuthorityRequest, dict])
def test_update_certificate_authority_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'certificate_authority': {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}}
    request_init['certificate_authority'] = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3', 'type_': 1, 'tier': 1, 'config': {'subject_config': {'subject': {'country_code': 'country_code_value', 'organization': 'organization_value', 'organizational_unit': 'organizational_unit_value', 'locality': 'locality_value', 'province': 'province_value', 'street_address': 'street_address_value', 'postal_code': 'postal_code_value'}, 'common_name': 'common_name_value', 'subject_alt_name': {'dns_names': ['dns_names_value1', 'dns_names_value2'], 'uris': ['uris_value1', 'uris_value2'], 'email_addresses': ['email_addresses_value1', 'email_addresses_value2'], 'ip_addresses': ['ip_addresses_value1', 'ip_addresses_value2'], 'custom_sans': [{'object_id': {'object_id_path': [1456, 1457]}, 'critical': True, 'value': b'value_blob'}]}}, 'reusable_config': {'reusable_config': 'reusable_config_value', 'reusable_config_values': {'key_usage': {'base_key_usage': {'digital_signature': True, 'content_commitment': True, 'key_encipherment': True, 'data_encipherment': True, 'key_agreement': True, 'cert_sign': True, 'crl_sign': True, 'encipher_only': True, 'decipher_only': True}, 'extended_key_usage': {'server_auth': True, 'client_auth': True, 'code_signing': True, 'email_protection': True, 'time_stamping': True, 'ocsp_signing': True}, 'unknown_extended_key_usages': {}}, 'ca_options': {'is_ca': {'value': True}, 'max_issuer_path_length': {'value': 541}}, 'policy_ids': {}, 'aia_ocsp_servers': ['aia_ocsp_servers_value1', 'aia_ocsp_servers_value2'], 'additional_extensions': {}}}, 'public_key': {'type_': 1, 'key': b'key_blob'}}, 'lifetime': {'seconds': 751, 'nanos': 543}, 'key_spec': {'cloud_kms_key_version': 'cloud_kms_key_version_value', 'algorithm': 1}, 'certificate_policy': {'allowed_config_list': {'allowed_config_values': {}}, 'overwrite_config_values': {}, 'allowed_locations_and_organizations': {}, 'allowed_common_names': ['allowed_common_names_value1', 'allowed_common_names_value2'], 'allowed_sans': {'allowed_dns_names': ['allowed_dns_names_value1', 'allowed_dns_names_value2'], 'allowed_uris': ['allowed_uris_value1', 'allowed_uris_value2'], 'allowed_email_addresses': ['allowed_email_addresses_value1', 'allowed_email_addresses_value2'], 'allowed_ips': ['allowed_ips_value1', 'allowed_ips_value2'], 'allow_globbing_dns_wildcards': True, 'allow_custom_sans': True}, 'maximum_lifetime': {}, 'allowed_issuance_modes': {'allow_csr_based_issuance': True, 'allow_config_based_issuance': True}}, 'issuing_options': {'include_ca_cert_url': True, 'include_crl_access_url': True}, 'subordinate_config': {'certificate_authority': 'certificate_authority_value', 'pem_issuer_chain': {'pem_certificates': ['pem_certificates_value1', 'pem_certificates_value2']}}, 'state': 1, 'pem_ca_certificates': ['pem_ca_certificates_value1', 'pem_ca_certificates_value2'], 'ca_certificate_descriptions': [{'subject_description': {'subject': {}, 'common_name': 'common_name_value', 'subject_alt_name': {}, 'hex_serial_number': 'hex_serial_number_value', 'lifetime': {}, 'not_before_time': {'seconds': 751, 'nanos': 543}, 'not_after_time': {}}, 'config_values': {}, 'public_key': {}, 'subject_key_id': {'key_id': 'key_id_value'}, 'authority_key_id': {}, 'crl_distribution_points': ['crl_distribution_points_value1', 'crl_distribution_points_value2'], 'aia_issuing_certificate_urls': ['aia_issuing_certificate_urls_value1', 'aia_issuing_certificate_urls_value2'], 'cert_fingerprint': {'sha256_hash': 'sha256_hash_value'}}], 'gcs_bucket': 'gcs_bucket_value', 'access_urls': {'ca_certificate_access_url': 'ca_certificate_access_url_value', 'crl_access_url': 'crl_access_url_value'}, 'create_time': {}, 'update_time': {}, 'delete_time': {}, 'labels': {}}
    test_field = service.UpdateCertificateAuthorityRequest.meta.fields['certificate_authority']

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
    for (field, value) in request_init['certificate_authority'].items():
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
                for i in range(0, len(request_init['certificate_authority'][field])):
                    del request_init['certificate_authority'][field][i][subfield]
            else:
                del request_init['certificate_authority'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_certificate_authority(request)
    assert response.operation.name == 'operations/spam'

def test_update_certificate_authority_rest_required_fields(request_type=service.UpdateCertificateAuthorityRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_certificate_authority._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_certificate_authority._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_certificate_authority(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_certificate_authority_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_certificate_authority._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('certificateAuthority', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_certificate_authority_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_update_certificate_authority') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_update_certificate_authority') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateCertificateAuthorityRequest.pb(service.UpdateCertificateAuthorityRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.UpdateCertificateAuthorityRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_certificate_authority(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_certificate_authority_rest_bad_request(transport: str='rest', request_type=service.UpdateCertificateAuthorityRequest):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'certificate_authority': {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_certificate_authority(request)

def test_update_certificate_authority_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'certificate_authority': {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}}
        mock_args = dict(certificate_authority=resources.CertificateAuthority(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_certificate_authority(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{certificate_authority.name=projects/*/locations/*/certificateAuthorities/*}' % client.transport._host, args[1])

def test_update_certificate_authority_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_certificate_authority(service.UpdateCertificateAuthorityRequest(), certificate_authority=resources.CertificateAuthority(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_certificate_authority_rest_error():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetCertificateRevocationListRequest, dict])
def test_get_certificate_revocation_list_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificateRevocationLists/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CertificateRevocationList(name='name_value', sequence_number=1601, pem_crl='pem_crl_value', access_url='access_url_value', state=resources.CertificateRevocationList.State.ACTIVE)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CertificateRevocationList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_certificate_revocation_list(request)
    assert isinstance(response, resources.CertificateRevocationList)
    assert response.name == 'name_value'
    assert response.sequence_number == 1601
    assert response.pem_crl == 'pem_crl_value'
    assert response.access_url == 'access_url_value'
    assert response.state == resources.CertificateRevocationList.State.ACTIVE

def test_get_certificate_revocation_list_rest_required_fields(request_type=service.GetCertificateRevocationListRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_certificate_revocation_list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_certificate_revocation_list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.CertificateRevocationList()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.CertificateRevocationList.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_certificate_revocation_list(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_certificate_revocation_list_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_certificate_revocation_list._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_certificate_revocation_list_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_get_certificate_revocation_list') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_get_certificate_revocation_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetCertificateRevocationListRequest.pb(service.GetCertificateRevocationListRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.CertificateRevocationList.to_json(resources.CertificateRevocationList())
        request = service.GetCertificateRevocationListRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.CertificateRevocationList()
        client.get_certificate_revocation_list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_certificate_revocation_list_rest_bad_request(transport: str='rest', request_type=service.GetCertificateRevocationListRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificateRevocationLists/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_certificate_revocation_list(request)

def test_get_certificate_revocation_list_rest_flattened():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.CertificateRevocationList()
        sample_request = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificateRevocationLists/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.CertificateRevocationList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_certificate_revocation_list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/certificateAuthorities/*/certificateRevocationLists/*}' % client.transport._host, args[1])

def test_get_certificate_revocation_list_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_certificate_revocation_list(service.GetCertificateRevocationListRequest(), name='name_value')

def test_get_certificate_revocation_list_rest_error():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListCertificateRevocationListsRequest, dict])
def test_list_certificate_revocation_lists_rest(request_type):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListCertificateRevocationListsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListCertificateRevocationListsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_certificate_revocation_lists(request)
    assert isinstance(response, pagers.ListCertificateRevocationListsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_certificate_revocation_lists_rest_required_fields(request_type=service.ListCertificateRevocationListsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_certificate_revocation_lists._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_certificate_revocation_lists._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListCertificateRevocationListsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListCertificateRevocationListsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_certificate_revocation_lists(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_certificate_revocation_lists_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_certificate_revocation_lists._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_certificate_revocation_lists_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_list_certificate_revocation_lists') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_list_certificate_revocation_lists') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListCertificateRevocationListsRequest.pb(service.ListCertificateRevocationListsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListCertificateRevocationListsResponse.to_json(service.ListCertificateRevocationListsResponse())
        request = service.ListCertificateRevocationListsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListCertificateRevocationListsResponse()
        client.list_certificate_revocation_lists(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_certificate_revocation_lists_rest_bad_request(transport: str='rest', request_type=service.ListCertificateRevocationListsRequest):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_certificate_revocation_lists(request)

def test_list_certificate_revocation_lists_rest_flattened():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListCertificateRevocationListsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListCertificateRevocationListsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_certificate_revocation_lists(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*/certificateAuthorities/*}/certificateRevocationLists' % client.transport._host, args[1])

def test_list_certificate_revocation_lists_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_certificate_revocation_lists(service.ListCertificateRevocationListsRequest(), parent='parent_value')

def test_list_certificate_revocation_lists_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList(), resources.CertificateRevocationList(), resources.CertificateRevocationList()], next_page_token='abc'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[], next_page_token='def'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList()], next_page_token='ghi'), service.ListCertificateRevocationListsResponse(certificate_revocation_lists=[resources.CertificateRevocationList(), resources.CertificateRevocationList()]))
        response = response + response
        response = tuple((service.ListCertificateRevocationListsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/certificateAuthorities/sample3'}
        pager = client.list_certificate_revocation_lists(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.CertificateRevocationList) for i in results))
        pages = list(client.list_certificate_revocation_lists(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.UpdateCertificateRevocationListRequest, dict])
def test_update_certificate_revocation_list_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'certificate_revocation_list': {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificateRevocationLists/sample4'}}
    request_init['certificate_revocation_list'] = {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificateRevocationLists/sample4', 'sequence_number': 1601, 'revoked_certificates': [{'certificate': 'certificate_value', 'hex_serial_number': 'hex_serial_number_value', 'revocation_reason': 1}], 'pem_crl': 'pem_crl_value', 'access_url': 'access_url_value', 'state': 1, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}}
    test_field = service.UpdateCertificateRevocationListRequest.meta.fields['certificate_revocation_list']

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
    for (field, value) in request_init['certificate_revocation_list'].items():
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
                for i in range(0, len(request_init['certificate_revocation_list'][field])):
                    del request_init['certificate_revocation_list'][field][i][subfield]
            else:
                del request_init['certificate_revocation_list'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_certificate_revocation_list(request)
    assert response.operation.name == 'operations/spam'

def test_update_certificate_revocation_list_rest_required_fields(request_type=service.UpdateCertificateRevocationListRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_certificate_revocation_list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_certificate_revocation_list._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_certificate_revocation_list(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_certificate_revocation_list_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_certificate_revocation_list._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('certificateRevocationList', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_certificate_revocation_list_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_update_certificate_revocation_list') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_update_certificate_revocation_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateCertificateRevocationListRequest.pb(service.UpdateCertificateRevocationListRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.UpdateCertificateRevocationListRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_certificate_revocation_list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_certificate_revocation_list_rest_bad_request(transport: str='rest', request_type=service.UpdateCertificateRevocationListRequest):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'certificate_revocation_list': {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificateRevocationLists/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_certificate_revocation_list(request)

def test_update_certificate_revocation_list_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'certificate_revocation_list': {'name': 'projects/sample1/locations/sample2/certificateAuthorities/sample3/certificateRevocationLists/sample4'}}
        mock_args = dict(certificate_revocation_list=resources.CertificateRevocationList(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_certificate_revocation_list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{certificate_revocation_list.name=projects/*/locations/*/certificateAuthorities/*/certificateRevocationLists/*}' % client.transport._host, args[1])

def test_update_certificate_revocation_list_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_certificate_revocation_list(service.UpdateCertificateRevocationListRequest(), certificate_revocation_list=resources.CertificateRevocationList(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_certificate_revocation_list_rest_error():
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetReusableConfigRequest, dict])
def test_get_reusable_config_rest(request_type):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/reusableConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.ReusableConfig(name='name_value', description='description_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.ReusableConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_reusable_config(request)
    assert isinstance(response, resources.ReusableConfig)
    assert response.name == 'name_value'
    assert response.description == 'description_value'

def test_get_reusable_config_rest_required_fields(request_type=service.GetReusableConfigRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_reusable_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_reusable_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.ReusableConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.ReusableConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_reusable_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_reusable_config_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_reusable_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_reusable_config_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_get_reusable_config') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_get_reusable_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetReusableConfigRequest.pb(service.GetReusableConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.ReusableConfig.to_json(resources.ReusableConfig())
        request = service.GetReusableConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.ReusableConfig()
        client.get_reusable_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_reusable_config_rest_bad_request(transport: str='rest', request_type=service.GetReusableConfigRequest):
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/reusableConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_reusable_config(request)

def test_get_reusable_config_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.ReusableConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/reusableConfigs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.ReusableConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_reusable_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/reusableConfigs/*}' % client.transport._host, args[1])

def test_get_reusable_config_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_reusable_config(service.GetReusableConfigRequest(), name='name_value')

def test_get_reusable_config_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListReusableConfigsRequest, dict])
def test_list_reusable_configs_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListReusableConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListReusableConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_reusable_configs(request)
    assert isinstance(response, pagers.ListReusableConfigsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_reusable_configs_rest_required_fields(request_type=service.ListReusableConfigsRequest):
    if False:
        return 10
    transport_class = transports.CertificateAuthorityServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_reusable_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_reusable_configs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListReusableConfigsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListReusableConfigsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_reusable_configs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_reusable_configs_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_reusable_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_reusable_configs_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CertificateAuthorityServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CertificateAuthorityServiceRestInterceptor())
    client = CertificateAuthorityServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'post_list_reusable_configs') as post, mock.patch.object(transports.CertificateAuthorityServiceRestInterceptor, 'pre_list_reusable_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListReusableConfigsRequest.pb(service.ListReusableConfigsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListReusableConfigsResponse.to_json(service.ListReusableConfigsResponse())
        request = service.ListReusableConfigsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListReusableConfigsResponse()
        client.list_reusable_configs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_reusable_configs_rest_bad_request(transport: str='rest', request_type=service.ListReusableConfigsRequest):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_reusable_configs(request)

def test_list_reusable_configs_rest_flattened():
    if False:
        return 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListReusableConfigsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListReusableConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_reusable_configs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*}/reusableConfigs' % client.transport._host, args[1])

def test_list_reusable_configs_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_reusable_configs(service.ListReusableConfigsRequest(), parent='parent_value')

def test_list_reusable_configs_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig(), resources.ReusableConfig(), resources.ReusableConfig()], next_page_token='abc'), service.ListReusableConfigsResponse(reusable_configs=[], next_page_token='def'), service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig()], next_page_token='ghi'), service.ListReusableConfigsResponse(reusable_configs=[resources.ReusableConfig(), resources.ReusableConfig()]))
        response = response + response
        response = tuple((service.ListReusableConfigsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_reusable_configs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.ReusableConfig) for i in results))
        pages = list(client.list_reusable_configs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.CertificateAuthorityServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CertificateAuthorityServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CertificateAuthorityServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CertificateAuthorityServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CertificateAuthorityServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CertificateAuthorityServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CertificateAuthorityServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CertificateAuthorityServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.CertificateAuthorityServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CertificateAuthorityServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.CertificateAuthorityServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CertificateAuthorityServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CertificateAuthorityServiceGrpcTransport, transports.CertificateAuthorityServiceGrpcAsyncIOTransport, transports.CertificateAuthorityServiceRestTransport])
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
        return 10
    transport = CertificateAuthorityServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CertificateAuthorityServiceGrpcTransport)

def test_certificate_authority_service_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CertificateAuthorityServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_certificate_authority_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.security.privateca_v1beta1.services.certificate_authority_service.transports.CertificateAuthorityServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CertificateAuthorityServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_certificate', 'get_certificate', 'list_certificates', 'revoke_certificate', 'update_certificate', 'activate_certificate_authority', 'create_certificate_authority', 'disable_certificate_authority', 'enable_certificate_authority', 'fetch_certificate_authority_csr', 'get_certificate_authority', 'list_certificate_authorities', 'restore_certificate_authority', 'schedule_delete_certificate_authority', 'update_certificate_authority', 'get_certificate_revocation_list', 'list_certificate_revocation_lists', 'update_certificate_revocation_list', 'get_reusable_config', 'list_reusable_configs')
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

def test_certificate_authority_service_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.security.privateca_v1beta1.services.certificate_authority_service.transports.CertificateAuthorityServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CertificateAuthorityServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_certificate_authority_service_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.security.privateca_v1beta1.services.certificate_authority_service.transports.CertificateAuthorityServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CertificateAuthorityServiceTransport()
        adc.assert_called_once()

def test_certificate_authority_service_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CertificateAuthorityServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CertificateAuthorityServiceGrpcTransport, transports.CertificateAuthorityServiceGrpcAsyncIOTransport])
def test_certificate_authority_service_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CertificateAuthorityServiceGrpcTransport, transports.CertificateAuthorityServiceGrpcAsyncIOTransport, transports.CertificateAuthorityServiceRestTransport])
def test_certificate_authority_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CertificateAuthorityServiceGrpcTransport, grpc_helpers), (transports.CertificateAuthorityServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_certificate_authority_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('privateca.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='privateca.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CertificateAuthorityServiceGrpcTransport, transports.CertificateAuthorityServiceGrpcAsyncIOTransport])
def test_certificate_authority_service_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        print('Hello World!')
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

def test_certificate_authority_service_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.CertificateAuthorityServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_certificate_authority_service_rest_lro_client():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_certificate_authority_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='privateca.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('privateca.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://privateca.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_certificate_authority_service_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='privateca.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('privateca.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://privateca.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_certificate_authority_service_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = CertificateAuthorityServiceClient(credentials=creds1, transport=transport_name)
    client2 = CertificateAuthorityServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_certificate._session
    session2 = client2.transport.create_certificate._session
    assert session1 != session2
    session1 = client1.transport.get_certificate._session
    session2 = client2.transport.get_certificate._session
    assert session1 != session2
    session1 = client1.transport.list_certificates._session
    session2 = client2.transport.list_certificates._session
    assert session1 != session2
    session1 = client1.transport.revoke_certificate._session
    session2 = client2.transport.revoke_certificate._session
    assert session1 != session2
    session1 = client1.transport.update_certificate._session
    session2 = client2.transport.update_certificate._session
    assert session1 != session2
    session1 = client1.transport.activate_certificate_authority._session
    session2 = client2.transport.activate_certificate_authority._session
    assert session1 != session2
    session1 = client1.transport.create_certificate_authority._session
    session2 = client2.transport.create_certificate_authority._session
    assert session1 != session2
    session1 = client1.transport.disable_certificate_authority._session
    session2 = client2.transport.disable_certificate_authority._session
    assert session1 != session2
    session1 = client1.transport.enable_certificate_authority._session
    session2 = client2.transport.enable_certificate_authority._session
    assert session1 != session2
    session1 = client1.transport.fetch_certificate_authority_csr._session
    session2 = client2.transport.fetch_certificate_authority_csr._session
    assert session1 != session2
    session1 = client1.transport.get_certificate_authority._session
    session2 = client2.transport.get_certificate_authority._session
    assert session1 != session2
    session1 = client1.transport.list_certificate_authorities._session
    session2 = client2.transport.list_certificate_authorities._session
    assert session1 != session2
    session1 = client1.transport.restore_certificate_authority._session
    session2 = client2.transport.restore_certificate_authority._session
    assert session1 != session2
    session1 = client1.transport.schedule_delete_certificate_authority._session
    session2 = client2.transport.schedule_delete_certificate_authority._session
    assert session1 != session2
    session1 = client1.transport.update_certificate_authority._session
    session2 = client2.transport.update_certificate_authority._session
    assert session1 != session2
    session1 = client1.transport.get_certificate_revocation_list._session
    session2 = client2.transport.get_certificate_revocation_list._session
    assert session1 != session2
    session1 = client1.transport.list_certificate_revocation_lists._session
    session2 = client2.transport.list_certificate_revocation_lists._session
    assert session1 != session2
    session1 = client1.transport.update_certificate_revocation_list._session
    session2 = client2.transport.update_certificate_revocation_list._session
    assert session1 != session2
    session1 = client1.transport.get_reusable_config._session
    session2 = client2.transport.get_reusable_config._session
    assert session1 != session2
    session1 = client1.transport.list_reusable_configs._session
    session2 = client2.transport.list_reusable_configs._session
    assert session1 != session2

def test_certificate_authority_service_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CertificateAuthorityServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_certificate_authority_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CertificateAuthorityServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CertificateAuthorityServiceGrpcTransport, transports.CertificateAuthorityServiceGrpcAsyncIOTransport])
def test_certificate_authority_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.CertificateAuthorityServiceGrpcTransport, transports.CertificateAuthorityServiceGrpcAsyncIOTransport])
def test_certificate_authority_service_transport_channel_mtls_with_adc(transport_class):
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

def test_certificate_authority_service_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_certificate_authority_service_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_certificate_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    certificate_authority = 'whelk'
    certificate = 'octopus'
    expected = 'projects/{project}/locations/{location}/certificateAuthorities/{certificate_authority}/certificates/{certificate}'.format(project=project, location=location, certificate_authority=certificate_authority, certificate=certificate)
    actual = CertificateAuthorityServiceClient.certificate_path(project, location, certificate_authority, certificate)
    assert expected == actual

def test_parse_certificate_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'oyster', 'location': 'nudibranch', 'certificate_authority': 'cuttlefish', 'certificate': 'mussel'}
    path = CertificateAuthorityServiceClient.certificate_path(**expected)
    actual = CertificateAuthorityServiceClient.parse_certificate_path(path)
    assert expected == actual

def test_certificate_authority_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    certificate_authority = 'scallop'
    expected = 'projects/{project}/locations/{location}/certificateAuthorities/{certificate_authority}'.format(project=project, location=location, certificate_authority=certificate_authority)
    actual = CertificateAuthorityServiceClient.certificate_authority_path(project, location, certificate_authority)
    assert expected == actual

def test_parse_certificate_authority_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone', 'location': 'squid', 'certificate_authority': 'clam'}
    path = CertificateAuthorityServiceClient.certificate_authority_path(**expected)
    actual = CertificateAuthorityServiceClient.parse_certificate_authority_path(path)
    assert expected == actual

def test_certificate_revocation_list_path():
    if False:
        i = 10
        return i + 15
    project = 'whelk'
    location = 'octopus'
    certificate_authority = 'oyster'
    certificate_revocation_list = 'nudibranch'
    expected = 'projects/{project}/locations/{location}/certificateAuthorities/{certificate_authority}/certificateRevocationLists/{certificate_revocation_list}'.format(project=project, location=location, certificate_authority=certificate_authority, certificate_revocation_list=certificate_revocation_list)
    actual = CertificateAuthorityServiceClient.certificate_revocation_list_path(project, location, certificate_authority, certificate_revocation_list)
    assert expected == actual

def test_parse_certificate_revocation_list_path():
    if False:
        print('Hello World!')
    expected = {'project': 'cuttlefish', 'location': 'mussel', 'certificate_authority': 'winkle', 'certificate_revocation_list': 'nautilus'}
    path = CertificateAuthorityServiceClient.certificate_revocation_list_path(**expected)
    actual = CertificateAuthorityServiceClient.parse_certificate_revocation_list_path(path)
    assert expected == actual

def test_reusable_config_path():
    if False:
        i = 10
        return i + 15
    project = 'scallop'
    location = 'abalone'
    reusable_config = 'squid'
    expected = 'projects/{project}/locations/{location}/reusableConfigs/{reusable_config}'.format(project=project, location=location, reusable_config=reusable_config)
    actual = CertificateAuthorityServiceClient.reusable_config_path(project, location, reusable_config)
    assert expected == actual

def test_parse_reusable_config_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'clam', 'location': 'whelk', 'reusable_config': 'octopus'}
    path = CertificateAuthorityServiceClient.reusable_config_path(**expected)
    actual = CertificateAuthorityServiceClient.parse_reusable_config_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CertificateAuthorityServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'nudibranch'}
    path = CertificateAuthorityServiceClient.common_billing_account_path(**expected)
    actual = CertificateAuthorityServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CertificateAuthorityServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'mussel'}
    path = CertificateAuthorityServiceClient.common_folder_path(**expected)
    actual = CertificateAuthorityServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CertificateAuthorityServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'nautilus'}
    path = CertificateAuthorityServiceClient.common_organization_path(**expected)
    actual = CertificateAuthorityServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = CertificateAuthorityServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'abalone'}
    path = CertificateAuthorityServiceClient.common_project_path(**expected)
    actual = CertificateAuthorityServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CertificateAuthorityServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = CertificateAuthorityServiceClient.common_location_path(**expected)
    actual = CertificateAuthorityServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CertificateAuthorityServiceTransport, '_prep_wrapped_messages') as prep:
        client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CertificateAuthorityServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = CertificateAuthorityServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CertificateAuthorityServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = CertificateAuthorityServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CertificateAuthorityServiceClient, transports.CertificateAuthorityServiceGrpcTransport), (CertificateAuthorityServiceAsyncClient, transports.CertificateAuthorityServiceGrpcAsyncIOTransport)])
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