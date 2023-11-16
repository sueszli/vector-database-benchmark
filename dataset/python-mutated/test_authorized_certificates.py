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
from google.cloud.appengine_admin_v1.services.authorized_certificates import AuthorizedCertificatesAsyncClient, AuthorizedCertificatesClient, pagers, transports
from google.cloud.appengine_admin_v1.types import appengine, certificate

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
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert AuthorizedCertificatesClient._get_default_mtls_endpoint(None) is None
    assert AuthorizedCertificatesClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert AuthorizedCertificatesClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert AuthorizedCertificatesClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert AuthorizedCertificatesClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert AuthorizedCertificatesClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(AuthorizedCertificatesClient, 'grpc'), (AuthorizedCertificatesAsyncClient, 'grpc_asyncio'), (AuthorizedCertificatesClient, 'rest')])
def test_authorized_certificates_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('appengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://appengine.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.AuthorizedCertificatesGrpcTransport, 'grpc'), (transports.AuthorizedCertificatesGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.AuthorizedCertificatesRestTransport, 'rest')])
def test_authorized_certificates_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(AuthorizedCertificatesClient, 'grpc'), (AuthorizedCertificatesAsyncClient, 'grpc_asyncio'), (AuthorizedCertificatesClient, 'rest')])
def test_authorized_certificates_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('appengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://appengine.googleapis.com')

def test_authorized_certificates_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = AuthorizedCertificatesClient.get_transport_class()
    available_transports = [transports.AuthorizedCertificatesGrpcTransport, transports.AuthorizedCertificatesRestTransport]
    assert transport in available_transports
    transport = AuthorizedCertificatesClient.get_transport_class('grpc')
    assert transport == transports.AuthorizedCertificatesGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AuthorizedCertificatesClient, transports.AuthorizedCertificatesGrpcTransport, 'grpc'), (AuthorizedCertificatesAsyncClient, transports.AuthorizedCertificatesGrpcAsyncIOTransport, 'grpc_asyncio'), (AuthorizedCertificatesClient, transports.AuthorizedCertificatesRestTransport, 'rest')])
@mock.patch.object(AuthorizedCertificatesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AuthorizedCertificatesClient))
@mock.patch.object(AuthorizedCertificatesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AuthorizedCertificatesAsyncClient))
def test_authorized_certificates_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(AuthorizedCertificatesClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(AuthorizedCertificatesClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(AuthorizedCertificatesClient, transports.AuthorizedCertificatesGrpcTransport, 'grpc', 'true'), (AuthorizedCertificatesAsyncClient, transports.AuthorizedCertificatesGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (AuthorizedCertificatesClient, transports.AuthorizedCertificatesGrpcTransport, 'grpc', 'false'), (AuthorizedCertificatesAsyncClient, transports.AuthorizedCertificatesGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (AuthorizedCertificatesClient, transports.AuthorizedCertificatesRestTransport, 'rest', 'true'), (AuthorizedCertificatesClient, transports.AuthorizedCertificatesRestTransport, 'rest', 'false')])
@mock.patch.object(AuthorizedCertificatesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AuthorizedCertificatesClient))
@mock.patch.object(AuthorizedCertificatesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AuthorizedCertificatesAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_authorized_certificates_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [AuthorizedCertificatesClient, AuthorizedCertificatesAsyncClient])
@mock.patch.object(AuthorizedCertificatesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AuthorizedCertificatesClient))
@mock.patch.object(AuthorizedCertificatesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AuthorizedCertificatesAsyncClient))
def test_authorized_certificates_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AuthorizedCertificatesClient, transports.AuthorizedCertificatesGrpcTransport, 'grpc'), (AuthorizedCertificatesAsyncClient, transports.AuthorizedCertificatesGrpcAsyncIOTransport, 'grpc_asyncio'), (AuthorizedCertificatesClient, transports.AuthorizedCertificatesRestTransport, 'rest')])
def test_authorized_certificates_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AuthorizedCertificatesClient, transports.AuthorizedCertificatesGrpcTransport, 'grpc', grpc_helpers), (AuthorizedCertificatesAsyncClient, transports.AuthorizedCertificatesGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (AuthorizedCertificatesClient, transports.AuthorizedCertificatesRestTransport, 'rest', None)])
def test_authorized_certificates_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_authorized_certificates_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.appengine_admin_v1.services.authorized_certificates.transports.AuthorizedCertificatesGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = AuthorizedCertificatesClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AuthorizedCertificatesClient, transports.AuthorizedCertificatesGrpcTransport, 'grpc', grpc_helpers), (AuthorizedCertificatesAsyncClient, transports.AuthorizedCertificatesGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_authorized_certificates_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('appengine.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/appengine.admin', 'https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), scopes=None, default_host='appengine.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [appengine.ListAuthorizedCertificatesRequest, dict])
def test_list_authorized_certificates(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_authorized_certificates), '__call__') as call:
        call.return_value = appengine.ListAuthorizedCertificatesResponse(next_page_token='next_page_token_value')
        response = client.list_authorized_certificates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.ListAuthorizedCertificatesRequest()
    assert isinstance(response, pagers.ListAuthorizedCertificatesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_authorized_certificates_empty_call():
    if False:
        i = 10
        return i + 15
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_authorized_certificates), '__call__') as call:
        client.list_authorized_certificates()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.ListAuthorizedCertificatesRequest()

@pytest.mark.asyncio
async def test_list_authorized_certificates_async(transport: str='grpc_asyncio', request_type=appengine.ListAuthorizedCertificatesRequest):
    client = AuthorizedCertificatesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_authorized_certificates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(appengine.ListAuthorizedCertificatesResponse(next_page_token='next_page_token_value'))
        response = await client.list_authorized_certificates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.ListAuthorizedCertificatesRequest()
    assert isinstance(response, pagers.ListAuthorizedCertificatesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_authorized_certificates_async_from_dict():
    await test_list_authorized_certificates_async(request_type=dict)

def test_list_authorized_certificates_field_headers():
    if False:
        while True:
            i = 10
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials())
    request = appengine.ListAuthorizedCertificatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_authorized_certificates), '__call__') as call:
        call.return_value = appengine.ListAuthorizedCertificatesResponse()
        client.list_authorized_certificates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_authorized_certificates_field_headers_async():
    client = AuthorizedCertificatesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = appengine.ListAuthorizedCertificatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_authorized_certificates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(appengine.ListAuthorizedCertificatesResponse())
        await client.list_authorized_certificates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_authorized_certificates_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_authorized_certificates), '__call__') as call:
        call.side_effect = (appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate()], next_page_token='abc'), appengine.ListAuthorizedCertificatesResponse(certificates=[], next_page_token='def'), appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate()], next_page_token='ghi'), appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_authorized_certificates(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, certificate.AuthorizedCertificate) for i in results))

def test_list_authorized_certificates_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_authorized_certificates), '__call__') as call:
        call.side_effect = (appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate()], next_page_token='abc'), appengine.ListAuthorizedCertificatesResponse(certificates=[], next_page_token='def'), appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate()], next_page_token='ghi'), appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate()]), RuntimeError)
        pages = list(client.list_authorized_certificates(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_authorized_certificates_async_pager():
    client = AuthorizedCertificatesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_authorized_certificates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate()], next_page_token='abc'), appengine.ListAuthorizedCertificatesResponse(certificates=[], next_page_token='def'), appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate()], next_page_token='ghi'), appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate()]), RuntimeError)
        async_pager = await client.list_authorized_certificates(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, certificate.AuthorizedCertificate) for i in responses))

@pytest.mark.asyncio
async def test_list_authorized_certificates_async_pages():
    client = AuthorizedCertificatesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_authorized_certificates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate()], next_page_token='abc'), appengine.ListAuthorizedCertificatesResponse(certificates=[], next_page_token='def'), appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate()], next_page_token='ghi'), appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_authorized_certificates(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [appengine.GetAuthorizedCertificateRequest, dict])
def test_get_authorized_certificate(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_authorized_certificate), '__call__') as call:
        call.return_value = certificate.AuthorizedCertificate(name='name_value', id='id_value', display_name='display_name_value', domain_names=['domain_names_value'], visible_domain_mappings=['visible_domain_mappings_value'], domain_mappings_count=2238)
        response = client.get_authorized_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.GetAuthorizedCertificateRequest()
    assert isinstance(response, certificate.AuthorizedCertificate)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.display_name == 'display_name_value'
    assert response.domain_names == ['domain_names_value']
    assert response.visible_domain_mappings == ['visible_domain_mappings_value']
    assert response.domain_mappings_count == 2238

def test_get_authorized_certificate_empty_call():
    if False:
        print('Hello World!')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_authorized_certificate), '__call__') as call:
        client.get_authorized_certificate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.GetAuthorizedCertificateRequest()

@pytest.mark.asyncio
async def test_get_authorized_certificate_async(transport: str='grpc_asyncio', request_type=appengine.GetAuthorizedCertificateRequest):
    client = AuthorizedCertificatesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_authorized_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate.AuthorizedCertificate(name='name_value', id='id_value', display_name='display_name_value', domain_names=['domain_names_value'], visible_domain_mappings=['visible_domain_mappings_value'], domain_mappings_count=2238))
        response = await client.get_authorized_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.GetAuthorizedCertificateRequest()
    assert isinstance(response, certificate.AuthorizedCertificate)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.display_name == 'display_name_value'
    assert response.domain_names == ['domain_names_value']
    assert response.visible_domain_mappings == ['visible_domain_mappings_value']
    assert response.domain_mappings_count == 2238

@pytest.mark.asyncio
async def test_get_authorized_certificate_async_from_dict():
    await test_get_authorized_certificate_async(request_type=dict)

def test_get_authorized_certificate_field_headers():
    if False:
        print('Hello World!')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials())
    request = appengine.GetAuthorizedCertificateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_authorized_certificate), '__call__') as call:
        call.return_value = certificate.AuthorizedCertificate()
        client.get_authorized_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_authorized_certificate_field_headers_async():
    client = AuthorizedCertificatesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = appengine.GetAuthorizedCertificateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_authorized_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate.AuthorizedCertificate())
        await client.get_authorized_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [appengine.CreateAuthorizedCertificateRequest, dict])
def test_create_authorized_certificate(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_authorized_certificate), '__call__') as call:
        call.return_value = certificate.AuthorizedCertificate(name='name_value', id='id_value', display_name='display_name_value', domain_names=['domain_names_value'], visible_domain_mappings=['visible_domain_mappings_value'], domain_mappings_count=2238)
        response = client.create_authorized_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.CreateAuthorizedCertificateRequest()
    assert isinstance(response, certificate.AuthorizedCertificate)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.display_name == 'display_name_value'
    assert response.domain_names == ['domain_names_value']
    assert response.visible_domain_mappings == ['visible_domain_mappings_value']
    assert response.domain_mappings_count == 2238

def test_create_authorized_certificate_empty_call():
    if False:
        i = 10
        return i + 15
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_authorized_certificate), '__call__') as call:
        client.create_authorized_certificate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.CreateAuthorizedCertificateRequest()

@pytest.mark.asyncio
async def test_create_authorized_certificate_async(transport: str='grpc_asyncio', request_type=appengine.CreateAuthorizedCertificateRequest):
    client = AuthorizedCertificatesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_authorized_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate.AuthorizedCertificate(name='name_value', id='id_value', display_name='display_name_value', domain_names=['domain_names_value'], visible_domain_mappings=['visible_domain_mappings_value'], domain_mappings_count=2238))
        response = await client.create_authorized_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.CreateAuthorizedCertificateRequest()
    assert isinstance(response, certificate.AuthorizedCertificate)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.display_name == 'display_name_value'
    assert response.domain_names == ['domain_names_value']
    assert response.visible_domain_mappings == ['visible_domain_mappings_value']
    assert response.domain_mappings_count == 2238

@pytest.mark.asyncio
async def test_create_authorized_certificate_async_from_dict():
    await test_create_authorized_certificate_async(request_type=dict)

def test_create_authorized_certificate_field_headers():
    if False:
        print('Hello World!')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials())
    request = appengine.CreateAuthorizedCertificateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_authorized_certificate), '__call__') as call:
        call.return_value = certificate.AuthorizedCertificate()
        client.create_authorized_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_authorized_certificate_field_headers_async():
    client = AuthorizedCertificatesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = appengine.CreateAuthorizedCertificateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_authorized_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate.AuthorizedCertificate())
        await client.create_authorized_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [appengine.UpdateAuthorizedCertificateRequest, dict])
def test_update_authorized_certificate(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_authorized_certificate), '__call__') as call:
        call.return_value = certificate.AuthorizedCertificate(name='name_value', id='id_value', display_name='display_name_value', domain_names=['domain_names_value'], visible_domain_mappings=['visible_domain_mappings_value'], domain_mappings_count=2238)
        response = client.update_authorized_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.UpdateAuthorizedCertificateRequest()
    assert isinstance(response, certificate.AuthorizedCertificate)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.display_name == 'display_name_value'
    assert response.domain_names == ['domain_names_value']
    assert response.visible_domain_mappings == ['visible_domain_mappings_value']
    assert response.domain_mappings_count == 2238

def test_update_authorized_certificate_empty_call():
    if False:
        while True:
            i = 10
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_authorized_certificate), '__call__') as call:
        client.update_authorized_certificate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.UpdateAuthorizedCertificateRequest()

@pytest.mark.asyncio
async def test_update_authorized_certificate_async(transport: str='grpc_asyncio', request_type=appengine.UpdateAuthorizedCertificateRequest):
    client = AuthorizedCertificatesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_authorized_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate.AuthorizedCertificate(name='name_value', id='id_value', display_name='display_name_value', domain_names=['domain_names_value'], visible_domain_mappings=['visible_domain_mappings_value'], domain_mappings_count=2238))
        response = await client.update_authorized_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.UpdateAuthorizedCertificateRequest()
    assert isinstance(response, certificate.AuthorizedCertificate)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.display_name == 'display_name_value'
    assert response.domain_names == ['domain_names_value']
    assert response.visible_domain_mappings == ['visible_domain_mappings_value']
    assert response.domain_mappings_count == 2238

@pytest.mark.asyncio
async def test_update_authorized_certificate_async_from_dict():
    await test_update_authorized_certificate_async(request_type=dict)

def test_update_authorized_certificate_field_headers():
    if False:
        print('Hello World!')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials())
    request = appengine.UpdateAuthorizedCertificateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_authorized_certificate), '__call__') as call:
        call.return_value = certificate.AuthorizedCertificate()
        client.update_authorized_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_authorized_certificate_field_headers_async():
    client = AuthorizedCertificatesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = appengine.UpdateAuthorizedCertificateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_authorized_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(certificate.AuthorizedCertificate())
        await client.update_authorized_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [appengine.DeleteAuthorizedCertificateRequest, dict])
def test_delete_authorized_certificate(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_authorized_certificate), '__call__') as call:
        call.return_value = None
        response = client.delete_authorized_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.DeleteAuthorizedCertificateRequest()
    assert response is None

def test_delete_authorized_certificate_empty_call():
    if False:
        print('Hello World!')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_authorized_certificate), '__call__') as call:
        client.delete_authorized_certificate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.DeleteAuthorizedCertificateRequest()

@pytest.mark.asyncio
async def test_delete_authorized_certificate_async(transport: str='grpc_asyncio', request_type=appengine.DeleteAuthorizedCertificateRequest):
    client = AuthorizedCertificatesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_authorized_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_authorized_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == appengine.DeleteAuthorizedCertificateRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_authorized_certificate_async_from_dict():
    await test_delete_authorized_certificate_async(request_type=dict)

def test_delete_authorized_certificate_field_headers():
    if False:
        print('Hello World!')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials())
    request = appengine.DeleteAuthorizedCertificateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_authorized_certificate), '__call__') as call:
        call.return_value = None
        client.delete_authorized_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_authorized_certificate_field_headers_async():
    client = AuthorizedCertificatesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = appengine.DeleteAuthorizedCertificateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_authorized_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_authorized_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [appengine.ListAuthorizedCertificatesRequest, dict])
def test_list_authorized_certificates_rest(request_type):
    if False:
        return 10
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'apps/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = appengine.ListAuthorizedCertificatesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = appengine.ListAuthorizedCertificatesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_authorized_certificates(request)
    assert isinstance(response, pagers.ListAuthorizedCertificatesPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_authorized_certificates_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AuthorizedCertificatesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AuthorizedCertificatesRestInterceptor())
    client = AuthorizedCertificatesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AuthorizedCertificatesRestInterceptor, 'post_list_authorized_certificates') as post, mock.patch.object(transports.AuthorizedCertificatesRestInterceptor, 'pre_list_authorized_certificates') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = appengine.ListAuthorizedCertificatesRequest.pb(appengine.ListAuthorizedCertificatesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = appengine.ListAuthorizedCertificatesResponse.to_json(appengine.ListAuthorizedCertificatesResponse())
        request = appengine.ListAuthorizedCertificatesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = appengine.ListAuthorizedCertificatesResponse()
        client.list_authorized_certificates(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_authorized_certificates_rest_bad_request(transport: str='rest', request_type=appengine.ListAuthorizedCertificatesRequest):
    if False:
        while True:
            i = 10
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'apps/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_authorized_certificates(request)

def test_list_authorized_certificates_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate()], next_page_token='abc'), appengine.ListAuthorizedCertificatesResponse(certificates=[], next_page_token='def'), appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate()], next_page_token='ghi'), appengine.ListAuthorizedCertificatesResponse(certificates=[certificate.AuthorizedCertificate(), certificate.AuthorizedCertificate()]))
        response = response + response
        response = tuple((appengine.ListAuthorizedCertificatesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'apps/sample1'}
        pager = client.list_authorized_certificates(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, certificate.AuthorizedCertificate) for i in results))
        pages = list(client.list_authorized_certificates(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [appengine.GetAuthorizedCertificateRequest, dict])
def test_get_authorized_certificate_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'apps/sample1/authorizedCertificates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = certificate.AuthorizedCertificate(name='name_value', id='id_value', display_name='display_name_value', domain_names=['domain_names_value'], visible_domain_mappings=['visible_domain_mappings_value'], domain_mappings_count=2238)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate.AuthorizedCertificate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_authorized_certificate(request)
    assert isinstance(response, certificate.AuthorizedCertificate)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.display_name == 'display_name_value'
    assert response.domain_names == ['domain_names_value']
    assert response.visible_domain_mappings == ['visible_domain_mappings_value']
    assert response.domain_mappings_count == 2238

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_authorized_certificate_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AuthorizedCertificatesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AuthorizedCertificatesRestInterceptor())
    client = AuthorizedCertificatesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AuthorizedCertificatesRestInterceptor, 'post_get_authorized_certificate') as post, mock.patch.object(transports.AuthorizedCertificatesRestInterceptor, 'pre_get_authorized_certificate') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = appengine.GetAuthorizedCertificateRequest.pb(appengine.GetAuthorizedCertificateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = certificate.AuthorizedCertificate.to_json(certificate.AuthorizedCertificate())
        request = appengine.GetAuthorizedCertificateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = certificate.AuthorizedCertificate()
        client.get_authorized_certificate(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_authorized_certificate_rest_bad_request(transport: str='rest', request_type=appengine.GetAuthorizedCertificateRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'apps/sample1/authorizedCertificates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_authorized_certificate(request)

def test_get_authorized_certificate_rest_error():
    if False:
        i = 10
        return i + 15
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [appengine.CreateAuthorizedCertificateRequest, dict])
def test_create_authorized_certificate_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'apps/sample1'}
    request_init['certificate'] = {'name': 'name_value', 'id': 'id_value', 'display_name': 'display_name_value', 'domain_names': ['domain_names_value1', 'domain_names_value2'], 'expire_time': {'seconds': 751, 'nanos': 543}, 'certificate_raw_data': {'public_certificate': 'public_certificate_value', 'private_key': 'private_key_value'}, 'managed_certificate': {'last_renewal_time': {}, 'status': 1}, 'visible_domain_mappings': ['visible_domain_mappings_value1', 'visible_domain_mappings_value2'], 'domain_mappings_count': 2238}
    test_field = appengine.CreateAuthorizedCertificateRequest.meta.fields['certificate']

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
        return_value = certificate.AuthorizedCertificate(name='name_value', id='id_value', display_name='display_name_value', domain_names=['domain_names_value'], visible_domain_mappings=['visible_domain_mappings_value'], domain_mappings_count=2238)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate.AuthorizedCertificate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_authorized_certificate(request)
    assert isinstance(response, certificate.AuthorizedCertificate)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.display_name == 'display_name_value'
    assert response.domain_names == ['domain_names_value']
    assert response.visible_domain_mappings == ['visible_domain_mappings_value']
    assert response.domain_mappings_count == 2238

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_authorized_certificate_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AuthorizedCertificatesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AuthorizedCertificatesRestInterceptor())
    client = AuthorizedCertificatesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AuthorizedCertificatesRestInterceptor, 'post_create_authorized_certificate') as post, mock.patch.object(transports.AuthorizedCertificatesRestInterceptor, 'pre_create_authorized_certificate') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = appengine.CreateAuthorizedCertificateRequest.pb(appengine.CreateAuthorizedCertificateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = certificate.AuthorizedCertificate.to_json(certificate.AuthorizedCertificate())
        request = appengine.CreateAuthorizedCertificateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = certificate.AuthorizedCertificate()
        client.create_authorized_certificate(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_authorized_certificate_rest_bad_request(transport: str='rest', request_type=appengine.CreateAuthorizedCertificateRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'apps/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_authorized_certificate(request)

def test_create_authorized_certificate_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [appengine.UpdateAuthorizedCertificateRequest, dict])
def test_update_authorized_certificate_rest(request_type):
    if False:
        while True:
            i = 10
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'apps/sample1/authorizedCertificates/sample2'}
    request_init['certificate'] = {'name': 'name_value', 'id': 'id_value', 'display_name': 'display_name_value', 'domain_names': ['domain_names_value1', 'domain_names_value2'], 'expire_time': {'seconds': 751, 'nanos': 543}, 'certificate_raw_data': {'public_certificate': 'public_certificate_value', 'private_key': 'private_key_value'}, 'managed_certificate': {'last_renewal_time': {}, 'status': 1}, 'visible_domain_mappings': ['visible_domain_mappings_value1', 'visible_domain_mappings_value2'], 'domain_mappings_count': 2238}
    test_field = appengine.UpdateAuthorizedCertificateRequest.meta.fields['certificate']

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
        return_value = certificate.AuthorizedCertificate(name='name_value', id='id_value', display_name='display_name_value', domain_names=['domain_names_value'], visible_domain_mappings=['visible_domain_mappings_value'], domain_mappings_count=2238)
        response_value = Response()
        response_value.status_code = 200
        return_value = certificate.AuthorizedCertificate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_authorized_certificate(request)
    assert isinstance(response, certificate.AuthorizedCertificate)
    assert response.name == 'name_value'
    assert response.id == 'id_value'
    assert response.display_name == 'display_name_value'
    assert response.domain_names == ['domain_names_value']
    assert response.visible_domain_mappings == ['visible_domain_mappings_value']
    assert response.domain_mappings_count == 2238

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_authorized_certificate_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AuthorizedCertificatesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AuthorizedCertificatesRestInterceptor())
    client = AuthorizedCertificatesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AuthorizedCertificatesRestInterceptor, 'post_update_authorized_certificate') as post, mock.patch.object(transports.AuthorizedCertificatesRestInterceptor, 'pre_update_authorized_certificate') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = appengine.UpdateAuthorizedCertificateRequest.pb(appengine.UpdateAuthorizedCertificateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = certificate.AuthorizedCertificate.to_json(certificate.AuthorizedCertificate())
        request = appengine.UpdateAuthorizedCertificateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = certificate.AuthorizedCertificate()
        client.update_authorized_certificate(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_authorized_certificate_rest_bad_request(transport: str='rest', request_type=appengine.UpdateAuthorizedCertificateRequest):
    if False:
        return 10
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'apps/sample1/authorizedCertificates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_authorized_certificate(request)

def test_update_authorized_certificate_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [appengine.DeleteAuthorizedCertificateRequest, dict])
def test_delete_authorized_certificate_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'apps/sample1/authorizedCertificates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_authorized_certificate(request)
    assert response is None

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_authorized_certificate_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AuthorizedCertificatesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AuthorizedCertificatesRestInterceptor())
    client = AuthorizedCertificatesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AuthorizedCertificatesRestInterceptor, 'pre_delete_authorized_certificate') as pre:
        pre.assert_not_called()
        pb_message = appengine.DeleteAuthorizedCertificateRequest.pb(appengine.DeleteAuthorizedCertificateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = appengine.DeleteAuthorizedCertificateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_authorized_certificate(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_authorized_certificate_rest_bad_request(transport: str='rest', request_type=appengine.DeleteAuthorizedCertificateRequest):
    if False:
        return 10
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'apps/sample1/authorizedCertificates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_authorized_certificate(request)

def test_delete_authorized_certificate_rest_error():
    if False:
        print('Hello World!')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AuthorizedCertificatesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.AuthorizedCertificatesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AuthorizedCertificatesClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.AuthorizedCertificatesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AuthorizedCertificatesClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AuthorizedCertificatesClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.AuthorizedCertificatesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AuthorizedCertificatesClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AuthorizedCertificatesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = AuthorizedCertificatesClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.AuthorizedCertificatesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.AuthorizedCertificatesGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.AuthorizedCertificatesGrpcTransport, transports.AuthorizedCertificatesGrpcAsyncIOTransport, transports.AuthorizedCertificatesRestTransport])
def test_transport_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        print('Hello World!')
    transport = AuthorizedCertificatesClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.AuthorizedCertificatesGrpcTransport)

def test_authorized_certificates_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.AuthorizedCertificatesTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_authorized_certificates_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.appengine_admin_v1.services.authorized_certificates.transports.AuthorizedCertificatesTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.AuthorizedCertificatesTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_authorized_certificates', 'get_authorized_certificate', 'create_authorized_certificate', 'update_authorized_certificate', 'delete_authorized_certificate')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_authorized_certificates_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.appengine_admin_v1.services.authorized_certificates.transports.AuthorizedCertificatesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AuthorizedCertificatesTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/appengine.admin', 'https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), quota_project_id='octopus')

def test_authorized_certificates_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.appengine_admin_v1.services.authorized_certificates.transports.AuthorizedCertificatesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AuthorizedCertificatesTransport()
        adc.assert_called_once()

def test_authorized_certificates_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        AuthorizedCertificatesClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/appengine.admin', 'https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.AuthorizedCertificatesGrpcTransport, transports.AuthorizedCertificatesGrpcAsyncIOTransport])
def test_authorized_certificates_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/appengine.admin', 'https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.AuthorizedCertificatesGrpcTransport, transports.AuthorizedCertificatesGrpcAsyncIOTransport, transports.AuthorizedCertificatesRestTransport])
def test_authorized_certificates_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.AuthorizedCertificatesGrpcTransport, grpc_helpers), (transports.AuthorizedCertificatesGrpcAsyncIOTransport, grpc_helpers_async)])
def test_authorized_certificates_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('appengine.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/appengine.admin', 'https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), scopes=['1', '2'], default_host='appengine.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.AuthorizedCertificatesGrpcTransport, transports.AuthorizedCertificatesGrpcAsyncIOTransport])
def test_authorized_certificates_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_authorized_certificates_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.AuthorizedCertificatesRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_authorized_certificates_host_no_port(transport_name):
    if False:
        return 10
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='appengine.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('appengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://appengine.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_authorized_certificates_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='appengine.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('appengine.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://appengine.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_authorized_certificates_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = AuthorizedCertificatesClient(credentials=creds1, transport=transport_name)
    client2 = AuthorizedCertificatesClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_authorized_certificates._session
    session2 = client2.transport.list_authorized_certificates._session
    assert session1 != session2
    session1 = client1.transport.get_authorized_certificate._session
    session2 = client2.transport.get_authorized_certificate._session
    assert session1 != session2
    session1 = client1.transport.create_authorized_certificate._session
    session2 = client2.transport.create_authorized_certificate._session
    assert session1 != session2
    session1 = client1.transport.update_authorized_certificate._session
    session2 = client2.transport.update_authorized_certificate._session
    assert session1 != session2
    session1 = client1.transport.delete_authorized_certificate._session
    session2 = client2.transport.delete_authorized_certificate._session
    assert session1 != session2

def test_authorized_certificates_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AuthorizedCertificatesGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_authorized_certificates_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AuthorizedCertificatesGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.AuthorizedCertificatesGrpcTransport, transports.AuthorizedCertificatesGrpcAsyncIOTransport])
def test_authorized_certificates_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.AuthorizedCertificatesGrpcTransport, transports.AuthorizedCertificatesGrpcAsyncIOTransport])
def test_authorized_certificates_transport_channel_mtls_with_adc(transport_class):
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

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = AuthorizedCertificatesClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'clam'}
    path = AuthorizedCertificatesClient.common_billing_account_path(**expected)
    actual = AuthorizedCertificatesClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = AuthorizedCertificatesClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'octopus'}
    path = AuthorizedCertificatesClient.common_folder_path(**expected)
    actual = AuthorizedCertificatesClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = AuthorizedCertificatesClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'nudibranch'}
    path = AuthorizedCertificatesClient.common_organization_path(**expected)
    actual = AuthorizedCertificatesClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = AuthorizedCertificatesClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'mussel'}
    path = AuthorizedCertificatesClient.common_project_path(**expected)
    actual = AuthorizedCertificatesClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = AuthorizedCertificatesClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = AuthorizedCertificatesClient.common_location_path(**expected)
    actual = AuthorizedCertificatesClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.AuthorizedCertificatesTransport, '_prep_wrapped_messages') as prep:
        client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.AuthorizedCertificatesTransport, '_prep_wrapped_messages') as prep:
        transport_class = AuthorizedCertificatesClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = AuthorizedCertificatesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = AuthorizedCertificatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(AuthorizedCertificatesClient, transports.AuthorizedCertificatesGrpcTransport), (AuthorizedCertificatesAsyncClient, transports.AuthorizedCertificatesGrpcAsyncIOTransport)])
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